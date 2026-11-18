//! Min-error LAMa in the sampled setting at t = 2 WITHOUT materialising the t = 2 tables.
//!
//! `CpModel::min_error_lns` needs every t = 2 table in memory (and a HashMap copy of it). That is
//! O(n^2 * |C|^2) rows: 1.4 billion for Shanghai at 50x50 / 30% queries, which OOMs at 240 GB, and
//! far more at 1-5% queries or in higher dimensions. Every row of those tables is just
//! `validate_candidate(pair, values)`, so this module runs the SAME search (LAMa t = 1 step, then
//! LNS steps that free a violated pair plus random records and re-solve them in Pumpkin) but asks
//! `validate_candidate` for a pair's cost when it needs it. Memory is O(n * |C|).
//!
//! Differences to `min_error_lns`, all on the search side (what counts as a valid database is
//! unchanged: every record's value in its t = 1 table, every pair within epsilon):
//!   * the t = 1 start is solved once and shared by every restart (it was recomputed identically
//!     per restart before, the Pumpkin brancher is deterministic);
//!   * a pair with BOTH records freed keeps at most `max_rows_per_pair` lowest-cost rows in the
//!     Pumpkin model (at 1% queries a pair can have ~10^5 valid rows);
//!   * an optional wall-clock deadline, after which restarts stop and report what they have;
//!   * with a deadline, a restart that is still outside epsilon after `steps_per_restart` steps is
//!     abandoned once it has gone `stall_steps` steps without improving, and new restarts (fresh seeds,
//!     past `restarts` if needed) keep starting until the deadline or `target_outputs` databases.
//!     Restarts that reach consistency are never abandoned (phase 2 is unchanged);
//!   * if Pumpkin finds no t = 1 start within 60 s, `start_failed` is set and main.rs reruns the whole
//!     pipeline (new query sample) instead of giving up.

use crate::LAMA::mini_solver::{CpModel, ScoredReconstruction};
use log::{debug, info, warn};
use pumpkin_solver::conflict_resolvers::resolvers::ResolutionResolver;
use pumpkin_solver::core::predicate;
use pumpkin_solver::core::predicates::Predicate;
use pumpkin_solver::core::results::{ProblemSolution, SatisfactionResultUnderAssumptions};
use pumpkin_solver::core::termination::TimeBudget;
use pumpkin_solver::core::variables::{DomainId, TransformableVariable};
use pumpkin_solver::{all_different, equals, Solver as PumpkinSolver};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::{BTreeSet, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

pub type Validator<'a> = dyn Fn(&[i64], &[i64]) -> (bool, f64) + Sync + 'a;

/// Search budget. Defaults match `CpModel::sampled_reconstructions`.
#[derive(Clone, Debug)]
pub struct LnsConfig {
    pub restarts: usize,
    pub steps_per_restart: usize,
    pub neighbourhood_size: usize,
    pub time_per_neighbourhood: Duration,
    pub max_rows_per_pair: usize,
    /// How many distinct databases to output (see `sample_neighbour`)
    pub target_outputs: usize,
    pub deadline: Option<Instant>,
    /// With a deadline: abandon a still-inconsistent restart after this many steps without improving
    pub stall_steps: usize,
}

impl Default for LnsConfig {
    fn default() -> Self {
        Self {
            restarts: 64,
            steps_per_restart: 1500,
            neighbourhood_size: 4,
            time_per_neighbourhood: Duration::from_millis(250),
            max_rows_per_pair: 1_000,
            target_outputs: 1000,
            deadline: None,
            stall_steps: 1500,
        }
    }
}

/// What the search found, plus how the TRUE database scores under the same objective.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, Default)]
pub struct LnsSummary {
    /// Databases written to the reconstruction file (sorted by total error, lowest first)
    pub returned: usize,
    pub target_outputs: usize,
    /// Restarts that reached a fully consistent database and then sampled around it
    pub restarts_consistent: usize,
    /// Of the restarts' final databases: how many have every pair within epsilon
    pub fully_consistent: usize,
    /// false = no restart reached a database with every pair within epsilon, so the output is the
    /// lowest-error databases found (NOT valid LAMa solutions)
    pub output_is_consistent: bool,
    pub best_total_error: i64,
    pub best_pairs_outside: i64,
    /// Same objective evaluated on the true database
    pub true_db_total_error: i64,
    pub true_db_pairs_outside: usize,
    /// true database missing from some record's t = 1 table (it could never be found)
    pub true_db_t1_missing: usize,
    pub true_db_in_output: bool,
    pub restarts_finished: usize,
    /// Restarts given up while still outside epsilon (no improvement in stall_steps steps)
    #[serde(default)]
    pub restarts_abandoned: usize,
    /// Pumpkin found no t = 1 start within 60 s (the pipeline is then restarted from scratch)
    #[serde(default)]
    pub start_failed: bool,
    /// Which run of the whole pipeline produced this result (1 = first; >1 after start failures)
    #[serde(default)]
    pub pipeline_attempt: usize,
    pub lns_steps: usize,
    pub hit_deadline: bool,
    pub lns_seconds: f64,
    pub max_rows_per_pair: usize,
}

pub struct PairLns<'a> {
    n: usize,
    /// var index -> encrypted record (its true value; what validate_candidate is keyed on)
    enc: Vec<i64>,
    /// var index -> values its t = 1 table allows
    candidates: Vec<Vec<i64>>,
    t1_cost: Vec<FxHashMap<i64, i64>>,
    validate: &'a Validator<'a>,
}

impl<'a> PairLns<'a> {
    pub const OUTSIDE_EPSILON: i64 = CpModel::OUTSIDE_EPSILON;

    /// `t1_model`: the model after `Translator::process_t1` (one singleton table per record).
    /// `enc_of_var[v]`: the encrypted record held by solver variable v.
    pub fn new(t1_model: &CpModel, enc_of_var: Vec<i64>, largest_val: i64, validate: &'a Validator<'a>) -> Self {
        let n = enc_of_var.len();
        let mut candidates: Vec<Vec<i64>> = vec![Vec::new(); n];
        let mut t1_cost: Vec<FxHashMap<i64, i64>> = vec![FxHashMap::default(); n];
        for tc in &t1_model.table_constraints {
            if let [var] = tc.vars[..] {
                for (row, &cost) in tc.values.iter().zip(&tc.costs) {
                    candidates[var].push(row[0]);
                    t1_cost[var].insert(row[0], cost);
                }
            }
        }
        for var in 0..n {
            if candidates[var].is_empty() {
                // Same fallback as min_error_lns: no t = 1 table -> the whole domain, cost 0
                warn!("Record {} has no t = 1 candidates, using the whole domain", enc_of_var[var]);
                candidates[var] = (0..=largest_val).collect();
                t1_cost[var] = candidates[var].iter().map(|&v| (v, 0)).collect();
            }
            candidates[var].sort_unstable();
            candidates[var].dedup();
        }
        Self { n, enc: enc_of_var, candidates, t1_cost, validate }
    }

    fn cost_of(err: f64) -> i64 {
        // Same scale as Translator (prob.powf(2.0) * 1_000_00.0)
        (err.powf(2.0) * 1_000_00.0).round() as i64
    }

    /// Cost of the pair (i, j) holding values (a, b), None if outside epsilon (or a == b).
    #[inline]
    fn pair_cost(&self, i: usize, j: usize, a: i64, b: i64) -> Option<i64> {
        if a == b {
            return None;
        }
        let (ok, err) = (self.validate)(&[self.enc[i], self.enc[j]], &[a, b]);
        ok.then(|| Self::cost_of(err))
    }

    #[inline]
    fn t1(&self, var: usize, value: i64) -> i64 {
        *self.t1_cost[var].get(&value).unwrap_or(&Self::OUTSIDE_EPSILON)
    }

    /// (total error, pairs outside epsilon per var) of a full database.
    fn score(&self, assignment: &[i64]) -> (i64, Vec<usize>) {
        let per_var: Vec<(i64, Vec<usize>)> = (0..self.n)
            .into_par_iter()
            .map(|i| {
                let mut err = self.t1(i, assignment[i]);
                let mut outside = Vec::new();
                for j in (i + 1)..self.n {
                    match self.pair_cost(i, j, assignment[i], assignment[j]) {
                        Some(c) => err += c,
                        None => {
                            err += Self::OUTSIDE_EPSILON;
                            outside.push(j);
                        }
                    }
                }
                (err, outside)
            })
            .collect();
        let mut viol = vec![0usize; self.n];
        let mut total = 0;
        for (i, (err, outside)) in per_var.into_iter().enumerate() {
            total += err;
            viol[i] += outside.len();
            for j in outside {
                viol[j] += 1;
            }
        }
        (total, viol)
    }

    pub fn run(&self, cfg: &LnsConfig) -> (Vec<ScoredReconstruction>, LnsSummary) {
        let started = Instant::now();
        let mut summary =
            LnsSummary { max_rows_per_pair: cfg.max_rows_per_pair, target_outputs: cfg.target_outputs, ..Default::default() };

        // How the true database scores: is it a valid LAMa solution at all?
        let truth: Vec<i64> = self.enc.clone();
        summary.true_db_t1_missing = (0..self.n).filter(|&v| !self.t1_cost[v].contains_key(&truth[v])).count();
        let (true_err, true_viol) = self.score(&truth);
        summary.true_db_total_error = true_err;
        summary.true_db_pairs_outside = true_viol.iter().sum::<usize>() / 2;
        info!(
            "True database: {} pairs outside epsilon, {} records outside their t = 1 table, total error {}",
            summary.true_db_pairs_outside, summary.true_db_t1_missing, true_err
        );

        // 1. LAMa t = 1 step, shared by every restart.
        let all: Vec<usize> = (0..self.n).collect();
        let empty = vec![-1i64; self.n];
        let Some(start_values) = self.solve_neighbourhood(&all, &empty, None, Duration::from_secs(60), cfg.max_rows_per_pair, true)
        else {
            // The caller restarts the whole pipeline (new query sample) when this is set
            warn!("Pumpkin found no all-different database from the t = 1 tables within 60 s: restart needed");
            summary.start_failed = true;
            summary.lns_seconds = started.elapsed().as_secs_f64();
            return (Vec::new(), summary);
        };
        info!("t = 1 start found {:.1}s after the LNS began", started.elapsed().as_secs_f64());
        let mut start = empty.clone();
        for (var, value) in start_values {
            start[var] = value;
        }
        let (start_err, start_viol) = self.score(&start);
        info!(
            "t = 1 start: {} pairs outside epsilon, starting {} restarts x {} LNS steps",
            start_viol.iter().sum::<usize>() / 2,
            cfg.restarts,
            cfg.steps_per_restart
        );

        let steps_done = AtomicUsize::new(0);
        let restarts_finished = AtomicUsize::new(0);
        let hit_deadline = std::sync::atomic::AtomicBool::new(false);
        let finals: Mutex<BTreeSet<(i64, Vec<i64>)>> = Mutex::new(BTreeSet::new());
        // Every distinct fully consistent database found (phase 2), with its total error
        let pool: Mutex<FxHashMap<Vec<i64>, i64>> = Mutex::new(FxHashMap::default());
        let restarts_consistent = AtomicUsize::new(0);
        let restarts_abandoned = AtomicUsize::new(0);
        let restarts_started = AtomicUsize::new(0);
        let attempts_per_restart = 20 * cfg.target_outputs.div_ceil(cfg.restarts.max(1)) + 50;
        let pb = indicatif::ProgressBar::new((cfg.restarts * cfg.steps_per_restart) as u64);
        let last_log = Mutex::new(Instant::now());

        let run_restart = |restart: usize| {
            let mut rng = rand::rngs::StdRng::seed_from_u64(restart as u64);
            let mut assignment = start.clone();
            let mut viol = start_viol.clone();
            let mut total = start_err;

            let mut step = 0usize;
            let mut steps_since_improvement = 0usize;
            loop {
                if cfg.deadline.is_some_and(|d| Instant::now() >= d) {
                    hit_deadline.store(true, Ordering::Relaxed);
                    break;
                }
                // enough consistent databases already: no need to keep searching
                if pool.lock().unwrap().len() >= cfg.target_outputs {
                    break;
                }
                // steps_per_restart steps; with a deadline, a restart still outside epsilon keeps
                // going until it is consistent or the time runs out
                if step >= cfg.steps_per_restart && (cfg.deadline.is_none() || total < Self::OUTSIDE_EPSILON) {
                    break;
                }
                // ...but one that has stopped improving is abandoned, so a fresh restart gets the thread
                if step >= cfg.steps_per_restart && steps_since_improvement >= cfg.stall_steps {
                    restarts_abandoned.fetch_add(1, Ordering::Relaxed);
                    info!(
                        "Restart {restart} abandoned after {step} steps: {} pairs outside epsilon, no improvement in {} steps",
                        total / Self::OUTSIDE_EPSILON,
                        steps_since_improvement
                    );
                    break;
                }
                step += 1;
                steps_since_improvement += 1;
                // Free a random violated pair (none left: purely random), topped up with random records.
                let violated_vars: Vec<usize> = (0..self.n).filter(|&v| viol[v] > 0).collect();
                let mut freed: Vec<usize> = Vec::new();
                if let Some(&v) = violated_vars.choose(&mut rng) {
                    let partners: Vec<usize> = (0..self.n)
                        .filter(|&u| u != v && self.pair_cost(v.min(u), v.max(u), assignment[v.min(u)], assignment[v.max(u)]).is_none())
                        .collect();
                    freed.push(v);
                    if let Some(&u) = partners.choose(&mut rng) {
                        freed.push(u);
                    }
                }
                let target = cfg.neighbourhood_size.min(self.n).max(freed.len());
                while freed.len() < target {
                    let v = rng.gen_range(0..self.n);
                    if !freed.contains(&v) {
                        freed.push(v);
                    }
                }

                let before = self.local_error(&freed, &assignment);
                if let Some(new_values) = self.solve_neighbourhood(
                    &freed,
                    &assignment,
                    Some(before.0),
                    cfg.time_per_neighbourhood,
                    cfg.max_rows_per_pair,
                    false,
                ) {
                    let old: Vec<(usize, i64)> = freed.iter().map(|&v| (v, assignment[v])).collect();
                    for (v, value) in new_values {
                        assignment[v] = value;
                    }
                    // Re-score exactly; keep the change only if it is better.
                    let after = self.local_error(&freed, &assignment);
                    if after.0 < before.0 {
                        steps_since_improvement = 0;
                        total += after.0 - before.0;
                        for &(i, j) in &before.1 {
                            viol[i] -= 1;
                            viol[j] -= 1;
                        }
                        for &(i, j) in &after.1 {
                            viol[i] += 1;
                            viol[j] += 1;
                        }
                    } else {
                        for (v, value) in old {
                            assignment[v] = value;
                        }
                    }
                }
                let done = steps_done.fetch_add(1, Ordering::Relaxed) + 1;
                pb.inc(1);
                // Plain log line about once a minute (the progress bar is hidden in slurm logs); run_me.sh status reads it
                let total_steps = cfg.restarts * cfg.steps_per_restart;
                let due = {
                    let mut last = last_log.lock().unwrap();
                    let due = last.elapsed() >= Duration::from_secs(60);
                    if due {
                        *last = Instant::now();
                    }
                    due
                };
                if due {
                    info!(
                        "LNS progress: {done}/{total_steps} steps ({:.0}%), {:.0}s in, this restart has {} pairs outside epsilon",
                        100.0 * done as f64 / total_steps as f64,
                        started.elapsed().as_secs_f64(),
                        total / Self::OUTSIDE_EPSILON
                    );
                }
            }
            if step > 0 {
                finals.lock().unwrap().insert((total, assignment.clone()));
            }

            // Phase 2: from a fully consistent database, walk to neighbouring fully consistent
            // databases (sample_neighbour) and keep every distinct one, until the pool is full.
            if total < Self::OUTSIDE_EPSILON {
                restarts_consistent.fetch_add(1, Ordering::Relaxed);
                pool.lock().unwrap().insert(assignment.clone(), total);
                for _ in 0..attempts_per_restart {
                    if pool.lock().unwrap().len() >= cfg.target_outputs
                        || cfg.deadline.is_some_and(|d| Instant::now() >= d)
                    {
                        break;
                    }
                    let mut freed: Vec<usize> = Vec::new();
                    while freed.len() < cfg.neighbourhood_size.min(self.n) {
                        let v = rng.gen_range(0..self.n);
                        if !freed.contains(&v) {
                            freed.push(v);
                        }
                    }
                    let Some(new_values) =
                        self.sample_neighbour(&freed, &assignment, &mut rng, cfg.time_per_neighbourhood, cfg.max_rows_per_pair)
                    else {
                        continue;
                    };
                    let before = self.local_error(&freed, &assignment);
                    let old: Vec<(usize, i64)> = freed.iter().map(|&v| (v, assignment[v])).collect();
                    for (v, value) in new_values {
                        assignment[v] = value;
                    }
                    let after = self.local_error(&freed, &assignment);
                    if after.1.is_empty() {
                        total += after.0 - before.0;
                        pool.lock().unwrap().insert(assignment.clone(), total);
                    } else {
                        // can't happen (every row/value is within epsilon), but never keep an invalid database
                        for (v, value) in old {
                            assignment[v] = value;
                        }
                    }
                }
            }
            restarts_finished.fetch_add(1, Ordering::Relaxed);
        };
        // One worker per thread pulls restart ids. Without a deadline: exactly cfg.restarts restarts (as
        // before). With one: new restarts keep starting until the deadline or the pool is full.
        let next_restart = AtomicUsize::new(0);
        (0..rayon::current_num_threads().max(1)).into_par_iter().for_each(|_| loop {
            if pool.lock().unwrap().len() >= cfg.target_outputs || cfg.deadline.is_some_and(|d| Instant::now() >= d) {
                break;
            }
            let restart = next_restart.fetch_add(1, Ordering::Relaxed);
            if cfg.deadline.is_none() && restart >= cfg.restarts {
                break;
            }
            restarts_started.fetch_add(1, Ordering::Relaxed);
            run_restart(restart);
        });
        pb.finish();
        if cfg.deadline.is_some_and(|d| Instant::now() >= d) {
            hit_deadline.store(true, Ordering::Relaxed);
        }
        // every restart started too late to take a step: report the t = 1 start
        if finals.lock().unwrap().is_empty() {
            finals.lock().unwrap().insert((start_err, start.clone()));
        }

        let finals: Vec<ScoredReconstruction> = finals
            .into_inner()
            .unwrap()
            .into_iter()
            .map(|(total_error, assignment)| ScoredReconstruction { assignment, total_error })
            .collect();

        summary.restarts_finished = restarts_finished.load(Ordering::Relaxed);
        summary.restarts_abandoned = restarts_abandoned.load(Ordering::Relaxed);
        info!(
            "{} restarts started, {} abandoned (stalled), {} reached consistency",
            restarts_started.load(Ordering::Relaxed),
            summary.restarts_abandoned,
            restarts_consistent.load(Ordering::Relaxed)
        );
        summary.lns_steps = steps_done.load(Ordering::Relaxed);
        summary.hit_deadline = hit_deadline.load(Ordering::Relaxed);
        summary.fully_consistent = finals.iter().filter(|r| r.total_error < Self::OUTSIDE_EPSILON).count();
        if let Some(best) = finals.first() {
            summary.best_total_error = best.total_error;
            summary.best_pairs_outside = best.total_error / Self::OUTSIDE_EPSILON;
        }

        summary.restarts_consistent = restarts_consistent.load(Ordering::Relaxed);

        // Valid LAMa solutions only (lowest error first), unless there are none: then the
        // restarts' lowest-error databases, flagged.
        let pool = pool.into_inner().unwrap();
        summary.output_is_consistent = !pool.is_empty();
        let mut out: Vec<ScoredReconstruction> = if summary.output_is_consistent {
            let mut v: Vec<ScoredReconstruction> =
                pool.into_iter().map(|(assignment, total_error)| ScoredReconstruction { assignment, total_error }).collect();
            v.sort_by(|a, b| a.total_error.cmp(&b.total_error).then_with(|| a.assignment.cmp(&b.assignment)));
            v
        } else {
            warn!("No restart reached a database with every pair within epsilon; returning the lowest-error ones");
            finals
        };
        out.truncate(cfg.target_outputs);
        summary.returned = out.len();
        summary.true_db_in_output = out.iter().any(|r| r.assignment == truth);
        summary.lns_seconds = started.elapsed().as_secs_f64();
        info!("Pair LNS: {summary:?}");
        (out, summary)
    }

    /// Error of every t-tuple touching a freed record (their t = 1 tables and every pair with a
    /// freed record), and the pairs among those that are outside epsilon.
    fn local_error(&self, freed: &[usize], assignment: &[i64]) -> (i64, Vec<(usize, usize)>) {
        let freed_set: HashSet<usize> = freed.iter().copied().collect();
        let mut err = 0;
        let mut outside = Vec::new();
        for &f in freed {
            err += self.t1(f, assignment[f]);
            for u in 0..self.n {
                // a pair with two freed records counts once
                if u == f || (freed_set.contains(&u) && u < f) {
                    continue;
                }
                let (i, j) = (f.min(u), f.max(u));
                match self.pair_cost(i, j, assignment[i], assignment[j]) {
                    Some(c) => err += c,
                    None => {
                        err += Self::OUTSIDE_EPSILON;
                        outside.push((i, j));
                    }
                }
            }
        }
        (err, outside)
    }

    /// Phase 2 step: re-assign the `freed` records to values that are DIFFERENT from their current
    /// ones (at least one record changes) such that every t-tuple touching them stays within
    /// epsilon. `assignment` must be fully consistent; then so is the result. Values are tried in
    /// a random order (each record's domain is shuffled), so repeated calls walk randomly over the
    /// consistent databases near `assignment`. None if there is no such neighbour (or no time).
    fn sample_neighbour(
        &self,
        freed: &[usize],
        assignment: &[i64],
        rng: &mut rand::rngs::StdRng,
        time_limit: Duration,
        max_rows_per_pair: usize,
    ) -> Option<Vec<(usize, i64)>> {
        let freed_set: HashSet<usize> = freed.iter().copied().collect();
        let taken: HashSet<i64> = (0..self.n).filter(|v| !freed_set.contains(v)).map(|v| assignment[v]).collect();

        // Domain of each freed record: t = 1 values, not held by a fixed record, and within epsilon
        // with EVERY fixed record. Shuffled; the Pumpkin variable is an index into it.
        let mut domain_of: FxHashMap<usize, Vec<i64>> = FxHashMap::default();
        for &v in freed {
            let mut d: Vec<i64> = self.candidates[v]
                .iter()
                .copied()
                .filter(|a| !taken.contains(a))
                .filter(|&a| {
                    (0..self.n).filter(|u| *u != v && !freed_set.contains(u)).all(|u| {
                        let (i, j, x, y) = if v < u { (v, u, a, assignment[u]) } else { (u, v, assignment[u], a) };
                        self.pair_cost(i, j, x, y).is_some()
                    })
                })
                .collect();
            if d.is_empty() {
                return None;
            }
            d.shuffle(rng);
            domain_of.insert(v, d);
        }

        let mut solver = PumpkinSolver::default();
        let idx_var: FxHashMap<usize, DomainId> = freed
            .iter()
            .map(|&v| (v, solver.new_bounded_integer(0, domain_of[&v].len() as i32 - 1)))
            .collect();
        let pos = |v: usize, value: i64| domain_of[&v].iter().position(|&x| x == value).map(|p| p as i32);

        // all-different on the values: no two freed records may take the same value
        let tag = solver.new_constraint_tag();
        for (x, &f) in freed.iter().enumerate() {
            for &g in &freed[x + 1..] {
                for (pf, &a) in domain_of[&f].iter().enumerate() {
                    if let Some(pg) = pos(g, a) {
                        solver
                            .add_clause(vec![predicate![idx_var[&f] != pf as i32], predicate![idx_var[&g] != pg]], tag)
                            .ok()?;
                    }
                }
            }
        }
        // Each pair of freed records takes one of its within-epsilon rows (lowest cost first if capped)
        for (x, &f) in freed.iter().enumerate() {
            for &g in &freed[x + 1..] {
                let (i, j) = (f.min(g), f.max(g));
                let mut rows: Vec<(i32, i32, i64)> = Vec::new();
                for (pi, &a) in domain_of[&i].iter().enumerate() {
                    for (pj, &b) in domain_of[&j].iter().enumerate() {
                        if let Some(c) = self.pair_cost(i, j, a, b) {
                            rows.push((pi as i32, pj as i32, c));
                        }
                    }
                }
                if rows.is_empty() {
                    return None;
                }
                if rows.len() > max_rows_per_pair {
                    rows.select_nth_unstable_by_key(max_rows_per_pair, |r| r.2);
                    rows.truncate(max_rows_per_pair);
                }
                let mut chosen = Vec::with_capacity(rows.len());
                for &(pi, pj, _) in &rows {
                    let row = solver.new_literal();
                    solver.add_clause(vec![row.get_false_predicate(), predicate![idx_var[&i] == pi]], tag).ok()?;
                    solver.add_clause(vec![row.get_false_predicate(), predicate![idx_var[&j] == pj]], tag).ok()?;
                    chosen.push(row.get_true_predicate());
                }
                solver.add_clause(chosen, tag).ok()?;
            }
        }
        // ... and it is not the current database
        let mut differs = Vec::new();
        for &v in freed {
            if let Some(p) = pos(v, assignment[v]) {
                differs.push(predicate![idx_var[&v] != p]);
            } else {
                return None; // current value not in the domain: assignment wasn't consistent
            }
        }
        solver.add_clause(differs, tag).ok()?;

        let mut brancher = solver.default_brancher();
        let mut resolver = ResolutionResolver::default();
        let mut termination = TimeBudget::starting_now(time_limit);
        let result = match solver.satisfy_under_assumptions(&mut brancher, &mut termination, &mut resolver, &[]) {
            SatisfactionResultUnderAssumptions::Satisfiable(found) => {
                let s = found.solution();
                Some(freed.iter().map(|&v| (v, domain_of[&v][s.get_integer_value(idx_var[&v]) as usize])).collect())
            }
            _ => None,
        };
        result
    }

    /// One LNS step in Pumpkin (same encoding as CpModel::solve_neighbourhood): re-assign the
    /// `freed` records, everything else keeps its value. With `t1_only`, only the t = 1 tables
    /// score (the LAMa t = 1 step). Returns the new values if Pumpkin found an error strictly below
    /// `error_before` (with None: any solution, then it keeps improving until the time runs out).
    fn solve_neighbourhood(
        &self,
        freed: &[usize],
        assignment: &[i64],
        error_before: Option<i64>,
        time_limit: Duration,
        max_rows_per_pair: usize,
        t1_only: bool,
    ) -> Option<Vec<(usize, i64)>> {
        let mut solver = PumpkinSolver::default();
        let freed_set: HashSet<usize> = freed.iter().copied().collect();
        let taken: HashSet<i64> = (0..self.n).filter(|v| !freed_set.contains(v)).map(|v| assignment[v]).collect();

        // Domains: t = 1 candidates (plus the current value) minus values fixed records hold.
        let mut domain_of: FxHashMap<usize, Vec<i64>> = FxHashMap::default();
        let mut record_var: FxHashMap<usize, DomainId> = FxHashMap::default();
        for &v in freed {
            let mut d: Vec<i64> = self.candidates[v].iter().copied().filter(|x| !taken.contains(x)).collect();
            if assignment[v] >= 0 {
                d.push(assignment[v]);
            }
            d.sort_unstable();
            d.dedup();
            if d.is_empty() {
                return None;
            }
            record_var.insert(v, solver.new_sparse_integer(d.iter().map(|&x| x as i32).collect::<Vec<_>>()));
            domain_of.insert(v, d);
        }
        let tag = solver.new_constraint_tag();
        solver
            .add_constraint(all_different(freed.iter().map(|v| record_var[v]).collect::<Vec<_>>(), tag))
            .post()
            .ok()?;

        // Per freed record and value: t = 1 cost + every pair with a FIXED record (None = outside).
        let mut value_terms: Vec<(usize, i64, i64, usize)> = Vec::new(); // (var, value, in-eps cost, #outside)
        let mut max_cost: i64 = 0;
        for &v in freed {
            for &a in &domain_of[&v] {
                let mut cost = self.t1(v, a);
                let mut outside = 0usize;
                if !t1_only {
                    for u in 0..self.n {
                        if u == v || freed_set.contains(&u) {
                            continue;
                        }
                        let (i, j, x, y) = if v < u { (v, u, a, assignment[u]) } else { (u, v, assignment[u], a) };
                        match self.pair_cost(i, j, x, y) {
                            Some(c) => {
                                max_cost = max_cost.max(c);
                                cost += c;
                            }
                            None => outside += 1,
                        }
                    }
                }
                value_terms.push((v, a, cost, outside));
            }
        }

        // Pairs with both records freed: their valid rows (at most max_rows_per_pair, lowest cost).
        let mut pair_rows: Vec<((usize, usize), Vec<(i64, i64, i64)>)> = Vec::new();
        if !t1_only {
            for (x, &f) in freed.iter().enumerate() {
                for &g in &freed[x + 1..] {
                    let (i, j) = (f.min(g), f.max(g));
                    let mut rows: Vec<(i64, i64, i64)> = domain_of[&i]
                        .par_iter()
                        .flat_map_iter(|&a| {
                            domain_of[&j].iter().filter_map(move |&b| self.pair_cost(i, j, a, b).map(|c| (a, b, c)))
                        })
                        .collect();
                    if rows.len() > max_rows_per_pair {
                        rows.select_nth_unstable_by_key(max_rows_per_pair, |r| r.2);
                        rows.truncate(max_rows_per_pair);
                    }
                    for r in &rows {
                        max_cost = max_cost.max(r.2);
                    }
                    pair_rows.push(((i, j), rows));
                }
            }
        }

        // A pair outside epsilon costs more than any single in-epsilon row (kept small so Pumpkin's
        // i32 sum can't overflow; the caller re-scores exactly with OUTSIDE_EPSILON).
        let penalty: i64 = max_cost + 1;
        let mut error_terms = Vec::new();
        let mut max_error: i64 = 0;
        let mut per_var_max: FxHashMap<usize, i64> = FxHashMap::default();
        for &(v, a, cost, outside) in &value_terms {
            let c = cost.min(Self::OUTSIDE_EPSILON) + outside as i64 * penalty; // t1 outside can't happen: domains are t1 values
            let e = per_var_max.entry(v).or_insert(0);
            *e = (*e).max(c);
            if c > 0 {
                let tag = solver.new_constraint_tag();
                let lit = solver.new_literal_for_predicate(predicate![record_var[&v] == a as i32], tag);
                error_terms.push(lit.get_integer_variable().scaled(i32::try_from(c).ok()?));
            }
        }
        max_error += per_var_max.values().sum::<i64>();
        for ((i, j), rows) in &pair_rows {
            let tag = solver.new_constraint_tag();
            let mut chosen = Vec::with_capacity(rows.len() + 1);
            for &(a, b, c) in rows {
                let row = solver.new_literal();
                solver.add_clause(vec![row.get_false_predicate(), predicate![record_var[i] == a as i32]], tag).ok()?;
                solver.add_clause(vec![row.get_false_predicate(), predicate![record_var[j] == b as i32]], tag).ok()?;
                if c > 0 {
                    error_terms.push(row.get_integer_variable().scaled(c as i32));
                }
                chosen.push(row.get_true_predicate());
            }
            let outside = solver.new_literal();
            error_terms.push(outside.get_integer_variable().scaled(penalty as i32));
            chosen.push(outside.get_true_predicate());
            solver.add_clause(chosen, tag).ok()?;
            max_error += penalty;
        }
        if max_error >= i32::MAX as i64 {
            debug!("Neighbourhood objective could overflow i32 ({max_error}), skipping");
            return None;
        }
        let total_error = solver.new_bounded_integer(0, max_error as i32);
        error_terms.push(total_error.scaled(-1));
        let tag = solver.new_constraint_tag();
        solver.add_constraint(equals(error_terms, 0, tag)).post().ok()?;

        let to_pumpkin = |caller: i64| (caller / Self::OUTSIDE_EPSILON) * penalty + caller % Self::OUTSIDE_EPSILON;
        let mut brancher = solver.default_brancher();
        let mut resolver = ResolutionResolver::default();
        let mut termination = TimeBudget::starting_now(time_limit);
        let mut bound = error_before.map(to_pumpkin);
        let mut best = None;
        loop {
            let assumptions: Vec<Predicate> = match bound {
                Some(b) if b <= 0 => return best,
                Some(b) => vec![predicate![total_error <= (b - 1) as i32]],
                None => vec![],
            };
            match solver.satisfy_under_assumptions(&mut brancher, &mut termination, &mut resolver, &assumptions) {
                SatisfactionResultUnderAssumptions::Satisfiable(found) => {
                    let s = found.solution();
                    bound = Some(s.get_integer_value(total_error) as i64);
                    best = Some(freed.iter().map(|&v| (v, s.get_integer_value(record_var[&v]) as i64)).collect());
                }
                _ => return best,
            }
        }
    }
}
