use crate::dataloader::unflatten_nd;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use itertools::Itertools;
use log::{debug, info, warn};
use pumpkin_solver::conflict_resolvers::resolvers::ResolutionResolver;
use pumpkin_solver::core::results::solution_iterator::IteratedSolution;
use pumpkin_solver::core::results::ProblemSolution;
use pumpkin_solver::core::termination::Indefinite;
use pumpkin_solver::{all_different, equals, table, Solver as PumpkinSolver};
use rayon::prelude::*;
use serde::Serialize;
use std::collections::HashMap;
use std::env;
use std::env::var;
use std::fs::File;
use std::io::Write;
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::thread;
use rand::{Rng};
use pumpkin_solver::core::predicate;
use pumpkin_solver::core::predicates::Predicate;
use pumpkin_solver::core::results::SatisfactionResultUnderAssumptions;
use pumpkin_solver::core::termination::TimeBudget;
use pumpkin_solver::core::variables::{DomainId, TransformableVariable};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::collections::HashSet;
use std::time::Duration;
use std::collections::BTreeSet;
use std::sync::Mutex;


/// One reconstruction found by `CpModel::min_error_lns`, with its total matching error.
#[derive(Clone, Debug)]
pub struct ScoredReconstruction {
    /// assignment[var_index] = plaintext value given to that record
    pub assignment: Vec<i64>,
    /// Sum over every t-tuple of its cost (squared frequency error), or `OUTSIDE_EPSILON`
    /// for a t-tuple whose values are not in its table. Below `OUTSIDE_EPSILON` = every
    /// t-tuple is within epsilon, i.e. a database plain LAMa would also accept.
    pub total_error: i64,
}

/// A mock for the original cp_sat IntVar
#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
pub struct IntVar(pub usize);

/// Our drop-in replacement for CpModelProto.
/// It holds the global state of the model as it builds up across t-rounds.
#[derive(Clone, Debug, Default)]
pub struct CpModel {
    pub num_vars: usize,
    pub all_different: bool,
    pub table_constraints: Vec<TableConstraint>,
    run_id: String,
    pub limit: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct TableConstraint {
    pub vars: Vec<usize>,
    pub values: Vec<Vec<i64>>,
    pub costs: Vec<i64>,
}

#[derive(Serialize)]
pub struct ConstraintMeta {
    pub var_ids: Vec<usize>,
    pub start_idx: usize,
    pub length: usize,
    pub tuple_size: usize,
}

impl CpModel {
    pub fn new(limit: Option<usize>) -> Self {
        // Generate a unique ID using the system clock (no extra crates needed)
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
            .to_string();

        Self {
            run_id: id,
            limit,
            ..Default::default()
        }
    }


    // Runs a quick debug check to make sure that what you're doing is actually going to complete
    // (i.e. that the true solution exists in here)
    pub fn validate(&self, var_index_map: &HashMap<IntVar, (i32, i64)>) {
        let mut found_tracker = HashMap::new();
        let mut found_all = true;
        let mut optimal_cost = 0;

        // Initialize the tracker
        for tc in &self.table_constraints {
            for vars in tc.vars.iter() {
                let orig_val = var_index_map.get(&(IntVar(vars.clone()))).unwrap().1;
                found_tracker.insert(orig_val, false);
            }
        }

        // Validate cohesive rows and calculate true cost
        for tc in &self.table_constraints {
            let mut target_tuple = Vec::with_capacity(tc.vars.len());
            for var_idx in &tc.vars {
                let orig_val = var_index_map.get(&(IntVar(*var_idx))).unwrap().1;
                target_tuple.push(orig_val);
            }

            let mut found_matching_row = false;

            for (row_idx, assignment) in tc.values.iter().enumerate() {
                if assignment == &target_tuple {
                    found_matching_row = true;
                    optimal_cost += tc.costs[row_idx];
                    for val in assignment {
                        found_tracker.insert(*val, true);
                    }
                    break;
                }
            }

            if !found_matching_row {
                warn!(
                    "Table constraint for vars {:?} does not contain the true assignment {:?}",
                    tc.vars, target_tuple
                );
            }
        }

        for (k, v) in &found_tracker {
            if !v {
                found_all = false;
                warn!("There is no valid assignment for {k}!! Solver is VERY LIKELY to crash.");
            }
        }

        if found_all {
            debug!("Every variable has at least 1 valid constraint! (They may still contradict...)")
        }
        debug!("Value for optimal cost was: {}!!", optimal_cost);
    }

    /// Mirrors the CP-SAT solve_with_parameters logic
    /// Replaces the Python Subprocess with a pure Rust native parallel solver
    pub fn solve(&self, largest_val: i64, get_one: bool) -> Option<Vec<Vec<i64>>> {
        // Sampled setting (limit is only Some when !perfect): best-fitting databases, no enumeration
        if self.limit.is_some() {
            let best = self.sampled_reconstructions(largest_val);
            return if best.is_empty() { None } else { Some(best) };
        }


        let largest_val_i32 = largest_val as i32;

        let all_solutions = if get_one {
            // Single thread, break on first result
            Self::solve_partition(self, largest_val_i32, None, true)
        } else {
            // Split the search space across 64 cores by partitioning the root variable
            let root_domain: Vec<i32> = (0..=largest_val_i32).collect();

            let pb = ProgressBar::new(root_domain.len() as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {bar:40.cyan/blue} {percent}% {msg} (ETA: {eta_precise})")
                    .unwrap()
            );

            let sols: Vec<Vec<i64>> = root_domain
                .into_par_iter()
                .progress_with(pb) // Attach the progress bar here
                .flat_map(|fixed_val| {
                    Self::solve_partition(self, largest_val_i32, Some(fixed_val), false)
                })
                .collect();
            sols
        };

        if all_solutions.is_empty() {
            None
        } else {
            Some(all_solutions)
        }
    }

    /// Internal function to build and solve a localized Pumpkin instance in memory.
    fn solve_partition(
        model: &CpModel,
        largest_val: i32,
        fixed_val_for_var_0: Option<i32>,
        get_one: bool,
    ) -> Vec<Vec<i64>> {
        let mut solver = PumpkinSolver::default();

        // 1. Initialize variables (0..=largest_val)
        let mut solver_vars = Vec::with_capacity(model.num_vars);
        for _ in 0..model.num_vars {
            solver_vars.push(solver.new_bounded_integer(0, largest_val));
        }

        // 2. Lock root variable for parallel space partitioning
        if let Some(fixed_val) = fixed_val_for_var_0 {
            if model.num_vars > 0 {
                let tag = solver.new_constraint_tag();
                solver.add_constraint(
                    equals(vec![solver_vars[0]], fixed_val, tag)
                ).post();
            }
        }

        // 3. AllDifferent global constraint
        if model.all_different && model.num_vars > 0 {
            let tag = solver.new_constraint_tag();
            solver.add_constraint(all_different(solver_vars.clone(), tag)).post();
        }

        // 4. Allowed Assignments (Table Constraints)
        for tc in &model.table_constraints {
            let vars: Vec<_> = tc.vars.iter().map(|&idx| solver_vars[idx]).collect();

            // Cast values to i32 for Pumpkin, discarding the costs array entirely
            let tuples: Vec<Vec<i32>> = tc.values.iter()
                .map(|t| t.iter().map(|&v| v as i32).collect())
                .collect();

            let tag = solver.new_constraint_tag();
            solver.add_constraint(table(vars, tuples, tag)).post();
        }

        // 5. Execute search using Pumpkin's native Iterator
        let mut solutions = Vec::new();
        let mut brancher = solver.default_brancher();
        let mut termination = Indefinite;
        let mut resolver = ResolutionResolver::default();

        let mut solution_iterator = solver.get_solution_iterator(
            &mut brancher,
            &mut termination,
            &mut resolver,
        );

        loop {
            match solution_iterator.next_solution() {
                IteratedSolution::Solution(solution, _, _, _) => {
                    // Reconstruct the array of assigned values
                    let current_assignment: Vec<i64> = solver_vars.iter()
                        .map(|&v| solution.get_integer_value(v) as i64)
                        .collect();

                    solutions.push(current_assignment);

                    if get_one || solutions.len() >= model.limit.unwrap_or(usize::MAX) {
                        break;
                    }
                }
                IteratedSolution::Finished | IteratedSolution::Unknown | IteratedSolution::Unsatisfiable => {
                    break;
                }
            }
        }

        solutions
    }

    /// Penalty for a t-tuple whose values fell outside epsilon (not a row in its table).
    pub const OUTSIDE_EPSILON: i64 = 10_000_000;
    // true: top-k over every database any restart visits; false: only each restart's final database
    pub const KEEP_EVERY_VISITED: bool = false;

    /// Sampled setting: the top-10 lowest-error databases from `min_error_lns`, keeping only those
    /// where every t-tuple is within epsilon (so they are valid LAMa solutions).
    pub fn sampled_reconstructions(&self, largest_val: i64) -> Vec<Vec<i64>> {
        debug!("About to start trying to find solutions in the imperfect world...");
        self.min_error_lns(largest_val, 250, 50, 2, Duration::from_millis(300), 1000)
            .into_iter()
            .filter(|r| r.total_error < Self::OUTSIDE_EPSILON)
            .map(|r| r.assignment)
            .collect()
    }

    /// Min-error LAMa: Large Neighbourhood Search (LNS) where every step is a small Pumpkin solve.
    ///
    /// # Arguments
    ///
    /// * `largest_val` - Largest encoded plaintext value (`0..=largest_val`); fallback candidates
    ///   for a record with no t = 1 table.
    /// * `restarts` - Number of independent searches, run in parallel (Rayon), seeded by index.
    /// * `neighbourhoods_per_restart` - Number of LNS steps per restart.
    /// * `neighbourhood_size` - Number of records freed (re-solved by Pumpkin) per LNS step.
    /// * `time_per_neighbourhood` - Pumpkin time limit for one LNS step.
    /// * `top_k` - Maximum number of distinct, lowest-error reconstructions to return.
    ///
    /// # Behaviour
    ///
    /// 1. **Set-up**: per t-tuple a lookup values -> cost, per record the t-tuples it is in,
    ///    and per record its t = 1 candidates (values within epsilon for that record alone).
    /// 2. **LAMa t = 1 step**: Pumpkin picks an all-different database from the t = 1 tables
    ///    (lowest t = 1 error it finds within the time limit). Its t = 2 tuples can still be
    ///    outside epsilon; those cost `OUTSIDE_EPSILON` and LNS removes them.
    /// 3. **LNS step**: free `neighbourhood_size` random records and fix the rest. Pumpkin gets
    ///    only the t-tuples touching a freed record, with each table cut down to the rows that
    ///    agree with the fixed records, and finds the lowest-error re-assignment of the freed
    ///    records (all-different, t-tuples within epsilon unless that is impossible). Kept only
    ///    if the total error goes down.
    /// 4. **Collect**: sort restarts by total error, drop duplicate databases, keep `top_k`.
    ///
    /// Each LNS step is a few thousand rows, so it never builds the full model (millions of
    /// rows untruncated) that makes plain enumeration blow up in the sampled setting.
    pub fn min_error_lns(
        &self,
        largest_val: i64,
        restarts: usize,
        neighbourhoods_per_restart: usize,
        neighbourhood_size: usize,
        time_per_neighbourhood: Duration,
        top_k: usize,
    ) -> Vec<ScoredReconstruction> {
        // 1. Set-up.
        // One map per t-tuple: candidate values (a row of its table) -> its cost (squared error).
        let t_tuple_costs: Vec<HashMap<Vec<i64>, i64>> = self
            .table_constraints
            .iter()
            .map(|tc| tc.values.iter().cloned().zip(tc.costs.iter().copied()).collect())
            .collect();

        // For each record (var): the t-tuples it appears in.
        let mut t_tuples_of_var: Vec<Vec<usize>> = vec![Vec::new(); self.num_vars];
        for (t_tuple_idx, tc) in self.table_constraints.iter().enumerate() {
            for &var in &tc.vars {
                t_tuples_of_var[var].push(t_tuple_idx);
            }
        }

        // For each record: the values its t = 1 table allows (fallback: the whole domain).
        let candidate_values: Vec<Vec<i64>> = (0..self.num_vars)
            .map(|var| {
                let mut values: Vec<i64> = self
                    .table_constraints
                    .iter()
                    .filter(|tc| tc.vars == [var])
                    .flat_map(|tc| tc.values.iter().map(|row| row[0]))
                    .collect();
                if values.is_empty() {
                    values = (0..=largest_val).collect();
                }
                values
            })
            .collect();
        let singleton_t_tuples: Vec<usize> =
            (0..self.table_constraints.len()).filter(|&i| self.table_constraints[i].vars.len() == 1).collect();

        // Cost of ONE t-tuple under a full database.
        let t_tuple_error = |assignment: &[i64], t_tuple_idx: usize| -> i64 {
            let values: Vec<i64> =
                self.table_constraints[t_tuple_idx].vars.iter().map(|&var| assignment[var]).collect();
            *t_tuple_costs[t_tuple_idx].get(&values).unwrap_or(&Self::OUTSIDE_EPSILON)
        };

        let pb = ProgressBar::new((restarts * neighbourhoods_per_restart) as u64).with_style(
            ProgressStyle::with_template("{bar:40} {pos}/{len} [{elapsed_precise}<{eta_precise}]").unwrap(),
        );


        // Running top-k over every database any restart visits: (total_error, assignment), worst last.
        let best_seen: Mutex<BTreeSet<(i64, Vec<i64>)>> = Mutex::new(BTreeSet::new());
        // Offer one database to the top-k; it is only cloned if it beats the current worst.
        let offer = |assignment: &[i64], total_error: i64| {
            if total_error >= Self::OUTSIDE_EPSILON { return; } // a t-tuple is outside epsilon: not a LAMa solution
            let mut best = best_seen.lock().unwrap();
            if best.len() >= top_k && total_error >= best.last().map_or(i64::MAX, |w| w.0) { return; }
            best.insert((total_error, assignment.to_vec())); // BTreeSet drops exact duplicates
            if best.len() > top_k { best.pop_last(); }
        };

        (0..restarts).into_par_iter().for_each(|restart| {
                let mut rng = rand::rngs::StdRng::seed_from_u64(restart as u64);
                let all_records: Vec<usize> = (0..self.num_vars).collect();

                // 2. LAMa t = 1 step: every record free, only the t = 1 tables.
                let mut assignment = vec![-1i64; self.num_vars];
            let Some((start_values, _)) = self.solve_neighbourhood(
                    &all_records,
                    &assignment,
                    &singleton_t_tuples,
                    &candidate_values,
                    None,
                    Duration::from_secs(5),
                ) else { return; };
                for (var, value) in start_values {
                    assignment[var] = value;
                }
                let mut total_error: i64 =
                    (0..self.table_constraints.len()).map(|t| t_tuple_error(&assignment, t)).sum();
                if Self::KEEP_EVERY_VISITED { offer(&assignment, total_error); }

                // 3. LNS steps.
                for _ in 0..neighbourhoods_per_restart {
                    let freed: Vec<usize> = all_records
                        .choose_multiple(&mut rng, neighbourhood_size.min(self.num_vars))
                        .copied()
                        .collect();
                    // Only t-tuples touching a freed record can change cost.
                    let mut touched: Vec<usize> =
                        freed.iter().flat_map(|&var| t_tuples_of_var[var].iter().copied()).collect();
                    touched.sort_unstable();
                    touched.dedup(); // a t-tuple with two freed records counts once
                    let error_before: i64 = touched.iter().map(|&t| t_tuple_error(&assignment, t)).sum();

                    if let Some((new_values, _)) = self.solve_neighbourhood(
                        &freed,
                        &assignment,
                        &touched,
                        &candidate_values,
                        Some(error_before),
                        time_per_neighbourhood,
                    ) {
                        let old_values: Vec<(usize, i64)> = freed.iter().map(|&var| (var, assignment[var])).collect();
                        for (var, value) in new_values {
                            assignment[var] = value;
                        }
                        // Re-score exactly from the lookups; keep the change only if it is better.
                        let error_after: i64 = touched.iter().map(|&t| t_tuple_error(&assignment, t)).sum();
                        if error_after < error_before {
                            total_error += error_after - error_before;
                            if Self::KEEP_EVERY_VISITED { offer(&assignment, total_error); }
                        } else {
                            for (var, value) in old_values {
                                assignment[var] = value;
                            }
                        }
                    }
                    pb.inc(1);
                }
            // Each restart's final database (already offered above if KEEP_EVERY_VISITED; the set drops the duplicate)
            offer(&assignment, total_error);
        });
        pb.finish();


        best_seen.into_inner().unwrap().into_iter()
            .map(|(total_error, assignment)| ScoredReconstruction { assignment, total_error })
            .collect()
    }

    /// One LNS step in Pumpkin: re-assign the `freed` records, every other record keeps its value
    /// in `assignment`, scoring only the `touched` t-tuples.
    ///
    /// Returns the new (record, value) pairs and their error over `touched`, if Pumpkin found one
    /// strictly below `error_before` (with `None`, any solution counts, then it keeps improving).
    fn solve_neighbourhood(
        &self,
        freed: &[usize],
        assignment: &[i64],
        touched: &[usize],
        candidate_values: &[Vec<i64>],
        error_before: Option<i64>,
        time_limit: Duration,
    ) -> Option<(Vec<(usize, i64)>, i64)> {
        let mut solver = PumpkinSolver::default();
        let freed_set: HashSet<usize> = freed.iter().copied().collect();
        // Values the fixed records already hold: off-limits for freed records (all-different).
        let taken: HashSet<i64> =
            (0..self.num_vars).filter(|v| !freed_set.contains(v)).map(|v| assignment[v]).collect();

        // A Pumpkin variable per freed record: its t = 1 candidates (plus its current value,
        // so the current database is always a solution), minus the taken values.
        let mut record_var: HashMap<usize, DomainId> = HashMap::new();
        let mut domain_of: HashMap<usize, HashSet<i64>> = HashMap::new();
        for &var in freed {
            let mut domain: Vec<i64> =
                candidate_values[var].iter().copied().filter(|v| !taken.contains(v)).collect();
            if assignment[var] >= 0 {
                domain.push(assignment[var]);
            }
            domain.sort_unstable();
            domain.dedup();
            record_var.insert(var, solver.new_sparse_integer(domain.iter().map(|&v| v as i32).collect::<Vec<_>>()));
            domain_of.insert(var, domain.into_iter().collect());
        }
        let tag = solver.new_constraint_tag();
        solver.add_constraint(all_different(freed.iter().map(|v| record_var[v]).collect::<Vec<_>>(), tag)).post().ok()?;

        // A t-tuple outside epsilon costs more than any single in-epsilon row. Kept small so
        // Pumpkin's i32 sum can't overflow (t = 3 touches thousands of t-tuples); the caller
        // re-scores exactly with OUTSIDE_EPSILON and only keeps real improvements.
        let outside_penalty: i64 = touched
            .iter()
            .map(|&t| self.table_constraints[t].costs.iter().copied().max().unwrap_or(0))
            .max()
            .unwrap_or(0)
            + 1;

        // Terms of: sum(cost * row_selected) + outside_penalty * outside - total_error = 0
        let mut error_terms = Vec::new();
        let mut max_error: i64 = 0;
        // A t-tuple with only ONE freed record just scores that record's value, so instead of
        // table rows its cost is added to value_cost[record][value] (much smaller model).
        let mut value_cost: HashMap<usize, HashMap<i64, i64>> =
            freed.iter().map(|&var| (var, domain_of[&var].iter().map(|&v| (v, 0)).collect())).collect();
        for &t_tuple_idx in touched {
            let tc = &self.table_constraints[t_tuple_idx];
            max_error += outside_penalty;
            // Rows of this t-tuple's table that agree with the fixed records and fit the
            // freed records' domains.
            let fitting_rows = tc.values.iter().zip(&tc.costs).filter(|(row, _)| {
                tc.vars.iter().zip(row.iter()).all(|(var, value)| match domain_of.get(var) {
                    Some(domain) => domain.contains(value),
                    None => assignment[*var] == *value,
                })
            });
            let freed_positions: Vec<usize> =
                (0..tc.vars.len()).filter(|&pos| record_var.contains_key(&tc.vars[pos])).collect();

            if let [pos] = freed_positions[..] {
                // One freed record: value -> cost of this t-tuple (outside_penalty if no row).
                let var = tc.vars[pos];
                let mut cost_of_value: HashMap<i64, i64> = HashMap::new();
                for (row, &cost) in fitting_rows {
                    cost_of_value.insert(row[pos], cost);
                }
                for (value, total) in value_cost.get_mut(&var).unwrap().iter_mut() {
                    *total += *cost_of_value.get(value).unwrap_or(&outside_penalty);
                }
                continue;
            }

            // Two or more freed records: keep the table rows.
            let tag = solver.new_constraint_tag();
            let mut chosen_rows = Vec::new();
            for (row, &cost) in fitting_rows {
                // row selected -> each freed record of the t-tuple takes the row's value
                let row_selected = solver.new_literal();
                for &pos in &freed_positions {
                    let record = record_var[&tc.vars[pos]];
                    let value = row[pos] as i32;
                    solver.add_clause(vec![row_selected.get_false_predicate(), predicate![record == value]], tag).ok()?;
                }
                if cost > 0 {
                    error_terms.push(row_selected.get_integer_variable().scaled(cost as i32));
                }
                chosen_rows.push(row_selected.get_true_predicate());
            }
            // The t-tuple takes one of its rows (within epsilon), or pays the penalty.
            let outside = solver.new_literal();
            error_terms.push(outside.get_integer_variable().scaled(outside_penalty as i32));
            chosen_rows.push(outside.get_true_predicate());
            solver.add_clause(chosen_rows, tag).ok()?;
        }
        // value_cost as objective terms: literal [record == value] times its summed cost.
        for (&var, costs) in &value_cost {
            let record = record_var[&var];
            for (&value, &cost) in costs {
                if cost > 0 {
                    let tag = solver.new_constraint_tag();
                    let value = value as i32;
                    let takes_value = solver.new_literal_for_predicate(predicate![record == value], tag);
                    error_terms.push(takes_value.get_integer_variable().scaled(cost as i32));
                }
            }
        }
        let total_error = solver.new_bounded_integer(0, max_error.min(i32::MAX as i64) as i32);
        error_terms.push(total_error.scaled(-1));
        let tag = solver.new_constraint_tag();
        solver.add_constraint(equals(error_terms, 0, tag)).post().ok()?;

        // Ask Pumpkin for strictly better solutions until there are none (or time runs out).
        // Pumpkin's error counts a t-tuple outside epsilon as outside_penalty; convert back to
        // OUTSIDE_EPSILON so the caller's total stays on one scale.
        let to_caller_scale = |pumpkin_error: i64| -> i64 {
            let outside = pumpkin_error / outside_penalty;
            outside * Self::OUTSIDE_EPSILON + pumpkin_error % outside_penalty
        };
        let to_pumpkin_scale = |caller_error: i64| -> i64 {
            (caller_error / Self::OUTSIDE_EPSILON) * outside_penalty + caller_error % Self::OUTSIDE_EPSILON
        };
        let mut brancher = solver.default_brancher();
        let mut resolver = ResolutionResolver::default();
        let mut termination = TimeBudget::starting_now(time_limit);
        let mut bound = error_before.map(to_pumpkin_scale);
        let mut best = None;
        loop {
            let assumptions: Vec<Predicate> = match bound {
                Some(b) => {
                    let must_beat = (b - 1) as i32;
                    vec![predicate![total_error <= must_beat]]
                }
                None => vec![],
            };
            match solver.satisfy_under_assumptions(&mut brancher, &mut termination, &mut resolver, &assumptions) {
                SatisfactionResultUnderAssumptions::Satisfiable(found) => {
                    let solution = found.solution();
                    let error = solution.get_integer_value(total_error) as i64;
                    let values: Vec<(usize, i64)> =
                        freed.iter().map(|&var| (var, solution.get_integer_value(record_var[&var]) as i64)).collect();
                    bound = Some(error);
                    best = Some((values, to_caller_scale(error)));
                }
                // Nothing better in this neighbourhood (or out of time): keep what we have.
                _ => return best,
            }
        }
    }

}

