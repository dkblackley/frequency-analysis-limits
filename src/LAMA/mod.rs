use crate::dataloader::tester::testDB;
use crate::dataloader::three_d::ThreeDMap;
use crate::dataloader::two_d::TwoDMap;
use crate::dataloader::{flatten_nd, unflatten_nd, Searchable};
use crate::plotting::{DbResult, ReconstructionDataPoint};
use crate::LAMA::error::LAMAError;
use crate::LAMA::lazy_lns::{LnsConfig, PairLns};
use crate::LAMA::query::QueryDistribution;
use crate::LAMA::selector::Selector;
use crate::LAMA::solver::CpSolverStatus::{Feasible, Optimal};
use crate::LAMA::solver::Solver;
use crate::LAMA::translator::Translator;
use crate::LAMA::utility::get_mbq;
use crate::{DomPair, Record, Value};
use log::{debug, error, info, trace, warn};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::BufWriter;
use std::time::{Duration, Instant};

mod error;
pub mod lazy_lns;
pub mod lazy_pairs;
pub mod mini_solver;
pub mod query;
pub mod selector;
pub mod solver;
pub mod translator;
pub mod utility;

/// What lama_attack did. RestartNeeded: Pumpkin found no t = 1 start (sampled setting); the caller
/// reruns the whole pipeline, which draws a new query sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttackOutcome {
    Finished,
    RestartNeeded,
    Failed,
}

/// Knobs for the sampled (query_percent < 1) search, see lazy_lns::LnsConfig.
#[derive(Clone, Debug)]
pub struct SampledSearch {
    pub restarts: usize,
    /// Wall-clock budget for the LNS in hours (0 = none). Restarts that run out stop and report
    /// what they have, so a job writes results instead of hitting the slurm time limit.
    pub lns_hours: f64,
    pub max_rows_per_pair: usize,
    /// How many distinct databases to output
    pub outputs: usize,
    /// With lns_hours: abandon a still-inconsistent restart after this many steps without improving
    pub stall_steps: usize,
    /// Restarts at most this many pairs outside epsilon get 4x stall_steps
    pub near_pairs: i64,
    /// Which run of the whole pipeline this is (recorded in the results; main.rs reruns on start failure)
    pub pipeline_attempt: usize,
    /// Seconds Pumpkin gets to find the t = 1 start
    pub start_secs: f64,
    /// Perfect setting (--query-percent 1.0), t = 2: check pairs on demand instead of building the
    /// pair tables (--lazy-pairs, lazy_pairs.rs). Same databases, far less memory. Default: tables.
    pub lazy_pairs: bool,
}

impl Default for SampledSearch {
    fn default() -> Self {
        Self { restarts: 64, lns_hours: 0.0, max_rows_per_pair: 1_000, outputs: 1000, stall_steps: 5000, near_pairs: 20, pipeline_attempt: 1, start_secs: 60.0, lazy_pairs: false }
    }
}

/// For every pair p in `pairs`: the sum of `weights[q]` over every q in `pairs` that encloses p
/// (q.0 <= p.0 and q.1 >= p.1 in every dimension). The same prefix-sum sweep as
/// QueryDistribution::precompute_cumulative_results, but on indices, so no DomPair is cloned.
/// `pairs` must hold every valid query of the grid [low, high].
pub fn enclosing_sums(pairs: &[DomPair], weights: &[f64], low: &[Value], high: &[Value]) -> Vec<f64> {
    let index: FxHashMap<&DomPair, usize> = pairs.iter().enumerate().map(|(i, p)| (p, i)).collect();
    let mut dp = weights.to_vec();
    let dims = low.len();
    let mut order: Vec<usize> = (0..pairs.len()).collect();
    let mut key: DomPair = (vec![0; dims], vec![0; dims]);

    // 1. lower bounds: dp[l, u] += dp[l - e_d, u], ascending in l[d]
    for d in 0..dims {
        order.par_sort_unstable_by_key(|&i| pairs[i].0[d]);
        for &i in &order {
            let p = &pairs[i];
            if p.0[d] > low[d] {
                key.0.clone_from(&p.0);
                key.1.clone_from(&p.1);
                key.0[d] -= 1;
                if let Some(&j) = index.get(&key) {
                    dp[i] += dp[j];
                }
            }
        }
    }
    // 2. upper bounds: dp[l, u] += dp[l, u + e_d], descending in u[d]
    for d in 0..dims {
        order.par_sort_unstable_by_key(|&i| std::cmp::Reverse(pairs[i].1[d]));
        for &i in &order {
            let p = &pairs[i];
            if p.1[d] < high[d] {
                key.0.clone_from(&p.0);
                key.1.clone_from(&p.1);
                key.1[d] += 1;
                if let Some(&j) = index.get(&key) {
                    dp[i] += dp[j];
                }
            }
        }
    }
    dp
}

/// Minimum bounding query of some flattened plaintexts, from a precomputed coordinate table.
#[inline]
fn mbq_of(vals: &[i64], coords: &[Record]) -> DomPair {
    let mut lo = coords[vals[0] as usize].clone();
    let mut hi = lo.clone();
    for &v in &vals[1..] {
        for (d, &x) in coords[v as usize].iter().enumerate() {
            if x < lo[d] {
                lo[d] = x;
            }
            if x > hi[d] {
                hi[d] = x;
            }
        }
    }
    (lo, hi)
}

pub fn lama_attack(
    db_name: &String,
    dir_path: &String,
    dist: &String,
    t: &u64,
    dim: &usize,
    padding: &Value,
    _save: &bool,
    query_percent: &f64,
    dist_noise: &f64,
    sampled_search: &SampledSearch,
) -> AttackOutcome {
    let loaded_db: Box<dyn Searchable + Sync>;
    let full_datapath = format!("{0}/{1}", dir_path, db_name);

    if db_name == "grid" {
        loaded_db = Box::new(testDB::new(*dim, 6, 35));
    } else if db_name == "nh" {
        info!("Starting LAMA attack using {} dataset", db_name);
        debug!("Loading data from {full_datapath}/{db_name}.json");

        let loaded_locs =
            ThreeDMap::load_array_locations_from_file(&format!("{full_datapath}/{db_name}.json"))
                .unwrap();
        loaded_db = Box::new(ThreeDMap::new_unscaled(loaded_locs, db_name.as_str()).unwrap());
    } else {
        info!("Starting LAMA attack using {} dataset", db_name);
        debug!("Loading data from {full_datapath}/{db_name}.json");
        let loaded_locs =
            TwoDMap::load_array_locations_from_file(&format!("{full_datapath}/{db_name}.json"))
                .unwrap();

        loaded_db = Box::new(
            TwoDMap::new_unscaled(
                loaded_locs,
                db_name.as_str(),
                (*padding, *padding),
                (*padding, *padding),
            )
                .unwrap(),
        );
    }

    debug!(
        "Loaded {}, a {}-dim DB with {} records and highest/lowest records {:?}/{:?}",
        loaded_db.get_name(),
        loaded_db.get_dims(),
        loaded_db.get_universe().len(),
        loaded_db.get_dom_pair().0,
        loaded_db.get_dom_pair().1
    );

    let (low_pair, high_pair) = loaded_db.get_dom_pair();
    let largest_possible_val: i64 = flatten_nd(&high_pair, &high_pair, &low_pair);
    // The selector holds the TRUE query distribution (noisy if dist_noise > 0)
    let selector = Selector::new_with_noise(dist, &loaded_db, *dist_noise);
    // The attacker's model of the distribution: the noiseless one
    let attacker_dist = if *dist_noise > 0.0 {
        // Same pairs (same order) as the selector: get_dom_pairs' order is not deterministic and
        // the Gaussian/Beta weights are assigned by index into this vec
        Some(QueryDistribution::new(
            selector.query_distribution.pairs.clone(),
            &loaded_db,
            dist.parse().unwrap(),
        ))
    } else {
        None
    };
    // With all queries observed, enumerate every database within epsilon. With dist noise the attacker
    // sees the (noisy) frequencies in the limit and knows the exact error bound (eps = max_dist_err).
    // The min-error search misses the true DB at 15x15 even though it is within eps, so don't use it here.
    let perfect = *query_percent == 1.0;

    debug!(
        "Working with padding: {}, low_pair: {:?}, high_pair: {:?}",
        padding, low_pair, high_pair
    );
    debug!(
        "Largest possible value is {largest_possible_val}, working with {} dompairs",
        selector
            .query_distribution
            .dom_pair_to_known_raw_weight
            .len()
    );

    info!("Selector computing values");

    let mut translator = Translator::new(
        largest_possible_val,
        loaded_db.get_universe(),
        &(t.clone() as usize),
        perfect,
    );
    if perfect && sampled_search.lazy_pairs {
        info!("--lazy-pairs: t = 2 pair constraints are checked on demand (no pair tables built)");
        translator.set_lazy_pairs(true);
    }

    let query_dist_ref = &selector.query_distribution;
    let attacker_dist_ref = attacker_dist.as_ref().unwrap_or(query_dist_ref);

    // How far the attacker's expected probabilities actually are from the truth (bounded by dist_noise).
    // Perfect setting: the attacker is assumed to know this bound, so it is epsilon. Sampled setting: the
    // noise is unknown to the attacker (it just sits on the query distribution), so epsilon is the VC
    // bound alone; max_dist_err is only recorded in the results.
    let max_dist_err = if *dist_noise > 0.0 {
        let max_dist_err = query_dist_ref
            .mbq_to_cumulative_prob
            .iter()
            .map(|(mbq, p)| (p - attacker_dist_ref.cumulative_prob_lookup(mbq)).abs())
            .fold(0.0, f64::max);
        info!("Distribution noise {dist_noise}: largest actual error in any enclosing probability is {max_dist_err}");
        max_dist_err
    } else {
        0.0
    };

    // Coordinates of every flattened value, so validating a candidate doesn't unflatten/allocate per record
    let coords: Vec<Record> = (0..=largest_possible_val)
        .map(|v| unflatten_nd(v, &high_pair, &low_pair))
        .collect();
    let coords_ref = &coords;

    let get_expected_prob = |plaintexts: &[i64]| -> f64 {
        attacker_dist_ref.cumulative_prob_lookup(&mbq_of(plaintexts, coords_ref))
    };

    let eps;
    let delta = 0.1; // fix to 90% confidence
    let num_queries;

    let get_observed_prob: Box<dyn Sync + Send + Fn(&[i64]) -> f64 + '_>;

    if *query_percent != 1.0 {
        // we're going to sample. (The all-possible-responses argument is unused by the sampler and
        // was the single biggest allocation in high dimensions, so it is no longer computed.)
        let (observed_queries, _responses, epsil) =
            selector.sample_percent_responses(*query_percent, delta, &Vec::new());
        // Sampling error only: the attacker does not know the distribution is noisy
        eps = epsil;
        if *dist_noise > 0.0 {
            info!("Sampled setting with unknown distribution noise: eps = VC bound {epsil} (the actual \
                   distribution error {max_dist_err} is NOT added)");
        }
        num_queries = observed_queries.len();

        info!("Making observed mapping to frequency");
        // Observed frequency of an MBQ = fraction of the sampled queries enclosing it. Exactly the old
        // per-pair count, but as one prefix sum over the histogram of sampled queries
        // (O(#pairs * dims) instead of O(#pairs * #sampled queries)).
        let all_dompairs = &query_dist_ref.pairs;
        let pair_index: FxHashMap<&DomPair, usize> =
            all_dompairs.iter().enumerate().map(|(i, p)| (p, i)).collect();
        let mut hist = vec![0.0; all_dompairs.len()];
        for q in &observed_queries {
            hist[pair_index[q]] += 1.0;
        }
        let queries_len = observed_queries.len() as f64;
        let observed: Vec<f64> = enclosing_sums(all_dompairs, &hist, &low_pair, &high_pair)
            .into_iter()
            .map(|c| c / queries_len)
            .collect();
        drop(observed_queries);

        get_observed_prob = Box::new(move |enc_tuple: &[i64]| -> f64 {
            let target_mbq = mbq_of(enc_tuple, coords_ref);
            pair_index.get(&target_mbq).map_or(0.0, |&i| observed[i])
        });
    } else {
        eps = max_dist_err;
        // How many queries can we do? A query is a hyperrectangle, The total number of hyper rectangles
        // in a space is n*(n+1)/2 where n is the length of a dim, then mutliply for each dim.
        let mut total = 1;

        for item in high_pair.clone() {
            total *= ((item + 1) * (item + 2)) / 2;
        }
        num_queries = total as usize;

        get_observed_prob = Box::new(move |enc_tuple: &[i64]| -> f64 {
            query_dist_ref.cumulative_prob_lookup(&mbq_of(enc_tuple, coords_ref))
        });
    }

    // if in the 'perfect' world only use things within 0.001\% of the true
    let active_eps = if eps == 0.0 { 1e-6 } else { eps };
    let validate_candidate = |enc_tuple: &[i64], proposed_plaintexts: &[i64]| -> (bool, f64) {
        let obs_prob = get_observed_prob(enc_tuple);
        let exp_prob = get_expected_prob(proposed_plaintexts);

        (
            (obs_prob - exp_prob).abs() <= active_eps,
            (obs_prob - exp_prob).abs(),
        )
    };

    let universe = loaded_db.get_universe();

    info!("Starting LAMA!");
    let start = Instant::now();
    translator.process_t1(&universe, &validate_candidate);

    let mut lns_summary = None;
    let responses: HashMap<i64, Vec<i64>>;
    let translater_met;

    if !perfect && *t >= 2 {
        // Sampled setting: min-error LNS straight from the t = 1 tables, scoring pairs on demand
        // (the t = 2 tables are never built; they were what ran out of memory at 50x50).
        if *t > 2 {
            warn!("Sampled setting searches t = 2 (pairs) only; --t {t} is treated as 2");
        }
        let var_index_map = translator.get_var_index_map();
        let mut enc_of_var = vec![0i64; var_index_map.len()];
        for (idx, enc) in var_index_map.values() {
            enc_of_var[*idx as usize] = *enc;
        }
        let lns = PairLns::new(translator.proto_model_ref(), enc_of_var.clone(), largest_possible_val, &validate_candidate);
        let cfg = LnsConfig {
            restarts: sampled_search.restarts,
            max_rows_per_pair: sampled_search.max_rows_per_pair,
            target_outputs: sampled_search.outputs,
            stall_steps: sampled_search.stall_steps,
            near_pairs: sampled_search.near_pairs,
            start_secs: sampled_search.start_secs,
            deadline: (sampled_search.lns_hours > 0.0)
                .then(|| Instant::now() + Duration::from_secs_f64(sampled_search.lns_hours * 3600.0)),
            ..LnsConfig::default()
        };
        let (found, mut summary) = lns.run(&cfg);
        summary.pipeline_attempt = sampled_search.pipeline_attempt;
        if summary.start_failed {
            error!("Pumpkin found no t = 1 start: restart needed (nothing written)");
            return AttackOutcome::RestartNeeded;
        }
        if found.is_empty() {
            error!("Pair LNS found no reconstruction at all");
            return AttackOutcome::Failed;
        }
        translator.metadata.recons_per_t.insert(2, found.len());
        responses = enc_of_var
            .iter()
            .enumerate()
            .map(|(idx, &enc)| (enc, found.iter().map(|r| r.assignment[idx]).collect()))
            .collect();
        lns_summary = Some(summary);
        translater_met = translator.metadata.clone();
    } else {
        for i in 2..(*t as usize + 1) {
            let start_t = Instant::now();
            info!("Recursively computing frequencies for {i} tuples");

            translator.process_t_greater_than_1(i, &universe, &validate_candidate);

            let end_t = Instant::now();
            debug!(
                "Round {i} took {} seconds",
                end_t.duration_since(start_t).as_secs()
            )
        }

        translater_met = translator.metadata.clone();
        if let Some(dbs) = translator.final_databases() {
            // --lazy-pairs, t = 2: the pair round already enumerated every database (same set the final
            // solve would find, Translator::final_databases); the final solve rebuilt every pair table
            // and OOMed at 128 GB on the noisy 50x50 runs.
            info!("Output = the {} databases of the lazy pair round (final solve skipped)",
                  dbs.values().next().map_or(0, Vec::len));
            responses = dbs;
        } else {
        info!("Building and executing the CP-SAT Solver for the final constraint graph...");
        let mut solver = Solver::new(translator.get_var_index_map());

        let mut model = translator.get_proto_model();

        // NOTE: The only change in the test is passing largest_enc_val here
        responses = solver.solve(&mut model, largest_possible_val, false);
        if solver.solution_stat != Optimal && solver.solution_stat != Feasible {
            // minimum number of expected reconstructions
            warn!(
                "Solver did not return the expected number of reconstructions. No solutions were found."
            );

            debug!("Solver failed to reconstruct full universe!!");
            trace!("'encrypted/encoded' universe of plaintexts: {universe:?}");
            trace!("Intvars workings: {:?}", solver);
            trace!("CPModel: {:?}", model);
            debug!("Solver Status: {:?}", solver.solution_stat);
            return AttackOutcome::Failed;
        }
        }
    }
    let end = Instant::now();

    let unique_name = if *dist_noise > 0.0 {
        format!("{db_name}_{dist}_p{query_percent}_noise{dist_noise}")
    } else {
        format!("{db_name}_{dist}_p{query_percent}")
    };

    let mut correct = 0;
    for (key, val) in &responses {
        correct += val.iter().filter(|v| *v == key).count();
    }
    // Number of reconstructed DATABASES (responses has one entry per record)
    let num_dbs = responses.values().next().map_or(0, Vec::len);

    debug!("{correct} correct out of {} total", responses.len());

    // Sampled setting: MSE to the truth, for information only (see LnsSummary::centroid_mse)
    if let Some(summary) = lns_summary.as_mut() {
        let (centroid, mean_db, first_db) = mse_summary(&into_recon_data(&responses, &loaded_db));
        summary.centroid_mse = centroid;
        summary.mean_db_mse = mean_db;
        summary.first_db_mse = first_db;
        info!("MSE to the truth (info only): centroid {centroid:?}, mean per database {mean_db:?}, database 0 {first_db:?}");
    }

    // Sampled setting: an earlier result for this config may be better (reruns are random). Keep the
    // better one in limits/ and put the other in limits/archive_other_runs/<timestamp>/. "Better" uses
    // only what the attacker sees, never the truth: see sampled_rank.
    let limits_dir = format!("{full_datapath}/limits");
    let results_path = format!("{limits_dir}/results_{unique_name}.json");
    let mut save_dir = limits_dir.clone();
    if let Some(new) = lns_summary.as_ref() {
        let ts = chrono_like_timestamp();
        let other_dir = format!("{limits_dir}/archive_other_runs/{ts}");
        let existing: Option<DbResult> =
            fs::read_to_string(&results_path).ok().and_then(|s| serde_json::from_str(&s).ok());
        let existing_rank = existing.as_ref().map(|e| sampled_rank(e.lns.as_ref()));
        let new_rank = sampled_rank(Some(new));
        match existing_rank {
            Some(old_rank) if old_rank > new_rank => {
                warn!(
                    "The existing result is better ({old_rank:?} > {new_rank:?}: consistent, #consistent DBs, \
                     -pairs outside, -error); keeping it and writing this run to {other_dir}"
                );
                save_dir = other_dir;
            }
            Some(old_rank) => {
                info!("This run beats the existing result ({new_rank:?} >= {old_rank:?}); moving the old one to {other_dir}");
                if fs::create_dir_all(&other_dir).is_ok() {
                    for f in [results_path.clone(), format!("{limits_dir}/{unique_name}_reconstruction.json")] {
                        let to = format!("{other_dir}/{}", f.rsplit('/').next().unwrap_or_default());
                        if let Err(e) = fs::rename(&f, &to) {
                            warn!("could not move {f} to {to}: {e}");
                        }
                    }
                }
            }
            None => {}
        }
    }

    info!("Saving correct solutions to {save_dir}/{unique_name}_reconstruction.json");
    save_reconstruction_data(&responses, &save_dir, &unique_name, &loaded_db);

    info!(
        "LAMa completely finished in {}",
        end.duration_since(start).as_secs_f64()
    );

    let final_res = DbResult {
        name: loaded_db.get_name().parse().unwrap(),
        method: "LAMA".to_string(),
        dims: loaded_db.get_dims() as u32,
        mse: Some(0.0),
        match_rate: Some(1.0),
        chamfer: Some(0.0),
        number_of_reconstructions: format!("{}", responses.len()),
        time_taken: end.duration_since(start).as_secs_f64(),
        total_db_size: loaded_db.get_universe().len() as u64,
        percent_queries_used: *query_percent,
        num_queries_used: num_queries as u64,
        eps: Some(eps),
        delt: Some(delta),
        translator_meta: translater_met,
        number_of_databases: Some(num_dbs),
        lns: lns_summary,
        query_order: None, // query list order = get_dom_pairs (par_bridge): not fixed between runs
        true_db_in_output: Some((0..num_dbs).any(|d| responses.iter().all(|(enc, vals)| vals[d] == *enc))),
        dist_noise: (*dist_noise > 0.0).then_some(*dist_noise),
        max_dist_err: (*dist_noise > 0.0).then_some(max_dist_err),
        noise_known: (*dist_noise > 0.0).then_some(perfect),
    };

    if let Err(e) = save_results(final_res, &format!("{save_dir}/results_{unique_name}.json")) {
        error!("failed writing to {save_dir}/results_{unique_name}.json: {e}");
    }

    info!("LAMA finished running on {db_name}");
    AttackOutcome::Finished
}
/// How good a sampled-setting result is, from what the ATTACKER sees only (never the true database):
/// (output is fully consistent, number of distinct consistent databases, -pairs outside epsilon of the
/// best database, -its total error). Larger is better. A result without an "lns" field (old code) is worst.
pub fn sampled_rank(lns: Option<&crate::LAMA::lazy_lns::LnsSummary>) -> (bool, bool, usize, i64, i64) {
    match lns {
        None => (false, false, 0, i64::MIN, i64::MIN),
        Some(l) => (
            true,
            l.output_is_consistent,
            if l.output_is_consistent { l.returned } else { 0 },
            -l.best_pairs_outside,
            -l.best_total_error,
        ),
    }
}

/// (centroid MSE, mean per-database MSE, MSE of database 0); squared Euclidean distance, no alignment.
fn mse_summary(data: &[Vec<ReconstructionDataPoint>]) -> (Option<f64>, Option<f64>, Option<f64>) {
    let sq = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>();
    let db_mse = |db: &Vec<ReconstructionDataPoint>| {
        db.iter().map(|p| sq(&p.true_points, &p.reconstructed_points)).sum::<f64>() / db.len().max(1) as f64
    };
    if data.is_empty() || data[0].is_empty() {
        return (None, None, None);
    }
    let n_rec = data[0].len();
    let dims = data[0][0].true_points.len();
    let mut centroid_err = 0.0;
    for r in 0..n_rec {
        let mut mean = vec![0.0; dims];
        for db in data {
            for (m, x) in mean.iter_mut().zip(&db[r].reconstructed_points) {
                *m += x / data.len() as f64;
            }
        }
        centroid_err += sq(&data[0][r].true_points, &mean);
    }
    let mean_db = data.iter().map(db_mse).sum::<f64>() / data.len() as f64;
    (Some(centroid_err / n_rec as f64), Some(mean_db), Some(db_mse(&data[0])))
}

/// YYYYmmdd_HHMMSS (UTC) from the system clock, without a date crate.
fn chrono_like_timestamp() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()) as i64;
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // civil-from-days (Howard Hinnant)
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}{m:02}{d:02}_{:02}{:02}{:02}", rem / 3600, rem % 3600 / 60, rem % 60)
}

fn save_results(result: DbResult, file_path: &str) -> Result<(), LAMAError> {
    // Create the file and wrap it in a BufWriter for better performance
    let file = File::create(file_path)?;
    let writer = BufWriter::new(file);

    serde_json::to_writer_pretty(writer, &result)?;
    Ok(())
}

pub fn into_recon_data(
    responses: &HashMap<i64, Vec<i64>>,
    loaded_db: &Box<dyn Searchable + Sync>,
) -> Vec<Vec<ReconstructionDataPoint>> {
    let mut data: Vec<Vec<ReconstructionDataPoint>> = Vec::new();

    // Iterate over the HashMap
    for (&encrypted_true, encrypted_recons) in responses {
        let true_vec = loaded_db.decrypt_point_f64(&encrypted_true);

        // Iterate over the reconstructions and get their index
        for (i, &encrypted_recon) in encrypted_recons.iter().enumerate() {
            let recon_vec = loaded_db.decrypt_point_f64(&encrypted_recon);

            let saved_point = ReconstructionDataPoint {
                true_points: true_vec.clone(),
                reconstructed_points: recon_vec,
                unscaled_points: None,
            };

            // If we have more reconstructions for this point than we have
            // outer arrays, we need to push a new empty Vec to hold them.
            if data.len() <= i {
                data.push(Vec::new());
            }

            // Push the saved point to the correct reconstruction index
            data[i].push(saved_point);
        }
    }
    data
}

fn save_reconstruction_data(
    responses: &HashMap<i64, Vec<i64>>,
    file_path: &str,
    unique_name: &str,
    loaded_db: &Box<dyn Searchable + Sync>,
) {
    let data: Vec<Vec<ReconstructionDataPoint>> = into_recon_data(responses, loaded_db);

    fs::create_dir_all(file_path).unwrap();

    // Create the file and wrap it in a BufWriter for better performance
    let file = File::create(format!("{file_path}/{unique_name}_reconstruction.json")).unwrap();
    let writer = BufWriter::new(file);

    serde_json::to_writer_pretty(writer, &data).unwrap();
}


#[cfg(test)]
mod enclosing_sums_tests {
    use super::*;
    use crate::LAMA::utility::encloses;
    use rand::Rng;

    /// The prefix sum must give exactly the old brute-force count of enclosing sampled queries.
    #[test]
    fn enclosing_sums_match_brute_force() {
        for dim in [1usize, 2, 3] {
            let db: Box<dyn Searchable + Sync> = Box::new(testDB::new(dim, 5, 35));
            let (low, high) = db.get_dom_pair();
            let pairs = Selector::get_dom_pairs(&db);
            let mut rng = rand::thread_rng();
            let sampled: Vec<DomPair> = (0..500).map(|_| pairs[rng.gen_range(0..pairs.len())].clone()).collect();
            let index: FxHashMap<&DomPair, usize> = pairs.iter().enumerate().map(|(i, p)| (p, i)).collect();
            let mut hist = vec![0.0; pairs.len()];
            for q in &sampled {
                hist[index[q]] += 1.0;
            }
            let fast = enclosing_sums(&pairs, &hist, &low, &high);
            for (i, p) in pairs.iter().enumerate() {
                let brute = sampled.iter().filter(|q| encloses(q, p)).count() as f64;
                assert_eq!(fast[i], brute, "dim {dim}, pair {p:?}");
            }
        }
    }
}

#[cfg(test)]
mod metadata_tests {
    use super::*;
    use crate::dataloader::tester::testDB;
    use crate::dataloader::three_d::ThreeDMap;
    use crate::dataloader::two_d::TwoDMap;
    use crate::dataloader::Searchable;

    // Helper function to handle math and printing for any dimension
    fn print_stats(name: &str, size_str: &str, items: usize, high: &[i64]) {
        // Calculate total possible grid cells
        let capacity: i64 = high.iter().map(|&val| val + 1).product();
        let density = (items as f64 / capacity as f64) * 100.0;

        // Calculate total queries using f64 to avoid overflow on higher dimensions
        let mut total_queries: f64 = 1.0;
        for &item in high {
            total_queries *= ((item as f64 + 1.0) * (item as f64 + 2.0)) / 2.0;
        }

        let q_10 = (total_queries * 0.10).ceil() as i64;
        let q_20 = (total_queries * 0.20).ceil() as i64;
        let q_30 = (total_queries * 0.30).ceil() as i64;

        println!(
            "{:<12} | {:<15} | {:<10} | {:>6.2}%   | {:<12} | {:<12} | {:<12} | {:<15}",
            name, size_str, items, density, q_10, q_20, q_30, total_queries as i64
        );
    }

    #[test]
    fn print_all_database_metadata() {
        println!(
            "{:<12} | {:<15} | {:<10} | {:<10} | {:<12} | {:<12} | {:<12} | {:<15}",
            "Dataset", "Grid", "Items", "Density", "10% Queries", "20% Queries", "30% Queries", "100% Queries"
        );
        println!("{:-<115}", "");

        // 1. Process 2D Maps
        let grid_sizes = ["10x10", "20x20", "30x30", "40x40", "50x50"];
        let datasets = ["spitz", "cali", "shopparis", "highway", "busstop", "drink"];

        for name in datasets {
            for size in grid_sizes {
                let path = format!("databases/{}/{}/{}.json", size, name, name);

                if std::path::Path::new(&path).exists() {
                    let locs = TwoDMap::load_array_locations_from_file(&path).unwrap();
                    let db = TwoDMap::new_unscaled(locs, name, (0, 0), (0, 0)).unwrap();

                    let items = db.get_universe().len();
                    let (_low, high) = db.get_dom_pair();

                    print_stats(name, size, items, &high);
                }
            }
        }

        // 2. Process 3D Map (nh)
        let path_3d = "databases/16x16x14/nh/nh.json";
        if std::path::Path::new(&path_3d).exists() {
            let locs = ThreeDMap::load_array_locations_from_file(path_3d).unwrap();
            let db = ThreeDMap::new_unscaled(locs, "nh").unwrap();

            let items = db.get_universe().len();
            let (_low, high) = db.get_dom_pair();

            print_stats("nh", "16x16x14", items, &high);
        }

        // 3. Process N-Dimensional Synthetic Grids
        for dim in 1..=6 {
            let size_str = vec!["6"; dim].join("x");

            // Uses your exact test parameters: dim, size_per_dim=6, density=35
            let db = testDB::new(dim, 6, 35);
            let items = db.get_universe().len();
            let (_low, high) = db.get_dom_pair();

            print_stats("grid", &size_str, items, &high);
        }

        println!("Done!");
        panic!()
    }
}