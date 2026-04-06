use crate::dataloader::tester::testDB;
use crate::dataloader::{flatten_nd, unflatten_nd, Searchable};
use crate::LAMA::error::LAMAError;
use crate::LAMA::query::QueryDistribution;
use crate::LAMA::utility::{
    binomial_coefficient, dominates, get_all_dominating_values, get_mbq, DistributionType,
};
use crate::{Coord, DomPair, Frequency, Probability, Record, Value};
use good_lp::{
    default_solver, variable, Constraint, Expression, ProblemVariables, Solution, SolverModel,
};
use indicatif::{ProgressBar, ProgressIterator, ProgressStyle};
use itertools::Itertools;
use log::info;
use rand::distributions::{Distribution, WeightedIndex};
use rand::thread_rng;
use rayon::prelude::*;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use statrs::distribution::Normal;
use statrs::distribution::{Beta, Continuous};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::mpsc::sync_channel;
use std::thread;
use std::time::Instant;

/// Selector: Choosing Record-Retrieval Events.
/// This component determines which record-retriewval set expressions are used.
/// Specifically, it generates the frequencies of dominating pairs, as it corresponds to some dist
pub struct Selector<'a> {
    dim: Value,
    lowest_rec: Record,
    largest_rec: Record,
    evc: usize,
    epsilon: f64,
    delta: f64,
    encrypted_db: &'a Box<dyn Searchable + Sync>,
    pub query_distribution: Box<QueryDistribution<'a>>,
}

impl<'a> Selector<'a> {
    pub fn new(
        dist: &str,
        loaded_db: &'a Box<dyn Searchable + Sync>,
        eps: f64,
        delt: f64,
    ) -> Box<Self> {
        let (low_pair, high_pair) = loaded_db.get_dom_pair();
        let dim = loaded_db.get_dims();

        let query_distribution = QueryDistribution::new(
            Self::get_dom_pairs(&**loaded_db),
            loaded_db,
            dist.parse().unwrap(),
        );

        let mut selector = Selector {
            encrypted_db: loaded_db,
            dim,
            lowest_rec: low_pair.clone(),
            largest_rec: high_pair.clone(),
            evc: (loaded_db.get_universe().len() - 1),
            epsilon: eps,
            delta: delt,
            query_distribution,
        };

        Box::new(selector)
    }

    /// Given A query distribution (DomPair -> Probability mapping) find the frequency of a t-tuple
    /// of records. Note that this is NOT the frequency of the specific response that is exactly
    /// that t-tuple. Instead, given t encrypted records, how frequently do we see these across ALL
    /// responses. For uniform, we expect the center to be largest.
    ///
    /// # Arguments
    ///
    /// # Returns
    ///
    /// A mapping from a Query (dominating pair) to the frequency we'd expect that Query to be
    /// served.
    ///
    pub fn precompute_perfect_t_observed(&self, t: usize) -> HashMap<u64, Vec<Vec<i64>>> {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] {msg}")
                .unwrap(),
        );
        pb.set_message("Fetching entire database...");

        // 1. Do a single search to extract the entire encrypted database
        let mut response = self
            .encrypted_db
            .do_search(&self.lowest_rec, &self.largest_rec);
        response.retain(|&x| x != i64::MIN); // Filter out empty space placeholders
        response.sort_unstable(); // Ensure canonical combination ordering

        pb.set_message(format!("Processing combinations for t={}...", t));

        let processed_count = AtomicU64::new(0);

        let freq_to_observed_map: HashMap<u64, Vec<Vec<i64>>> = response
            .into_iter()
            .combinations(t)
            .par_bridge()
            .fold(
                HashMap::new,
                |mut local_map: HashMap<u64, Vec<Vec<i64>>>, t_tuple| {
                    let decoded_points: Vec<Record> = t_tuple
                        .iter()
                        .map(|&v| unflatten_nd(v, &self.largest_rec, &self.lowest_rec))
                        .collect();

                    let dom_pair = get_mbq(&decoded_points);

                    // NEW: Just ask the struct directly!
                    let freq = self.query_distribution.get_true_freq(&dom_pair);

                    if freq > 0 {
                        local_map.entry(freq).or_default().push(t_tuple);
                    }

                    // Update progress bar safely
                    let current = processed_count.fetch_add(1, AtomicOrdering::Relaxed);
                    if current % 10_000 == 0 {
                        pb.set_message(format!("Processed {} tuples...", current));
                    }

                    local_map
                },
            )
            .reduce(HashMap::new, |mut map1, map2| {
                // Merge thread-local HashMaps by extending the vectors
                for (freq, mut tuples) in map2 {
                    map1.entry(freq).or_default().append(&mut tuples);
                }
                map1
            });

        pb.finish_with_message("Done computing observed frequencies via direct MBQ.");
        info!("Precomputation complete.");

        freq_to_observed_map
    }

    /// Precomputes and serializes TRUE probabilities for dominant pairs using only the query
    /// distribution. This function iterates through the domain to calculate how many queries cover
    /// specific point pairs (dominant pairs). It doesn't say anything about how many records
    /// are found/the frequency of records. This is purely the Query Distribution. The calculation
    /// is: What is the probability of this specific dominating pair being issued as a query added
    /// to the probability of every pair that also dominates this pair. Hence, the probability is:
    /// what's the probability of OBSERVING this dominating pair.
    ///
    /// # Returns
    ///
    /// A mapping from a Query (dominating pair) to the (true) probability we'd expect that Query to be
    /// served.
    ///
    pub fn get_dominant_pair_to_perfect_prob_map(
        &self,
    ) -> Result<HashMap<DomPair, Probability>, LAMAError> {
        info!("Task 1: Computing dominant pair frequencies...");

        let timer = Instant::now();
        let lowest_rec = self.lowest_rec.clone();
        let largest_rec = self.largest_rec.clone();

        let space_size: f64 = lowest_rec
            .iter()
            .zip(largest_rec.iter())
            .map(|(&low, &high)| (high - low + 1) as f64)
            .product();

        let total_pairs = space_size.powi(2);
        let total_dom_pairs = (total_pairs / 2_f64.powi((self.dim - 1) as i32)) as u64;

        let domain_iter = lowest_rec
            .iter()
            .zip(largest_rec.iter())
            .map(|(&low, &high)| low..=high)
            .multi_cartesian_product();

        let pb = ProgressBar::new(total_dom_pairs / 2);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})").unwrap()
                .progress_chars("#>-"),
        );
        let processed_count = AtomicU64::new(0);

        let largest_rec_ref = &largest_rec;
        let processed_count_ref = &processed_count;

        let true_pair_frequency_dict: HashMap<DomPair, Probability> = domain_iter
            .par_bridge()
            .flat_map(|v| {
                let dom_iter = v
                    .iter()
                    .zip(largest_rec_ref.iter())
                    .map(|(&v_val, &max_val)| v_val..=max_val)
                    .multi_cartesian_product();
                let pb_inner = pb.clone();

                dom_iter.par_bridge().map({
                    let v_clone = v.clone();

                    move |dv| {
                        let pair = (v_clone.clone(), dv);

                        let prob = self.query_distribution.get_query_prob(&pair);

                        let count = processed_count_ref.fetch_add(1, AtomicOrdering::Relaxed);
                        if count % 100_000 == 0 {
                            pb_inner.set_position(count);
                        }
                        (pair, prob)
                    }
                })
            })
            .collect();

        pb.finish_with_message("Done computing dominant pair frequencies");
        info!("Finished DP frequencies in {:?}", timer.elapsed());

        Ok(true_pair_frequency_dict)
    }
    pub fn get_freq_val_t_tup_dict(
        &self,
        t: usize,
    ) -> Result<HashMap<(Value, Frequency), Vec<Vec<Value>>>, LAMAError> {
        let lowest_rec = self.lowest_rec.clone();
        let largest_rec = self.largest_rec.clone();

        let mut vals: Vec<Record> = lowest_rec
            .iter()
            .zip(largest_rec.iter())
            .map(|(&low, &high)| low..=high)
            .multi_cartesian_product()
            .collect();

        let n = vals.len();
        let total_combinations = binomial_coefficient(n, t);

        vals.sort();

        let pb = ProgressBar::new(total_combinations);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})",
                )?
                .progress_chars("#>-"),
        );

        let processed_count = AtomicU64::new(0);

        let val_tup_freq_dict: HashMap<(Value, Frequency), Vec<Vec<Value>>> = vals
            .into_iter()
            .combinations(t)
            .par_bridge()
            .fold(
                HashMap::new,
                |mut local_map: HashMap<(Value, Frequency), Vec<Vec<Value>>>, val_tuple| {
                    let bounding_pair = get_mbq(&val_tuple);
                    let freq = self.query_distribution.get_true_freq(&bounding_pair);

                    let flattened_tuple: Vec<Value> = val_tuple
                        .iter()
                        .map(|record| flatten_nd(record, &*largest_rec, &*lowest_rec))
                        .collect();

                    local_map
                        .entry((flattened_tuple.len() as Value, freq))
                        .or_default()
                        .push(flattened_tuple);

                    let current = processed_count.fetch_add(1, AtomicOrdering::Relaxed);
                    if current % 10000 == 0 {
                        pb.set_position(current);
                    }

                    local_map
                },
            )
            .reduce(HashMap::new, |mut map1, map2| {
                for (k, mut v) in map2 {
                    map1.entry(k).or_default().append(&mut v);
                }
                map1
            });

        pb.finish_with_message("Done computing value tuple frequencies");

        Ok(val_tup_freq_dict)
    }

    /// Computes the theoretical expected frequencies for all t-tuples
    pub fn build_theoretical_t_dict(
        &self,
        lowest_rec: &[i64],
        largest_rec: &[i64],
        t: usize,
    ) -> HashMap<u64, Vec<Vec<i64>>> {
        let mut vals: Vec<Record> = lowest_rec
            .iter()
            .zip(largest_rec.iter())
            .map(|(&low, &high)| low..=high)
            .multi_cartesian_product()
            .collect();

        let n = vals.len();
        let total_combinations = binomial_coefficient(n, t);
        vals.sort_unstable();

        let pb = ProgressBar::new(total_combinations as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})").unwrap()
                .progress_chars("#>-"),
        );

        let processed_count = AtomicU64::new(0);

        let theoretical_dict: HashMap<u64, Vec<Vec<i64>>> = vals
            .into_iter()
            .combinations(t)
            .par_bridge()
            .fold(
                HashMap::new,
                |mut local_map: HashMap<u64, Vec<Vec<i64>>>, val_tuple| {
                    let bounding_pair = get_mbq(&val_tuple);
                    let freq = self.query_distribution.get_true_freq(&bounding_pair);

                    let flattened_tuple: Vec<i64> = val_tuple
                        .iter()
                        .map(|record| flatten_nd(record, largest_rec, lowest_rec))
                        .collect();

                    local_map.entry(freq).or_default().push(flattened_tuple);

                    let current = processed_count.fetch_add(1, AtomicOrdering::Relaxed);
                    if current % 10_000 == 0 {
                        pb.set_position(current);
                    }

                    local_map
                },
            )
            .reduce(HashMap::new, |mut map1, map2| {
                for (freq, mut tuples) in map2 {
                    map1.entry(freq).or_default().append(&mut tuples);
                }
                map1
            });

        pb.finish_with_message(format!("Finished theoretical mapping for t={}", t));

        theoretical_dict
    }

    pub fn get_dom_pairs(encrypted_db: &(dyn Searchable + Sync)) -> Vec<DomPair> {
        info!("Finding all possible responses...");

        let (lowest_rec, largest_rec) = encrypted_db.get_dom_pair();

        // Product of (max - min + 1) for each dimension
        let space_size: f64 = lowest_rec
            .iter()
            .zip(largest_rec.iter())
            .map(|(&low, &high)| (high - low + 1) as f64)
            .product();

        let total_pairs = space_size.powi(2);
        let total_dom_pairs =
            (total_pairs / 2_f64.powi((encrypted_db.get_dims() - 1) as i32)) as u64;

        let domain_iter = lowest_rec
            .iter()
            .zip(largest_rec.iter())
            .map(|(&low, &high)| low..=high)
            .multi_cartesian_product();

        let pb = ProgressBar::new(total_dom_pairs / 2);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})").unwrap()
                .progress_chars("#>-"),
        );
        let processed_count = AtomicU64::new(0);

        let largest_rec_ref = &largest_rec;
        let processed_count_ref = &processed_count;

        let pair_vec: Vec<DomPair> = domain_iter
            .par_bridge() // Parallelize the outer loop
            .flat_map(|v| {
                // Build the iterator lazily without allocating a Vec
                let dom_iter = v
                    .iter()
                    .zip(largest_rec_ref.iter())
                    .map(|(&v_val, &max_val)| v_val..=max_val)
                    .multi_cartesian_product();
                let pb_inner = pb.clone();

                // Bridge the inner iterator to run concurrently as well
                dom_iter.par_bridge().map({
                    // Clone `v` once per outer iteration so the inner closure can own its own copy
                    let v_clone = v.clone();
                    move |dv| {
                        let count = processed_count_ref.fetch_add(1, AtomicOrdering::Relaxed);
                        if count % 100_000 == 0 {
                            pb_inner.set_position(count);
                        }
                        (v_clone.clone(), dv)
                    }
                })
            })
            .collect();

        pb.finish_with_message("Done computing dominant pair frequencies");
        pair_vec
    }

    // TODO: Sample instead.

    pub fn get_all_possible_responses(&self) -> Vec<Vec<Value>> {
        info!("Finding all possible responses...");

        let timer = Instant::now();

        let dom_pairs = Selector::get_dom_pairs(&**self.encrypted_db);

        let pb = ProgressBar::new(dom_pairs.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})").unwrap()
                .progress_chars("#>-"),
        );
        let processed_count = AtomicU64::new(0);

        let responses: Vec<Vec<i64>> = dom_pairs
            .par_iter()
            .map(|pair| {
                let resp = self.encrypted_db.do_search(&pair.0, &pair.1);

                let count = processed_count.fetch_add(1, AtomicOrdering::Relaxed);
                if count % 100_000 == 0 {
                    pb.set_position(count);
                }
                resp
            })
            .collect();

        pb.finish_with_message("Getting all possible responses");

        info!("Finished getting all responses in {:?}", timer.elapsed());
        responses
    }

    /// Evaluates the SUKP to find the bounding profit `q` for empirical VC-dimension.
    /// `capacity`: The maximum transaction length (l)
    /// `itemsets`: A slice of vectors, where each vector contains the indices of items in that set.
    /// `num_items`: The total number of unique items in the universe |U|.
    pub fn get_vc_sukp_bound(&self) -> f64 {
        let responses = self.get_all_possible_responses();
        // The size of the largest response
        let largest_resp = self
            .encrypted_db
            .do_search(
                &self.encrypted_db.get_dom_pair().0,
                &self.encrypted_db.get_dom_pair().1,
            )
            .len();
        let universe_size: usize = largest_resp;
        let capacity = largest_resp;

        // 1. Calculate Upper Bound via LP Relaxation
        let q_upper = Self::solve_sukp_internal(capacity, &responses, universe_size, false);

        // Simple heuristic for lower bound (e.g., just count itemsets smaller than capacity)
        // In a real implementation, you might want a slightly smarter greedy heuristic here.
        let q_lower = responses.iter().filter(|s| s.len() <= capacity).count() as f64;

        // The Power of 2 check
        let b_upper = q_upper.log2().floor();
        let b_lower = q_lower.log2().floor();

        if b_upper == b_lower {
            // We proved no power of 2 exists between the bounds.
            return q_upper;
        }

        // 2. Fallback to exact MILP if the bounds straddle a power of 2
        Self::solve_sukp_internal(capacity, &responses, universe_size, true)
    }

    fn solve_sukp_internal(
        capacity: usize,
        itemsets: &[Vec<Value>],
        num_items: usize,
        is_integer: bool,
    ) -> f64 {
        let mut vars = ProblemVariables::new();

        // x_j: 1 if item j is included in the knapsack capacity
        let mut x = Vec::with_capacity(num_items);
        for _ in 0..num_items {
            let mut var = variable().min(0.0).max(1.0);
            if is_integer {
                var = var.integer();
            }
            x.push(vars.add(var));
        }

        // y_i: 1 if itemset i is fully included
        let mut y = Vec::with_capacity(itemsets.len());
        for _ in 0..itemsets.len() {
            let mut var = variable().min(0.0).max(1.0);
            if is_integer {
                var = var.integer();
            }
            y.push(vars.add(var));
        }

        // Objective: Maximize sum(y_i)
        let objective: Expression = y.iter().sum();
        let mut model = vars.maximise(objective).using(default_solver);

        // Constraint 1: Capacity limit -> sum(x_j) <= l
        let weight_expr: Expression = x.iter().sum();
        model = model.with(weight_expr << capacity as f64);

        // Constraint 2: Union constraint -> y_i <= x_j for all j in A_i
        for (i, itemset) in itemsets.iter().enumerate() {
            for &j in itemset {
                model = model.with(y[i] - x[j as usize] << 0.0);
            }
        }

        let solution = model.solve().unwrap();
        solution.eval(y.iter().sum::<Expression>())
    }

    /// Calculates the guaranteed error bound (epsilon) for a given sample size.
    ///
    /// Ref: "Finding the True Frequent Itemsets" (Riondato & Vandin, 2014)
    pub fn calculate_epsilon(num_items: usize, num_samples: usize, delta: f64) -> f64 {
        // Corollary 2: The absolute worst-case VC-dimension for the power set of items.
        // VC(R(2^I)) <= |I| - 1
        let d = (num_items - 1) as f64;

        let l = num_samples as f64;

        // Theorem 1[cite: 147, 148]: Universal constant c is estimated to be <= 0.5
        let c = 0.5;

        // Equation 1[cite: 145]: epsilon = sqrt( (c / l) * (d + ln(1 / delta)) )
        // Note: We use natural log (.ln()) as is standard for Chernoff/VC bounds.
        let epsilon = ((c / l) * (d + (1.0 / delta).ln())).sqrt();

        epsilon
    }

    /// Calculates the number of samples required to hit a target epsilon and delta.
    pub fn calculate_required_samples(num_items: usize, target_epsilon: f64, delta: f64) -> usize {
        let d = (num_items - 1) as f64;
        let c = 0.5;

        // Algebraic rearrangement of Equation 1 [cite: 145] to solve for l
        let l = (c / target_epsilon.powi(2)) * (d + (1.0 / delta).ln());

        l.ceil() as usize
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::LAMA::selector::Selector;
//     use crate::LAMA::utility::encloses;
//     use rand::Rng;
//
//     #[test]
//     fn test_vc_bounds_and_empirical_reality_testdb() {
//         // 1. Setup a 10x10 test database at 50% density
//         let rows = 50;
//         let cols = 50;
//         let db = testDB::new(rows, cols, 100);
//         let boxed_db: Box<dyn Searchable + Sync> = Box::new(db);
//
//         let selector = Selector {
//             dist: "uniform".to_string(),
//             encrypted_db: &boxed_db,
//             dim: 2,
//             lowest_rec: vec![0, 0],
//             largest_rec: vec![(rows - 1) as i64, (cols - 1) as i64],
//             evc: boxed_db.get_universe().len(),
//             epsilon: 0.05,
//             delta: 0.1,
//             query_percent: -1.0,
//         };
//
//         // Universe size is the number of actual non-empty records in the DB
//         let universe_size = boxed_db.get_universe().len();
//
//         let target_epsilon = 0.15; // 5% error margin
//         let confidence = 0.70; // 90% confidence
//         let delta = 1.0 - confidence;
//
//         // Calculate theoretical required queries based on the universe size [cite: 145, 147]
//         let required_samples =
//             Selector::calculate_required_samples(universe_size, target_epsilon, delta);
//
//         println!("Universe Size: {}", universe_size);
//         println!(
//             "Theoretical queries required for <= 5% error: {}",
//             required_samples
//         );
//
//         let dom_freq_map = selector.get_dominant_pair_to_freq_map().unwrap();
//         let all_queries: Vec<DomPair> = dom_freq_map.keys().cloned().collect();
//
//         // Pick a random a response to act as our 'test'
//
//         let mut rng = rand::thread_rng();
//         let target_response_idx = (rng.gen::<u64>() % all_queries.len() as u64) as usize;
//         let target_response = boxed_db.do_search(
//             &all_queries[target_response_idx].0,
//             &all_queries[target_response_idx].1,
//         );
//
//         let mut true_tuples = Vec::new();
//
//         for tuple in target_response.iter() {
//             true_tuples.push(unflatten_nd(
//                 *tuple,
//                 &selector.largest_rec,
//                 &selector.lowest_rec,
//             ))
//         }
//
//         // let target_response = all_responses
//         //     .iter()
//         //     .find(|r| !r.is_empty())
//         //     .unwrap()
//         //     .clone();
//
//         let dom_pair = get_mbq(&*true_tuples);
//         assert_eq!(dom_pair, all_queries[target_response_idx]); // This should be obvious. Keeping it as a sanity check
//
//         // Calculate the exact TRUE probability.
//         // Under a uniform distribution, this is the count of queries returning a superset
//         // divided by the total number of possible queries (dominating pairs).
//         let mut true_frequency = dom_freq_map.get(&dom_pair).unwrap();
//         let total_dom_pairs = dom_freq_map.len() as f64;
//         let true_probability: f64 = (true_frequency.clone() as f64) / total_dom_pairs;
//
//         // 3. Empirical Simulation
//         let mut rng = rand::thread_rng();
//         let mut observed_hits = 0;
//
//         for _ in 0..required_samples {
//             // Uniformly sample a query from the map
//             let random_query_idx: usize = (rng.gen::<u64>() % all_queries.len() as u64) as usize;
//             let sampled_query = &all_queries[random_query_idx];
//
//             // If the sampled query dominates our target MBQ, it's a hit!
//             if encloses(sampled_query, &dom_pair) {
//                 observed_hits += 1;
//             }
//         }
//
//         let empirical_probability = observed_hits as f64 / required_samples as f64;
//         let actual_error = (true_probability - empirical_probability).abs();
//
//         let empirical_probability = observed_hits as f64 / required_samples as f64;
//         let actual_error = (true_probability - empirical_probability).abs();
//
//         println!("True Probability: {:.4}", true_probability);
//         println!("Empirical Probability: {:.4}", empirical_probability);
//         println!("Actual Empirical Error: {:.6}", actual_error);
//
//         // The empirical error should be vastly smaller than the worst-case epsilon bound.
//         assert!(actual_error <= target_epsilon);
//     }
//
//     #[test]
//     fn test_sukp_bound_on_testdb() {
//         // Create a 2x2 grid at 100% density.
//         // We know exactly how many geometric subsets this creates.
//         let rows = 2;
//         let cols = 2;
//         let db = testDB::new(rows, cols, 100);
//         let boxed_db: Box<dyn Searchable + Sync> = Box::new(db);
//
//         let selector = Selector {
//             dist: "uniform".to_string(),
//             encrypted_db: &boxed_db,
//             dim: 2,
//             lowest_rec: vec![0, 0],
//             largest_rec: vec![(rows - 1) as i64, (cols - 1) as i64],
//             evc: 0,
//             epsilon: 0.0,
//             delta: 0.0,
//             query_percent: -1.0,
//         };
//
//         // For a dense 2x2 grid:
//         // - Universe size (capacity) = 4 items.
//         // - Total possible bounding boxes (responses) = 9
//         //   (Four 1x1s, two 1x2s, two 2x1s, one 2x2).
//         // Since every response contains only a subset of the 4 universe items,
//         // the Set-Union Knapsack Problem can pick ALL 9 itemsets without
//         // exceeding the weight capacity of 4 items.
//
//         let q = selector.get_vc_sukp_bound();
//
//         // The maximum profit 'q' should be exactly 9.0.
//         assert!(
//             (q - 9.0).abs() < 1e-5,
//             "Expected SUKP profit of 9.0, got {}",
//             q
//         );
//
//         // VC-bound b = floor(log2(q)) + 1
//         // floor(log2(9)) = 3.
//         // b = 3 + 1 = 4.
//         let b = q.log2().floor() + 1.0;
//         assert_eq!(b, 4.0);
//     }
// }
