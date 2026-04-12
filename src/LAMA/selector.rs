use crate::dataloader::{flatten_nd, unflatten_nd, Searchable};
use crate::LAMA::error::LAMAError;
use crate::LAMA::query::QueryDistribution;
use crate::LAMA::utility::{
    binomial_coefficient, get_mbq,
};
use crate::{DomPair, Frequency, Record, Value};
use good_lp::{
    default_solver, variable, Expression, ProblemVariables, Solution, SolverModel,
};
use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use log::info;
use rand::distributions::Distribution;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
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

        let selector = Selector {
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

        let mut response = self
            .encrypted_db
            .do_search(&self.lowest_rec, &self.largest_rec);
        response.retain(|&x| x != i64::MIN);
        response.sort_unstable();

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

                    // Get TRUE CUMULATIVE PROBABILITY!
                    let prob = self.query_distribution.get_cumulative_prob(&dom_pair);

                    if prob > 0.0 {
                        // Cast f64 to u64 for HashMap storage
                        local_map.entry(prob.to_bits()).or_default().push(t_tuple);
                    }

                    let current = processed_count.fetch_add(1, AtomicOrdering::Relaxed);
                    if current % 10_000 == 0 {
                        pb.set_message(format!("Processed {} tuples...", current));
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

        pb.finish_with_message("Done computing observed frequencies via direct MBQ.");
        info!("Precomputation complete.");

        freq_to_observed_map
    }

    /// Computes the theoretical expected frequencies for all t-tuples
    pub fn build_theoretical_t_dict(&self, t: usize) -> HashMap<u64, Vec<Vec<i64>>> {
        let (lowest_rec, largest_rec) = self.encrypted_db.get_dom_pair();

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
                .template("[{elapsed_precise}] [{bar:40}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("=> "),
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
                    let prob = self.query_distribution.get_cumulative_prob(&bounding_pair);

                    let flattened_tuple: Vec<i64> = val_tuple
                        .iter()
                        .map(|record| flatten_nd(record, &largest_rec, &lowest_rec))
                        .collect();

                    // Convert the f64 probability back to bits for the hashmap
                    local_map
                        .entry(prob.to_bits())
                        .or_default()
                        .push(flattened_tuple);

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
                .template("[{elapsed_precise}] [{bar:40}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("=> "),
        );

        let processed_count = AtomicU64::new(0);
        let largest_rec_ref = &largest_rec;
        let processed_count_ref = &processed_count;

        let pair_vec: Vec<DomPair> = domain_iter
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
                        let count = processed_count_ref.fetch_add(1, AtomicOrdering::Relaxed);
                        if count % 10_000 == 0 {
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
                .template("[{elapsed_precise}] [{bar:40}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("=> "),
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

                    // NEW: Use the cumulative probability natively
                    let prob = self.query_distribution.get_cumulative_prob(&bounding_pair);

                    let flattened_tuple: Vec<Value> = val_tuple
                        .iter()
                        .map(|record| flatten_nd(record, &*largest_rec, &*lowest_rec))
                        .collect();

                    // Convert f64 probability to bits (u64 / Frequency) for safe O(1) hashing
                    local_map
                        .entry((flattened_tuple.len() as Value, prob.to_bits()))
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

    // TODO: Sample instead.

    pub fn get_all_possible_responses(&self) -> Vec<Vec<Value>> {
        info!("Finding all possible responses...");

        let timer = Instant::now();

        let dom_pairs = Selector::get_dom_pairs(&**self.encrypted_db);

        let pb = ProgressBar::new(dom_pairs.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] [{bar:40}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("=> "),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataloader::tester::testDB;

    #[test]
    fn test_5x5_corner_probabilities() {
        let db = testDB::new(5, 5, 100);
        let boxed_db: Box<dyn Searchable + Sync> = Box::new(db);
        let selector = Selector::new("gaussian", &boxed_db, 0.0, 0.0);

        // A 5x5 grid operates on indices 0..4
        let top_right = (vec![4, 4], vec![4, 4]);
        let bottom_left = (vec![0, 0], vec![0, 0]);
        let top_left = (vec![0, 4], vec![0, 4]);
        let bottom_right = (vec![4, 0], vec![4, 0]);
        let center = (vec![2, 2], vec![2, 2]);

        let p_tr = selector.query_distribution.get_cumulative_prob(&top_right);
        let p_bl = selector
            .query_distribution
            .get_cumulative_prob(&bottom_left);
        let p_tl = selector.query_distribution.get_cumulative_prob(&top_left);
        let p_br = selector
            .query_distribution
            .get_cumulative_prob(&bottom_right);
        let p_center = selector.query_distribution.get_cumulative_prob(&center);

        // As proven mathematically:
        // Total queries = 15 * 15 = 225
        // Corners = 1*5*1*5 = 25 queries -> 25 / 225 = 1/9 = 0.1111...
        // Center = 3*3*3*3 = 81 queries -> 81 / 225 = 0.36

        // Prove corners are identical
        assert_eq!(p_tr, p_bl);
        assert_eq!(p_bl, p_tl);
        assert_eq!(p_tl, p_br);

        // Prove the math perfectly matches the float math from our function
        assert_eq!(p_tr, 25.0 / 225.0);
        assert_eq!(p_center, 81.0 / 225.0);

        // Prove center is larger than edges
        assert!(p_tr < p_center);
    }
}
