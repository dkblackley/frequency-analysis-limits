use crate::dataloader::{flatten_nd, unflatten_nd, Searchable};
use crate::LAMA::error::LAMAError;
use crate::LAMA::query::QueryDistribution;
use crate::LAMA::utility::{binomial_coefficient, get_mbq};
use crate::{DomPair, Frequency, Probability, Record, Value};
use good_lp::{default_solver, variable, Expression, ProblemVariables, Solution, SolverModel};
use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use log::{info, warn};
use rand::distributions::Distribution;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
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
    encrypted_db: &'a Box<dyn Searchable + Sync>,
    pub query_distribution: Box<QueryDistribution<'a>>,
}

impl<'a> Selector<'a> {
    pub fn new(dist: &str, loaded_db: &'a Box<dyn Searchable + Sync>) -> Box<Self> {
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
            // evc: (loaded_db.get_universe().len() - 1),
            query_distribution,
        };

        Box::new(selector)
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

    /// The core math logic! Returns the sum of probabilities of all queries that ENCLOSE the target MBQ.
    pub fn compute_cumulative_prob_map(
        // we're allowed to know these, because I just put a '0' if not observed...
        lowest_rec: &[Value],
        largest_rec: &[Value],
        dom_pair_to_raw_observed_weight: &FxHashMap<DomPair, f64>,
        total_weight: f64,
    ) -> FxHashMap<DomPair, Probability> {
        let mut dom_pair_to_known_prob = FxHashMap::<DomPair, Probability>::default();

        for (mbq, _weight) in dom_pair_to_raw_observed_weight {
            // Every other dist: Sum Cartesian space of enclosing queries.
            let (target_lower, target_upper) = mbq;

            let lower_combos = lowest_rec
                .iter()
                .zip(target_lower.iter())
                .map(|(&min_val, &t_val)| min_val..=t_val)
                .multi_cartesian_product();

            let upper_combos = target_upper
                .iter()
                .zip(largest_rec.iter())
                .map(|(&t_val, &max_val)| t_val..=max_val)
                .multi_cartesian_product();

            // Parallelize the evaluation of the cartesian product space
            let true_prob: f64 = lower_combos
                .cartesian_product(upper_combos)
                .par_bridge() //
                .map(|(c_lower, c_upper)| {
                    // Look up the weight, default to 0.0 if not found, then return it to be summed
                    *dom_pair_to_raw_observed_weight
                        .get(&(c_lower, c_upper))
                        .unwrap_or(&0.0)
                })
                .sum(); // Rayon handles the thread-safe accumulation here
            dom_pair_to_known_prob.insert(mbq.clone(), true_prob / total_weight);
        }
        dom_pair_to_known_prob
    }

    pub fn get_raw_dompair_weight_map(
        observed: Vec<Vec<Value>>,
        encrypted_db: &Box<dyn Searchable + Sync>,
    ) -> FxHashMap<DomPair, Probability> {
        //First, get a vec of all dompairs:

        let mut dom_pair_weight = FxHashMap::<DomPair, f64>::default();

        // to make the calculation easier we 'cheat' by peaking at the true max/min dompair
        let (low_pair, high_pair) = encrypted_db.get_dom_pair();
        for response in observed {
            let mut observed_points = Vec::new();
            for record in response {
                let unflattened = unflatten_nd(record, &high_pair, &low_pair);
                observed_points.push(unflattened);
            }
            let mbq = get_mbq(&observed_points);
            if !dom_pair_weight.contains_key(&mbq) {
                dom_pair_weight.insert(mbq, 1.0);
            } else {
                let prev = dom_pair_weight[&mbq];
                dom_pair_weight.insert(mbq, prev + 1.0);
            }
        }
        dom_pair_weight
    }

    pub fn get_responses_from_queries(&self, dom_pairs: Vec<DomPair>) -> Vec<Vec<Value>> {
        info!("Running {} queries", dom_pairs.len());
        let timer = Instant::now();

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
    pub fn get_vc_sukp_bound(&self, responses: Vec<Vec<Value>>) -> f64 {
        //let responses = self.get_all_possible_responses();
        // The size of the largest response
        let largest_resp = responses.iter().map(|s| s.len()).max().unwrap();
        // let universe_size: usize = self.encrypted_db.get_universe().len();
        let capacity = largest_resp;

        // 1. Calculate Upper Bound via LP Relaxation
        let q_upper = Self::solve_sukp_internal(&responses, false);

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

        warn!("Couldn't find exact power of 2, fallback to raw SUKP");
        // 2. Fallback to exact MILP if the bounds straddle a power of 2
        Self::solve_sukp_internal(&responses, true)
    }

    fn solve_sukp_internal(itemsets: &[Vec<Value>], is_integer: bool) -> f64 {
        let max_item = itemsets
            .iter()
            .flat_map(|r| r.iter())
            .max()
            .copied()
            .unwrap_or(0);
        let num_items = (max_item + 1) as usize;

        let capacity = itemsets.iter().map(|s| s.len()).max().unwrap_or(0);

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
    pub fn calculate_epsilon(empirical_vc: f64, num_samples: usize, delta: f64) -> f64 {
        // Corollary 2: The absolute worst-case VC-dimension for the power set of items.
        // VC(R(2^I)) <= |I| - 1
        // let d = (num_items - 1) as f64;

        let l = num_samples as f64;

        // Theorem 1: Universal constant c is estimated to be <= 0.5
        let c = 0.5;

        // Equation 1: epsilon = sqrt( (c / l) * (d + ln(1 / delta)) )
        // Note: We use natural log (.ln()) as is standard for Chernoff/VC bounds.
        let epsilon = ((c / l) * (empirical_vc + (1.0 / delta).ln())).sqrt();

        epsilon
    }

    /// Calculates the number of samples required to hit a target epsilon and delta.
    pub fn calculate_required_samples(empirical_vc: f64, target_epsilon: f64, delta: f64) -> usize {
        let c = 0.5;

        // Algebraic rearrangement of Equation 1 [cite: 145] to solve for l
        let l = (c / target_epsilon.powi(2)) * (empirical_vc + (1.0 / delta).ln());

        l.ceil() as usize
    }
}

#[cfg(test)]
mod tests {

    // #[test]
    // fn test_5x5_corner_probabilities() {
    //     let db = testDB::new(5, 5, 100);
    //     let boxed_db: Box<dyn Searchable + Sync> = Box::new(db);
    //     let selector = Selector::new("gaussian", &boxed_db, 0.0, 0.0);
    //
    //     // A 5x5 grid operates on indices 0..4
    //     let top_right = (vec![4, 4], vec![4, 4]);
    //     let bottom_left = (vec![0, 0], vec![0, 0]);
    //     let top_left = (vec![0, 4], vec![0, 4]);
    //     let bottom_right = (vec![4, 0], vec![4, 0]);
    //     let center = (vec![2, 2], vec![2, 2]);
    //
    //     let p_tr = selector.query_distribution.get_cumulative_prob(&top_right);
    //     let p_bl = selector
    //         .query_distribution
    //         .get_cumulative_prob(&bottom_left);
    //     let p_tl = selector.query_distribution.get_cumulative_prob(&top_left);
    //     let p_br = selector
    //         .query_distribution
    //         .get_cumulative_prob(&bottom_right);
    //     let p_center = selector.query_distribution.get_cumulative_prob(&center);
    //
    //     // As proven mathematically:
    //     // Total queries = 15 * 15 = 225
    //     // Corners = 1*5*1*5 = 25 queries -> 25 / 225 = 1/9 = 0.1111...
    //     // Center = 3*3*3*3 = 81 queries -> 81 / 225 = 0.36
    //
    //     // Prove corners are identical
    //     assert_eq!(p_tr, p_bl);
    //     assert_eq!(p_bl, p_tl);
    //     assert_eq!(p_tl, p_br);
    //
    //     // Prove the math perfectly matches the float math from our function
    //     assert_eq!(p_tr, 25.0 / 225.0);
    //     assert_eq!(p_center, 81.0 / 225.0);
    //
    //     // Prove center is larger than edges
    //     assert!(p_tr < p_center);
    // }
}
