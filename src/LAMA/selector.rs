use crate::dataloader::{flatten_nd, unflatten_nd, Searchable};
use crate::LAMA::error::LAMAError;
use crate::LAMA::query::QueryDistribution;
use crate::LAMA::utility::{binomial_coefficient, get_mbq};
use crate::{DomPair, Frequency, Probability, Record, Value};
use good_lp::{
    default_solver, highs, variable, Expression, ProblemVariables, Solution, SolverModel, Variable,
};
use indicatif::{ProgressBar, ProgressStyle};
use itertools::{all, Itertools};
use log::{debug, info, trace, warn};
use rand::distributions::Distribution;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::{HashMap, HashSet};
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
            Self::get_dom_pairs(loaded_db),
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

    pub fn get_dom_pairs(encrypted_db: &Box<dyn Searchable + Sync>) -> Vec<DomPair> {
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

    pub fn compute_antichain_bound<Value: PartialEq + Clone>(
        dataset: Vec<Vec<Value>>,
        universe: Vec<Value>,
    ) -> f64 {
        let mut iter = dataset.into_iter();

        let first_tx = match iter.next() {
            Some(tx) => tx,
            None => return 0.0, // Edge case: empty dataset
        };

        // T <- {τ}
        let mut t: Vec<Vec<Value>> = vec![first_tx];

        // q <- 1
        let mut q = 1;

        // while scanIsNotComplete do (iterating over the rest)
        for tau in iter {
            //if |τ| > q and τ != I and ¬∃a ∈ T such that τ = a then
            if tau.len() > q && tau != universe && !t.contains(&tau) {
                // R <- T ∪ {τ}
                let mut r = t.clone();
                r.push(tau.clone());

                // q <- max integer such that R contains at least q transactions of length at least q
                // To compute this efficiently sort `t` by length descending.
                r.sort_unstable_by(|a, b| b.len().cmp(&a.len()));

                let mut new_q = 1;
                // The largest possible q is the length of the longest transaction (t[0].len()),
                // but it also can't be larger than the total number of transactions we have (t.len()).
                let mut test_q = std::cmp::min(r[0].len(), r.len());

                while test_q > 0 {
                    let mut all_valid = true;

                    // Explicitly check the first `test_q` itemsets
                    for tx in r.iter().take(test_q) {
                        if tx.len() < test_q {
                            all_valid = false;
                            break;
                        }
                    }

                    // If they all passed the length check, we found our max integer!
                    if all_valid {
                        new_q = test_q;
                        break;
                    }

                    // If not, decrease the target by 1 and repeat
                    test_q -= 1;
                }

                q = new_q;

                // T <- set of the q longest transactions from R
                r.truncate(q);
                t = r;
            }
        }

        // 12. return q (cast to f64 to match your required signature)
        q as f64
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
            if observed_points.len() == 0 {
                warn!("Saw nothing?");
                continue;
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

        let dom_pairs = Selector::get_dom_pairs(self.encrypted_db);

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

    pub fn sample_percent_responses(
        &self,
        target_query_percentage: f64,
        delta: f64,
        all_possible_responses: &Vec<Vec<Value>>,
    ) -> (Vec<DomPair>, Vec<Vec<Value>>, f64) {
        let largest_dom = self.encrypted_db.get_dom_pair();

        // How many queries can we do? A query is a hyperrectangle, The total number of hyper rectangles
        // in a space is n*(n+1)/2 where n is the length of a dim, then mutliply for each dim.
        let mut total = 1;

        for item in largest_dom.1 {
            total *= ((item + 1) * (item + 2)) / 2;
        }

        let total_possible_queries = total;
        let num_queries_to_observe =
            ((total_possible_queries as f64) * target_query_percentage).ceil() as usize;

        info!("--- Sampling Parameters ---");
        info!(
            "Targeting {}% of queries ({} samples).",
            target_query_percentage * 100.0,
            num_queries_to_observe
        );

        // 5. Sample the specific number of queries based on the distribution weights
        let mut rng = rand::thread_rng();
        let mut observed_queries: Vec<DomPair> = Vec::with_capacity(num_queries_to_observe);

        info!("Sampling {} queries...", num_queries_to_observe);
        for _ in 0..num_queries_to_observe {
            // Use the WeightedIndex sampler you built in QueryDistribution
            let sampled_idx = self.query_distribution.sampler.sample(&mut rng);
            let sampled_pair = self.query_distribution.pairs[sampled_idx].clone();
            observed_queries.push(sampled_pair);
        }

        // 3. Calculate Empirical VC Dimension (Fixing Issue A)
        let responses = self.get_responses_from_queries(observed_queries.clone());
        let emp_vc_dim = self.get_emp_vc_sukp_bound(responses.clone(), all_possible_responses);

        // A simple bound as per corollary 2
        let vc_dim =
            Selector::simple_bound_corollary2(&responses, &self.encrypted_db.get_universe());

        let eps_empr = Selector::calculate_epsilon_emp_vc(
            emp_vc_dim, // Pass the EVC
            num_queries_to_observe,
            delta,
        );

        let eps_reg = Selector::calculate_epsilon_real_vc(vc_dim, num_queries_to_observe, delta);

        let eps = eps_reg.min(eps_empr);

        info!(
            "VC Dimension: {}, Empirical VC Dimension: {}",
            vc_dim, emp_vc_dim
        );
        info!(
            "Empirical Epsilon Bound: {}, Regular Epsilon Bound: {}",
            eps_empr, eps_reg
        );
        info!("---------------------------");

        return (observed_queries, responses, eps);
    }

    pub fn simple_bound_corollary2(responses: &Vec<Vec<Value>>, universe: &Vec<Value>) -> f64 {
        let mut total = 0;
        for record in universe {
            for response in responses {
                if response.contains(record) {
                    total += 1;
                }
            }
        }
        let q = Self::solve_sukp_internal(&responses, total);
        q.log2().floor() + 1.0
    }

    /// Evaluates the SUKP to find the bounding profit `q` for empirical VC-dimension.
    /// What we need for setup: Universe - iterate over the response and set of encrypted records.
    /// Set of elements that are subsets of U. The set of responses. Weights and profits are 1 for
    /// everything. Weights are assigned to encrypted records but profits to subsets. Capacity -
    /// From corollary 1 we know that the capacity we want to solve for is the largest response. TO
    /// get a tighter bound, we have to iterate over each l and L.
    pub fn get_emp_vc_sukp_bound(
        &self,
        responses: Vec<Vec<Value>>,
        all_respones: &Vec<Vec<Value>>,
    ) -> f64 {
        // 1. Find all distinct transaction lengths (ell_i) (computed wrt D)
        let mut distinct_lengths = Vec::new();
        for response in all_respones {
            let len = response.len();
            let mut found = false;

            // Basic loop to check if we already recorded this lengt
            // each length is indexed by i, as in the paper
            for i in 0..distinct_lengths.len() {
                if distinct_lengths[i] == len {
                    found = true;
                    break;
                }
            }

            if !found {
                distinct_lengths.push(len);
            }
        }
        // Sort lengths in decreasing order so: ell_1 > ell_2 > ... > ell_w
        distinct_lengths.sort_unstable_by(|a, b| b.cmp(a));

        let pb = ProgressBar::new(distinct_lengths.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] [{bar:40}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("=> "),
        );

        info!("Finding all responses of length ell with ");

        // Now we want capital L_i, that is, for each l_i, find all transactions of length AT LEAST
        // l_i and then find the largest set such that no to items are subset of each-other.

        let l_sets: Vec<_> = distinct_lengths
            .par_iter()
            .map(|&current_l| {
                let mut transactions_at_least_l = Vec::new();

                for tx in all_respones {
                    if tx.len() >= current_l {
                        transactions_at_least_l.push(tx);
                    }
                }

                let mut largest_set_no_subsets = Vec::new();

                // Cleaned up: .enumerate() gives us the index (j) and the transaction (current_tx)
                for (j, &current_tx) in transactions_at_least_l.iter().enumerate() {
                    let mut is_subset_of_something_else = false;

                    for (k, &other_tx) in transactions_at_least_l.iter().enumerate() {
                        if j == k {
                            continue;
                        }

                        if current_tx.len() <= other_tx.len() {
                            // --- THE MAGIC LINE ---
                            // This says: "For every item in current_tx, does other_tx contain it?"
                            // It immediately stops and returns false if it finds a missing item.
                            let is_subset = current_tx.iter().all(|item| other_tx.contains(item));

                            if is_subset {
                                if other_tx.len() > current_tx.len() || k < j {
                                    is_subset_of_something_else = true;
                                    break;
                                }
                            }
                        }
                    }

                    if !is_subset_of_something_else {
                        largest_set_no_subsets.push(current_tx);
                    }
                }

                // 3. Update the progress bar inside the thread and return the result
                // (If `pb` is an `indicatif::ProgressBar`, it is completely thread-safe)
                pb.inc(1);

                largest_set_no_subsets
            })
            .collect(); // 4. Collect gathers the parallel results back into a Vec in the correct order
        let pb = ProgressBar::new(distinct_lengths.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] [{bar:40}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("=> "),
        );

        return (0..distinct_lengths.len())
            .into_par_iter()
            .find_map_first(|j| {
                let ell_i = distinct_lengths[j];
                let cap_l_i = l_sets[j][0].len();

                // 4. Compute q_i (Optimal SUKP profit with capacity ell_i)
                // Note: `responses` is captured by reference, which is perfectly safe in Rayon
                let q_i = Self::solve_sukp_internal(&responses, ell_i);

                // Progress bar increments safely from any thread
                pb.inc(1);

                // Safety check: log2(0) is negative infinity, so handle a profit of 0
                let b_i = if q_i <= 0.0 {
                    0.0
                } else {
                    // 5. Compute b_i = floor(log_2(q_i)) + 1
                    q_i.log2().floor() + 1.0
                };

                // 6. Lemma 1 check
                // If the condition is met, we return Some(b_i).
                // Rayon will stop processing remaining chunks and return this value.
                if b_i <= cap_l_i as f64 {
                    Some(b_i)
                } else {
                    trace!(
                        "Missed profit val, log2 profit was {} and cap was {}",
                        b_i,
                        cap_l_i
                    );
                    None
                }
            })
            .unwrap_or(0.0); // Fallback if no such j is found
    }

    fn solve_sukp_internal(itemsets: &[Vec<Value>], capacity: usize) -> f64 {
        // 1. Gather unique items and create a stable mapping to indices 0..num_items
        let mut unique_items = HashSet::new();
        for itemset in itemsets {
            for item in itemset {
                unique_items.insert(*item);
            }
        }

        let item_to_index: HashMap<Value, usize> = unique_items
            .into_iter()
            .enumerate()
            .map(|(idx, item)| (item, idx))
            .collect();

        let num_items = item_to_index.len();
        let mut vars = ProblemVariables::new();

        let x: Vec<Variable> = (0..num_items)
            .map(|_| vars.add(variable().min(0.0).max(1.0)))
            .collect();

        let y: Vec<Variable> = (0..itemsets.len())
            .map(|_| vars.add(variable().min(0.0).max(1.0)))
            .collect();

        let objective: Expression = y.iter().sum();

        // Note: Consider using highs instead of default_solver for better performance
        let mut model = highs(vars.maximise(objective));

        let weight_expr: Expression = x.iter().sum();
        model = model.with(weight_expr << capacity as f64);

        for (i, itemset) in itemsets.iter().enumerate() {
            for item in itemset {
                // 2. Safely look up the contiguous index for this specific item
                let safe_j = item_to_index[item];
                model = model.with(y[i] - x[safe_j] << 0.0);
            }
        }

        let solution = model.solve().unwrap();
        solution.eval(y.iter().sum::<Expression>())
    }

    /// Calculates the guaranteed error bound (epsilon) for a given sample size.
    ///
    /// Ref: "Finding the True Frequent Itemsets" (Riondato & Vandin, 2014)
    pub fn calculate_epsilon_real_vc(vc: f64, num_samples: usize, delta: f64) -> f64 {
        let l = num_samples as f64;

        // Theorem 1: Universal constant c is estimated to be <= 0.5
        let c = 0.5;

        // Equation 1: epsilon = sqrt( (c / l) * (d + ln(1 / delta)) )
        // Note: We use natural log (.ln()) as is standard for Chernoff/VC bounds.
        let epsilon = ((c / l) * (vc + (1.0 / delta).ln())).sqrt();

        epsilon
    }

    pub fn calculate_epsilon_emp_vc(d: f64, num_samples: usize, delta: f64) -> f64 {
        let l = num_samples as f64;

        let part_1 = 2.0 * ((2.0 * d * (l + 1.0).ln()) / l).sqrt();
        let part_2 = ((2.0 * (2.0 / delta).ln()) / l).sqrt();

        let epsilon = part_1 + part_2;

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
    use crate::dataloader::tester::testDB;
    use crate::dataloader::Searchable;
    use crate::LAMA::selector::Selector;

    use super::*;

    #[cfg(test)]
    mod statistical_bounds_tests {
        use super::*;
        use crate::dataloader::tester::testDB;
        use crate::LAMA::utility::{dominates, encloses, get_mbq};
        use log::error;
        use rand::Rng;

        #[test]
        fn test_simple_fit_all() {
            // All items fit within the capacity.
            let itemsets = vec![vec![1, 2], vec![2, 3]];
            let capacity = 5;

            assert_eq!(Selector::solve_sukp_internal(&itemsets, capacity), 2.0);
        }

        #[test]
        fn test_overlapping_sets() {
            // Sets 0 and 1 share a lot of items. Set 2 is totally disjoint.
            let itemsets = vec![
                vec![1, 2, 3],      // Set 0
                vec![2, 3, 4],      // Set 1
                vec![8, 9, 10, 11], // Set 2
            ];
            let capacity = 4;

            // Upper bound will be >= the exact answer. By relaxing integers, it might
            // take fractions of Set 2, yielding a slightly higher theoretical profit limit.
            let upper_bound = Selector::solve_sukp_internal(&itemsets, capacity);
            assert!(upper_bound >= 2.0);
        }

        #[test]
        fn test_capacity_too_small() {
            // Impossible to pick even a single set.
            let itemsets = vec![vec![1, 2, 3], vec![4, 5, 6]];
            let capacity = 2;

            assert_eq!(Selector::solve_sukp_internal(&itemsets, capacity), 0.0);
        }

        #[test]
        fn verify_vc_cumulative_probability_bounds() {
            let _ = env_logger::builder()
                .is_test(true)
                .filter_level(log::LevelFilter::Info)
                .try_init();

            // 1. Setup a dense 10x10 DB to ensure we have plenty of overlapping queries
            let dim = 2;
            let size_per_dim = 10;
            let db: Box<dyn Searchable + Sync + 'static> =
                Box::new(testDB::new(dim, size_per_dim, 100));

            // Use uniform distribution to easily verify the true probabilities
            let mut selector = Selector::new("beta", &db);

            // 2. Sample 15% of the total query space
            let target_pct = 0.15;
            let total_queries = selector.query_distribution.pairs.len();
            let num_samples = ((total_queries as f64) * target_pct).ceil() as usize;

            let mut rng = rand::thread_rng();
            let mut sampled_queries = Vec::with_capacity(num_samples);
            for _ in 0..num_samples {
                let idx = selector.query_distribution.sampler.sample(&mut rng);
                sampled_queries.push(selector.query_distribution.pairs[idx].clone());
            }

            // 3. Calculate VC bounds based on the sample
            let responses = selector.get_responses_from_queries(sampled_queries.clone());
            let all_resposnes = selector.get_all_possible_responses();
            let empirical_vc = selector.get_emp_vc_sukp_bound(responses, &all_resposnes);
            // let empirical_vc = q_profit.log2().floor() + 1.0;

            let delta = 0.1; // 90% confidence that the maximum error across ALL itemsets <= epsilon
            let epsilon = Selector::calculate_epsilon_real_vc(empirical_vc, num_samples, delta);

            info!("Total Query Space: {}", total_queries);
            info!("Sample Size: {}", num_samples);
            info!("Empirical VC Dim: {}", empirical_vc);
            info!("Guaranteed Epsilon Error Bound: {}", epsilon);

            // 4. Generate 100 random itemsets (t=2 tuples) to test the theorem
            let universe = db.get_universe();
            let mut test_tuples = Vec::new();
            for _ in 0..100 {
                let pt_a = universe[rng.gen_range(0..universe.len())];
                let pt_b = universe[rng.gen_range(0..universe.len())];
                test_tuples.push(vec![pt_a, pt_b]);
            }

            let (low_pair, high_pair) = db.get_dom_pair();
            let mut violations = 0;
            let mut max_observed_error = 0.0;

            // 5. Run the Cumulative Probability Verification
            for tuple in test_tuples {
                // Unflatten to get actual coordinates to compute the Minimum Bounding Query (MBQ)
                let pt_a = unflatten_nd(tuple[0], &high_pair, &low_pair);
                let pt_b = unflatten_nd(tuple[1], &high_pair, &low_pair);
                let target_mbq = get_mbq(&[pt_a, pt_b]);

                // A. True Cumulative Probability (Calculated by your exact prefix-sum algorithm)
                let p_true = selector
                    .query_distribution
                    .cumulative_prob_lookup(&target_mbq);

                // B. Empirical Cumulative Probability (Calculated by observing our sample)
                let mut enclose_count = 0;
                for obs_mbq in &sampled_queries {
                    // The "Transaction" (obs_mbq) contains the "Itemset" if the query's
                    // bounding box stretches beyond the itemset's Minimum Bounding Query.
                    if encloses(obs_mbq, &target_mbq) {
                        enclose_count += 1;
                    }
                }

                let p_emp = (enclose_count as f64) / (num_samples as f64);

                // C. Validate Theorem
                let error = (p_emp - p_true).abs();
                if error > max_observed_error {
                    max_observed_error = error;
                }

                if error > epsilon {
                    violations += 1;
                    error!("VIOLATION! Tuple: {:?} | p_true: {:.4} | p_emp: {:.4} | error: {:.4} > eps: {:.4}",
                       tuple, p_true, p_emp, error, epsilon);
                }
            }

            info!(
                "Maximum observed error: {:.4} (Allowed: {:.4})",
                max_observed_error, epsilon
            );
            info!("Total Violations: {} / 100", violations);

            // The theorem states that with 1 - delta probability, the error for ALL itemsets is <= epsilon.
            // Assuming we didn't hit the 10% bad luck draw, violations should be exactly 0.
            assert_eq!(
                violations, 0,
                "The empirical probability deviated beyond the guaranteed VC bound!"
            );
        }
    }
}
