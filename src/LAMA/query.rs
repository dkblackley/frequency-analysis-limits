// Adjust imports as necessary for DomPair
use crate::dataloader::tester::testDB;
use crate::dataloader::{flatten_dompair, flatten_nd, unflatten_nd, Searchable};
use crate::LAMA::selector::Selector;
use crate::LAMA::utility::{encloses, get_mbq, DistributionType};
use crate::{Coord, DomPair, Probability, Record, Value};
use indicatif::{ParallelProgressIterator, ProgressBar};
use itertools::Itertools;
use log::{debug, error, info, warn};
use rand::distributions::WeightedIndex;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use statrs::distribution::{Beta, Continuous, Normal};
use std::collections::{BTreeMap, HashMap, HashSet};

pub struct QueryDistribution<'a> {
    encrypted_db: &'a Box<dyn Searchable + Sync>,
    pub pairs: Vec<DomPair>,
    weights: Vec<f64>,
    pub dom_pair_to_known_raw_weight: FxHashMap<DomPair, f64>,
    pub mbq_to_cumulative_prob: FxHashMap<DomPair, Probability>,
    pub cumulative_probs_and_dom_pairs: Vec<(Probability, DomPair)>,
    pub total_weight: f64,
    pub sampler: WeightedIndex<f64>,
    pub dist: DistributionType,
    pub lowest_rec: Vec<Value>,
    pub largest_rec: Vec<Value>,
}

impl<'a> QueryDistribution<'a> {
    pub fn new(
        pairs: Vec<DomPair>,
        encrypted_db: &'a Box<dyn Searchable + Sync>,
        dist: DistributionType,
    ) -> Box<Self> {
        Self::new_with_noise(pairs, encrypted_db, dist, 0.0)
    }

    /// Same as `new`, but the per-query probabilities are perturbed so that the result is exactly
    /// `noise` away (in total variation distance) from `dist`. Every cumulative (enclosing)
    /// probability therefore differs from the noiseless one by at most `noise`.
    pub fn new_with_noise(
        pairs: Vec<DomPair>,
        encrypted_db: &'a Box<dyn Searchable + Sync>,
        dist: DistributionType,
        noise: f64,
    ) -> Box<Self> {
        let (lowest_rec, largest_rec) = encrypted_db.get_dom_pair();

        info!("Beginning to set up {dist} distribution");
        let (_probs_and_dom_pairs_raw, sampler, weights, total_weight, dom_pair_to_known_weight) =
            match dist {
                DistributionType::Uniform => Self::new_uniform_internal(pairs.clone()),
                DistributionType::Gaussian => Self::new_gaussian(&pairs),
                DistributionType::Beta => Self::new_beta(&pairs),
                DistributionType::Flat => Self::new_flat(&pairs, encrypted_db),
            };

        let (sampler, weights, total_weight, dom_pair_to_known_weight) = if noise > 0.0 {
            Self::add_noise(&pairs, &weights, total_weight, noise)
        } else {
            (sampler, weights, total_weight, dom_pair_to_known_weight)
        };

        info!("Computing true cumulative probabilities for every MBQ");

        let dims = lowest_rec.len();

        // Create a vector of vectors, one for each dimension
        let mut sorted_by_low_pairs = vec![pairs.to_vec(); dims];
        let mut sorted_by_high_pairs = vec![pairs.to_vec(); dims];

        // Sort each inner vector by its respective dimension
        for dim in 0..dims {
            sorted_by_low_pairs[dim].sort_unstable_by_key(|p| p.0[dim]);
            sorted_by_high_pairs[dim].sort_unstable_by(|a, b| b.1[dim].cmp(&a.1[dim]));
        }

        let mbq_to_cumulative_prob = Self::precompute_cumulative_results(
            &sorted_by_low_pairs,
            &sorted_by_high_pairs,
            &dom_pair_to_known_weight,
            &lowest_rec,
            &largest_rec,
            total_weight,
            false,
        );

        debug!("Finished computing true cumulative probabilities");
        // Sequential Insertion Phase
        let mut cumulative_probs_and_dom_pairs = Vec::with_capacity(pairs.len());
        for (pair, &cum_prob) in &mbq_to_cumulative_prob {
            cumulative_probs_and_dom_pairs.push((cum_prob, pair.clone()));
        }
        // probs_and_dom_pairs.par_sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        debug!("Done pre-computing probability dist");
        Box::new(Self {
            encrypted_db,
            pairs,
            weights,
            dom_pair_to_known_raw_weight: dom_pair_to_known_weight,
            mbq_to_cumulative_prob,
            cumulative_probs_and_dom_pairs,
            total_weight,
            sampler,
            dist,
            lowest_rec,
            largest_rec,
        })
    }

    /// Dynamically precomputes the cumulative probabilities for all possible DomPairs
    /// using an N-dimensional prefix sum approach.
    /// Dynamically precomputes the cumulative probabilities for all possible DomPairs
    /// using an N-dimensional prefix sum approach.
    fn precompute_cumulative_results(
        sorted_by_lowest: &[Vec<DomPair>],  // Notice the type change here
        sorted_by_highest: &[Vec<DomPair>], // Notice the type change here
        dom_pair_to_known_weight: &FxHashMap<DomPair, f64>,
        lowest_rec: &[Value],
        largest_rec: &[Value],
        total_weight: f64,
        return_weights: bool,
    ) -> FxHashMap<DomPair, Probability> {
        let mut dp = dom_pair_to_known_weight.clone();

        // Safety check to ensure we have data
        if sorted_by_highest.is_empty() || sorted_by_highest[0].is_empty() {
            return dp;
        }

        let dims = lowest_rec.len();

        // 1. Sweep over LOWER bounds
        for dim in 0..dims {
            // Grab the vector specifically sorted for THIS dimension
            for pair in &sorted_by_lowest[dim] {
                if pair.0[dim] > lowest_rec[dim] {
                    let mut prev_pair = pair.clone();
                    prev_pair.0[dim] -= 1;

                    if let Some(&prev_val) = dp.get(&prev_pair) {
                        let current_val = *dp.get(pair).unwrap_or(&0.0);
                        dp.insert(pair.clone(), current_val + prev_val);
                    }
                }
            }
        }

        // 2. Sweep over UPPER bounds
        for dim in 0..dims {
            // Grab the vector specifically sorted for THIS dimension
            for pair in &sorted_by_highest[dim] {
                if pair.1[dim] < largest_rec[dim] {
                    let mut next_pair = pair.clone();
                    next_pair.1[dim] += 1;

                    if let Some(&next_val) = dp.get(&next_pair) {
                        let current_val = *dp.get(pair).unwrap_or(&0.0);
                        dp.insert(pair.clone(), current_val + next_val);
                    }
                }
            }
        }

        let mut final_probs = FxHashMap::default();

        if return_weights {
            for (pair, cumulative_weight) in dp {
                final_probs.insert(pair, cumulative_weight);
            }
        } else {
            for (pair, cumulative_weight) in dp {
                final_probs.insert(pair, cumulative_weight / total_weight);
            }
        }

        final_probs
    }

    /// The core math logic! Returns the sum of probabilities of all queries that ENCLOSE the target MBQ.
    fn compute_cumulative_prob(
        mbq: &DomPair,
        dist: &DistributionType,
        lowest_rec: &[Value],
        largest_rec: &[Value],
        dom_pair_to_known_prob: &FxHashMap<DomPair, f64>,
        //dom_pair_slice: &[(DomPair, f64)], // Now a sorted flat slice
        total_weight: f64,
    ) -> Probability {
        match dist {
            DistributionType::Uniform => {
                // This is faster than a hash lookup for uniform (I think)
                let mut dominated_vals: u64 = 1;
                for (&l_val, &min_val) in mbq.0.iter().zip(lowest_rec.iter()) {
                    dominated_vals *= (l_val - min_val + 1) as u64;
                }
                let mut dominating_vals: u64 = 1;
                for (&u_val, &max_val) in mbq.1.iter().zip(largest_rec.iter()) {
                    dominating_vals *= (max_val - u_val + 1) as u64;
                }
                let count = (dominated_vals * dominating_vals) as f64;
                count / total_weight
            }

            _ => {
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
                    //.par_bridge() //
                    .map(|(c_lower, c_upper)| {
                        // Look up the weight, default to 0.0 if not found, then return it to be summed
                        *dom_pair_to_known_prob
                            .get(&(c_lower, c_upper))
                            .unwrap_or(&0.0)
                    })
                    .sum(); // Rayon handles the thread-safe accumulation here

                true_prob / total_weight
            }
        }
    }

    pub fn cumulative_prob_lookup(&self, query: &DomPair) -> Probability {
        self.mbq_to_cumulative_prob.get(query).unwrap().clone()
    }

    // pub fn get_cumulative_prob(&self, p0: &DomPair) -> Probability {
    //     *self.mbq_to_cumulative_prob.get(p0).unwrap_or(&0.0)
    // }

    pub fn get_candidate_pairs_by_probability(
        &self,
        observed_prob: Probability,
        epsilon: f64,
    ) -> Vec<DomPair> {
        let lower_bound = observed_prob - epsilon;
        let upper_bound = observed_prob + epsilon;

        let start_idx = self
            .cumulative_probs_and_dom_pairs
            .partition_point(|x| x.0 < lower_bound);
        let end_idx = self
            .cumulative_probs_and_dom_pairs
            .partition_point(|x| x.0 <= upper_bound);

        self.cumulative_probs_and_dom_pairs[start_idx..end_idx]
            .iter()
            .map(|(_, pair)| pair.clone())
            .collect()
    }

    /// Mixes the distribution `p` (from `weights`) with a random one `q`:
    /// p' = (1 - lambda) p + lambda q, with lambda = noise / TV(p, q), so TV(p, p') = noise.
    /// Seeded, so the same `noise` always gives the same noisy distribution.
    fn add_noise(
        pairs: &[DomPair],
        weights: &[f64],
        total_weight: f64,
        noise: f64,
    ) -> (WeightedIndex<f64>, Vec<f64>, f64, FxHashMap<DomPair, f64>) {
        let mut rng = StdRng::seed_from_u64(1337);
        // Hand out the random weights in sorted-pair order: `pairs` comes from get_dom_pairs (par_bridge),
        // whose order changes per run, so indexing by position would give a different Q every run.
        let mut order: Vec<usize> = (0..pairs.len()).collect();
        order.sort_unstable_by(|&a, &b| pairs[a].cmp(&pairs[b]));
        let mut random_weights = vec![0.0; weights.len()];
        for idx in order {
            random_weights[idx] = rng.gen::<f64>();
        }
        let random_total: f64 = random_weights.iter().sum();

        let tv_pq: f64 = weights
            .iter()
            .zip(random_weights.iter())
            .map(|(w, r)| (w / total_weight - r / random_total).abs())
            .sum::<f64>()
            / 2.0;

        let lambda = noise / tv_pq;
        if lambda > 1.0 {
            warn!("Noise {noise} is larger than TV(p, q) = {tv_pq}, using q as the distribution");
        }
        let lambda = lambda.min(1.0);
        info!("Adding noise: TV(p, q) = {tv_pq}, lambda = {lambda}, TV(p, p') = {}", lambda * tv_pq);

        let noisy_weights: Vec<f64> = weights
            .iter()
            .zip(random_weights.iter())
            .map(|(w, r)| (1.0 - lambda) * (w / total_weight) + lambda * (r / random_total))
            .collect();

        let mut dom_to_weight = FxHashMap::default();
        for (pair, w) in pairs.iter().zip(noisy_weights.iter()) {
            dom_to_weight.insert(pair.clone(), *w);
        }
        let sampler = WeightedIndex::new(&noisy_weights).expect("Failed to create WeightedIndex");

        // The noisy weights are already normalised
        (sampler, noisy_weights, 1.0, dom_to_weight)
    }

    fn new_uniform_internal(
        pairs: Vec<DomPair>,
    ) -> (
        Vec<(f64, DomPair)>,
        WeightedIndex<f64>,
        Vec<f64>,
        f64,
        FxHashMap<DomPair, f64>,
    ) {
        let weights = vec![1.0; pairs.len()];
        let weights_clone = weights.clone();
        let total_weight = pairs.len() as f64;
        let sampler = WeightedIndex::new(&weights).unwrap();

        let probs_and_pairs: Vec<_> = pairs
            .into_par_iter()
            .progress()
            .zip(weights.par_iter())
            // Destructure the reference here with &w
            .map(|(p, &w)| (w, p))
            .collect();
        let mut dom_pair_to_known_prob: FxHashMap<DomPair, f64> = FxHashMap::default();
        for (prob, pair) in &probs_and_pairs {
            dom_pair_to_known_prob.insert(pair.clone(), *prob);
        }

        (
            probs_and_pairs,
            sampler,
            weights_clone,
            total_weight,
            dom_pair_to_known_prob,
        )
    }

    pub fn new_gaussian(
        pairs: &[DomPair],
    ) -> (
        Vec<(f64, DomPair)>,
        WeightedIndex<f64>,
        Vec<f64>,
        f64,
        FxHashMap<DomPair, f64>,
    ) {
        // Params taken from REMIN
        let mu = pairs.len() as f64 / 2.0;
        let sigma = pairs.len() as f64 / 5.0;
        let mut dom_to_weight = FxHashMap::default();

        let normal_dist = Normal::new(mu, sigma).expect("Sigma must be > 0.0");
        debug!("Finished creating normal distribution");

        // Calculate the 'weights' - The actual prob of sampling this index/Dom pair
        let weights: Vec<f64> = pairs
            .par_iter()
            .progress()
            .enumerate()
            .map(|(i, _)| normal_dist.pdf(i as f64))
            .collect();
        for i in 0..weights.len() {
            dom_to_weight.insert(pairs[i].clone(), weights[i]);
        }

        debug!("Finished calculating weights");

        let total_weight: f64 = weights.par_iter().progress().sum();
        debug!("Finished calculating total weight");

        let sampler = WeightedIndex::new(&weights).expect("Failed to create WeightedIndex");
        debug!("Finished creating WeightedIndex sampler");

        // Store each pairs probability of being sampled in a vec
        let mut weight_pair: Vec<(f64, DomPair)> = weights
            .par_iter()
            .copied()
            .zip(pairs.par_iter().cloned())
            .progress()
            .collect();
        weight_pair.par_sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        debug!("Finished storing and sorting weight pairs");

        (weight_pair, sampler, weights, total_weight, dom_to_weight)
    }

    pub fn new_beta(
        pairs: &[DomPair],
    ) -> (
        Vec<(f64, DomPair)>,
        WeightedIndex<f64>,
        Vec<f64>,
        f64,
        FxHashMap<DomPair, f64>,
    ) {
        // The other methods were slightly 'off' when calculating this dist I think.
        let beta_dist = Beta::new(2.0, 1.0).expect("Alpha/Beta must be > 0.0");
        let mut dom_to_weight = FxHashMap::default();

        let n = pairs.len() as f64;
        let weights: Vec<f64> = pairs
            .par_iter()
            .progress()
            .enumerate()
            .map(|(i, _)| {
                // this prevents 0 prob or 1 prob (I think)
                let x = (i as f64 + 0.5) / n;
                beta_dist.pdf(x)
            })
            .collect();
        for i in 0..weights.len() {
            dom_to_weight.insert(pairs[i].clone(), weights[i]);
        }
        debug!("Finished calculating weights");

        let total_weight: f64 = weights.par_iter().progress().sum();
        debug!("Finished calculating total weight");

        let sampler = WeightedIndex::new(&weights).expect("Failed to create WeightedIndex");
        debug!("Finished creating WeightedIndex sampler");

        let mut weight_pair: Vec<(f64, DomPair)> = weights
            .par_iter()
            .copied()
            .zip(pairs.par_iter().cloned())
            .progress()
            .collect();
        debug!("Finished storing weight pairs");
        weight_pair.par_sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        debug!("Finished storing and sorting weight pairs");

        (weight_pair, sampler, weights, total_weight, dom_to_weight)
    }

    pub fn new_flat(
        pairs: &[DomPair],
        encrypted_db: &'a Box<dyn Searchable + Sync + 'static>,
    ) -> (
        Vec<(f64, DomPair)>,
        WeightedIndex<f64>,
        Vec<f64>,
        f64,
        FxHashMap<DomPair, f64>,
    ) {
        debug!("Making internal QD for flatten");
        let old_qd = Self::new(Vec::from(pairs), encrypted_db, DistributionType::Uniform);
        let mut mapping = old_qd.dom_pair_to_known_raw_weight;
        let mut total_weight = old_qd.total_weight;

        // as per algorithm 2: Start at the largest possible query and work back. THis updates
        // the old QD as we go and works over every pair, so time might be n^2
        let (low_pair, high_pair) = encrypted_db.get_dom_pair();
        let all_recs = encrypted_db.do_search(&low_pair, &high_pair);
        let largest_l1 = taxicab_distance(&low_pair, &high_pair);

        fn taxicab_distance(v1: &[i64], v2: &[i64]) -> u64 {
            v1.iter().zip(v2.iter()).map(|(a, b)| a.abs_diff(*b)).sum()
        }

        debug!("About to start big loop...");
        // make a 'dummy hashmap'. Index is just the distance 'd' and we get: the two records,
        // as a tuple, and the mbq that covers them.
        let mut distance_groups: Vec<Vec<((Value, Value), DomPair)>> =
            vec![Vec::new(); largest_l1 as usize + 1];

        let domain_iter = low_pair
            .iter()
            .zip(high_pair.iter())
            .map(|(&low, &high)| low..=high)
            .multi_cartesian_product();

        let all_possible_coords: Vec<Vec<i64>> = domain_iter.collect();

        // 2. Iterate over the entire domain, not just the DB records
        for outer in &all_possible_coords {
            for inner in &all_possible_coords {
                let taxicab = taxicab_distance(outer, inner);
                let query = get_mbq(&[outer.clone(), inner.clone()]);

                // You will need to flatten these coordinates if your distance_groups
                // expects the 1D Value representation
                let flat_outer = flatten_nd(outer, &high_pair, &low_pair);
                let flat_inner = flatten_nd(inner, &high_pair, &low_pair);

                distance_groups[taxicab as usize].push(((flat_outer, flat_inner), query));
            }
        }

        let dims = low_pair.len();

        let mut sorted_by_low_pairs = vec![pairs.to_vec(); dims];
        let mut sorted_by_high_pairs = vec![pairs.to_vec(); dims];

        for dim in 0..dims {
            sorted_by_low_pairs[dim].sort_unstable_by_key(|p| p.0[dim]);
            sorted_by_high_pairs[dim].sort_unstable_by(|a, b| b.1[dim].cmp(&a.1[dim]));
        }

        let pb = ProgressBar::new(largest_l1 + 1);

        for d in (0..=largest_l1).rev() {
            let cumul_weight_mapping = Self::precompute_cumulative_results(
                &sorted_by_low_pairs,
                &sorted_by_high_pairs,
                &mapping,
                &low_pair,
                &high_pair,
                total_weight,
                true,
            );

            let initial_tmx = encrypted_db.get_dom_pair();

            // If there are no pairs with this distance, just skip to the next 'd'
            let pairs_for_d: &Vec<((Value, Value), DomPair)> = {
                match distance_groups.get(d as usize) {
                    Some(pairs) => pairs,
                    None => {
                        warn!(
                            "Skipped all items of l1 distance {d}, Are you using a sparse dataset?"
                        );
                        pb.inc(1);
                        continue;
                    }
                }
            };

            let (equi_dist_pairs, unique_queries, max_weight, tmx) = pairs_for_d
                .par_iter()
                .map(|(original_pair, query)| {
                    let cand_prob_val = cumul_weight_mapping.get(query).unwrap() / total_weight;

                    // Wrap the single query in a HashSet so it matches the reduce type!
                    let mut unique_set = HashSet::new();
                    unique_set.insert(query.clone());

                    // Return: (Vec, HashSet, f64, DomPair)
                    (
                        vec![*original_pair],
                        unique_set,
                        cand_prob_val,
                        query.clone(),
                    )
                })
                .reduce(
                    || (Vec::new(), HashSet::new(), -1.0, initial_tmx.clone()),
                    |mut a, mut b| {
                        a.0.extend(b.0);
                        a.1.extend(b.1.into_iter()); // Combine the HashSets properly
                        if b.2 > a.2 {
                            a.2 = b.2;
                            a.3 = b.3;
                        }
                        a
                    },
                );

            // As per line 3 - the sum of weights covering tmx
            let smx = cumul_weight_mapping.get(&tmx).unwrap();

            for mbq in unique_queries.iter() {
                let st = cumul_weight_mapping.get(mbq).unwrap();
                let old_weight = mapping.get(&mbq).unwrap();
                mapping.insert(mbq.clone(), old_weight.clone() + (smx - st));
                total_weight += smx - st;
            }
            pb.inc(1);
        }

        pb.finish_with_message("Done");

        debug!("Finished calculating weights");

        let weights: Vec<f64> = pairs
            .iter()
            .map(|p| *mapping.get(p).expect("Pair missing from mapping!"))
            .collect();
        let final_total_weight: f64 = mapping.values().sum();
        const EPSILON: f64 = 1e-9; // Tolerance for floating point comparison

        debug!("Flat distribution made");
        let final_weight_dom_pairs: Vec<(f64, DomPair)> =
            mapping.clone().into_iter().map(|(d, w)| (w, d)).collect();
        let sampler_res = WeightedIndex::new(&weights);
        let sampler;
        match sampler_res {
            Ok(int_sampler) => {
                debug!("Finished creating WeightedIndex sampler");
                sampler = int_sampler
            }
            Err(e) => {
                error!("Failed to create WeightedIndex sampler: {}", e);
                error!("Degubg: Total weight: {total_weight}, Weights: {weights:?}");
                panic!("Failed to create WeightedIndex sampler");
            }
        }

        // 4. Return the tuple matching the expected signature
        (
            final_weight_dom_pairs,
            sampler,
            weights,
            total_weight,
            mapping,
        )
    }

    fn make_vec_to_prob_map(pairs: Vec<DomPair>, weights: Vec<f64>) -> FxHashMap<DomPair, f64> {
        //let mut vec_to_index = FxHashMap::default();

        pairs.into_par_iter().zip(weights.into_par_iter()).collect()
        // for (pair, &weight) in pairs.into_par_iter().zip(weights.par_iter()) {
        //     vec_to_index.insert(pair, weight);
        // }
        // vec_to_index
    }
}

// TEST FUNCTIONS FOR FLAT DB ------------------------------------------------------------


#[test]
fn test_exact_probability_matching_uniform() {
    let _ = env_logger::builder().is_test(true).try_init();

    // 1. Setup a small 3x3 test DB.
    // We use a small grid so the total number of queries is manageable for a brute-force check.
    let dim = 2;
    let size_per_dim = 3;
    let db: Box<dyn Searchable + Sync + 'static> = Box::new(testDB::new(dim, size_per_dim, 10));

    // 2. Get all possible domain pairs (every possible bounding box)
    let all_pairs = Selector::get_dom_pairs(&db);

    // 3. Initialize the QueryDistribution with Uniform
    // This will pre-calculate the "Expected" probabilities using your math
    let qd = QueryDistribution::new(all_pairs.clone(), &db, DistributionType::Uniform);

    // 4. Pretend we observed ALL possible queries EXACTLY ONCE (100% sample)
    let observed_queries = all_pairs.clone();
    let num_queries = observed_queries.len() as f64;

    let mut max_diff = 0.0;
    let mut failures = 0;

    // 5. For every possible pair of points, calculate Observed vs Expected
    for pair in &all_pairs {
        // --- Calculate OBSERVED Probability (Empirical) ---
        let mut enclose_count = 0;
        for q in &observed_queries {
            // Using your utility function
            if encloses(q, pair) {
                enclose_count += 1;
            }
        }
        let obs_prob = enclose_count as f64 / num_queries;

        // --- Calculate EXPECTED Probability (Mathematical) ---
        let exp_prob = qd.cumulative_prob_lookup(pair);

        let diff = (obs_prob - exp_prob).abs();
        if diff > max_diff {
            max_diff = diff;
        }

        // Compare with a small float tolerance
        if diff > 1e-9 {
            failures += 1;
            error!(
                "Mismatch on pair {:?}! Observed: {:.4}, Expected: {:.4}, Diff: {}",
                pair, obs_prob, exp_prob, diff
            );
        }
    }

    info!("Maximum probability difference: {}", max_diff);
    assert_eq!(
        failures, 0,
        "Observed probabilities did not match Expected probabilities!"
    );
}
