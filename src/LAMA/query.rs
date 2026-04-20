use crate::dataloader::{flatten_dompair, flatten_nd, unflatten_nd, Searchable};
use crate::LAMA::utility::{get_mbq, DistributionType};
// Adjust imports as necessary for DomPair
use crate::{Coord, DomPair, Probability, Record, Value};
use indicatif::{ParallelProgressIterator, ProgressBar};
use itertools::Itertools;
use log::{debug, error, info, warn};
use plotters::prelude::*;
use rand::distributions::WeightedIndex;
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
        let (lowest_rec, largest_rec) = encrypted_db.get_dom_pair();

        info!("Beginning to set up {dist} distribution");
        let (_probs_and_dom_pairs_raw, sampler, weights, total_weight, dom_pair_to_known_weight) =
            match dist {
                DistributionType::Uniform => Self::new_uniform_internal(pairs.clone()),
                DistributionType::Gaussian => Self::new_gaussian(&pairs),
                DistributionType::Beta => Self::new_beta(&pairs),
                DistributionType::Flat => Self::new_flat(&pairs, encrypted_db),
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

pub fn plot_qd_heatmap(
    dom_pair_to_known_prob: &FxHashMap<DomPair, f64>,
    largest: Record,
    lowest: Record,
    total_weight: f64,
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // 1. Force the exact axis sequence requested (0-indexed)
    let records: Vec<Vec<Coord>> = vec![
        vec![0, 0], // 1,1
        vec![0, 1], // 1,2
        vec![1, 0], // 2,1
        vec![0, 2], // 1,3
        vec![1, 1], // 2,2
        vec![2, 0], // 3,1
        vec![0, 3], // 1,4
        vec![1, 2], // 2,3
        vec![2, 1], // 3,2
        vec![3, 0], // 4,1
        vec![1, 3], // 2,4
        vec![2, 2], // 3,3
        vec![3, 1], // 4,2
        vec![2, 3], // 3,4
        vec![3, 2], // 4,3
        vec![3, 3], // 4,4
    ];

    // 2. Compute the 16x16 probability matrix
    let mut matrix = vec![vec![0.0; 16]; 16];
    let mut max_prob = 0.0f64;

    for (i, v1) in records.iter().enumerate() {
        for (j, v2) in records.iter().enumerate() {
            let query = get_mbq(&[(*v1.clone()).to_owned(), (*v2.clone()).to_owned()]);

            let prob = QueryDistribution::compute_cumulative_prob(
                &query,
                &DistributionType::Flat,
                &lowest,
                &largest,
                &dom_pair_to_known_prob,
                total_weight,
            );
            matrix[i][j] = prob;
            if prob > max_prob {
                max_prob = prob;
            }
        }
    }

    // 3. Draw the Heatmap using Plotters
    let root = BitMapBackend::new(filename, (900, 900)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Probability of Simultaneous Retrieval (uni)",
            ("sans-serif", 30).into_font(),
        )
        .margin(60)
        .x_label_area_size(80)
        .right_y_label_area_size(80)
        .build_cartesian_2d(-0.5f64..15.5f64, -0.5f64..15.5f64)?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .x_labels(16)
        .y_labels(16)
        .x_label_formatter(&|x| {
            let idx = x.round() as isize;
            if (0..16).contains(&idx) {
                format!(
                    "({},{})",
                    records[idx as usize][0] + 1,
                    records[idx as usize][1] + 1
                )
            } else {
                "".to_string()
            }
        })
        .y_label_formatter(&|y| {
            let idx = y.round() as isize;
            if (0..16).contains(&idx) {
                format!(
                    "({},{})",
                    records[idx as usize][0] + 1,
                    records[idx as usize][1] + 1
                )
            } else {
                "".to_string()
            }
        })
        .x_label_style(
            ("sans-serif", 15)
                .into_font()
                .transform(FontTransform::Rotate270),
        )
        .y_label_style(("sans-serif", 15).into_font())
        .draw()?;

    chart.draw_series(matrix.iter().enumerate().flat_map(|(y, row)| {
        row.iter().enumerate().map(move |(x, &prob)| {
            let intensity = if max_prob > 0.0 { prob / max_prob } else { 0.0 };
            let hue = (1.0 - intensity) * 240.0 / 360.0;
            let color = HSLColor(hue, 1.0, 0.5);

            let x_f = x as f64;
            let y_f = y as f64;

            let mut rect = Rectangle::new(
                [(x_f - 0.5, y_f - 0.5), (x_f + 0.5, y_f + 0.5)],
                color.filled(),
            );
            rect.set_margin(1, 1, 1, 1);
            rect
        })
    }))?;

    root.present()?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataloader::tester::testDB;
    // Bring plot_qd_heatmap and testDB into scope

    #[test]
    fn test_generate_and_plot_heatmap() {
        let _ = env_logger::try_init();
        // 1. Setup a 4x4 test DB using your dynamically sizing struct
        let dim = 2;
        let size_per_dim = 4;

        // Note: Using a boxed trait object as required by the `new_flat` signature
        let db: Box<dyn Searchable + Sync + 'static> =
            Box::new(testDB::new(dim, size_per_dim, 100));

        // 2. Generate `pairs` (all valid bounding boxes for a 4x4 grid)
        // Since it's a 4x4, there are precisely 100 bounding queries
        let mut all_pairs = Vec::new();
        for x1 in 0..4 {
            for y1 in 0..4 {
                for x2 in x1..4 {
                    for y2 in y1..4 {
                        all_pairs.push((vec![x1, y1], vec![x2, y2]));
                    }
                }
            }
        }

        // 3. Run the frequency-flattening algorithm
        let (final_weights, _sampler, weights, total_weight, dom_pair_to_weight) =
            QueryDistribution::new_flat(&all_pairs, &db);

        let (low, high) = db.get_dom_pair();

        // 4. Generate the heatmap!
        plot_qd_heatmap(
            &dom_pair_to_weight,
            high,
            low,
            total_weight,
            "figures/qd_distribution_heatmap.png",
        )
        .expect("Failed to plot the probability heatmap");
    }
}
