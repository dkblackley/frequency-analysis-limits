use crate::dataloader::{unflatten_nd, Searchable};
use crate::LAMA::utility::{get_mbq, DistributionType};
// Adjust imports as necessary for DomPair
use crate::{Coord, DomPair, Probability, Record, Value};
use indicatif::ParallelProgressIterator;
use itertools::Itertools;
use log::{debug, error, info};
use plotters::prelude::*;
use rand::distributions::WeightedIndex;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use statrs::distribution::{Beta, Continuous, Normal};
use std::collections::{HashMap, HashSet};

pub struct QueryDistribution<'a> {
    encrypted_db: &'a Box<dyn Searchable + Sync>,
    pairs: Vec<DomPair>,
    weights: Vec<f64>,
    pub dom_pair_to_known_prob: FxHashMap<DomPair, f64>,
    mbq_to_cumulative_prob: FxHashMap<DomPair, Probability>,
    pub probs_and_dom_pairs: Vec<(Probability, DomPair)>,
    pub total_weight: f64,
    sampler: WeightedIndex<f64>,
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

        let mut mbq_to_cumulative_prob = FxHashMap::default();
        let mut probs_and_dom_pairs = Vec::with_capacity(pairs.len());

        info!("Computing true cumulative probabilities for every MBQ");

        // Compute the True CUMULATIVE probability for every possible MBQ
        let computed_results: Vec<_> = pairs
            .par_iter()
            .progress() // Attaches the indicatif progress bar to Rayon
            .map(|pair| {
                let cum_prob = Self::compute_cumulative_prob(
                    pair,
                    &dist, // Note: `dist`, `lowest_rec`, etc. must implement `Sync`
                    &lowest_rec,
                    &largest_rec,
                    &dom_pair_to_known_weight,
                    total_weight,
                );

                // let current = atom_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                // if current % 1000 == 0 {
                //     pb.set_position(current);
                // }

                // Return a tuple of references/clones needed for insertion
                (pair, cum_prob)
            })
            .collect();

        debug!("Finished computing true cumulative probabilities");
        // 3. Sequential Insertion Phase
        // Iterating over the pre-computed results to insert is virtually instantaneous.
        for (pair, cum_prob) in computed_results {
            mbq_to_cumulative_prob.insert(pair.clone(), cum_prob);
            probs_and_dom_pairs.push((cum_prob, pair.clone()));
        }

        // probs_and_dom_pairs.par_sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        debug!("Done pre-computing probability dist");
        Box::new(Self {
            encrypted_db,
            pairs,
            weights,
            dom_pair_to_known_prob: dom_pair_to_known_weight,
            mbq_to_cumulative_prob,
            probs_and_dom_pairs,
            total_weight,
            sampler,
            dist,
            lowest_rec,
            largest_rec,
        })
    }

    /// The core math logic! Returns the sum of probabilities of all queries that ENCLOSE the target MBQ.
    pub fn compute_cumulative_prob(
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
                    .par_bridge() // <--- The magic parallel bit
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

    /// Only used for the 'flattened' dist. Returns the direct weight, not the prob.
    fn compute_cumulative_weight(
        mbq: &DomPair,
        dist: &DistributionType,
        lowest_rec: &[Value],
        largest_rec: &[Value],
        dom_pair_to_known_prob: &FxHashMap<DomPair, f64>,
        //dom_pair_slice: &[(DomPair, f64)], // Now a sorted flat slice
        _total_weight: f64,
    ) -> f64 {
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
                count
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
                    .par_bridge() // <--- The magic parallel bit
                    .map(|(c_lower, c_upper)| {
                        // Look up the weight, default to 0.0 if not found, then return it to be summed
                        *dom_pair_to_known_prob
                            .get(&(c_lower, c_upper))
                            .unwrap_or(&0.0)
                    })
                    .sum(); // Rayon handles the thread-safe accumulation here

                true_prob
            }
        }
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
            .probs_and_dom_pairs
            .partition_point(|x| x.0 < lower_bound);
        let end_idx = self
            .probs_and_dom_pairs
            .partition_point(|x| x.0 <= upper_bound);

        self.probs_and_dom_pairs[start_idx..end_idx]
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
        let mut mapping = old_qd.dom_pair_to_known_prob;
        let mut total_weight = old_qd.total_weight;

        // as per algorithm 2: Start at the largest possible query and work back. THis updates
        // the old QD as we go and works over every pair, so time might be n^2
        let (low_pair, high_pair) = encrypted_db.get_dom_pair();
        let all_recs = encrypted_db.do_search(&low_pair, &high_pair);
        let largest_l1 = taxicab_distance(&low_pair, &high_pair);

        fn taxicab_distance(v1: &[i64], v2: &[i64]) -> u64 {
            v1.iter().zip(v2.iter()).map(|(a, b)| a.abs_diff(*b)).sum()
        }

        let mut debug_res = HashMap::new();
        let mut debug_query_set = HashSet::new();

        for d in (0..=largest_l1).rev() {
            // find everything of this distance. Distance is defined as taxicab/L1
            let mut equi_dist_pairs: Vec<(Value, Value)> = Vec::new();
            let mut unique_queries = HashSet::new();
            let mut max_weight = -1.0;
            let mut tmx = encrypted_db.get_dom_pair();

            for outer in &all_recs {
                for inner in &all_recs {
                    let unflat_out = unflatten_nd(*outer, &high_pair, &low_pair);
                    let unflat_in = unflatten_nd(*inner, &high_pair, &low_pair);

                    let taxicab = taxicab_distance(&unflat_out, &unflat_in);

                    if taxicab == d {
                        let query = get_mbq(&[unflat_out.clone(), unflat_in.clone()]);
                        debug_query_set.insert(query.clone());
                        // This might need to get updated dynamically?
                        // let cand_weight = mapping.get(&query).unwrap();
                        let cand_prob = &Self::compute_cumulative_prob(
                            &query,
                            &DistributionType::Flat,
                            &low_pair,
                            &high_pair,
                            &mapping,
                            total_weight,
                        );

                        if *cand_prob > max_weight {
                            max_weight = *cand_prob;
                            tmx = query.clone();
                        }
                        equi_dist_pairs.push((*outer, *inner));
                        unique_queries.insert(query);
                    }
                }
            }

            // As per line 3 - the sum of weights covering tmx
            let smx = Self::compute_cumulative_weight(
                &tmx,
                &DistributionType::Flat,
                &low_pair,
                &high_pair,
                &mapping,
                total_weight,
            );

            for mbq in unique_queries.iter() {
                let st = Self::compute_cumulative_weight(
                    &mbq,
                    &DistributionType::Flat,
                    &low_pair,
                    &high_pair,
                    &mapping,
                    total_weight,
                );
                let old_weight = mapping.get(&mbq).unwrap();
                debug_res.insert(d, (mbq.clone(), mapping.clone()));
                mapping.insert(mbq.clone(), *old_weight + (smx - st));
                total_weight += smx - st;
            }
        }

        debug!("Finished calculating weights");

        let weights: Vec<f64> = pairs
            .iter()
            .map(|p| *mapping.get(p).expect("Pair missing from mapping!"))
            .collect();
        let final_total_weight: f64 = mapping.values().sum();
        const EPSILON: f64 = 1e-9; // Tolerance for floating point comparison

        let mut matches = HashMap::new();

        for d in (0..=largest_l1).rev() {
            let mut expected_weight: Option<f64> = None;
            let mut reference_pair: Option<(Value, Value)> = None;

            for outer in &all_recs {
                for inner in &all_recs {
                    let unflat_out = unflatten_nd(*outer, &high_pair, &low_pair);
                    let unflat_in = unflatten_nd(*inner, &high_pair, &low_pair);

                    if taxicab_distance(&unflat_out, &unflat_in) == d {
                        *matches.entry(d).or_insert(0) += 1;
                        let mbq = get_mbq(&[unflat_out.clone(), unflat_in.clone()]);

                        // Important: Use the FINAL `mapping` and `final_total_weight` here
                        let current_cumu_weight = Self::compute_cumulative_weight(
                            &mbq,
                            &DistributionType::Flat,
                            &low_pair,
                            &high_pair,
                            &mapping,
                            final_total_weight,
                        );

                        match expected_weight {
                            None => {
                                // Set the baseline for this distance 'd'
                                expected_weight = Some(current_cumu_weight);
                                reference_pair = Some((*outer, *inner));
                            }
                            Some(expected) => {
                                // Compare current pair against the baseline
                                if (expected - current_cumu_weight).abs() > EPSILON {
                                    error!("================ FLATNESS ASSERTION FAILED ================");
                                    error!("Distance group (d): {}", d);
                                    error!("Total distribution weight: {}", final_total_weight);
                                    error!("--- Reference Pair ---");
                                    error!("Values: {:?}", reference_pair.unwrap());
                                    error!(
                                        "Un-flattened: {:?}, {:?}",
                                        unflatten_nd(
                                            reference_pair.unwrap().0,
                                            &high_pair,
                                            &low_pair
                                        ),
                                        unflatten_nd(
                                            reference_pair.unwrap().1,
                                            &high_pair,
                                            &low_pair
                                        )
                                    );
                                    error!("Cumulative Weight: {}", expected);
                                    error!("Cumulative Prob:   {}", expected / final_total_weight);
                                    error!("--- Failing Pair ---");
                                    error!("Values: ({:?}, {:?})", *outer, *inner);
                                    error!(
                                        "Un-flattened: {:?}, {:?}",
                                        unflatten_nd(*outer, &high_pair, &low_pair),
                                        unflatten_nd(*inner, &high_pair, &low_pair)
                                    );
                                    error!("MBQ: {:?}", mbq);
                                    error!("Cumulative Weight: {}", current_cumu_weight);
                                    error!(
                                        "Cumulative Prob:   {}",
                                        current_cumu_weight / final_total_weight
                                    );
                                    error!(
                                        "Difference: {}",
                                        (expected - current_cumu_weight).abs()
                                    );
                                    error!("===========================================================");
                                    panic!("Equidistant pairs do not have the same probability! Algorithm failed at d={}", d);
                                }
                            }
                        }
                    }
                }
            }
        }
        debug!("total matches: {:?}", matches);
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
