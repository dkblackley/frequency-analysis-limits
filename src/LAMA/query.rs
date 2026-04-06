use crate::dataloader::Searchable;
use crate::LAMA::utility::DistributionType;
use crate::{DomPair, Probability, Value};
use itertools::Itertools;
use rand::distributions::WeightedIndex;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use statrs::distribution::{Beta, Continuous, Normal};

pub struct QueryDistribution<'a> {
    encrypted_db: &'a Box<dyn Searchable + Sync>,
    pairs: Vec<DomPair>,
    weights: Vec<f64>,
    dom_pair_to_known_prob: FxHashMap<DomPair, f64>,
    mbq_to_cumulative_prob: FxHashMap<DomPair, Probability>,
    pub probs_and_dom_pairs: Vec<(Probability, DomPair)>,
    pub total_weight: f64,
    sampler: WeightedIndex<f64>,
    pub(crate) dist: DistributionType,
    lowest_rec: Vec<Value>,
    largest_rec: Vec<Value>,
}

impl<'a> QueryDistribution<'a> {
    pub fn new(
        pairs: Vec<DomPair>,
        encrypted_db: &'a Box<dyn Searchable + Sync>,
        dist: DistributionType,
    ) -> Box<Self> {
        let (mut probs_and_dom_pairs_raw, sampler, weights, total_weight) = match dist {
            DistributionType::Uniform => Self::new_uniform_internal(&pairs),
            DistributionType::Gaussian => Self::new_gaussian(&pairs),
            DistributionType::Beta => Self::new_beta(&pairs),
        };

        let (lowest_rec, largest_rec) = encrypted_db.get_dom_pair();
        let dom_pair_to_known_prob = Self::make_vec_to_prob_map(&pairs, &weights);

        let mut mbq_to_cumulative_prob = FxHashMap::default();
        let mut probs_and_dom_pairs = Vec::with_capacity(pairs.len());

        // Compute the True CUMULATIVE probability for every possible MBQ
        for pair in &pairs {
            let cum_prob = Self::compute_cumulative_prob(
                pair,
                &dist,
                &lowest_rec,
                &largest_rec,
                &dom_pair_to_known_prob,
                total_weight,
            );
            mbq_to_cumulative_prob.insert(pair.clone(), cum_prob);
            probs_and_dom_pairs.push((cum_prob, pair.clone()));
        }

        probs_and_dom_pairs.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        Box::new(Self {
            encrypted_db,
            pairs,
            weights,
            dom_pair_to_known_prob,
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
        total_weight: f64,
    ) -> Probability {
        match dist {
            DistributionType::Uniform => {
                // O(1) Analytical Calculation for Uniform distributions
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
                // Fallback: Sum Cartesian space of enclosing queries for Gaussian/Beta
                let (target_lower, target_upper) = mbq;
                let mut true_prob: f64 = 0.0;

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

                for (c_lower, c_upper) in lower_combos.cartesian_product(upper_combos) {
                    if let Some(&weight) = dom_pair_to_known_prob.get(&(c_lower, c_upper)) {
                        true_prob += weight;
                    }
                }
                true_prob / total_weight
            }
        }
    }

    pub fn get_cumulative_prob(&self, p0: &DomPair) -> Probability {
        *self.mbq_to_cumulative_prob.get(p0).unwrap_or(&0.0)
    }

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
        pairs: &[DomPair],
    ) -> (Vec<(f64, DomPair)>, WeightedIndex<f64>, Vec<f64>, f64) {
        let weights = vec![1.0; pairs.len()];
        let total_weight = pairs.len() as f64;
        let sampler = WeightedIndex::new(&weights).unwrap();
        let probs_and_pairs = pairs
            .iter()
            .zip(weights.iter())
            .map(|(p, &w)| (w, p.clone()))
            .collect();
        (probs_and_pairs, sampler, weights, total_weight)
    }

    pub fn new_gaussian(
        pairs: &[DomPair],
    ) -> (Vec<(f64, DomPair)>, WeightedIndex<f64>, Vec<f64>, f64) {
        let mu = pairs.len() as f64 / 2.0;
        let sigma = pairs.len() as f64 / 5.0;
        let normal_dist = Normal::new(mu, sigma).expect("Sigma must be > 0.0");
        let weights: Vec<f64> = pairs
            .par_iter()
            .enumerate()
            .map(|(i, _)| normal_dist.pdf(i as f64))
            .collect();
        let total_weight: f64 = weights.par_iter().sum();
        let sampler = WeightedIndex::new(&weights).expect("Failed to create WeightedIndex");
        let mut weight_pair: Vec<(f64, DomPair)> =
            weights.iter().copied().zip(pairs.iter().cloned()).collect();
        weight_pair.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        (weight_pair, sampler, weights, total_weight)
    }

    pub fn new_beta(pairs: &[DomPair]) -> (Vec<(f64, DomPair)>, WeightedIndex<f64>, Vec<f64>, f64) {
        let beta_dist = Beta::new(2.0, 1.0).expect("Alpha/Beta must be > 0.0");
        let n = pairs.len() as f64;
        let weights: Vec<f64> = pairs
            .par_iter()
            .enumerate()
            .map(|(i, _)| {
                let x = (i as f64 + 0.5) / n;
                beta_dist.pdf(x)
            })
            .collect();
        let total_weight: f64 = weights.par_iter().sum();
        let sampler = WeightedIndex::new(&weights).expect("Failed to create WeightedIndex");
        let weight_pair: Vec<(f64, DomPair)> =
            weights.iter().copied().zip(pairs.iter().cloned()).collect();
        (weight_pair, sampler, weights, total_weight)
    }

    fn make_vec_to_prob_map(pairs: &[DomPair], weights: &[f64]) -> FxHashMap<DomPair, f64> {
        let mut vec_to_index = FxHashMap::default();
        for (pair, &weight) in pairs.iter().zip(weights.iter()) {
            vec_to_index.insert(pair.clone(), weight);
        }
        vec_to_index
    }
}
