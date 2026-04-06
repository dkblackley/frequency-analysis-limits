use crate::dataloader::Searchable;
use crate::LAMA::utility::DistributionType;
use crate::{DomPair, Frequency, Probability, Value};
use log::info;
use rand::distributions::WeightedIndex;
use rand::thread_rng;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use statrs::distribution::{Beta, Continuous, Normal};
use std::collections::HashMap;

pub struct QueryDistribution<'a> {
    encrypted_db: &'a Box<dyn Searchable + Sync>,
    pairs: Vec<DomPair>,
    weights: Vec<f64>,
    dom_pair_to_known_prob: FxHashMap<DomPair, f64>,
    probs_and_dom_pairs: Vec<(f64, DomPair)>,
    total_weight: f64,
    sampler: WeightedIndex<f64>,
    pub(crate) dist: DistributionType,
    lowest_rec: Vec<Value>,
    largest_rec: Vec<Value>,
    exact_frequencies: HashMap<DomPair, Frequency>,
    // NEW: Cache the total exact sum to easily compute probabilities
    pub total_exact_frequency: Frequency,
}

impl<'a> QueryDistribution<'a> {
    pub fn new(
        pairs: Vec<DomPair>,
        encrypted_db: &'a Box<dyn Searchable + Sync>,
        dist: DistributionType,
    ) -> Box<Self> {
        // Hardcoded distribution parameters as requested
        let (mut probs_and_dom_pairs_raw, sampler, weights, total_weight) = match dist {
            DistributionType::Uniform => Self::new_uniform_internal(&pairs),
            DistributionType::Gaussian => Self::new_gaussian(&pairs),
            DistributionType::Beta => Self::new_beta(&pairs),
        };

        let (lowest_rec, largest_rec) = encrypted_db.get_dom_pair();
        let dom_pair_to_known_prob = Self::make_vec_to_prob_map(&pairs, &weights);

        let (exact_frequencies, total_exact_frequency) = Self::calculate_exact_frequencies(
            &pairs,
            &weights,
            total_weight,
            pairs.len() as Frequency,
        );

        // 1. Build the Probability -> DomPair mapping
        let mut probs_and_dom_pairs: Vec<(Probability, DomPair)> = exact_frequencies
            .iter()
            .map(|(pair, &freq)| {
                let prob = (freq as f64) / (total_exact_frequency as f64);
                (prob, pair.clone())
            })
            .collect();

        // 2. Sort by probability for O(log n) lookups later
        probs_and_dom_pairs.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        Box::new(Self {
            probs_and_dom_pairs, // Now sorted and containing exact probabilities
            sampler,
            weights,
            total_weight,
            dist,
            lowest_rec,
            largest_rec,
            encrypted_db,
            pairs,
            dom_pair_to_known_prob,
            exact_frequencies,
            total_exact_frequency,
        })
    }

    pub fn get_candidate_pairs_by_probability(
        &self,
        observed_prob: Probability,
        epsilon: f64,
    ) -> Vec<DomPair> {
        let lower_bound = observed_prob - epsilon;
        let upper_bound = observed_prob + epsilon;

        // Binary search using Rust's partition_point
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

    /// Returns the true, exact discrete frequency of the given pair being queried.
    pub fn get_true_freq(&self, p0: &DomPair) -> Frequency {
        *self.exact_frequencies.get(p0).unwrap_or(&0)
    }

    /// NEW: Converts the exact frequency of this query into a Probability (0.0 to 1.0)
    pub fn get_query_prob(&self, p0: &DomPair) -> Probability {
        if self.total_exact_frequency == 0 {
            return 0.0;
        }
        let freq = self.get_true_freq(p0);
        (freq as f64) / (self.total_exact_frequency as f64)
    }

    /// Returns both the frequency map and the sum of all assigned frequencies
    pub fn calculate_exact_frequencies(
        pairs: &[DomPair],
        weights: &[f64],
        total_weight: f64,
        total_samples: Frequency,
    ) -> (HashMap<DomPair, Frequency>, Frequency) {
        let mut freq_map = HashMap::new();
        let mut total_freq = 0;

        for (pair, &weight) in pairs.iter().zip(weights.iter()) {
            let prob = weight / total_weight;
            let exact_count = (prob * (total_samples as f64)).round() as Frequency;
            let final_count = exact_count.max(1);

            total_freq += final_count;
            freq_map.insert(pair.clone(), final_count);
        }

        (freq_map, total_freq)
    }

    // Fixed uniform initialization
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

    // Fixed Gaussian initialization (Removed dangerous parallel mutable capture)
    pub fn new_gaussian(
        pairs: &[DomPair],
    ) -> (Vec<(f64, DomPair)>, WeightedIndex<f64>, Vec<f64>, f64) {
        let mu = pairs.len() as f64 / 2.0;
        let sigma = pairs.len() as f64 / 5.0;
        info!("Initialising gaussian dist with mu {mu}, std {sigma}");
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

    // Fixed Beta initialization
    pub fn new_beta(pairs: &[DomPair]) -> (Vec<(f64, DomPair)>, WeightedIndex<f64>, Vec<f64>, f64) {
        let beta_dist = Beta::new(2.0, 1.0).expect("Alpha/Beta must be > 0.0");
        let n = pairs.len() as f64;

        let weights: Vec<f64> = pairs
            .par_iter()
            .enumerate()
            .map(|(i, _)| {
                let x = (i as f64 + 0.5) / n;
                // beta_dist.pdf(x) * (n - 1.0) // Taken from REMIN dist, not sure what this does?
                beta_dist.pdf(x)
            })
            .collect();

        let total_weight: f64 = weights.par_iter().sum();
        let sampler = WeightedIndex::new(&weights).expect("Failed to create WeightedIndex");

        let weight_pair: Vec<(f64, DomPair)> =
            weights.iter().copied().zip(pairs.iter().cloned()).collect();

        (weight_pair, sampler, weights, total_weight)
    }

    // Helper function mapping
    fn make_vec_to_prob_map(pairs: &[DomPair], weights: &[f64]) -> FxHashMap<DomPair, f64> {
        let mut vec_to_index = FxHashMap::default();
        for (pair, &weight) in pairs.iter().zip(weights.iter()) {
            vec_to_index.insert(pair.clone(), weight);
        }
        vec_to_index
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataloader::tester::testDB;
    // Assuming testDB is in your crate root or imported

    fn generate_mock_pairs_for_db(db: &testDB) -> Vec<DomPair> {
        // Just generating a few random dominating pairs within the 50x50 grid for testing
        vec![
            (vec![0, 0], vec![10, 10]),
            (vec![5, 5], vec![20, 20]),
            (vec![0, 0], vec![49, 49]), // Covers the whole DB
        ]
    }

    #[test]
    fn test_frequency_to_probability_conversion() {
        let db = testDB::new(50, 50, 50);
        let pairs = generate_mock_pairs_for_db(&db);
        let boxed_db: Box<dyn Searchable + Sync> = Box::new(db);

        let q_dist = QueryDistribution::new(pairs.clone(), &boxed_db, DistributionType::Gaussian);

        // 1. Check frequencies
        let freq_0 = q_dist.get_true_freq(&pairs[0]);
        let freq_1 = q_dist.get_true_freq(&pairs[1]);
        let freq_2 = q_dist.get_true_freq(&pairs[2]);

        let calculated_total = freq_0 + freq_1 + freq_2;
        assert_eq!(
            q_dist.total_exact_frequency, calculated_total,
            "Total frequency mismatch"
        );

        // 2. Check probabilities
        let prob_0 = q_dist.get_query_prob(&pairs[0]);
        let prob_1 = q_dist.get_query_prob(&pairs[1]);
        let prob_2 = q_dist.get_query_prob(&pairs[2]);

        // Probabilities must sum to 1.0 (allowing for floating point epsilon)
        let total_prob = prob_0 + prob_1 + prob_2;
        assert!(
            (total_prob - 1.0).abs() < f64::EPSILON,
            "Probabilities do not sum to 1.0!"
        );

        // Ensure the mathematical conversion is perfectly accurate
        assert_eq!(prob_0, (freq_0 as f64) / (calculated_total as f64));
    }
}
