use crate::dataloader::datasets::Searchable;
use crate::LAMA::error::LAMAError;
use crate::LAMA::utility::{
    binomial_coefficient, compute_pair_weight, get_all_dominating_values, get_mbq,
};

use crate::{Coord, DomPair, Frequency, Record, Value};
use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use log::info;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Selector: Choosing Record-Retrieval Events.
/// This component determines which record-retrieval set expressions are used.
/// Specifically, it generates the frequencies of dominating pairs, as it corresponds to some dist
pub struct Selector<'a> {
    pub dist: String,
    pub encrypted_db: &'a (dyn Searchable + Sync),
    pub dim: Value,
    pub lowest_rec: Record,
    pub largest_rec: Record,
}

impl Selector<'_> {
    /// Precomputes and serializes TRUE frequencies for dominant pairs and t-tuples of values.
    ///
    /// This function iterates through the domain to calculate how many queries cover specific
    /// point pairs (dominant pairs) and groups of $t$ points (t-tuples). The results are
    /// saved as binary files using `bincode` for later use in frequency analysis.
    ///
    /// # Arguments
    /// * `t` - The size of the value tuples to analyze. Should always be 2 * dimension
    /// * `dim` - The dimensionality of the data.
    /// * `dist` - The distribution type (e.g., "uniform").
    ///
    /// # Returns
    ///
    pub fn get_dominant_pair_to_freq_map(&self) -> Result<HashMap<DomPair, Frequency>, LAMAError> {
        info!("Task 1: Computing dominant pair frequencies...");
        let timer = Instant::now();
        let lowest_rec = self.lowest_rec.clone();
        let largest_rec = self.largest_rec.clone();
        let dist = self.dist.clone();

        // Product of (max - min + 1) for each dimension
        let space_size: f64 = lowest_rec
            .iter()
            .zip(largest_rec.iter())
            .map(|(&low, &high)| (high - low + 1) as f64)
            .product();

        let total_pairs = space_size.powi(2);
        let total_dom_pairs = (total_pairs / 2_f64.powi((self.dim - 1) as i32)) as u64;

        // Assuming every query can occur, what is the frequency of each dominant pair?
        let mut true_pair_frequency_dict: HashMap<DomPair, Frequency> = HashMap::new();
        let domain_iter = lowest_rec
            .iter()
            .zip(largest_rec.iter())
            .map(|(&low, &high)| low..=high)
            .multi_cartesian_product();

        // Set up indicatif progress bar
        let pb = ProgressBar::new(total_dom_pairs / 2);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})",
                )?
                .progress_chars("#>-"),
        );

        for v in domain_iter {
            for dv in get_all_dominating_values(&v, &largest_rec) {
                let pair = (v.clone(), dv.clone());
                let frequency = compute_pair_weight(&pair, &dist, &lowest_rec, &largest_rec);
                true_pair_frequency_dict.insert(pair, frequency);

                pb.inc(1); // Increment the progress bar silently
            }
        }
        pb.finish_with_message("Done computing dominant pair frequencies");

        // let file = File::create(&path)?;
        // bincode::serialize_into(BufWriter::new(file), &true_pair_frequency_dict)?;

        info!("Finished DP frequencies in {:?}", timer.elapsed());
        Ok(true_pair_frequency_dict)
    }

    /// Instead of dumbly generating every sing t-tuple we take in a list of all dom pairs and consider
    /// only dom-pairs that are a certain distance away (for a dense DB we can exactly calculate the
    /// number of records returned in the response, otherwise we have to count up to t). Only considers
    /// dompairs that doesn't have 0 freq
    pub fn get_freq_val_possible_t_tup_dict(
        &self,
        t: usize,
        dom_pairs: HashMap<DomPair, Frequency>,
    ) -> Result<HashMap<(Value, Frequency), Vec<Vec<Value>>>, LAMAError> {
        let pb = ProgressBar::new(dom_pairs.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})",
                )?
                .progress_chars("#>-"),
        );

        let binding = self.encrypted_db.get_dom_pair();
        let min_domain_val = binding.0.iter().min().unwrap();

        // Thread-safe progress counter
        let processed_count = AtomicU64::new(0);

        // 1. Parallel Map-Reduce to accumulate frequencies
        let response_to_total_freq: HashMap<Vec<Value>, Frequency> = dom_pairs
            .into_par_iter()
            .filter(|(_, freq)| *freq > 0)
            .fold(
                || HashMap::new(), // Local map for each thread
                |mut local_map, (pair, freq)| {
                    let mut response = self.encrypted_db.do_search(pair.0, pair.1);

                    // Filter in-place to avoid allocating a new Vec
                    response.retain(|v| *v >= *min_domain_val);

                    if !response.is_empty() && response.len() <= t {
                        // Unstable sort is faster for primitives and perfectly fine here
                        response.sort_unstable();

                        *local_map.entry(response).or_insert(0) += freq;
                    }

                    // Update progress bar occasionally to avoid atomic bottleneck
                    let current = processed_count.fetch_add(1, Ordering::Relaxed);
                    if current % 1000 == 0 {
                        pb.set_position(current);
                    }

                    local_map
                },
            )
            .reduce(
                || HashMap::new(), // Merge local maps together
                |mut map1, map2| {
                    for (k, v) in map2 {
                        *map1.entry(k).or_insert(0) += v;
                    }
                    map1
                },
            );

        pb.finish_with_message("Done computing value tuple frequencies");

        // 2. Invert the map into (t, total_freq) -> Vec<Response>
        let mut val_tup_freq_dict: HashMap<(Value, Frequency), Vec<Vec<Value>>> = HashMap::new();

        for (response, total_freq) in response_to_total_freq {
            let t_val = response.len() as Value;
            val_tup_freq_dict
                .entry((t_val, total_freq))
                .or_default()
                .push(response);
        }

        Ok(val_tup_freq_dict)
    }

    #[deprecated(note = "Please use `get_freq_val_possible_t_tup_dict` instead.")]
    /// Calculate for just t=1 (and then use the solver to trim t down)
    pub fn get_freq_all_val_dict(self) -> Result<HashMap<Frequency, Vec<Record>>, LAMAError> {
        let lowest_rec = self.lowest_rec.clone();
        let largest_rec = self.largest_rec.clone();
        // 1. Generate multi-dimensional Cartesian product based on specific bounds
        let mut vals: Vec<Record> = lowest_rec
            .iter()
            .zip(largest_rec.iter())
            .map(|(&low, &high)| low..=high)
            .multi_cartesian_product()
            .collect();

        // 2. Calculate the total volume of the specific bounding box
        let total_combinations: u64 = lowest_rec
            .iter()
            .zip(largest_rec.iter())
            .map(|(&low, &high)| (high - low + 1) as u64)
            .product();

        vals.sort();

        // Approximate total combinations to drive the progress bar
        let pb = ProgressBar::new(total_combinations);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})",
                )?
                .progress_chars("#>-"),
        );

        let mut val_tup_freq_dict: HashMap<Frequency, Vec<Record>> = HashMap::new();

        for val_tuple in vals.into_iter() {
            // Bounding pair should always be val_tuple repeated
            let bounding_pair = get_mbq(&[val_tuple.clone()]);

            // Updated to pass both bounding records
            let freq = compute_pair_weight(&bounding_pair, &self.dist, &lowest_rec, &largest_rec);

            val_tup_freq_dict.entry(freq).or_default().push(val_tuple);

            pb.inc(1);
        }
        pb.finish_with_message("Done computing value tuple frequencies");

        Ok(val_tup_freq_dict)
    }
    #[deprecated(note = "Please use `get_freq_val_possible_t_tup_dict` instead.")]
    /// This takes years for any non-trivial t...
    pub fn get_freq_val_t_tup_dict(
        self,
        t: usize,
    ) -> Result<HashMap<Frequency, Vec<Vec<Record>>>, LAMAError> {
        let lowest_rec = self.lowest_rec.clone();
        let largest_rec = self.largest_rec.clone();
        // 1. Generate multi-dimensional Cartesian product based on specific bounds
        let mut vals: Vec<Record> = lowest_rec
            .iter()
            .zip(largest_rec.iter())
            .map(|(&low, &high)| low..=high)
            .multi_cartesian_product()
            .collect();

        let n = vals.len();
        let total_combinations = binomial_coefficient(n, t);

        vals.sort();

        // Approximate total combinations to drive the progress bar
        let pb = ProgressBar::new(total_combinations);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})",
                )?
                .progress_chars("#>-"),
        );

        let mut val_tup_freq_dict: HashMap<Frequency, Vec<Vec<Record>>> = HashMap::new();

        for val_tuple in vals.into_iter().combinations(t) {
            let bounding_pair = get_mbq(&val_tuple);

            // Updated to pass both bounding records
            let freq = compute_pair_weight(&bounding_pair, &self.dist, &lowest_rec, &largest_rec);

            val_tup_freq_dict.entry(freq).or_default().push(val_tuple);

            pb.inc(1);
        }
        pb.finish_with_message("Done computing value tuple frequencies");

        Ok(val_tup_freq_dict)
    }
}

#[cfg(test)]
mod tests {}
