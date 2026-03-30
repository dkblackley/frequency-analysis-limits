use crate::dataloader::{flatten_nd, unflatten_nd, Searchable};
use crate::LAMA::error::LAMAError;
use crate::LAMA::utility::{
    binomial_coefficient, compute_pair_weight, dominates, get_all_dominating_values, get_mbq,
    Distribution,
};
use crate::{Coord, DomPair, Frequency, Record, Value};
use indicatif::{ProgressBar, ProgressIterator, ProgressStyle};

use itertools::Itertools;
use log::info;
use rayon::prelude::*;
use rayon::prelude::*;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::fs::File;

use std::io::{BufReader, BufWriter, Write};

use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use std::sync::mpsc::sync_channel;

use std::thread;
use std::time::Instant;

/// Selector: Choosing Record-Retrieval Events.
/// This component determines which record-retrieval set expressions are used.
/// Specifically, it generates the frequencies of dominating pairs, as it corresponds to some dist
pub struct Selector<'a> {
    pub dist: String,
    pub encrypted_db: &'a Box<dyn Searchable + Sync>,
    pub dim: Value,
    pub lowest_rec: Record,
    pub largest_rec: Record,
}

// Struct for the Min-Heap used during the K-Way Merge
#[derive(Eq, PartialEq)]
struct HeapItem {
    freq: u64,
    tuple: Vec<i64>,
    chunk_idx: usize,
}

// Implement Ord to make BinaryHeap act as a Min-Heap based on frequency
impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .freq
            .cmp(&self.freq)
            .then_with(|| self.chunk_idx.cmp(&other.chunk_idx))
    }
}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Selector<'_> {
    /// Given A query distribution (DomPair -> frequency mapping) find the frequency of a t-tuple
    /// of records. Note that this is NOT the frequency of the specific response that is exactly
    /// that t-tuple. Instead, given t encrypted records, how frequently do we see these across ALL
    ///
    /// # Arguments
    ///
    /// # Returns
    ///
    /// A mapping from a Query (dominating pair) to the frequency we'd expect that Query to be
    /// served.
    ///
    pub fn precompute_t_observed(
        &self,
        t: usize,
        output_filepath: &str,
        dom_pairs: &HashMap<DomPair, Frequency>,
    ) -> HashMap<u64, Vec<Vec<i64>>> {
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

        // 2. Parallel Map-Reduce: Calculate MBQ for every combination directly
        let freq_to_observed_map: HashMap<u64, Vec<Vec<i64>>> = response
            .into_iter()
            .combinations(t)
            .par_bridge() // Distribute combinations to Rayon worker threads
            .fold(
                HashMap::new,
                |mut local_map: HashMap<u64, Vec<Vec<i64>>>, t_tuple| {
                    // Decode the point coordinates using the bounds
                    let decoded_points: Vec<Record> = t_tuple
                        .iter()
                        .map(|&v| unflatten_nd(v, &self.largest_rec, &self.lowest_rec))
                        .collect();

                    // Calculate the theoretical Minimum Bounding Query for these points
                    let dom_pair = get_mbq(&decoded_points);

                    // Directly look up its expected theoretical frequency
                    let freq = *dom_pairs.get(&dom_pair).unwrap();

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

        // 3. Serialize the final grouped map to disk
        info!("Writing grouped observed map to disk...");
        let file = File::create(output_filepath).expect("Failed to create file");
        let writer = BufWriter::with_capacity(8 * 1024 * 1024, file);

        bincode::serialize_into(writer, &freq_to_observed_map).expect("Failed to write to disk");

        info!("Precomputation complete.");

        freq_to_observed_map
    }

    /// Precomputes and serializes TRUE frequencies for dominant pairs using only the query
    /// distribution. This function iterates through the domain to calculate how many queries cover
    /// specific point pairs (dominant pairs). It doesn't say anything about how many records
    /// are found/ the frequency of records. This is purely the Query Distribution.
    ///
    /// # Returns
    ///
    /// A mapping from a Query (dominating pair) to the frequency we'd expect that Query to be
    /// served.
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

        let domain_iter = lowest_rec
            .iter()
            .zip(largest_rec.iter())
            .map(|(&low, &high)| low..=high)
            .multi_cartesian_product();

        let pb = ProgressBar::new(total_dom_pairs / 2);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})")?
                .progress_chars("#>-"),
        );
        let processed_count = AtomicU64::new(0);

        let parsed_dist = dist.parse().unwrap();

        let parsed_dist_ref = &parsed_dist;
        let lowest_rec_ref = &lowest_rec;
        let largest_rec_ref = &largest_rec;
        let processed_count_ref = &processed_count;

        let true_pair_frequency_dict: HashMap<DomPair, Frequency> = domain_iter
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
                        let pair = (v_clone.clone(), dv);
                        // Using the references we created outside
                        let frequency = compute_pair_weight(
                            &pair,
                            parsed_dist_ref,
                            lowest_rec_ref,
                            largest_rec_ref,
                        );
                        let count = processed_count_ref.fetch_add(1, AtomicOrdering::Relaxed);
                        if count % 100_000 == 0 {
                            pb_inner.set_position(count);
                        }
                        (pair, frequency)
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

        // Parse the distribution ONCE here
        let dist_enum = Distribution::from_str(&self.dist).unwrap();

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

        // Thread-safe progress counter
        let processed_count = AtomicU64::new(0);

        // 1. Generate combinations sequentially, but process them in parallel
        let val_tup_freq_dict: HashMap<(Value, Frequency), Vec<Vec<Value>>> = vals
            .into_iter()
            .combinations(t)
            .par_bridge() // <--- This hands off the generated combinations to Rayon worker threads
            .fold(
                HashMap::new,
                |mut local_map: HashMap<(Value, Frequency), Vec<Vec<Value>>>, val_tuple| {
                    let bounding_pair = get_mbq(&val_tuple);
                    let freq = compute_pair_weight(
                        &bounding_pair,
                        &dist_enum,
                        &*lowest_rec,
                        &*largest_rec,
                    );

                    // Flatten the n-dimensional records into 1D values AFTER computing the spatial frequency
                    let flattened_tuple: Vec<Value> = val_tuple
                        .iter()
                        .map(|record| flatten_nd(record, &*largest_rec, &*lowest_rec))
                        .collect();

                    local_map
                        .entry((flattened_tuple.len() as Value, freq))
                        .or_default()
                        .push(flattened_tuple);

                    // Update progress bar without bottlenecking threads
                    let current = processed_count.fetch_add(1, AtomicOrdering::Relaxed);
                    if current % 10000 == 0 {
                        pb.set_position(current);
                    }

                    local_map
                },
            )
            .reduce(HashMap::new, |mut map1, map2| {
                // Merge the thread-local hash maps
                for (k, mut v) in map2 {
                    map1.entry(k).or_default().append(&mut v);
                }
                map1
            });

        pb.finish_with_message("Done computing value tuple frequencies");

        Ok(val_tup_freq_dict)
    }

    /// Computes the theoretical (100% dense) expected frequencies for all t-tuples
    pub fn build_theoretical_t_dict(
        lowest_rec: &[i64],
        largest_rec: &[i64],
        dist: &str,
        t: usize,
    ) -> HashMap<u64, Vec<Vec<i64>>> {
        let dist_enum = Distribution::from_str(dist).expect("Invalid distribution");

        // 1. Generate the 100% dense universe (every single theoretical coordinate)
        let mut vals: Vec<Record> = lowest_rec
            .iter()
            .zip(largest_rec.iter())
            .map(|(&low, &high)| low..=high)
            .multi_cartesian_product()
            .collect();

        let n = vals.len();
        let total_combinations = binomial_coefficient(n, t);
        vals.sort_unstable(); // Ensure consistent lexicographical ordering

        let pb = ProgressBar::new(total_combinations as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})").unwrap()
                .progress_chars("#>-"),
        );

        let processed_count = AtomicU64::new(0);

        // 2. Parallel map-reduce over every single t-length combination
        let theoretical_dict: HashMap<u64, Vec<Vec<i64>>> = vals
            .into_iter()
            .combinations(t)
            .par_bridge() // Hand off combinations to the Rayon worker pool
            .fold(
                HashMap::new, // Give each thread its own local HashMap
                |mut local_map: HashMap<u64, Vec<Vec<i64>>>, val_tuple| {
                    // Calculate expected frequency (MBQ weight)
                    let bounding_pair = get_mbq(&val_tuple);
                    let freq =
                        compute_pair_weight(&bounding_pair, &dist_enum, lowest_rec, largest_rec);

                    // Flatten the n-dimensional points to their 1D target aliases
                    let flattened_tuple: Vec<i64> = val_tuple
                        .iter()
                        .map(|record| flatten_nd(record, largest_rec, lowest_rec))
                        .collect();

                    // Group by frequency
                    local_map.entry(freq).or_default().push(flattened_tuple);

                    // Update progress bar without causing thread lock contention
                    // let current = processed_count.fetch_add(1, AtomicOrdering::Relaxed);
                    // if current % 10_000 == 0 {
                    //     pb.set_position(current);
                    // }

                    local_map
                },
            )
            .reduce(
                HashMap::new, // Merge all the thread-local HashMaps back together
                |mut map1, map2| {
                    for (freq, mut tuples) in map2 {
                        map1.entry(freq).or_default().append(&mut tuples);
                    }
                    map1
                },
            );

        pb.finish_with_message(format!("Finished theoretical mapping for t={}", t));

        theoretical_dict
    }
}

#[cfg(test)]
mod tests {}
