use crate::dataloader::{flatten_nd, Searchable};
use crate::LAMA::error::LAMAError;
use crate::LAMA::utility::{
    binomial_coefficient, compute_pair_weight, dominates, get_all_dominating_values, get_mbq,
    Distribution,
};
use crate::{Coord, DomPair, Frequency, Record, Value};
use indicatif::{ProgressBar, ProgressStyle};

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
    pub fn precompute_sorted_to_disk(
        &self,
        t: usize,
        final_output_filepath: &str,
        chunk_limit: usize,
    ) {
        // chunk_limit = 5_000_000 for 5GB
        let lowest_rec = self.lowest_rec.clone();
        let largest_rec = self.largest_rec.clone();
        let dist_enum = Distribution::from_str(&self.dist).unwrap_or(Distribution::Uniform);

        let mut vals: Vec<Record> = lowest_rec
            .iter()
            .zip(largest_rec.iter())
            .map(|(&low, &high)| low..=high)
            .multi_cartesian_product()
            .collect();
        vals.sort();

        let n = vals.len();
        let total_combinations = binomial_coefficient(n, t);

        let pb = ProgressBar::new(total_combinations);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})").unwrap()
        );

        let mut processed_count = AtomicU64::new(0);

        // Channel for streaming computed tuples to the chunk manager
        let (sender, receiver) = sync_channel::<(u64, Vec<i64>)>(50_000);

        // --- PHASE 1: CHUNK GENERATION & IN-MEMORY SORTING ---
        let chunk_thread = thread::spawn(move || {
            let mut buffer: Vec<(u64, Vec<i64>)> = Vec::new();
            let mut chunk_files = Vec::new();
            let mut chunk_counter = 0;

            while let Ok(item) = receiver.recv() {
                buffer.push(item);

                if buffer.len() >= chunk_limit {
                    buffer.sort_unstable_by_key(|k| k.0);

                    let filename = format!("temp_chunk_{}.bin", chunk_counter);
                    let file = File::create(&filename).expect("Failed to create chunk");
                    let mut writer = BufWriter::new(file);

                    for buf_item in &buffer {
                        bincode::serialize_into(&mut writer, buf_item).unwrap();
                    }

                    chunk_files.push(filename);
                    chunk_counter += 1;
                    buffer.clear();
                }
            }

            // Process the final partial chunk
            if !buffer.is_empty() {
                buffer.sort_unstable_by_key(|k| k.0);
                let filename = format!("temp_chunk_{}.bin", chunk_counter);
                let file = File::create(&filename).expect("Failed to create chunk");
                let mut writer = BufWriter::new(file);
                for buf_item in &buffer {
                    bincode::serialize_into(&mut writer, buf_item).unwrap();
                }
                chunk_files.push(filename);
            }

            chunk_files
        });

        // Compute combinations in parallel
        vals.into_iter()
            .combinations(t)
            .par_bridge()
            .for_each_with(sender, |s, val_tuple| {
                let bounding_pair = get_mbq(&val_tuple);
                let freq =
                    compute_pair_weight(&bounding_pair, &dist_enum, &lowest_rec, &largest_rec);
                let flattened: Vec<i64> = val_tuple
                    .iter()
                    .map(|rec| flatten_nd(rec, &largest_rec, &lowest_rec))
                    .collect();

                s.send((freq, flattened)).ok();

                let current = processed_count.fetch_add(1, AtomicOrdering::Relaxed);
                if current % 10_000 == 0 {
                    pb.set_position(current);
                }
            });

        pb.finish_with_message("Done generating. Merging chunks...");

        let chunk_files = chunk_thread.join().unwrap();

        // --- PHASE 2: K-WAY MERGE INTO FINAL FILE ---
        let mut min_heap = BinaryHeap::new();
        let mut readers = Vec::new();

        // Open all chunk files for reading
        for (idx, filename) in chunk_files.iter().enumerate() {
            let file = File::open(filename).unwrap();
            let mut reader = BufReader::with_capacity(1024 * 1024 * 50, file); // 50MB buffer per file

            // Read the first item from each file to seed the heap
            if let Ok((freq, tuple)) = bincode::deserialize_from::<_, (u64, Vec<i64>)>(&mut reader)
            {
                min_heap.push(HeapItem {
                    freq,
                    tuple,
                    chunk_idx: idx,
                });
            }
            readers.push(reader);
        }

        let final_file = File::create(final_output_filepath).expect("Failed to create final file");
        let mut final_writer = BufWriter::with_capacity(16 * 1024 * 1024, final_file);

        // Continuously pop the minimum frequency and pull the next item from that specific chunk
        while let Some(min_item) = min_heap.pop() {
            let output_tuple = (min_item.freq, min_item.tuple);
            bincode::serialize_into(&mut final_writer, &output_tuple).unwrap();

            // Read the next item from the file we just pulled from
            let idx = min_item.chunk_idx;
            if let Ok((next_freq, next_tuple)) =
                bincode::deserialize_from::<_, (u64, Vec<i64>)>(&mut readers[idx])
            {
                min_heap.push(HeapItem {
                    freq: next_freq,
                    tuple: next_tuple,
                    chunk_idx: idx,
                });
            }
        }

        final_writer.flush().unwrap();

        // Cleanup temporary files
        for filename in chunk_files {
            std::fs::remove_file(filename).unwrap();
        }
    }

    pub fn precompute_ram_sorted_to_disk(
        &self,
        t: usize,
        output_filepath: &str,
        secret_map: HashMap<i64, i64>,
    ) {
        let lowest_rec = self.lowest_rec.clone();
        let largest_rec = self.largest_rec.clone();
        let dist_enum = Distribution::from_str(&self.dist).unwrap_or(Distribution::Uniform);

        let mut vals: Vec<Record> = lowest_rec
            .iter()
            .zip(largest_rec.iter())
            .map(|(&low, &high)| low..=high)
            .multi_cartesian_product()
            .collect();
        vals.sort();

        //TODO: Option for either enc or unenc (probably in flatten_nd)
        let mut encrypted_tuples: Vec<(u64, Vec<i64>)> = vals
            .into_iter()
            .combinations(t)
            .map(|val_tuple| {
                let bounding_pair = get_mbq(&val_tuple);
                let freq =
                    compute_pair_weight(&bounding_pair, &dist_enum, &lowest_rec, &largest_rec);

                let encrypted_flattened: Vec<i64> = val_tuple
                    .iter()
                    .map(|rec| {
                        let true_flat_id = flatten_nd(rec, &largest_rec, &lowest_rec);
                        // MASK THE ID HERE
                        *secret_map
                            .get(&true_flat_id)
                            .expect("ID missing from universe")
                    })
                    .collect();

                (freq, encrypted_flattened)
            })
            .collect();

        encrypted_tuples.sort_unstable_by_key(|k| k.0);

        // 3. Stream to disk sequentially
        let file = File::create(output_filepath).expect("Failed to create file");
        let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, file);

        for item in encrypted_tuples {
            bincode::serialize_into(&mut writer, &item).unwrap();
        }
    }

    pub fn precompute_observed_to_disk(
        &self,
        t: usize,
        output_filepath: &str,
        dom_pairs: &HashMap<DomPair, Frequency>,
    ) -> Vec<(u64, Vec<i64>)> {
        let pb = ProgressBar::new(dom_pairs.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );
        let processed_count = AtomicU64::new(0);

        info!("Simulating leakage and aggregating t-tuple frequencies...");

        // 3. Parallel Map-Reduce to accumulate true observed frequencies
        let t_tuple_to_freq: HashMap<Vec<i64>, u64> = dom_pairs
            .into_par_iter()
            .fold(
                HashMap::new, // Thread-local map
                |mut local_map, dom_freq| {
                    // Compute the weight of this specific query
                    let (pair, freq) = dom_freq;

                    if freq > &0 {
                        // Query the encrypted DB
                        let mut response = self.encrypted_db.do_search(&pair.0, &pair.1);

                        // If the response contains at least `t` items, we can extract t-tuples
                        if response.len() >= t {
                            response.sort_unstable();

                            // For every t-tuple present in this response, add the query's weight
                            for t_tuple in response.into_iter().combinations(t) {
                                *local_map.entry(t_tuple).or_insert(0) += freq;
                            }
                        }
                    }

                    // Update progress bar without bottlenecking threads
                    let current = processed_count.fetch_add(1, AtomicOrdering::Relaxed);
                    if current % 5000 == 0 {
                        pb.set_position(current);
                    }

                    local_map
                },
            )
            .reduce(
                HashMap::new, // Merge all thread-local maps
                |mut map1, map2| {
                    for (k, v) in map2 {
                        *map1.entry(k).or_insert(0) += v;
                    }
                    map1
                },
            );

        pb.finish_with_message("Done aggregating frequencies.");

        // 4. Transform into the (Frequency, Tuple) format expected by the Translator
        let mut observed_tuples: Vec<(u64, Vec<i64>)> = t_tuple_to_freq
            .into_iter()
            .map(|(tuple, freq)| (freq, tuple))
            .collect();

        // 5. Sort by frequency
        observed_tuples.sort_unstable_by_key(|k| k.0);

        // 6. Stream to disk sequentially
        info!(
            "Writing {} unique t-tuples to disk...",
            observed_tuples.len()
        );
        let file = File::create(output_filepath).expect("Failed to create file");
        let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, file);

        for item in observed_tuples {
            bincode::serialize_into(&mut writer, &item).unwrap();
        }

        info!("Precomputation complete.");

        return observed_tuples;
    }

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

        //TODO: Non-uniform

        // Assuming every query occurs once, what is the frequency of seeing each dominant pair?
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
                let frequency =
                    compute_pair_weight(&pair, &dist.parse().unwrap(), &lowest_rec, &largest_rec);
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
}

#[cfg(test)]
mod tests {}
