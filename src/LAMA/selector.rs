use crate::dataloader::{flatten_nd, unflatten_nd, Searchable};
use crate::LAMA::error::LAMAError;
use crate::LAMA::utility::{
    binomial_coefficient, compute_pair_weight, dominates, get_all_dominating_values, get_mbq,
    Distribution,
};
use crate::{Coord, DomPair, Frequency, Record, Value};
use good_lp::{
    default_solver, variable, Constraint, Expression, ProblemVariables, Solution, SolverModel,
};
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
/// This component determines which record-retriewval set expressions are used.
/// Specifically, it generates the frequencies of dominating pairs, as it corresponds to some dist
pub struct Selector<'a> {
    pub dist: String,
    pub encrypted_db: &'a Box<dyn Searchable + Sync>,
    pub dim: Value,
    pub lowest_rec: Record,
    pub largest_rec: Record,
    pub evc: usize,
    epsilon: Value,
    delta: Value,
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

    pub fn get_dom_pairs(&self) -> Vec<DomPair> {
        info!("Finding all possible responses...");

        let lowest_rec = self.lowest_rec.clone();
        let largest_rec = self.largest_rec.clone();

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
                .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})").unwrap()
                .progress_chars("#>-"),
        );
        let processed_count = AtomicU64::new(0);

        let largest_rec_ref = &largest_rec;
        let processed_count_ref = &processed_count;

        let pair_vec: Vec<DomPair> = domain_iter
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
                        let count = processed_count_ref.fetch_add(1, AtomicOrdering::Relaxed);
                        if count % 100_000 == 0 {
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

    // TODO: Sample instead.

    pub fn get_all_possible_responses(&self) -> Vec<Vec<Value>> {
        info!("Finding all possible responses...");

        let timer = Instant::now();
        let lowest_rec = self.lowest_rec.clone();
        let largest_rec = self.largest_rec.clone();
        let dist = self.dist.clone();

        let dom_pairs = self.get_dom_pairs();

        let pb = ProgressBar::new(dom_pairs.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})").unwrap()
                .progress_chars("#>-"),
        );
        let processed_count = AtomicU64::new(0);

        let processed_count_ref = &processed_count;

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
    pub fn get_vc_sukp_bound(&self) -> f64 {
        let responses = self.get_all_possible_responses();
        // The size of the largest response
        let largest_resp = self
            .encrypted_db
            .do_search(
                &self.encrypted_db.get_dom_pair().0,
                &self.encrypted_db.get_dom_pair().1,
            )
            .len();
        let universe_size: usize = largest_resp;
        let capacity = largest_resp;

        // 1. Calculate Upper Bound via LP Relaxation
        let q_upper = Self::solve_sukp_internal(capacity, &responses, universe_size, false);

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

        // 2. Fallback to exact MILP if the bounds straddle a power of 2
        Self::solve_sukp_internal(capacity, &responses, universe_size, true)
    }

    fn solve_sukp_internal(
        capacity: usize,
        itemsets: &[Vec<Value>],
        num_items: usize,
        is_integer: bool,
    ) -> f64 {
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
                model = model.with(y[i] - x[j] << 0.0);
            }
        }

        let solution = model.solve().unwrap();
        solution.eval(y.iter().sum::<Expression>())
    }

    /// Calculates the guaranteed error bound (epsilon) for a given sample size.
    ///
    /// Ref: "Finding the True Frequent Itemsets" (Riondato & Vandin, 2014)
    pub fn calculate_epsilon(num_items: usize, num_samples: usize, delta: f64) -> f64 {
        // Corollary 2: The absolute worst-case VC-dimension for the power set of items.
        // VC(R(2^I)) <= |I| - 1
        let d = (num_items - 1) as f64;

        let l = num_samples as f64;

        // Theorem 1[cite: 147, 148]: Universal constant c is estimated to be <= 0.5
        let c = 0.5;

        // Equation 1[cite: 145]: epsilon = sqrt( (c / l) * (d + ln(1 / delta)) )
        // Note: We use natural log (.ln()) as is standard for Chernoff/VC bounds.
        let epsilon = ((c / l) * (d + (1.0 / delta).ln())).sqrt();

        epsilon
    }

    /// Calculates the number of samples required to hit a target epsilon and delta.
    pub fn calculate_required_samples(num_items: usize, target_epsilon: f64, delta: f64) -> usize {
        let d = (num_items - 1) as f64;
        let c = 0.5;

        // Algebraic rearrangement of Equation 1 [cite: 145] to solve for l
        let l = (c / target_epsilon.powi(2)) * (d + (1.0 / delta).ln());

        l.ceil() as usize
    }
}

#[cfg(test)]
mod tests {
    use crate::LAMA::selector::Selector;
    use rand::RngExt;
    use std::collections::HashSet;

    #[test]
    fn test_vc_bounds_and_empirical_reality() {
        let num_items = 1000;
        let confidence = 0.90; // 90% confidence
        let delta = 1.0 - confidence; // delta = 0.10
        let target_epsilon = 0.05; // 5% error margin

        // 1. Calculate how many queries we theoretically need to be 90% sure
        // that NO subset deviates by more than 5%.
        let required_samples =
            Selector::calculate_required_samples(num_items, target_epsilon, delta);

        println!(
            "For {} items, to guarantee <= {} error with {} confidence:",
            num_items, target_epsilon, confidence
        );
        println!("Theoretical queries required: {}", required_samples);

        // Verify the math reverses correctly
        let calculated_eps = Selector::calculate_epsilon(num_items, required_samples, delta);
        assert!((calculated_eps - target_epsilon).abs() < 1e-5);

        // 2. Empirical Simulation
        // Let's create a single "target itemset" of 3 specific records we care about.
        let target_itemset: HashSet<usize> = vec![42, 105, 999].into_iter().collect();

        // Let's assume our actual (hidden) distribution returns this exact subset 15% of the time.
        let true_probability = 0.15;

        let mut rng = rand::rng();
        let mut observed_hits = 0;

        // Simulate drawing the required number of samples
        for _ in 0..required_samples {
            // Did our query return a superset of the target itemset?
            if rng.random_bool(true_probability) {
                observed_hits += 1;
            }
        }

        let empirical_probability = observed_hits as f64 / required_samples as f64;
        let actual_error = (true_probability - empirical_probability).abs();

        println!("True Probability: {:.4}", true_probability);
        println!(
            "Empirical Probability from {} samples: {:.4}",
            required_samples, empirical_probability
        );
        println!("Actual Empirical Error: {:.6}", actual_error);
        println!("Theoretical Max Error (Epsilon): {:.6}", target_epsilon);

        // The empirical error should be vastly smaller than the worst-case epsilon bound.
        assert!(actual_error <= target_epsilon);
    }

    #[test]
    fn test_sukp_disjoint_itemsets() {
        // 10 items, each item is its own itemset
        let num_items = 10;
        let itemsets: Vec<Vec<usize>> = (0..num_items).map(|i| vec![i]).collect();

        // Capacity 5: We should only be able to pick 5 itemsets
        let capacity = 5;
        let q = Selector::get_vc_sukp_bound(capacity, &itemsets, num_items);

        // Profit should be exactly 5.0
        assert!((q - 5.0).abs() < 1e-5);

        // VC-bound b = floor(log2(5)) + 1 = 2 + 1 = 3
        let b = q.log2().floor() + 1.0;
        assert_eq!(b, 3.0);
    }
    #[test]
    fn test_sukp_power_set() {
        let num_items = 3;
        // All non-empty subsets of {0, 1, 2}
        let itemsets = vec![
            vec![0],
            vec![1],
            vec![2], // Size 1
            vec![0, 1],
            vec![0, 2],
            vec![1, 2],    // Size 2
            vec![0, 1, 2], // Size 3
        ];

        // Capacity 3: We can include everything
        let q = get_vc_sukp_bound(3, &itemsets, num_items);
        assert!((q - 7.0).abs() < 1e-5);

        // Capacity 2: We can pick {0}, {1}, {0,1} but not anything containing {2}
        let q_limited = get_vc_sukp_bound(2, &itemsets, num_items);
        // Best is to pick all subsets of any 2 items: {0}, {1}, {0,1} -> Profit 3
        assert!((q_limited - 3.0).abs() < 1e-5);
    }
}
