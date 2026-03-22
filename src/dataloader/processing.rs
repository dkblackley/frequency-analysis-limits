use crate::dataloader::datasets::Searchable;
use crate::dataloader::error::DataProcessingError;
use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use log::{debug, error, info, warn};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::time::Instant;

// Type aliases to make the code more readable
pub type Coord = i64;
pub type Value = Coord;
pub type Record = Vec<Coord>;
pub type Responses = Vec<Record>;
pub type DomPair = (Record, Record);
// Note: Python dictionaries can use floats as keys, but Rust HashMaps cannot due to NaN ambiguity.
// Assuming frequency can be represented as an integer (e.g., scaled) or an ordered wrapper.
// Using u64 here as a placeholder for your frequency type.
pub type Frequency = u64;

struct Query {
    pub dom_pair: DomPair, // dom pair that defines this query
    pub size: Value,       // the distance between the two points
}

//TODO: Re-write this docsdtring with more precise language ALso only do this after discussing with Evgenios!
/// If you have a database that doesn't have discrete numbers to easily make rectangles over,
/// we need to calculate the continuous probability intervals. See Upfal Probability & Computing
/// chapter 8. For simplicity, This only works on 1D. We can get arbitrary D by intersecting probs
/// across dimension (See Chapter ). We can assume the probability of a single discrete item is
/// the probability of the maximum range that only includes that item. This is only true for the
/// RESPONSE distribution (which is what we want).
/// The real intricate part is that we're not sampling form a range. We're sampling TWO POINTS
/// uniformly from a range and then wanting the probability that a discrete point falls within
/// those two points. The first part naturally has probability 0, any single point on the real
/// line has infinetely small prob. To solve this
///
/// # Arguments
/// * `points` - Points on a real number line.
/// * `lower` - The lower bound of possible values.
/// * `upper` - The upper bound of possible values.
///
/// # Returns
/// * `HashMap<f64, f64>` - A mapping from every point (key) to it's probability
fn make_true_prob_uniform_continuous(
    points: Vec<f64>,
    lower: f64,
    upper: f64,
) -> HashMap<f64, f64> {
    return HashMap::new();
}

/// Calculates the true probability distribution for discrete points sampled uniformly.
///
/// # Arguments
/// * `points` - A vector of discrete points (u64).
/// * `lower` - The lower bound of the discrete range (inclusive).
/// * `upper` - The upper bound of the discrete range (inclusive).
///
/// # Returns
/// * `HashMap<u64, u64>` - A mapping of points to their respective counts or probabilities.
fn make_true_prob_uniform_discrete(points: Vec<u64>, lower: u64, upper: u64) -> HashMap<u64, u64> {
    return HashMap::new();
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
pub fn get_dominant_pair_to_freq_map(
    dim: Coord,
    lowest_val: Coord,
    largest_val: Coord,
    dist: &str,
) -> Result<HashMap<DomPair, Frequency>, DataProcessingError> {
    info!("Task 1: Computing dominant pair frequencies...");
    let timer = Instant::now();
    // let path = dp_freq_path;
    // if let Some(parent) = path.parent() {
    //     fs::create_dir_all(parent)?;
    // }

    // THis just assumes we start at 1
    let total_pairs = (largest_val.pow(dim as u32) as f64).powi(2);
    let total_dom_pairs = (total_pairs / 2_f64.powi((dim - 1) as i32)) as u64;

    // Assuming every query can occur, what is the frequency of each dominant pair?
    let mut true_pair_frequency_dict: HashMap<DomPair, Frequency> = HashMap::new();
    let domain_iter = (0..dim).map(|_| 1..=largest_val).multi_cartesian_product();

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
        for dv in get_all_dominating_values(&v, largest_val) {
            let pair = (v.clone(), dv.clone());
            let frequency = compute_pair_weight(&pair, dist, largest_val);
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
pub fn get_freq_to_dominant_pair_map(
    dim: Value,
    lowest_val: Value,
    largest_val: Value,
    dist: &str,
) -> Result<HashMap<Frequency, DomPair>, DataProcessingError> {
    info!("Task 1: Computing dominant pair frequencies...");
    let timer = Instant::now();

    // THis just assumes we start at 1
    let total_pairs = largest_val.pow(dim as u32) as Value;

    // Assuming every query can occur, what is the frequency of each dominant pair?
    let mut true_pair_frequency_dict: HashMap<Frequency, DomPair> = HashMap::new();
    let domain_iter = (0..dim).map(|_| 1..=largest_val).multi_cartesian_product();

    // Set up indicatif progress bar
    let pb = ProgressBar::new(total_pairs as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})",
            )?
            .progress_chars("#>-"),
    );

    for v in domain_iter {
        for dv in get_all_dominating_values(&v, largest_val) {
            let pair = (v.clone(), dv.clone());
            let frequency = compute_pair_weight(&pair, dist, largest_val);
            true_pair_frequency_dict.insert(frequency, pair);
        }
        pb.inc(1); // Increment the progress bar silently
    }
    pb.finish_with_message("Done computing dominant pair frequencies");

    // let file = File::create(&path)?;
    // bincode::serialize_into(BufWriter::new(file), &true_pair_frequency_dict)?;

    info!("Finished DP frequencies in {:?}", timer.elapsed());
    Ok(true_pair_frequency_dict)
}

/// Calculate for just t=1 (and then use the solver to trim t down)
pub fn get_freq_all_val_dict(
    dim: Value,
    lowest_val: Value,
    largest_val: Value,
    t: usize,
    dist: &str,
) -> Result<HashMap<Frequency, Vec<Record>>, DataProcessingError> {
    let mut vals: Vec<Record> = (0..dim)
        .map(|_| 1..=largest_val)
        .multi_cartesian_product()
        .collect();

    let total_combinations = largest_val.pow(dim as u32);

    vals.sort();
    // Approximate total combinations to drive the progress bar
    // Note: For large N, math::comb can overflow u64. We use a rough estimate or simply use an indeterminate spinner if it's too big.
    let pb = ProgressBar::new(total_combinations as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            // Added wide_bar, pos (current), len (total), and eta (time remaining)
            .template(
                "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})",
            )?
            .progress_chars("#>-"),
    );

    let mut val_tup_freq_dict: HashMap<Frequency, Vec<Record>> = HashMap::new();

    for val_tuple in vals.into_iter() {
        // Bounding pair should always be val_tuple repeated (?)
        let bounding_pair = get_mbq(&[val_tuple.clone()]);
        let freq = compute_pair_weight(&bounding_pair, dist, largest_val);

        val_tup_freq_dict.entry(freq).or_default().push(val_tuple);

        pb.inc(1);
    }
    pb.finish_with_message("Done computing value tuple frequencies");

    Ok(val_tup_freq_dict)
}

/// This takes years for any non-trivial t...
pub fn get_freq_val_t_tup_dict(
    dim: Value,
    lowest_val: Value,
    largest_val: Value,
    t: usize,
    dist: &str,
) -> Result<HashMap<Frequency, Vec<Vec<Record>>>, DataProcessingError> {
    let mut vals: Vec<Record> = (0..dim)
        .map(|_| 1..=largest_val)
        .multi_cartesian_product()
        .collect();

    let n = vals.len(); // Assuming vals is a Vec, slice, or has a .len() method
    let total_combinations = binomial_coefficient(n, t);

    vals.sort();
    // Approximate total combinations to drive the progress bar
    // Note: For large N, math::comb can overflow u64. We use a rough estimate or simply use an indeterminate spinner if it's too big.
    let pb = ProgressBar::new(total_combinations);
    pb.set_style(
        ProgressStyle::default_bar()
            // Added wide_bar, pos (current), len (total), and eta (time remaining)
            .template(
                "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})",
            )?
            .progress_chars("#>-"),
    );

    let mut val_tup_freq_dict: HashMap<Frequency, Vec<Vec<Record>>> = HashMap::new();

    for val_tuple in vals.into_iter().combinations(t) {
        let bounding_pair = get_mbq(&val_tuple);
        let freq = compute_pair_weight(&bounding_pair, dist, largest_val);

        val_tup_freq_dict.entry(freq).or_default().push(val_tuple);

        pb.inc(1);
    }
    pb.finish_with_message("Done computing value tuple frequencies");

    Ok(val_tup_freq_dict)
}

/// Instead of dumbly generating every sing t-tuple we take in a list of all dom pairs and consider
/// only dom-pairs that are a certain distance away (for a dense DB we can exactly calculate the
/// number of records returned in the response, otherwise we have to count up to t). Only considers
/// dompairs that doesn't have 0 freq
pub fn get_freq_val_possible_t_tup_dict(
    dim: Value,
    t: usize,
    dom_pairs: HashMap<DomPair, Frequency>,
    enc_db: &impl Searchable,
    dist: &str,
) -> Result<HashMap<(Value, Frequency), Vec<Vec<Value>>>, DataProcessingError> {
    // 1. Accumulate total frequency for each unique response
    let mut response_to_total_freq: HashMap<Vec<Value>, Frequency> = HashMap::new();

    let pb = ProgressBar::new(dom_pairs.keys().len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})",
            )?
            .progress_chars("#>-"),
    );

    for (pair, freq) in dom_pairs {
        if freq == 0 {
            continue;
        }

        let response = enc_db.do_search(pair.0, pair.1);

        if response.len() > 0 && response.len() <= t {
            let mut filtered_response: Vec<_> = response
                .into_iter()
                .filter(|v| v >= &enc_db.get_domain_range().0)
                .collect();

            if filtered_response.is_empty() {
                continue;
            }

            // Sort to ensure identical responses hash to the same key
            filtered_response.sort();

            // Sum up the frequencies of all queries returning this exact response
            *response_to_total_freq.entry(filtered_response).or_insert(0) += freq;
        }
        pb.inc(1);
    }

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

// Helper to determine if point u dominates point v (u_i >= v_i for all i)
fn dominates(u: &[Coord], v: &[Coord]) -> bool {
    u.iter().zip(v.iter()).all(|(u_val, v_val)| u_val >= v_val)
}

// Helper for L1 distance calculation
fn _l1_distance(p1: &[Coord], p2: &[Coord]) -> u64 {
    p1.iter().zip(p2.iter()).map(|(a, b)| a.abs_diff(*b)).sum()
}

fn get_all_dominating_values(v: &[Coord], n: Value) -> Vec<Record> {
    v.iter()
        // Instead of starting at 1, start at the value of v in this dimension
        .map(|&v_val| v_val..=n)
        .multi_cartesian_product()
        .collect()
}

fn compute_pair_weight(pair: &DomPair, dist: &str, n: Value) -> Frequency {
    let (lower, upper) = pair;

    // Under uniform, simply count the queries covering the pair
    if dist == "uniform" {
        let mut dominating_vals: u64 = 1;
        for u_val in upper {
            dominating_vals *= ((n + 1) - u_val) as u64;
        }

        let mut dominated_vals: u64 = 1;
        for l_val in lower {
            dominated_vals *= *l_val as u64;
        }

        return dominated_vals * dominating_vals;
    }
    // Fallback for unimplemented distributions ('random', 'flattened', etc.)
    //TODO: Cartesian prodect of all possible queries over any given 'rectangle'
    else {
        warn!(
            "Distribution '{}' is not fully implemented. Returning weight 0.",
            dist
        );
        return 0;
    }
}

/// Return the minimum bounding query (MBQ) of a t-tuple (i.e. dominating vals)
fn get_mbq(t_tup: &[Record]) -> DomPair {
    let dim = t_tup[0].len();
    let mut minima = vec![Value::MAX; dim];
    let mut maxima = vec![Value::MIN; dim];

    for p in t_tup {
        for d in 0..dim {
            if p[d] < minima[d] {
                minima[d] = p[d];
            }
            if p[d] > maxima[d] {
                maxima[d] = p[d];
            }
        }
    }
    (minima, maxima)
}

fn binomial_coefficient(n: usize, t: usize) -> u64 {
    if t > n {
        return 0;
    }
    // Take advantage of symmetry: C(n, t) == C(n, n-t)
    let t = std::cmp::min(t, n - t);
    let mut result = 1u64;
    for i in 1..=t {
        result = result * (n as u64 - i as u64 + 1) / (i as u64);
    }
    result
}

// Test functions
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dominates() {
        let p1 = vec![1, 2, 3];
        let p2 = vec![1, 1, 1];
        let p3 = vec![2, 2, 2];
        assert!(dominates(&p1, &p2));
        assert!(!dominates(&p2, &p1));
        assert!(!dominates(&p1, &p3));
        assert!(!dominates(&p2, &p3));
    }

    #[test]
    fn test_get_mbq() {
        let points = vec![vec![1, 10, 5], vec![5, 2, 8], vec![3, 5, 1]];
        let (minima, maxima) = get_mbq(&points);
        assert_eq!(minima, vec![1, 2, 1]);
        assert_eq!(maxima, vec![5, 10, 8]);
    }

    #[test]
    fn test_compute_pair_weight_uniform_1d() {
        // In 1D, for range [l, u] in domain [1, n], weight is l * (n + 1 - u)
        let lower = vec![2];
        let upper = vec![4];
        let n = 10;
        let weight = compute_pair_weight(&(lower, upper), "uniform", n);
        // 2 * (10 + 1 - 4) = 2 * 7 = 14
        assert_eq!(weight, 14);
    }

    #[test]
    fn test_compute_pair_weight_2d_exhaustive() {
        // 2D Domain where N=2.
        let n = 2;
        let dist = "uniform";

        // Point definitions for N=2
        let p_11 = vec![1, 1];
        let p_12 = vec![1, 2];
        let p_21 = vec![2, 1];
        let p_22 = vec![2, 2];

        // Format: (v, dv, expected_weight)
        let test_cases = vec![
            // What are all the possible 'rectangles' that can be made over these points?
            // for any one point there's pretty much always going to be 4. For a 'rectanlge'
            // it's always gonna be 2 and then there is one that covers all points.
            (&p_11, &p_11, 4), // (1*1) * (2*2) = 4
            (&p_11, &p_12, 2), // (1*1) * (2*1) = 2
            (&p_11, &p_21, 2), // (1*1) * (1*2) = 2
            (&p_11, &p_22, 1), // (1*1) * (1*1) = 1
            (&p_12, &p_12, 4), // (1*2) * (2*1) = 4
            (&p_12, &p_22, 2), // (1*2) * (1*1) = 2
            (&p_21, &p_21, 4), // (2*1) * (1*2) = 4
            (&p_21, &p_22, 2), // (2*1) * (1*1) = 2
            (&p_22, &p_22, 4), // (2*2) * (1*1) = 4
        ];

        for (v, dv, expected) in test_cases {
            let pair = (v.clone(), dv.clone());
            let weight = compute_pair_weight(&pair, dist, n);
            assert_eq!(weight, expected, "Failed for pair: v={:?}, dv={:?}", v, dv);
        }
    }
}
//     #[test]
//     fn test_e2e_precompute_pipeline() -> Result<(), Box<dyn std::error::Error>> {
//         let dp_file_path = NamedTempFile::new().expect("Failed to create temp file");
//         let val_tup_file_path = NamedTempFile::new().expect("Failed to create temp file");
//
//         // 2. Setup a small, deterministic database for the test
//         // Dimension = 2, max coordinate = 2
//         let db: HashMap<u64, Point> = HashMap::from([
//             (0, vec![1, 1]),
//             (1, vec![2, 2]),
//             (2, vec![3, 3]),
//             (3, vec![4, 4]),
//             (4, vec![5, 5]),
//         ]);
//
//         let dim = 2;
//         let t = dim * 2;
//         let dist = "uniform";
//
//         // 3. Run your main precomputation function
//         // Note: Using the dynamic n_val logic you implemented
//         compute_dominant_pair_freq(
//             t,
//             dim,
//             dist,
//             1,
//             5,
//             dp_file_path.path(),
//             val_tup_file_path.path(),
//         )?;
//
//
//
//         // The temp_dir goes out of scope here, wiping the generated files automatically!
//         Ok(())
//     }
// }
