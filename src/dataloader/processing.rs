use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use log::{debug, error, info, warn};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::time::Instant;

// Type aliases to make the code more readable
type Point = Vec<usize>;
type Pair = (Point, Point);
// Note: Python dictionaries can use floats as keys, but Rust HashMaps cannot due to NaN ambiguity.
// Assuming frequency can be represented as an integer (e.g., scaled) or an ordered wrapper.
// Using u64 here as a placeholder for your frequency type.
type Frequency = u64;

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
fn make_true_prob_uniform_continuous(points: Vec<f64>, lower: f64, upper: f64, ) -> HashMap<f64, f64> {

    return HashMap::new()
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
fn make_true_prob_uniform_discrete(points: Vec<u64>, lower: u64, upper: u64, ) -> HashMap<u64, u64> {

    return HashMap::new();
}


fn compute_dominant_pair_freq(t: usize, dim: usize, dist: &str, _n: usize, record_value_dict: HashMap<usize, Point>, base_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let compute_dp_freq = true;
    let compute_val_tup_freq = true;
    let compute_matches = true;


    info!("Starting precomputation for dim={}, dist={}, t={}", dim, dist, t);

    // n_val is the largest possible value in the values of record_value_dict
    let mut largest_val = 1;
    let mut lowest_val = 1;

    for (key, val,) in &record_value_dict  {
        for i in 0..val.len() {
            if val[i] > largest_val {
                largest_val = val[i];
            }
            if val[i] < lowest_val {
                lowest_val = val[i];
            }
        }
    }

    // 1. COMPUTE THE FREQUENCY OF EVERY DOMINANT PAIR
    if compute_dp_freq {
        info!("Task 1: Computing dominant pair frequencies...");
        let timer = Instant::now();
        let path = base_dir.join(format!("dp_frequencies/{}/{}_dimensions.bin", dist, dim));
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let total_pairs = (largest_val.pow(dim as u32) as f64).powi(2);
        let total_dom_pairs = (total_pairs / 2_f64.powi((dim - 1) as i32)) as u64;

        // Set up indicatif progress bar
        let pb = ProgressBar::new(total_dom_pairs);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})")?
            .progress_chars("#>-"));

        // Assuming every query can occur, what is the frequency of each dominant pair?
        let mut true_pair_frequency_dict: HashMap<Pair, Frequency> = HashMap::new();
        let domain_iter = (0..dim).map(|_| 1..=largest_val).multi_cartesian_product();

        for v in domain_iter {
            for dv in get_all_dominating_values(&v, largest_val) {
                let pair = (v.clone(), dv.clone());
                let frequency = compute_pair_weight(&pair, dist, largest_val);
                true_pair_frequency_dict.insert(pair, frequency);

                pb.inc(1); // Increment the progress bar silently
            }
        }
        pb.finish_with_message("Done computing dominant pair frequencies");

        let file = File::create(&path)?;
        bincode::serialize_into(BufWriter::new(file), &true_pair_frequency_dict)?;

        info!("Finished DP frequencies in {:?}", timer.elapsed());
    }

    // 2. COMPUTE THE FREQUENCY OF EVERY T-TUPLE OF VALUES
    if compute_val_tup_freq {
        info!("Task 2: Computing value tuple frequencies...");
        let timer = Instant::now();
        let path = base_dir.join(format!("val_tup_frequencies/{}/{}_dim/t{}.bin", dist, dim, t));
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let vals: Vec<Point> = (0..dim).map(|_| 1..=largest_val).multi_cartesian_product().collect();

        // Approximate total combinations to drive the progress bar
        // Note: For large N, math::comb can overflow u64. We use a rough estimate or simply use an indeterminate spinner if it's too big.
        let pb = ProgressBar::new_spinner();
        pb.set_style(ProgressStyle::default_spinner().template("{spinner:.green} [{elapsed_precise}]")?);

        let mut val_tup_freq_dict: HashMap<Frequency, Vec<Vec<Point>>> = HashMap::new();

        for mut val_tuple in vals.into_iter().combinations(t) {
            val_tuple.sort();

            let bounding_pair = get_mbq(&val_tuple);
            let freq = compute_pair_weight(&bounding_pair, dist, largest_val);

            val_tup_freq_dict.entry(freq).or_default().push(val_tuple);

            pb.inc(1);

        }
        pb.finish_with_message("Done computing value tuple frequencies");

        let file = File::create(&path)?;
        bincode::serialize_into(BufWriter::new(file), &val_tup_freq_dict)?;

        info!("Finished value tuple frequencies in {:?}", timer.elapsed());
    }


    info!("Finished precompute task entirely.");
    Ok(())
}


// Helper to determine if point u dominates point v (u_i >= v_i for all i)
fn dominates(u: &[usize], v: &[usize]) -> bool {
    u.iter().zip(v.iter()).all(|(u_val, v_val)| u_val >= v_val)
}

// Helper for L1 distance calculation
fn _l1_distance(p1: &[usize], p2: &[usize]) -> usize {
    p1.iter().zip(p2.iter()).map(|(a, b)| a.abs_diff(*b)).sum()
}

fn get_all_dominating_values(v: &[usize], n: usize) -> Vec<Point> {
    let dim = v.len();
    // Generate all points in the domain and filter by dominance
    (0..dim)
        .map(|_| 1..=n)
        .multi_cartesian_product()
        .filter(|u| dominates(u, v))
        .collect()
}

fn compute_pair_weight(pair: &Pair, dist: &str, n: usize) -> Frequency {
    let (lower, upper) = pair;
    let dim = lower.len();

    // Prefixing unused variables with an underscore to prevent compiler warnings
    let _max_query_size = dim * (n - 1);
    let _pair_distance = _l1_distance(lower, upper);
    let mut _pair_weight: Frequency = 0;

    let _pos_capacities: Vec<usize> = (0..dim).map(|i| n - upper[i]).collect();
    let _neg_capacities: Vec<usize> = (0..dim).map(|i| lower[i] - 1).collect();

    let mut _capacities: Vec<usize> = _pos_capacities.iter().copied().chain(_neg_capacities.iter().copied()).collect();
    _capacities.retain(|&c| c > 0);

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
    else {
        warn!("Distribution '{}' is not fully implemented. Returning weight 0.", dist);
        return 0;
    }
}

// Return the minimum bounding query (MBQ) of a t-tuple
fn get_mbq(t_tup: &[Point]) -> Pair {
    let dim = t_tup[0].len();
    let mut minima = vec![usize::MAX; dim];
    let mut maxima = vec![usize::MIN; dim];

    for p in t_tup {
        for d in 0..dim {
            if p[d] < minima[d] { minima[d] = p[d]; }
            if p[d] > maxima[d] { maxima[d] = p[d]; }
        }
    }
    (minima, maxima)
}


// Test functions
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

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
        let points = vec![
            vec![1, 10, 5],
            vec![5, 2, 8],
            vec![3, 5, 1],
        ];
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
            assert_eq!(
                weight, expected,
                "Failed for pair: v={:?}, dv={:?}", v, dv
            );
        }
    }


        #[test]
        fn test_e2e_precompute_pipeline() -> Result<(), Box<dyn std::error::Error>> {
            // 1. Create a temporary directory
            // This will automatically be cleaned up when the test finishes
            let temp_dir = tempdir()?;
            let base_path = temp_dir.path();

            // 2. Setup a small, deterministic database for the test
            // Dimension = 2, max coordinate = 2
            let db: HashMap<usize, Point> = HashMap::from([
                (0, vec![1, 1]),
                (1, vec![2, 2]),
                (2, vec![3, 3]),
                (3, vec![4, 4]),
                (4, vec![5, 5]),
            ]);

            let dim = 2;
            let t = dim*2;
            let dist = "uniform";

            // 3. Run your main precomputation function
            // Note: Using the dynamic n_val logic you implemented
            compute_dominant_pair_freq(t, dim, dist, 0, db, base_path)?;

            // 4. Verify the outputs exist on disk
            let dp_file_path = base_path.join(format!("dp_frequencies/{}/{}_dimensions.bin", dist, dim));
            assert!(
                dp_file_path.exists(),
                "Expected DP frequency file was not created at {:?}", dp_file_path
            );

            let val_tup_file_path = base_path.join(format!("val_tup_frequencies/{}/{}_dim/t{}.bin", dist, dim, t));
            assert!(
                val_tup_file_path.exists(),
                "Expected value tuple file was not created at {:?}", val_tup_file_path
            );

            // 5. Verify the files actually contain data (aren't just empty 0-byte files)
            let dp_metadata = fs::metadata(&dp_file_path)?;
            assert!(dp_metadata.len() > 0, "DP frequency file is empty");

            // The temp_dir goes out of scope here, wiping the generated files automatically!
            Ok(())
        }

}