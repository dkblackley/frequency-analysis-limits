#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]

use cp_sat::proto::CpSolverStatus;
use frequency_analysis_limits::dataloader::tester::testDB;
use frequency_analysis_limits::dataloader::{unflatten_nd, Searchable};
use frequency_analysis_limits::Record;
use frequency_analysis_limits::LAMA::solver::Solver;
use frequency_analysis_limits::LAMA::utility::get_mbq;
use frequency_analysis_limits::{DomPair, Frequency, Value};
use log::{debug, error, info, warn};
use num_rational::Ratio;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde_json::to_string;
use sha2::{Digest, Sha256};
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::BufReader;
use tempfile::NamedTempFile;

// Figure 2 of the paper but normalized. This assumption that we think the adversary knows
pub const PROB_MATRIX: [[f32; 5]; 5] = [
    [
        19.0 / 347.0,
        14.0 / 347.0,
        12.0 / 347.0,
        9.0 / 347.0,
        7.0 / 347.0,
    ],
    [
        14.0 / 347.0,
        23.0 / 347.0,
        17.0 / 347.0,
        12.0 / 347.0,
        9.0 / 347.0,
    ],
    [
        12.0 / 347.0,
        17.0 / 347.0,
        23.0 / 347.0,
        16.0 / 347.0,
        12.0 / 347.0,
    ],
    [
        9.0 / 347.0,
        12.0 / 347.0,
        16.0 / 347.0,
        19.0 / 347.0,
        14.0 / 347.0,
    ],
    [
        7.0 / 347.0,
        9.0 / 347.0,
        12.0 / 347.0,
        14.0 / 347.0,
        19.0 / 347.0,
    ],
];

// Assume we 'sampled' this as according to the above probabilities, and it just happens to be perfect.
pub const FREQ_MATRIX: [[i32; 5]; 5] = [
    [19, 14, 12, 9, 7],
    [14, 23, 17, 12, 9],
    [12, 17, 23, 16, 12],
    [9, 12, 16, 19, 14],
    [7, 9, 12, 14, 19],
];

// a 2d database with 25 total values. Each item is a single encrypted record. The above FREQ and
// PROB is the matching freq and tru prob of observing this, lets say.

/// Helper functions for LAMa tests.
/// Calculates the frequency (number of range queries covering a point) in a 2D grid.
/// In a perfect square grid of size `max_val` x `max_val`, a query [x1, x2] x [y1, y2]
/// covers (x, y) if x1 <= x <= x2 and y1 <= y <= y2.
/// The number of such intervals for 1D is (coordinate + 1) * (max_val - coordinate).
/// This all assumes a uniform response distribution
pub fn calculate_query_coverage(x: u32, y: u32, max_val: u32) -> u32 {
    let x_cov = (x + 1) * (max_val - x);
    let y_cov = (y + 1) * (max_val - y);
    x_cov * y_cov
}

/// Generates a perfectly uniform 2D database grid.
pub fn generate_grid(size: u32) -> Vec<(u32, u32)> {
    let mut grid = Vec::with_capacity((size * size) as usize);
    for x in 0..size {
        for y in 0..size {
            grid.push((x, y));
        }
    }

    grid
}

fn get_unique_resp_id(data: &Vec<Record>) -> u64 {
    let mut hasher = DefaultHasher::new();
    // The Hash trait is automatically implemented for Vecs and u32s
    data.hash(&mut hasher);
    hasher.finish()
}

fn generate_unique_id(vec: &Vec<i64>) -> String {
    let mut hasher = Sha256::new();

    for &num in vec {
        hasher.update(num.to_le_bytes());
    }

    let result = hasher.finalize();
    let hex_string = to_string(&result.to_ascii_uppercase()).unwrap();

    // Format the result as a hexadecimal string
    format!("{hex_string}")
}

/// Flattens an N-dimensional point with arbitrary upper and lower bounds into a 1D index.
/// Assumes `upper` bounds are inclusive (e.g., bounds 1 to 10 means 10 elements (0-9).
fn flatten_nd(point: &[i64], upper: &[i64], lower: &[i64]) -> i64 {
    let mut index = 0;
    let mut multiplier = 1;

    for i in (0..point.len()).rev() {
        let point_scaled = point[i] - lower[i];

        index += point_scaled * multiplier;

        let dimension_size = upper[i] - lower[i] + 1;
        multiplier *= dimension_size;
    }

    index
}

/// Maps points to unique IDs, bounds is the largest possible val
pub fn bounded_hyperrectangle_id(coords: &[u64], bounds: &[u64]) -> u64 {
    assert_eq!(
        coords.len(),
        bounds.len(),
        "Coordinates and bounds must match in dimensions"
    );

    let mut id = 0;
    let mut stride = 1;

    for (&c, &b) in coords.iter().zip(bounds.iter()) {
        id += c * stride;
        stride *= b;
    }

    id
}

// Extra tests for this specific file.
#[test]
fn test_2d_1_based_coordinates() {
    // Grid: X goes 1 to 10, Y goes 1 to 20
    let lower = [1, 1];
    let upper = [10, 20];

    // Your expected examples (adjusted to 0-based 1D output)
    // Note: The maximum index for a 10x20 grid is 199 (since we start at 0)

    // Point (1, 1) should be the very first index
    assert_eq!(flatten_nd(&[1, 1], &upper, &lower), 0);

    // Point (1, 10)
    assert_eq!(flatten_nd(&[1, 10], &upper, &lower), 9);

    // Point (2, 1) -> skips exactly one full Y row (20 elements)
    assert_eq!(flatten_nd(&[2, 1], &upper, &lower), 20);

    // Point (10, 20) -> The very last element
    assert_eq!(flatten_nd(&[10, 20], &upper, &lower), 199);
}

#[test]
fn test_3d_mixed_bounds() {
    // X: -5 to 5 (size 11)
    // Y: 0 to 9  (size 10)
    // Z: 1 to 2  (size 2)
    let lower = [-5, 0, 1];
    let upper = [5, 9, 2];

    // Minimum point maps to 0
    assert_eq!(flatten_nd(&[-5, 0, 1], &upper, &lower), 0);

    // Moving Z by 1
    assert_eq!(flatten_nd(&[-5, 0, 2], &upper, &lower), 1);

    // Moving Y by 1 skips one Z row (2 elements)
    assert_eq!(flatten_nd(&[-5, 1, 1], &upper, &lower), 2);
}

#[test]
fn test_calculate_query_coverage() {
    let numerators: [[i32; 5]; 5] = [
        [19, 14, 12, 9, 7],
        [14, 23, 17, 12, 9],
        [12, 17, 23, 16, 12],
        [9, 12, 16, 19, 14],
        [7, 9, 12, 14, 19],
    ];
    let denom = 347;

    let total_sum = numerators.iter().flatten().sum::<i32>();

    assert_eq!(total_sum, denom);

    // We construct the Ratios here for the exact test.
    let true_sum: Ratio<i32> = numerators
        .iter()
        .flatten()
        .map(|&num| Ratio::new(num, denom))
        .sum();

    assert_eq!(true_sum, Ratio::from_integer(1));

    let float_sum: f32 = PROB_MATRIX.iter().flatten().sum();

    let epsilon = 0.0001;

    assert!((float_sum - 1.0).abs() < epsilon, "Sum was {float_sum}");
}

use frequency_analysis_limits::LAMA::selector::Selector;
use frequency_analysis_limits::LAMA::translator::Translator;
pub fn generate_secret_mapping(universe: &[i64]) -> HashMap<i64, i64> {
    // 1. Copy the universe to act as our pool of available aliases
    let mut encrypted_aliases = universe.to_vec();

    // Create a seeded RNG (requires a 32-byte array)
    let mut rng = StdRng::seed_from_u64(42);

    encrypted_aliases.shuffle(&mut rng);

    // 3. Bind each true ID to a totally random (but unique) alias from the same domain
    let secret_map: HashMap<i64, i64> = universe
        .iter()
        .cloned()
        .zip(encrypted_aliases.into_iter())
        .collect();

    secret_map
}
pub fn generate_mapping(universe: &[i64]) -> HashMap<i64, i64> {
    // 1. Copy the universe to act as our pool of available aliases
    let mut plaintext = universe.to_vec();

    // 3. Bind each true ID to a totally random (but unique) alias from the same domain
    let secret_map: HashMap<i64, i64> = universe
        .iter()
        .cloned()
        .zip(plaintext.into_iter())
        .collect();

    secret_map
}

fn check_isomorphism(responses: &HashMap<i64, i64>, rows: i64, cols: i64) -> Option<&'static str> {
    // 2. Build the solver's actual mapping: True_Plaintext_ID -> Guessed_Plaintext_ID
    let mut true_to_guessed = HashMap::new();
    for (encrypted_alias, guessed_id) in responses {
        true_to_guessed.insert(encrypted_alias, *guessed_id);
    }

    let max_r = rows - 1;
    let max_c = cols - 1;

    // Helper closures to translate between 1D IDs and 2D Coordinates
    // This matches the math from your `flatten_nd` function
    let to_coord = |id: i64| -> (i64, i64) { (id / cols, id % cols) };
    let to_id = |r: i64, c: i64| -> i64 { r * cols + c };

    // 3. Define the 8 valid geometric transformations for a 2D grid
    let transformations: Vec<(&str, Box<dyn Fn(i64, i64) -> (i64, i64)>)> = vec![
        ("Identity (Perfect Match)", Box::new(|r, c| (r, c))),
        ("Rotated 90°", Box::new(move |r, c| (c, max_r - r))),
        ("Rotated 180°", Box::new(move |r, c| (max_r - r, max_c - c))),
        ("Rotated 270°", Box::new(move |r, c| (max_c - c, r))),
        (
            "Reflected Horizontal (Flip Y)",
            Box::new(move |r, c| (r, max_c - c)),
        ),
        (
            "Reflected Vertical (Flip X)",
            Box::new(move |r, c| (max_r - r, c)),
        ),
        ("Reflected Main Diagonal", Box::new(|r, c| (c, r))),
        (
            "Reflected Anti-Diagonal",
            Box::new(move |r, c| (max_c - c, max_r - r)),
        ),
    ];

    // 4. Test the solver's mapping against each transformation
    for (name, transform) in transformations {
        let mut is_match = true;

        for (&true_id, &guessed_id) in &true_to_guessed {
            let (r, c) = to_coord(*true_id);
            let (trans_r, trans_c) = transform(r, c);
            let expected_guessed_id = to_id(trans_r, trans_c);

            if guessed_id != expected_guessed_id {
                is_match = false;
                break;
            }
        }

        // If all 400 points conform to this specific transformation, we cracked it
        if is_match {
            return Some(name);
        }
    }

    None
}

#[test]
fn end_to_end() {
    let _ = env_logger::builder()
        .is_test(true)
        // This forces 'info' to be the default level if RUST_LOG isn't set
        .filter_level(log::LevelFilter::Debug)
        .try_init();

    let rows = 15;
    let cols = 15;

    info!("Loading test DB ({}x{})", rows, cols);
    let loaded_db: Box<dyn Searchable + Sync> = Box::new(testDB::new(rows, cols, 100));

    let dim = loaded_db.get_dims();
    let dist = "uniform";
    let (low_pair, high_pair) = loaded_db.get_dom_pair();
    let universe = loaded_db.get_universe();
    let largest_enc_val = universe.iter().max().unwrap();

    let selector = Selector {
        dist: dist.to_string(),
        encrypted_db: &loaded_db,
        dim,
        lowest_rec: low_pair.clone(),
        largest_rec: high_pair.clone(),
    };

    info!("1. Computing True Frequencies (DomPair -> Freq map)...");
    let dom_pair_freq = selector.get_dominant_pair_to_freq_map().unwrap();

    info!("2. Precomputing observed encrypted tuples for t=1...");
    let obs_t1 = selector.precompute_observed_to_disk(1, "dummy1", &dom_pair_freq);

    info!("3. Building plaintext dictionary for fast t=1 lookup...");
    let pt_t1_dict = build_plaintext_dict(&obs_t1);

    info!(
        "4. Initializing Translator with universe size: {}",
        universe.len()
    );
    let mut translator = Translator::new(*largest_enc_val, universe.clone());

    info!("--> Processing Base Case (t=1)");
    translator.process_t1(&obs_t1, &pt_t1_dict);

    // Explicitly bind read-only references so the closures cleanly pass the Sync+Send bounds
    // required by Rayon's worker threads in `process_t_greater_than_1`.
    let dom_freq_ref = &dom_pair_freq;
    let high_pair_ref = &high_pair;
    let low_pair_ref = &low_pair;

    info!("5. Compiling closures for dynamic frequency lookups...");

    // The Observed Frequency Closure
    let get_observed_freq = |enc_tuple: &[i64]| -> u64 {
        let true_plaintexts: Vec<Record> = enc_tuple
            .iter()
            .map(|rec| unflatten_nd(*rec, high_pair_ref, low_pair_ref))
            .collect();

        let dom_pair = get_mbq(&true_plaintexts);
        *dom_freq_ref.get(&dom_pair).unwrap_or(&0)
    };

    // The Expected Frequency Closure
    let get_expected_freq = |candidate_vals: &[i64]| -> u64 {
        let decoded_points: Vec<Record> = candidate_vals
            .iter()
            .map(|&v| unflatten_nd(v, high_pair_ref, low_pair_ref))
            .collect();

        let dom_pair = get_mbq(&decoded_points);
        *dom_freq_ref.get(&dom_pair).unwrap_or(&0)
    };

    info!("--> Processing Recursive Case (t=2) across thread pool...");
    translator.process_t_greater_than_1(2, &universe, get_observed_freq, get_expected_freq);
    info!("--> Processing Recursive Case (t=3) across thread pool...");
    translator.process_t_greater_than_1(3, &universe, get_observed_freq, get_expected_freq);
    info!("--> Processing Recursive Case (t=4) across thread pool...");
    translator.process_t_greater_than_1(4, &universe, get_observed_freq, get_expected_freq);

    info!("6. Building and executing the CP-SAT Solver for the final constraint graph...");
    let mut solver = Solver::new(translator.get_var_index_map());

    // Note: depending on your Translator method signatures, you may need to use
    // `mut model = translator.get_proto_model(); solver.solve(&mut model);`
    // if get_proto_model() consumes `self`.
    let responses = solver.solve(&mut translator.get_proto_model());

    if responses.len() != universe.len() {
        let mut freq_to_t_tuple: HashMap<(Value, Frequency), Vec<Vec<Value>>> =
            selector.get_freq_val_t_tup_dict(2).unwrap();

        error!("FATAL: Solver failed to reconstruct full universe!!");
        error!("Known frequency-to-plaintext mappings: {freq_to_t_tuple:?}");
        error!("'encrypted/encoded' universe of plaintexts: {universe:?}");
        error!("Frequency to t-tuple matches: {freq_to_t_tuple:?}");
        error!("Solver Status: {:?}", solver.solution_stat);
        panic!("Solver did not return a full assignment.");
    }

    let mut correct = 0;
    let mut incorrect = 0;

    // Key is actually the true encoded value.
    for (encrypted_alias, guessed_plaintext) in responses.clone() {
        if guessed_plaintext == encrypted_alias {
            correct += 1;
        } else {
            incorrect += 1;
        }
    }

    info!(
        "Direct matches: {} correct, {} incorrect",
        correct, incorrect
    );
    info!("7. Running Isomorphism Checks...");

    // Check for a valid rotation/reflection
    match check_isomorphism(&responses, rows as i64, cols as i64) {
        Some(transformation_name) => {
            info!(
                "SUCCESS! Solver found a valid isomorphism: {}",
                transformation_name
            );
        }
        None => {
            error!("Solver produced a mathematically invalid reconstruction.");
            panic!("Test Failed: Not a valid rotation or reflection.");
        }
    }

    info!("Test completed successfully.")
}

// Helper to group the plaintexts by frequency for fast O(1) lookups
fn build_plaintext_dict(tuples: &[(u64, Vec<i64>)]) -> HashMap<u64, Vec<Vec<i64>>> {
    let mut dict: HashMap<u64, Vec<Vec<i64>>> = HashMap::new();
    for (freq, tup) in tuples {
        dict.entry(*freq).or_default().push(tup.clone());
    }
    dict
}
