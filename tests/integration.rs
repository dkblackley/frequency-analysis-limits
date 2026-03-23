#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]

use frequency_analysis_limits::dataloader::Searchable;
use frequency_analysis_limits::Record;
use std::collections::hash_map::Entry;
mod common;

use crate::common::testDB;
use frequency_analysis_limits::LAMA::solver::Solver;
use frequency_analysis_limits::{DomPair, Frequency, Value};
use log::{debug, error, info, warn};
use num_rational::Ratio;
use serde_json::to_string;
use sha2::{Digest, Sha256};
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

#[test]
fn end_to_end() {
    let _ = env_logger::try_init();

    let rows = 200;
    let cols = 200;
    let loaded_db = testDB::new(rows, cols, 90);

    let dp_file_path = NamedTempFile::new().expect("Failed to create temp file");
    let val_tup_file_path = NamedTempFile::new().expect("Failed to create temp file");

    let t = loaded_db.get_dims() * 2;
    let dim = loaded_db.get_dims();
    let dist = "uniform";
    let (low_pair, high_pair) = loaded_db.get_dom_pair();
    let binding = loaded_db.get_universe();
    let largest_enc_val = binding.iter().max().unwrap();

    let selector = Selector {
        dist: dist.to_string(),
        encrypted_db: &loaded_db,
        dim,
        lowest_rec: low_pair,
        largest_rec: high_pair.clone(),
    };

    // THis is a bruteforce calculation of the TRUE frequency of all dompairs.
    let dom_pair_freq: HashMap<DomPair, Frequency> =
        selector.get_dominant_pair_to_freq_map().unwrap();

    // A bruteforce calculation of the TRUE frequency of all t-tuples
    // let freq_of_t_tups: HashMap<Frequency, Vec<Vec<Record>>> =
    //     get_freq_val_t_tup_dict(dim as u64, low, high, t as usize, dist).unwrap();

    // let freq_of_one_tups: HashMap<Frequency, Vec<Record>> =
    //     get_freq_all_val_dict(dim as Value, low, high, t as usize, dist).unwrap();

    // Remember, value in this case means encoded ID, i.e a single point. We assume the 'ideal' case
    // that is: Both the true freq-plaintext and observed are the same.
    let mut freq_to_t_tuple: HashMap<(Value, Frequency), Vec<Vec<Value>>> = selector
        .get_freq_val_possible_t_tup_dict((2 * dim) as usize, dom_pair_freq)
        .unwrap();

    let mut translator = Translator::new(freq_to_t_tuple.clone(), *largest_enc_val);

    let (mut model, int_var_map) = translator.translate(&freq_to_t_tuple, loaded_db.get_universe());

    let solver = Solver::new(int_var_map);

    let responses = solver.solve(&mut model);

    if responses.len() != loaded_db.get_universe().len() {
        error!("Solver failed!! Printout out debug info");
        error!("Known frequency-to-plaintext mappings: {freq_to_t_tuple:?}");
        let uni = loaded_db.get_universe();
        error!("'encrypted/encoded' universe of plaintexts: {uni:?}");
        error!("Frequency to t-tuple matches: {freq_to_t_tuple:?}");
        error!("Solver: {solver:?}");
        panic!();
    }

    let mut correct = 0;
    let mut incorrect = 0;

    //Key is actually the true value.
    for (key, val) in responses {
        if key == val {
            correct = correct + 1
        } else {
            incorrect = incorrect + 1
        }
    }

    info!("OK!")
}
