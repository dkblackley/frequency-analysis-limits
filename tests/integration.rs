#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]

use cp_sat::proto::CpSolverStatus;
use frequency_analysis_limits::dataloader::tester::testDB;
use frequency_analysis_limits::dataloader::{flatten_nd, unflatten_nd, Searchable};
use frequency_analysis_limits::LAMA::solver::Solver;
use frequency_analysis_limits::LAMA::utility::get_mbq;
use frequency_analysis_limits::{DomPair, Probability, Value};
use frequency_analysis_limits::{Frequency, Record};
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

use frequency_analysis_limits::LAMA::selector::Selector;
use frequency_analysis_limits::LAMA::translator::Translator;

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

        // If all points conform to this specific transformation, we cracked it
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
        .filter_level(log::LevelFilter::Debug)
        .try_init();

    // Force rayon to one thread
    rayon::ThreadPoolBuilder::new()
        .num_threads(0)
        .build_global()
        .unwrap();

    let rows_cols = 6;
    let dim = 3;

    info!("Loading test DB ({}x{})", rows_cols, rows_cols);
    let loaded_db: Box<dyn Searchable + Sync> = Box::new(testDB::new(dim, rows_cols, 25));

    let dist = "uniform";
    let eps = 0.0; // Perfect knowledge constraint
    let delt = 0.0;

    let selector = Selector::new(dist, &loaded_db, eps, delt);

    let (low_pair, high_pair) = loaded_db.get_dom_pair();
    let universe = loaded_db.get_universe();
    let largest_enc_val: i64 = flatten_nd(&high_pair, &high_pair, &low_pair);

    info!(
        "3. Initializing Translator with universe size: {}",
        universe.len()
    );
    let mut translator = Translator::new(largest_enc_val, universe.clone());

    // Grab references to avoid lifetime closure issues
    let query_dist_ref = &selector.query_distribution;
    let high_pair_ref = &high_pair;
    let low_pair_ref = &low_pair;

    info!("4. Compiling probabilistic closures for dynamic lookups...");

    // 1. Observed probability of the encrypted records
    let get_observed_prob = |enc_tuple: &[i64]| -> f64 {
        let true_plaintexts: Vec<Record> = enc_tuple
            .iter()
            .map(|rec| unflatten_nd(*rec, high_pair_ref, low_pair_ref))
            .collect();
        let dom_pair = get_mbq(&true_plaintexts);

        // Use the native cumulative probability directly
        query_dist_ref.get_cumulative_prob(&dom_pair)
    };

    // 2. Expected (true) probability of proposed plaintexts
    let get_expected_prob = |plaintexts: &[i64]| -> f64 {
        let pt_records: Vec<Record> = plaintexts
            .iter()
            .map(|&v| unflatten_nd(v, high_pair_ref, low_pair_ref))
            .collect();
        let pt_mbq = get_mbq(&pt_records);

        // Use the native cumulative probability directly
        query_dist_ref.get_cumulative_prob(&pt_mbq)
    };

    // 3. Unified Validator
    let active_eps = if eps == 0.0 { 1e-5 } else { eps };
    let validate_candidate = |enc_tuple: &[i64], proposed_plaintexts: &[i64]| -> bool {
        let obs_prob = get_observed_prob(enc_tuple);
        if obs_prob == 0.0 {
            return false;
        }

        let exp_prob = get_expected_prob(proposed_plaintexts);
        (obs_prob - exp_prob).abs() <= active_eps
    };

    info!("--> Processing Base Case (t=1)");
    translator.process_t1(&universe, &validate_candidate);

    info!("--> Processing Recursive Case (t=2) sequentially across chunk models...");
    translator.process_t_greater_than_1(2, &universe, &validate_candidate);

    info!("--> Processing Recursive Case (t=3) sequentially across chunk models...");
    translator.process_t_greater_than_1(3, &universe, &validate_candidate);

    // t=3 is usually good enough for every dist type, uniform mostly is good after t=2 but sometimes
    // gets better at t=3. t=4 is almost always actually overkill
    info!(
        "Remember: Computing for DB size {} by {}...",
        rows_cols, rows_cols
    );
    info!("--> Processing Recursive Case (t=4) sequentially across chunk models...");
    translator.process_t_greater_than_1(4, &universe, &validate_candidate);

    info!("5. Building and executing the CP-SAT Solver for the final constraint graph...");
    let mut solver = Solver::new(translator.get_var_index_map());

    let mut model = translator.get_proto_model();
    let responses = solver.solve(&mut model, false);

    if (solver.solution_stat != CpSolverStatus::Optimal
        && solver.solution_stat != CpSolverStatus::Feasible
        || dist == "uniform" && responses[&0].len() < 2)
        || (dist == "gaussian" && responses[&0].len() < 1)
    {
        // minimum number of expected reconstructions
        let freq_to_t_tuple: HashMap<(Value, Frequency), Vec<Vec<Value>>> =
            selector.get_freq_val_t_tup_dict(2).unwrap();

        error!("FATAL: Solver failed to reconstruct full universe!!");
        // error!("Known frequency-to-plaintext mappings: {freq_to_t_tuple:?}");
        error!("'encrypted/encoded' universe of plaintexts: {universe:?}");
        error!("Intvars workings: {:?}", solver);
        //error!("CPModel: {:?}", model);
        error!("Solver Status: {:?}", solver.solution_stat);
        panic!("Solver did not return a full assignment.");
    }

    let mut correct = 0;
    let mut incorrect = 0;

    for (encrypted_alias, guessed_plaintext) in responses.clone() {
        if guessed_plaintext[0] == encrypted_alias {
            correct += 1;
        } else {
            incorrect += 1;
        }
    }

    info!(
        "Direct matches: {} correct, {} incorrect",
        correct, incorrect
    );
    info!("6. Running Isomorphism Checks...");

    let total_responses = responses[&0].len();
    let mut found_truth = false;

    for i in 0..total_responses {
        let mut iso_map = HashMap::new();
        for (key, val) in responses.clone() {
            iso_map.insert(key, val[i]);
        }
        match check_isomorphism(&iso_map, rows_cols as i64, rows_cols as i64) {
            Some(transformation_name) => {
                if transformation_name.contains("Perfect") {
                    found_truth = true;
                }
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
    }

    assert!(found_truth);
    info!("Test completed successfully.")
}
