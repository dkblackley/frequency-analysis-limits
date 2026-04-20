#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]

use cp_sat::proto::CpSolverStatus;
use frequency_analysis_limits::dataloader::tester::testDB;
use frequency_analysis_limits::dataloader::two_d::{Location, TwoDMap};
use frequency_analysis_limits::dataloader::{flatten_nd, unflatten_nd, Searchable};
use frequency_analysis_limits::LAMA::query::QueryDistribution;
use frequency_analysis_limits::LAMA::solver::Solver;
use frequency_analysis_limits::LAMA::utility::{encloses, get_mbq};
use frequency_analysis_limits::{DomPair, Value};
use frequency_analysis_limits::{Frequency, Record};
use log::{debug, error, info};
use rand::distributions::Distribution;
use rustc_hash::FxHashMap;
use sha2::Digest;
use std::collections::HashMap;
use std::fmt::format;

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
use frequency_analysis_limits::LAMA::solver::CpSolverStatus::{Feasible, Optimal};
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
fn end_flat() {
    let _ = env_logger::builder()
        .is_test(true)
        .filter_level(log::LevelFilter::Debug)
        .try_init();

    // Force rayon to one thread
    rayon::ThreadPoolBuilder::new()
        .num_threads(0)
        .build_global()
        .unwrap();

    let point_1 = Location {
        longitude: 0.0,
        latitude: 0.0,
    };

    let point_2 = Location {
        longitude: 1.0,
        latitude: 0.0,
    };

    let loaded_db: Box<dyn Searchable + Sync> = Box::new(
        TwoDMap::new_unscaled(vec![point_1, point_2], "flat_test", (0, 2), (0, 3)).unwrap(),
    );

    let all_recs = loaded_db.get_universe();

    let mut found_first = true;
    let mut found_second = true;
    let wanted_flat_1 = flatten_nd(&[0, 0], &[3, 3], &[0, 0]);
    let wanted_flat_2 = flatten_nd(&[1, 0], &[3, 3], &[0, 0]);

    for rec in all_recs.clone() {
        if rec == wanted_flat_1 {
            found_first = true;
        } else if rec == wanted_flat_2 {
            found_second = true;
        } else {
            panic!("Found not wanted/ unknown item: {}, {:?}", rec, all_recs)
        }
    }

    assert!(found_first);
    assert!(found_second);

    let dist = "flat";
    let eps = 0.0; // Perfect knowledge constraint
    let delt = 0.0;

    let selector = Selector::new(dist, &loaded_db);

    let (low_pair, high_pair) = loaded_db.get_dom_pair();
    assert_eq!(high_pair, vec![3, 3]);
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
        query_dist_ref.cumulative_prob_lookup(&dom_pair)
    };

    // 2. Expected (true) probability of proposed plaintexts
    let get_expected_prob = |plaintexts: &[i64]| -> f64 {
        let pt_records: Vec<Record> = plaintexts
            .iter()
            .map(|&v| unflatten_nd(v, high_pair_ref, low_pair_ref))
            .collect();
        let pt_mbq = get_mbq(&pt_records);

        // Use the native cumulative probability directly
        query_dist_ref.cumulative_prob_lookup(&pt_mbq)
    };

    // 3. Unified Validator
    let active_eps = if eps == 0.0 { 1e-5 } else { eps };
    let validate_candidate = |enc_tuple: &[i64], proposed_plaintexts: &[i64]| -> (bool, f64) {
        let obs_prob = get_observed_prob(enc_tuple);

        let exp_prob = get_expected_prob(proposed_plaintexts);
        (
            (obs_prob - exp_prob).abs() <= active_eps,
            (obs_prob - exp_prob).abs(),
        )
    };

    info!("Bruteforcing t=2");
    let (mut model, index_map) = Translator::process_t_brute_force(
        2,
        largest_enc_val,
        &*universe.clone(),
        &validate_candidate,
    );
    let mut solver = Solver::new(index_map);
    let responses = solver.solve(&mut model, largest_enc_val, false);
    let mut universes = Vec::new();
    let true_values: Vec<&i64> = responses.keys().collect();
    let number_solutions = solver.num_sols;

    for i in 0..number_solutions {
        let mut universe = Vec::new();
        for true_val in true_values.clone() {
            let unflat_key = unflatten_nd(*true_val, high_pair_ref, low_pair_ref);
            let unflat_val =
                unflatten_nd(responses[true_val][i as usize], high_pair_ref, low_pair_ref);
            universe.push(format!(
                "True: {:?}, Reconstructed: {:?}",
                unflat_key, unflat_val
            ));
        }
        universes.push(universe);
    }

    info!("Possible reconstructions: {:?}", universes);
    let mut temp_map = HashMap::new();
    let mut temp_universe: Vec<i64> = (0..largest_enc_val).collect();

    for (key, vals) in responses {
        let difference: Vec<_> = temp_universe
            .clone()
            .into_iter()
            .filter(|item| !vals.contains(item))
            .collect();
        temp_map.insert(key, difference);
    }
    debug!("Impossible values: {:?}", temp_map);

    info!("--> Processing Base Case (t=1)");
    translator.process_t1(&universe, &validate_candidate);

    info!("--> Processing Recursive Case (t=2) sequentially across chunk models...");
    translator.process_t_greater_than_1(2, &universe, &validate_candidate);

    info!("5. Building and executing the CP-SAT Solver for the final constraint graph...");
    let mut solver = Solver::new(translator.get_var_index_map());

    let mut model = translator.get_proto_model();
    let responses = solver.solve(&mut model, largest_enc_val, false);

    if solver.solution_stat != Optimal && solver.solution_stat != Feasible {
        error!("FATAL: Solver failed to reconstruct full universe!!");
        error!("'encrypted/encoded' universe of plaintexts: {universe:?}");
        error!("Intvars workings: {:?}", solver);
        error!("Solver Status: {:?}", solver.solution_stat);
        panic!("Solver did not return a full assignment.");
    }

    let total_responses = solver.num_sols;
    // Remember, we cannot get rid of/hide the distances. Hence, after t=1 and t=2 the solver is
    // able to narrow down to the 4 items of equal distance.
    assert_eq!(total_responses, 4);
    let mut found_truth = false;

    for i in 0..total_responses {
        let mut true_count = 0;
        let mut iso_map = HashMap::new();
        for (key, val) in responses.clone() {
            for recon in &val {
                if *recon == key {
                    true_count += 1;
                    break;
                }
            }

            iso_map.insert(key, val[i as usize]);
        }
        if true_count == 2 {
            found_truth = true;
        }
    }

    assert!(found_truth);
    info!("Test completed successfully.")
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

    // This should take about a minute to run... (if not very sparse!)
    let rows_cols = 6;
    let dim = 2;

    info!("Loading test DB ({}x{})", rows_cols, rows_cols);
    let loaded_db: Box<dyn Searchable + Sync> = Box::new(testDB::new(dim, rows_cols, 80));

    let dist = "flat";
    let eps = 0.0; // Perfect knowledge constraint
    let delt = 0.0;

    let selector = Selector::new(dist, &loaded_db);

    let (low_pair, high_pair) = loaded_db.get_dom_pair();
    let universe = loaded_db.get_universe();
    let largest_enc_val: i64 = flatten_nd(&high_pair, &high_pair, &low_pair);

    info!(
        "3. Initializing Translator with universe size: {} and low/high pair {:?}, {:?}",
        universe.len(),
        low_pair,
        high_pair
    );
    let mut translator = Translator::new(largest_enc_val, universe.clone());

    let query_dist_ref = &selector.query_distribution;
    let high_pair_ref = &high_pair;
    let low_pair_ref = &low_pair;

    info!("4. Compiling probabilistic closures for dynamic lookups...");

    let get_observed_prob = |enc_tuple: &[i64]| -> f64 {
        let true_plaintexts: Vec<Record> = enc_tuple
            .iter()
            .map(|rec| unflatten_nd(*rec, high_pair_ref, low_pair_ref))
            .collect();
        let dom_pair = get_mbq(&true_plaintexts);

        query_dist_ref.cumulative_prob_lookup(&dom_pair)
    };

    let get_expected_prob = |plaintexts: &[i64]| -> f64 {
        let pt_records: Vec<Record> = plaintexts
            .iter()
            .map(|&v| unflatten_nd(v, high_pair_ref, low_pair_ref))
            .collect();
        let pt_mbq = get_mbq(&pt_records);

        query_dist_ref.cumulative_prob_lookup(&pt_mbq)
    };

    let active_eps = if eps == 0.0 { 1e-5 } else { eps };
    let validate_candidate = |enc_tuple: &[i64], proposed_plaintexts: &[i64]| -> (bool, f64) {
        let obs_prob = get_observed_prob(enc_tuple);

        let exp_prob = get_expected_prob(proposed_plaintexts);
        (
            (obs_prob - exp_prob).abs() <= active_eps,
            (obs_prob - exp_prob).abs(),
        )
    };

    info!("--> Processing Base Case (t=1)");
    translator.process_t1(&universe, &validate_candidate);

    info!("--> Processing Recursive Case (t=2) sequentially across chunk models...");
    translator.process_t_greater_than_1(2, &universe, &validate_candidate);

    info!("--> Processing Recursive Case (t=3) sequentially across chunk models...");
    translator.process_t_greater_than_1(3, &universe, &validate_candidate);

    info!("5. Building and executing the CP-SAT Solver for the final constraint graph...");
    let mut solver = Solver::new(translator.get_var_index_map());

    let mut model = translator.get_proto_model();

    // NOTE: The only change in the test is passing largest_enc_val here
    let responses = solver.solve(&mut model, largest_enc_val, false);

    if (solver.solution_stat != Optimal && solver.solution_stat != Feasible
        || dist == "uniform" && responses[&0].len() < 2)
        || (dist == "gaussian" && responses[&0].len() < 1)
    {
        error!("FATAL: Solver failed to reconstruct full universe!!");
        error!("'encrypted/encoded' universe of plaintexts: {universe:?}");
        error!("Intvars workings: {:?}", solver);
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

#[test]
fn end_to_end_sampled() {
    let _ = env_logger::builder()
        .is_test(true)
        .filter_level(log::LevelFilter::Debug)
        .try_init();

    // Force rayon to one thread
    rayon::ThreadPoolBuilder::new()
        .num_threads(0)
        .build_global()
        .unwrap();

    let rows_cols = 15;
    let dim = 2;

    info!("Loading test DB ({}x{})", rows_cols, rows_cols);
    let loaded_db: Box<dyn Searchable + Sync> = Box::new(testDB::new(dim, rows_cols, 60));

    let dist = "gaussian";
    let target_query_percentage = 0.5; // e.g., observe 5% of all possible queries
    let fixed_delta = 0.9; // 50% confidence that error <= epsilon

    // 1. Initialize a baseline selector to generate the distribution space
    // We pass 0.0 for eps/delt temporarily just to build the QueryDistribution
    let mut selector = Selector::new(dist, &loaded_db);

    let largest_dom = loaded_db.get_dom_pair();

    // How many queries can we do? A query is a hyperrectangle, The total number of hyper rectangles
    // in a space is n*(n+1)/2 where n is the length of a dim, then mutliply for each dim.
    let mut total = 1;

    for item in largest_dom.1 {
        total *= ((item + 1) * (item + 2)) / 2;
    }

    let total_possible_queries = total;
    let num_queries_to_observe =
        ((total_possible_queries as f64) * target_query_percentage).ceil() as usize;

    info!("--- Sampling Parameters ---");
    info!(
        "Targeting {}% of queries ({} samples).",
        target_query_percentage * 100.0,
        num_queries_to_observe
    );

    // 5. Sample the specific number of queries based on the distribution weights
    let mut rng = rand::thread_rng();
    let mut observed_queries: Vec<DomPair> = Vec::with_capacity(num_queries_to_observe);

    info!("Sampling {} queries...", num_queries_to_observe);
    for _ in 0..num_queries_to_observe {
        // Use the WeightedIndex sampler you built in QueryDistribution
        let sampled_idx = selector.query_distribution.sampler.sample(&mut rng);
        let sampled_pair = selector.query_distribution.pairs[sampled_idx].clone();
        observed_queries.push(sampled_pair);
    }

    // 3. Calculate Empirical VC Dimension (Fixing Issue A)
    // Assuming you update `get_vc_sukp_bound` to return the `q` profit, we calculate `b` here.
    // Ideally, `get_vc_sukp_bound` should just return `b` directly.
    let responses = selector.get_responses_from_queries(observed_queries.clone());
    let q_profit = selector.get_vc_sukp_bound(responses.clone());
    let empirical_vc_dim = q_profit.log2().floor() + 1.0;

    // 4. Reverse-calculate Epsilon (Fixing Issue B)
    // Note: You must update `calculate_epsilon` to accept `d` (the EVC) directly,
    // replacing the hardcoded `num_items - 1`.
    let eps = Selector::calculate_epsilon(
        empirical_vc_dim, // Pass the EVC, not num_items
        num_queries_to_observe,
        fixed_delta,
    );

    info!("Empirical VC Dimension: {}", empirical_vc_dim);
    info!("Guaranteed Epsilon Bound: {}", eps);
    info!("---------------------------");

    // 6. Update selector and Translator initialization with our real epsilon

    let (low_pair, high_pair) = loaded_db.get_dom_pair();
    let universe = loaded_db.get_universe();
    let largest_enc_val: i64 = flatten_nd(&high_pair, &high_pair, &low_pair);

    let mut translator = Translator::new(largest_enc_val, universe.clone());
    let query_dist_ref = &selector.query_distribution;
    let high_pair_ref = &high_pair;
    let low_pair_ref = &low_pair;

    let weight_map = Selector::get_raw_dompair_weight_map(responses.clone(), &loaded_db);
    let pair_to_prob = Selector::compute_cumulative_prob_map(
        &low_pair,
        &high_pair,
        &weight_map,
        num_queries_to_observe as f64,
    );

    info!("4. Compiling probabilistic closures for dynamic lookups...");

    let mut dom_to_enclosed_freq = FxHashMap::default();

    let all_dompairs = Selector::get_dom_pairs(&loaded_db);

    for pair in &all_dompairs {
        let mut enclose_count = 0;
        for que in &observed_queries {
            if encloses(que, pair) {
                enclose_count += 1;
            }
            dom_to_enclosed_freq.insert(
                pair.clone(),
                enclose_count as f64 / num_queries_to_observe as f64,
            );
        }
    }

    let get_observed_prob = |enc_tuple: &[i64]| -> f64 {
        let underlying_vals: Vec<Record> = enc_tuple
            .iter()
            .map(|rec| unflatten_nd(*rec, high_pair_ref, low_pair_ref))
            .collect();
        let target_mbq = get_mbq(&underlying_vals);
        dom_to_enclosed_freq
            .get(&target_mbq)
            .unwrap_or(&0.0)
            .clone()
    };

    let get_expected_prob = |plaintexts: &[i64]| -> f64 {
        let pt_records: Vec<Record> = plaintexts
            .iter()
            .map(|&v| unflatten_nd(v, high_pair_ref, low_pair_ref))
            .collect();
        let pt_mbq = get_mbq(&pt_records);

        query_dist_ref.cumulative_prob_lookup(&pt_mbq)
    };

    let active_eps = if eps == 0.0 { 1e-5 } else { eps };
    let validate_candidate = |enc_tuple: &[i64], proposed_plaintexts: &[i64]| -> (bool, f64) {
        let obs_prob = get_observed_prob(enc_tuple);

        let exp_prob = get_expected_prob(proposed_plaintexts);
        (
            (obs_prob - exp_prob).abs() <= active_eps,
            (obs_prob - exp_prob).abs(),
        )
    };

    info!("--> Processing Base Case (t=1)");
    translator.process_t1(&universe, &validate_candidate);

    info!("--> Processing Recursive Case (t=2) sequentially across chunk models...");
    translator.process_t_greater_than_1(2, &universe, &validate_candidate);

    info!("--> Processing Recursive Case (t=3) sequentially across chunk models...");
    translator.process_t_greater_than_1(3, &universe, &validate_candidate);

    info!("5. Building and executing the CP-SAT Solver for the final constraint graph...");
    let mut solver = Solver::new(translator.get_var_index_map());

    let mut model = translator.get_proto_model();

    // NOTE: The only change in the test is passing largest_enc_val here
    let responses = solver.solve(&mut model, largest_enc_val, false);

    if (solver.solution_stat != Optimal && solver.solution_stat != Feasible
        || dist == "uniform" && responses[&0].len() < 2)
        || (dist == "gaussian" && responses[&0].len() < 1)
    {
        error!("FATAL: Solver failed to reconstruct full universe!!");
        error!("'encrypted/encoded' universe of plaintexts: {universe:?}");
        error!("Intvars workings: {:?}", solver);
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
