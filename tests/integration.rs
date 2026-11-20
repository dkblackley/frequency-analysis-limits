#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]

use frequency_analysis_limits::dataloader::tester::testDB;
use frequency_analysis_limits::dataloader::two_d::{Location, TwoDMap};
use frequency_analysis_limits::dataloader::{flatten_nd, unflatten_nd, Searchable};
use frequency_analysis_limits::LAMA::mini_solver::CpModel;
use frequency_analysis_limits::LAMA::query::QueryDistribution;
use frequency_analysis_limits::LAMA::selector::Selector;
use frequency_analysis_limits::LAMA::solver::CpSolverStatus::{Feasible, Optimal};
use frequency_analysis_limits::LAMA::solver::Solver;
use frequency_analysis_limits::LAMA::translator::Translator;
use frequency_analysis_limits::LAMA::utility::{encloses, find_valid_solution, get_mbq};
use frequency_analysis_limits::{DomPair, Value};
use frequency_analysis_limits::{Frequency, Record};
use itertools::all;
use log::warn;
use log::{debug, error, info};
use rand::distributions::Distribution;
use rand::Rng;
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

/// Generates a specified number of random locations within a bounding box.
pub fn generate_random_locations(
    count: usize,
    min_lon: f64,
    max_lon: f64,
    min_lat: f64,
    max_lat: f64,
) -> Vec<Location> {
    let mut rng = rand::thread_rng();
    let mut locations = Vec::with_capacity(count);

    for _ in 0..count {
        locations.push(Location {
            // gen_range with `..=` ensures the maximum value is inclusive
            longitude: rng.gen_range(min_lon..=max_lon),
            latitude: rng.gen_range(min_lat..=max_lat),
        });
    }

    locations
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

    // let point_1 = Location {
    //     longitude: 0.0,
    //     latitude: 0.0,
    // };
    //
    // let point_2 = Location {
    //     longitude: 1.0,
    //     latitude: 0.0,
    // };
    //
    // let point_3 = Location {
    //     longitude: 0.0,
    //     latitude: 2.0,
    // };

    let min_longitude = 0.0;
    let max_longitude = 3.0;
    let min_latitude = 0.0;
    let max_latitude = 3.0;

    let points =
        generate_random_locations(10, min_longitude, max_longitude, min_latitude, max_latitude);

    // Scale to an 8x8 DB.
    let loaded_db: Box<dyn Searchable + Sync> =
        Box::new(TwoDMap::new_unscaled(points, "flat_test", (2, 2), (2, 2)).unwrap());

    let all_recs = loaded_db.get_universe();

    let mut found_first = true;
    let mut found_second = true;
    let wanted_flat_1 = flatten_nd(&[0, 0], &[3, 3], &[0, 0]);
    let wanted_flat_2 = flatten_nd(&[1, 0], &[3, 3], &[0, 0]);

    // for rec in all_recs.clone() {
    //     if rec == wanted_flat_1 {
    //         found_first = true;
    //     } else if rec == wanted_flat_2 {
    //         found_second = true;
    //     } else {
    //         panic!("Found not wanted/ unknown item: {}, {:?}", rec, all_recs)
    //     }
    // }

    assert!(found_first);
    assert!(found_second);

    let dist = "flat";
    let eps = 0.0; // Perfect knowledge constraint
    let delt = 0.0;

    let selector = Selector::new(dist, &loaded_db);

    let (low_pair, high_pair) = loaded_db.get_dom_pair();
    // assert_eq!(high_pair, vec![3, 3]);
    let universe = loaded_db.get_universe();
    let largest_enc_val: i64 = flatten_nd(&high_pair, &high_pair, &low_pair);

    info!(
        "3. Initializing Translator with universe size: {}",
        universe.len()
    );
    let mut translator = Translator::new(largest_enc_val, universe.clone(), &2, true);

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
    let model = CpModel::new(None);
    let (mut model, index_map) = Translator::process_t_brute_force(
        2,
        largest_enc_val,
        &*universe.clone(),
        model,
        &validate_candidate,
        true,
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

    info!("--> Processing Recursive Case (t=3) sequentially across chunk models...");
    translator.process_t_greater_than_1(3, &universe, &validate_candidate);

    // only have two records so t=2 is the max.

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

    let valid = find_valid_solution(&responses, &total_responses);

    assert_ne!(valid.0, -1);

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
    let rows_cols = 10;
    let dim = 2;

    info!("Loading test DB ({}x{})", rows_cols, rows_cols);
    let loaded_db: Box<dyn Searchable + Sync> = Box::new(testDB::new(dim, rows_cols, 80));

    let dist = "uniform";
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
    let mut translator = Translator::new(largest_enc_val, universe.clone(), &4, true);

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

    let active_eps = if eps == 0.0 { 1e-9 } else { eps };
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

    // info!("--> Processing Recursive Case (t=4) sequentially across chunk models...");
    // translator.process_t_greater_than_1(4, &universe, &validate_candidate);

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

    for i in 0..total_responses {
        let mut iso_map = HashMap::new();
        for (key, val) in responses.clone() {
            iso_map.insert(key, val[i]);
        }
        match check_isomorphism(&iso_map, rows_cols as i64, rows_cols as i64) {
            Some(transformation_name) => {
                info!(
                    "SUCCESS! Solver found a valid isomorphism: {}",
                    transformation_name
                );
            }
            None => {
                error!("Solver produced a mathematically invalid reconstruction.");
            }
        }
    }

    let valid = find_valid_solution(&responses, &(total_responses as i32));

    assert_ne!(valid.0, -1);

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

    let rows_cols = 8;
    let dim = 2;

    info!("Loading test DB ({}x{})", rows_cols, rows_cols);
    let loaded_db: Box<dyn Searchable + Sync> = Box::new(testDB::new(dim, rows_cols, 35));

    let dist = "gaussian";
    let target_query_percentage = 0.2; // e.g., observe 20% of all possible queries
    let fixed_delta = 0.1; // 90% confidence that error <= epsilon

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

    let all_resp = selector.get_all_possible_responses();

    let (observed_queries, responses, eps) =
        selector.sample_percent_responses(target_query_percentage, fixed_delta, &all_resp);

    let (low_pair, high_pair) = loaded_db.get_dom_pair();
    let universe = loaded_db.get_universe();
    let largest_enc_val: i64 = flatten_nd(&high_pair, &high_pair, &low_pair);

    let mut translator = Translator::new(largest_enc_val, universe.clone(), &4, false);
    let query_dist_ref = &selector.query_distribution;
    let high_pair_ref = &high_pair;
    let low_pair_ref = &low_pair;

    let weight_map = Selector::get_raw_dompair_weight_map(responses.clone(), &loaded_db);

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

    let active_eps = if eps == 0.0 { 1e-9 } else { eps };
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

    // info!("--> Processing Recursive Case (t=3) sequentially across chunk models...");
    // translator.process_t_greater_than_1(3, &universe, &validate_candidate);

    let index_map = translator.get_var_index_map();
    let mut model = translator.get_proto_model();

    // brute force instead!

    // let mut model = PythonCpModel::new();
    //
    // info!("--> Brute force up to (t=3)");
    // let (mut model, index_map) = Translator::process_t_brute_force(
    //     3,
    //     largest_enc_val,
    //     &*universe.clone(),
    //     model,
    //     &validate_candidate,
    // );
    // model.validate(&index_map);

    info!("5. Building and executing the CP-SAT Solver for the final constraint graph...");
    let mut solver = Solver::new(index_map);

    // NOTE: The only change in the test is passing largest_enc_val here
    let responses = solver.solve(&mut model, largest_enc_val, false);

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

    let total_responses = responses[&0].len();

    let valid = find_valid_solution(&responses, &(total_responses as i32));
    info!("Valid output: {}", valid.0);

    // ---- MSE of every reconstruction vs the truth (raw, and best over the 8 grid rotations/reflections) ----
    // responses[true_enc][j] = value reconstruction j gave that record
    let num_recons = responses.values().next().map_or(0, |v| v.len());
    let side = high_pair[0] - low_pair[0]; // assumes a square 2D grid
    let to_xy = |enc: i64| -> (i64, i64) {
        let p = unflatten_nd(enc, &high_pair, &low_pair);
        (p[0] - low_pair[0], p[1] - low_pair[1])
    };
    // The 8 rotations/reflections of a square grid
    let symmetry = |(x, y): (i64, i64), k: usize| -> (i64, i64) {
        match k {
            0 => (x, y), 1 => (side - x, y), 2 => (x, side - y), 3 => (side - x, side - y),
            4 => (y, x), 5 => (side - y, x), 6 => (y, side - x), _ => (side - y, side - x),
        }
    };
    let mut raw_mses = Vec::with_capacity(num_recons);
    let mut aligned_mses = Vec::with_capacity(num_recons);
    for j in 0..num_recons {
        // MSE of reconstruction j after applying symmetry k to it
        let mse_under = |k: usize| -> f64 {
            responses.iter().map(|(&true_enc, guesses)| {
                let (tx, ty) = to_xy(true_enc);
                let (gx, gy) = symmetry(to_xy(guesses[j]), k);
                ((tx - gx).pow(2) + (ty - gy).pow(2)) as f64
            }).sum::<f64>() / responses.len() as f64
        };
        raw_mses.push(mse_under(0));
        aligned_mses.push((0..8).map(|k| mse_under(k)).fold(f64::INFINITY, f64::min));
    }
    let summary = |mses: &[f64]| -> (f64, f64, f64) {
        let avg = mses.iter().sum::<f64>() / mses.len().max(1) as f64;
        let best = mses.iter().copied().fold(f64::INFINITY, f64::min);
        let worst = mses.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (avg, best, worst)
    };
    let (avg, best, worst) = summary(&raw_mses);
    info!("MSE over {} recons (raw):     avg {:.3}, best {:.3}, worst {:.3}", num_recons, avg, best, worst);
    let (avg, best, worst) = summary(&aligned_mses);
    info!("MSE over {} recons (aligned): avg {:.3}, best {:.3}, worst {:.3}", num_recons, avg, best, worst);

    // ---- Does the data prefer the truth? Same cost as the search: singles + pairs ----
    let encs: Vec<i64> = responses.keys().copied().collect();
    let cost = |enc_tuple: &[i64], pts: &[i64]| -> i64 {
        let (valid, err) = validate_candidate(enc_tuple, pts);
        if valid { (err.powf(2.0) * 1_000_00.0).round() as i64 } else { 1_000_000_000_000 }
    };
    let score = |guess: &dyn Fn(i64) -> i64| -> i64 {
        let mut total = 0;
        for (a, &e1) in encs.iter().enumerate() {
            total += cost(&[e1], &[guess(e1)]);
            for &e2 in &encs[a + 1..] {
                total += cost(&[e1, e2], &[guess(e1), guess(e2)]);
            }
        }
        total
    };
    let truth_error = score(&|e| e); // test DBs: encrypted id == true plaintext
    let best_found = (0..num_recons).map(|j| score(&|e| responses[&e][j])).min().unwrap_or(i64::MAX);
    info!("Total error: truth {}, best reconstruction {}", truth_error, best_found);

    info!("Test completed successfully.")
}

#[test]
fn end_to_end_sampled_noisy() {
    let _ = env_logger::builder()
        .is_test(true)
        .filter_level(log::LevelFilter::Debug)
        .try_init();

    // Force rayon to one thread
    rayon::ThreadPoolBuilder::new()
        .num_threads(0)
        .build_global()
        .unwrap();

    let rows_cols = 8;
    let dim = 2;

    info!("Loading test DB ({}x{})", rows_cols, rows_cols);
    let loaded_db: Box<dyn Searchable + Sync> = Box::new(testDB::new(dim, rows_cols, 35));

    let dist = "gaussian";
    let target_query_percentage = 0.2; // e.g., observe 20% of all possible queries
    let fixed_delta = 0.1; // 90% confidence that error <= epsilon
    let dist_noise = 0.01; // TV distance between the true and the attacker's query distribution

    // 1. Initialize a baseline selector to generate the distribution space
    // We pass 0.0 for eps/delt temporarily just to build the QueryDistribution
    // The selector holds (and samples from) the TRUE, noisy query distribution
    let mut selector = Selector::new_with_noise(dist, &loaded_db, dist_noise);
    // The attacker only knows the noiseless distribution. Same pairs (same order) as the selector,
    // since the Gaussian weights are assigned by index into this vec
    let attacker_dist = QueryDistribution::new(
        selector.query_distribution.pairs.clone(),
        &loaded_db,
        dist.parse().unwrap(),
    );

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

    let all_resp = selector.get_all_possible_responses();

    let (observed_queries, responses, sampling_eps) =
        selector.sample_percent_responses(target_query_percentage, fixed_delta, &all_resp);
    // Sampling error + distribution error
    let eps = sampling_eps + dist_noise;

    let max_dist_err = selector
        .query_distribution
        .mbq_to_cumulative_prob
        .iter()
        .map(|(mbq, p)| (p - attacker_dist.cumulative_prob_lookup(mbq)).abs())
        .fold(0.0, f64::max);
    info!("Sampling eps {sampling_eps}, dist noise {dist_noise}, largest actual distribution error {max_dist_err}");
    assert!(max_dist_err <= dist_noise + 1e-9, "distribution error exceeds dist_noise");

    let (low_pair, high_pair) = loaded_db.get_dom_pair();
    let universe = loaded_db.get_universe();
    let largest_enc_val: i64 = flatten_nd(&high_pair, &high_pair, &low_pair);

    let mut translator = Translator::new(largest_enc_val, universe.clone(), &4, false);
    let query_dist_ref = &attacker_dist;
    let high_pair_ref = &high_pair;
    let low_pair_ref = &low_pair;

    let weight_map = Selector::get_raw_dompair_weight_map(responses.clone(), &loaded_db);

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

    let active_eps = if eps == 0.0 { 1e-9 } else { eps };
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

    // info!("--> Processing Recursive Case (t=3) sequentially across chunk models...");
    // translator.process_t_greater_than_1(3, &universe, &validate_candidate);

    let index_map = translator.get_var_index_map();
    let mut model = translator.get_proto_model();

    // brute force instead!

    // let mut model = PythonCpModel::new();
    //
    // info!("--> Brute force up to (t=3)");
    // let (mut model, index_map) = Translator::process_t_brute_force(
    //     3,
    //     largest_enc_val,
    //     &*universe.clone(),
    //     model,
    //     &validate_candidate,
    // );
    // model.validate(&index_map);

    info!("5. Building and executing the CP-SAT Solver for the final constraint graph...");
    let mut solver = Solver::new(index_map);

    // NOTE: The only change in the test is passing largest_enc_val here
    let responses = solver.solve(&mut model, largest_enc_val, false);

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

    let total_responses = responses[&0].len();

    let valid = find_valid_solution(&responses, &(total_responses as i32));
    info!("Valid output: {}", valid.0);

    // ---- MSE of every reconstruction vs the truth (raw, and best over the 8 grid rotations/reflections) ----
    // responses[true_enc][j] = value reconstruction j gave that record
    let num_recons = responses.values().next().map_or(0, |v| v.len());
    let side = high_pair[0] - low_pair[0]; // assumes a square 2D grid
    let to_xy = |enc: i64| -> (i64, i64) {
        let p = unflatten_nd(enc, &high_pair, &low_pair);
        (p[0] - low_pair[0], p[1] - low_pair[1])
    };
    // The 8 rotations/reflections of a square grid
    let symmetry = |(x, y): (i64, i64), k: usize| -> (i64, i64) {
        match k {
            0 => (x, y), 1 => (side - x, y), 2 => (x, side - y), 3 => (side - x, side - y),
            4 => (y, x), 5 => (side - y, x), 6 => (y, side - x), _ => (side - y, side - x),
        }
    };
    let mut raw_mses = Vec::with_capacity(num_recons);
    let mut aligned_mses = Vec::with_capacity(num_recons);
    for j in 0..num_recons {
        // MSE of reconstruction j after applying symmetry k to it
        let mse_under = |k: usize| -> f64 {
            responses.iter().map(|(&true_enc, guesses)| {
                let (tx, ty) = to_xy(true_enc);
                let (gx, gy) = symmetry(to_xy(guesses[j]), k);
                ((tx - gx).pow(2) + (ty - gy).pow(2)) as f64
            }).sum::<f64>() / responses.len() as f64
        };
        raw_mses.push(mse_under(0));
        aligned_mses.push((0..8).map(|k| mse_under(k)).fold(f64::INFINITY, f64::min));
    }
    let summary = |mses: &[f64]| -> (f64, f64, f64) {
        let avg = mses.iter().sum::<f64>() / mses.len().max(1) as f64;
        let best = mses.iter().copied().fold(f64::INFINITY, f64::min);
        let worst = mses.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (avg, best, worst)
    };
    let (avg, best, worst) = summary(&raw_mses);
    info!("MSE over {} recons (raw):     avg {:.3}, best {:.3}, worst {:.3}", num_recons, avg, best, worst);
    let (avg, best, worst) = summary(&aligned_mses);
    info!("MSE over {} recons (aligned): avg {:.3}, best {:.3}, worst {:.3}", num_recons, avg, best, worst);

    // ---- Does the data prefer the truth? Same cost as the search: singles + pairs ----
    let encs: Vec<i64> = responses.keys().copied().collect();
    let cost = |enc_tuple: &[i64], pts: &[i64]| -> i64 {
        let (valid, err) = validate_candidate(enc_tuple, pts);
        if valid { (err.powf(2.0) * 1_000_00.0).round() as i64 } else { 1_000_000_000_000 }
    };
    let score = |guess: &dyn Fn(i64) -> i64| -> i64 {
        let mut total = 0;
        for (a, &e1) in encs.iter().enumerate() {
            total += cost(&[e1], &[guess(e1)]);
            for &e2 in &encs[a + 1..] {
                total += cost(&[e1, e2], &[guess(e1), guess(e2)]);
            }
        }
        total
    };
    let truth_error = score(&|e| e); // test DBs: encrypted id == true plaintext
    let best_found = (0..num_recons).map(|j| score(&|e| responses[&e][j])).min().unwrap_or(i64::MAX);
    info!("Total error: truth {}, best reconstruction {}", truth_error, best_found);

    info!("Test completed successfully.")
}

/// Everything one perfect-setting run produces, normalised so the two t = 2 paths can be compared
/// (Pumpkin may find solutions in a different order, and the original code already iterates
/// HashMaps, so only the SETS are meaningful).
#[derive(Debug, PartialEq)]
struct PerfectRun {
    /// t-tuple tables after each round, rows sorted
    tables: Vec<(usize, Vec<(Vec<i64>, Vec<Vec<i64>>)>)>,
    /// solutions of the last global round, sorted
    last_solutions: Vec<Vec<i64>>,
    constraints_per_t: Vec<(usize, usize)>,
    recons_per_t: Vec<(usize, usize)>,
    /// final databases (record -> value), sorted
    databases: Vec<Vec<(i64, i64)>>,
}

/// `loaded_db` and `selector` are shared by both paths: get_dom_pairs' query order (hence the
/// gaussian/beta weights) differs between two Selector::new calls, so two separately built runs
/// differ even with the same code.
fn perfect_run(
    loaded_db: &Box<dyn Searchable + Sync>,
    selector: &Selector,
    eps: f64,
    max_t: usize,
    lazy: bool,
) -> PerfectRun {
    let (low_pair, high_pair) = loaded_db.get_dom_pair();
    let universe = loaded_db.get_universe();
    let largest_enc_val: i64 = flatten_nd(&high_pair, &high_pair, &low_pair);
    let mut translator = Translator::new(largest_enc_val, universe.clone(), &max_t, true);
    translator.set_lazy_pairs(lazy);

    let q = &selector.query_distribution;
    let prob = |vals: &[i64]| -> f64 {
        let recs: Vec<Record> = vals.iter().map(|&v| unflatten_nd(v, &high_pair, &low_pair)).collect();
        q.cumulative_prob_lookup(&get_mbq(&recs))
    };
    // eps > 0 plays the role of the noise bound: many candidates per record, many databases
    let active_eps = if eps == 0.0 { 1e-9 } else { eps };
    let validate = |enc: &[i64], pt: &[i64]| -> (bool, f64) {
        let err = (prob(enc) - prob(pt)).abs();
        (err <= active_eps, err)
    };

    translator.process_t1(&universe, &validate);
    for t in 2..=max_t {
        translator.process_t_greater_than_1(t, &universe, &validate);
    }

    let mut tables = Vec::new();
    for t in 2..=max_t {
        let mut rows: Vec<(Vec<i64>, Vec<Vec<i64>>)> = translator.t_assignment_archive[&t]
            .iter()
            .map(|(k, v)| {
                let mut v = v.clone();
                v.sort();
                (k.clone(), v)
            })
            .collect();
        rows.sort();
        tables.push((t, rows));
    }
    let mut last_solutions = translator.last_solutions.clone().unwrap_or_default();
    last_solutions.sort();
    let mut constraints_per_t: Vec<_> = translator.metadata.constraints_per_t.clone().into_iter().collect();
    constraints_per_t.sort();
    let mut recons_per_t: Vec<_> = translator.metadata.recons_per_t.clone().into_iter().collect();
    recons_per_t.sort();

    // --lazy-pairs at t = 2: lama_attack takes the pair round's databases and skips the final solve
    // (Translator::final_databases); the table path still runs the final Solver.
    let responses = match translator.final_databases() {
        Some(dbs) => {
            assert!(lazy && max_t == 2, "final_databases is only for the lazy pair path at t = 2");
            dbs
        }
        None => {
            let mut solver = Solver::new(translator.get_var_index_map());
            let mut model = translator.get_proto_model();
            let responses = solver.solve(&mut model, largest_enc_val, false);
            assert!(solver.solution_stat == Optimal || solver.solution_stat == Feasible, "no databases found");
            responses
        }
    };
    let num_dbs = responses.values().next().map_or(0, Vec::len);
    let mut databases: Vec<Vec<(i64, i64)>> = (0..num_dbs)
        .map(|d| {
            let mut db: Vec<(i64, i64)> = responses.iter().map(|(enc, vals)| (*enc, vals[d])).collect();
            db.sort();
            db
        })
        .collect();
    databases.sort();

    PerfectRun { tables, last_solutions, constraints_per_t, recons_per_t, databases }
}

/// The lazy pair path (bits + propagator, lazy_pairs.rs) must give EXACTLY what the materialised
/// t = 2 tables give: same solutions, same bookkeeping, same final databases. At t = 2 the lazy path
/// skips the table rebuild and the final solve (its databases ARE the output), so its t = 2 tables are
/// not compared; the table path's final databases (after the final solve) must equal them. Exact frequencies and noise-sized eps; uniform (8 symmetric images), gaussian
/// and beta; t = 2 only (what the runs use).
#[test]
fn perfect_lazy_pairs_match_tables() {
    let _ = env_logger::builder().is_test(true).filter_level(log::LevelFilter::Info).try_init();
    // (dist, grid side, density %, eps, max t)
    let cases = [
        ("uniform", 6, 35, 0.0, 2),
        ("gaussian", 6, 35, 0.0, 2),
        ("uniform", 6, 35, 0.004, 2),
        ("gaussian", 6, 35, 0.004, 2),
        ("beta", 6, 35, 0.004, 2),
        ("uniform", 7, 30, 0.002, 2),
        ("gaussian", 8, 25, 0.002, 2),
        // wide eps: many candidates per record and many databases (the noisy 50x50 regime)
        ("uniform", 6, 35, 0.01, 2),
        ("gaussian", 6, 35, 0.01, 2),
        ("beta", 6, 35, 0.01, 2),
        ("gaussian", 7, 30, 0.006, 2),
        ("uniform", 10, 12, 0.002, 2),
        ("uniform", 6, 35, 0.03, 2),  // 24 databases, 4464 pair rows
        ("gaussian", 6, 35, 0.03, 2), // 18 databases, non-symmetric
    ];
    for (dist, side, density, eps, max_t) in cases {
        let loaded_db: Box<dyn Searchable + Sync> = Box::new(testDB::new(2, side, density));
        let selector = Selector::new(dist, &loaded_db);
        let tables = perfect_run(&loaded_db, &selector, eps, max_t, false);
        let lazy = perfect_run(&loaded_db, &selector, eps, max_t, true);
        // the table path must agree with itself too (guards the test against nondeterminism)
        let tables_again = perfect_run(&loaded_db, &selector, eps, max_t, false);
        assert_eq!(tables, tables_again, "table path not deterministic: {dist} {side}x{side} eps {eps}");
        info!(
            "{dist} {side}x{side} {density}% eps {eps} t<={max_t}: {} databases, t=2 rows {:?}",
            tables.databases.len(),
            tables.constraints_per_t
        );
        assert!(!tables.databases.is_empty());
        let ctx = format!("{dist} {side}x{side} eps {eps}");
        assert_eq!(tables.last_solutions, lazy.last_solutions, "lazy round solutions differ: {ctx}");
        assert_eq!(tables.constraints_per_t, lazy.constraints_per_t, "lazy bookkeeping differs: {ctx}");
        assert_eq!(tables.recons_per_t, lazy.recons_per_t, "lazy bookkeeping differs: {ctx}");
        assert_eq!(tables.databases, lazy.databases, "final databases differ (final solve skipped): {ctx}");
        // and the skipped final solve really changed nothing: final databases = the round's solutions
        assert_eq!(tables.databases.len(), tables.last_solutions.len(), "final solve changed the set: {ctx}");
    }
}


/// eps of the perfect-setting noise runs, exactly as lama_attack computes it (mod.rs: max over every MBQ
/// of |P_true'(enclose) - P_attacker(enclose)|, true' = noisy, attacker = noiseless), for each TV.
fn noise_eps(loaded_db: &Box<dyn Searchable + Sync>, dist: &str, tv: f64) -> f64 {
    let selector = Selector::new_with_noise(dist, loaded_db, tv);
    let attacker = QueryDistribution::new(selector.query_distribution.pairs.clone(), loaded_db, dist.parse().unwrap());
    selector
        .query_distribution
        .mbq_to_cumulative_prob
        .iter()
        .map(|(mbq, p)| (p - attacker.cumulative_prob_lookup(mbq)).abs())
        .fold(0.0, f64::max)
}

/// Debug report + checks for the TV-noise sweep. Prints eps at every TV per distribution
/// (`cargo test --release --test integration noise_eps_per_tv -- --nocapture`) and checks:
/// - deterministic: building the distributions again gives the SAME eps (needs the sorted query list,
///   selector.rs get_dom_pairs; before, gaussian / beta changed every run),
/// - eps = lambda * max|P_q - P_p| with lambda = TV / TV(p, q), i.e. eps / TV is the same at every TV
///   (this is what makes the output databases nested in TV),
/// - eps <= TV (an enclosing probability can't move by more than the total variation).
/// NOISE_EPS_SIDE=<n> sets the synthetic grid side (default 8; 15 is still quick).
#[test]
fn noise_eps_per_tv() {
    let side: usize = std::env::var("NOISE_EPS_SIDE").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let loaded_db: Box<dyn Searchable + Sync> = Box::new(testDB::new(2, side, 35));
    let tvs = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05];
    println!("{side}x{side} synthetic grid, 35% density");
    println!("{:<9} {:>7} {:>12} {:>10}", "dist", "TV", "eps", "eps/TV");
    for dist in ["uniform", "gaussian", "beta"] {
        let eps: Vec<f64> = tvs.iter().map(|&tv| noise_eps(&loaded_db, dist, tv)).collect();
        let again: Vec<f64> = tvs.iter().map(|&tv| noise_eps(&loaded_db, dist, tv)).collect();
        for (tv, e) in tvs.iter().zip(&eps) {
            println!("{dist:<9} {tv:>7} {e:>12.6e} {:>10.6}", e / tv);
        }
        assert_eq!(eps, again, "{dist}: eps differs between two identical builds (query order not fixed?)");
        let ratio = eps[0] / tvs[0];
        for (tv, e) in tvs.iter().zip(&eps) {
            assert!(*e <= tv + 1e-12, "{dist}: eps {e} > TV {tv}");
            assert!((e / tv - ratio).abs() <= 1e-9 * ratio.max(1e-300), "{dist}: eps/TV not constant ({} vs {ratio} at TV {tv})", e / tv);
        }
    }
}
