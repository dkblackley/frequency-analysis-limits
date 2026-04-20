use crate::dataloader::tester::testDB;
use crate::dataloader::three_d::ThreeDMap;
use crate::dataloader::two_d::TwoDMap;
use crate::dataloader::{flatten_nd, unflatten_nd, Searchable};
use crate::plotting::{DbResult, ReconstructionDataPoint};
use crate::LAMA::error::LAMAError;
use crate::LAMA::query::QueryDistribution;
use crate::LAMA::selector::Selector;
use crate::LAMA::solver::CpSolverStatus::{Feasible, Optimal};
use crate::LAMA::solver::Solver;
use crate::LAMA::translator::Translator;
use crate::LAMA::utility::get_mbq;
use crate::{Frequency, Record, Value};
use cp_sat::proto::CpSolverStatus;
use log::{debug, error, info, warn};
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::BufWriter;
use std::time::Instant;

mod error;
mod ortools_wrap;
pub mod query;
pub mod selector;
pub mod solver;
pub mod translator;
pub mod utility;

pub fn lama_attack(
    db_name: &String,
    dir_path: &String,
    dist: &String,
    t: &u64,
    dim: &usize,
    padding: &Value,
    _save: &bool,
    eps: &f64,
    delt: &f64,
) {
    let loaded_db: Box<dyn Searchable + Sync>;
    let full_datapath = format!("{0}/{1}", dir_path, db_name);
    let unique_name = format!("{db_name}_{dist}_e{eps}_d{delt}");

    if db_name == "grid" {
        loaded_db = Box::new(testDB::new(*dim, 5, 65));
    } else if db_name == "nh" {
        info!("Starting LAMA attack using {} dataset", db_name);
        debug!("Loading data from {full_datapath}/{db_name}.json");

        let loaded_locs =
            ThreeDMap::load_array_locations_from_file(&format!("{full_datapath}/{db_name}.json"))
                .unwrap();
        loaded_db = Box::new(ThreeDMap::new_unscaled(loaded_locs, db_name.as_str()).unwrap());
    } else {
        info!("Starting LAMA attack using {} dataset", db_name);
        debug!("Loading data from {full_datapath}/{db_name}.json");
        let loaded_locs =
            TwoDMap::load_array_locations_from_file(&format!("{full_datapath}/{db_name}.json"))
                .unwrap();

        loaded_db = Box::new(
            TwoDMap::new_unscaled(
                loaded_locs,
                db_name.as_str(),
                (*padding, *padding),
                (*padding, *padding),
            )
            .unwrap(),
        );
    }

    debug!(
        "Loaded {}, a {}-dim DB with {} records and highest/lowest records {:?}/{:?}",
        loaded_db.get_name(),
        loaded_db.get_dims(),
        loaded_db.get_universe().len(),
        loaded_db.get_dom_pair().0,
        loaded_db.get_dom_pair().1
    );

    debug!("Using eps: {}, delta: {}", eps, delt);

    let (low_pair, high_pair) = loaded_db.get_dom_pair();
    let largest_possible_val: i64 = flatten_nd(&high_pair, &high_pair, &low_pair);
    let selector = Selector::new(dist, &loaded_db);

    debug!(
        "Working with padding: {}, low_pair: {:?}, high_pair: {:?}",
        padding, low_pair, high_pair
    );
    debug!(
        "Largest possible value is {largest_possible_val}, working with {} dompairs",
        selector
            .query_distribution
            .dom_pair_to_known_raw_weight
            .len()
    );

    info!("Selector computing values");

    let mut translator = Translator::new(largest_possible_val, loaded_db.get_universe());

    let query_dist_ref = &selector.query_distribution;
    let high_pair_ref = &high_pair;
    let low_pair_ref = &low_pair;

    let get_observed_prob = |enc_tuple: &[i64]| -> f64 {
        let true_point: Vec<Record> = enc_tuple
            .iter()
            .map(|rec| unflatten_nd(*rec, high_pair_ref, low_pair_ref))
            .collect();
        let dom_pair = get_mbq(&true_point);

        // Use the native cumulative probability directly
        //query_dist_ref.get_cumulative_prob(&dom_pair)
        // QueryDistribution::compute_cumulative_prob(
        //     &dom_pair,
        //     &query_dist_ref.dist,
        //     &*query_dist_ref.lowest_rec,
        //     &*query_dist_ref.largest_rec,
        //     &query_dist_ref.dom_pair_to_known_prob,
        //     query_dist_ref.total_weight,
        // )

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
        // QueryDistribution::compute_cumulative_prob(
        //     &pt_mbq,
        //     &query_dist_ref.dist,
        //     &*query_dist_ref.lowest_rec,
        //     &*query_dist_ref.largest_rec,
        //     &query_dist_ref.dom_pair_to_known_prob,
        //     query_dist_ref.total_weight,
        // )

        query_dist_ref.cumulative_prob_lookup(&pt_mbq)
    };

    // if in the 'perfect' world only use things within 0.01\% of the true
    let active_eps = if *eps == 0.0 { 1e-5 } else { *eps };
    let validate_candidate = |enc_tuple: &[i64], proposed_plaintexts: &[i64]| -> (bool, f64) {
        let obs_prob = get_observed_prob(enc_tuple);

        let exp_prob = get_expected_prob(proposed_plaintexts);
        (
            (obs_prob - exp_prob).abs() <= active_eps,
            (obs_prob - exp_prob).abs(),
        )
    };

    let universe = loaded_db.get_universe();

    info!("Starting LAMA!");
    let start = Instant::now();
    translator.process_t1(&universe, &validate_candidate);

    for i in 2..(*t as usize + 1) {
        let start_t = Instant::now();
        info!("Recursively computing frequencies for {i} tuples");

        translator.process_t_greater_than_1(i, &universe, &validate_candidate);

        let end_t = Instant::now();
        debug!(
            "Round {i} took {} seconds",
            end_t.duration_since(start_t).as_secs()
        )
    }

    info!("Building and executing the CP-SAT Solver for the final constraint graph...");
    let mut solver = Solver::new(translator.get_var_index_map());

    let mut model = translator.get_proto_model();

    // NOTE: The only change in the test is passing largest_enc_val here
    let responses = solver.solve(&mut model, largest_possible_val, false);
    let end = Instant::now();
    if solver.solution_stat != Optimal && solver.solution_stat != Feasible {
        // minimum number of expected reconstructions
        warn!(
            "Solver did not return the expected number of reconstructions. No solutions were found."
        );

        debug!("Solver failed to reconstruct full universe!!");
        //debug!("Known frequency-to-plaintext 2-tuple mappings: {freq_to_t_tuple:?}");
        debug!("'encrypted/encoded' universe of plaintexts: {universe:?}");
        debug!("Intvars workings: {:?}", solver);
        debug!("CPModel: {:?}", model);
        debug!("Solver Status: {:?}", solver.solution_stat);
        return;
    }

    let mut correct = 0;
    let mut first_resp = HashMap::new();

    for (key, val) in responses.clone() {
        first_resp.insert(key, val[0]); // just pretend first resp is the correct one.
        for i in 0..val.len() {
            if key == val[i] {
                correct += 1;
            }
        }
    }

    debug!("{correct} correct out of {} total", responses.len());
    info!("Saving correct solutions to {full_datapath}/limits/{unique_name}_reconstruction.json");
    save_reconstruction_data(
        &responses,
        &format!("{full_datapath}/limits"),
        &unique_name,
        &loaded_db,
    );

    info!(
        "LAMa completely finished in {}",
        end.duration_since(start).as_secs_f64()
    );

    let final_res = DbResult {
        name: loaded_db.get_name().parse().unwrap(),
        method: "LAMA".to_string(),
        dims: loaded_db.get_dims() as u32,
        mse: Some(0.0),
        match_rate: Some(1.0),
        chamfer: Some(0.0),
        number_of_reconstructions: format!("{}", responses.len()),
        time_taken: end.duration_since(start).as_secs_f64(),
        total_db_size: loaded_db.get_universe().len() as u64,
        percent_queries_used: 1.0,
        num_queries_used: 1, //todo:
        eps: Some(*eps),
        delt: Some(*delt),
    };

    if let Err(e) = save_results(
        final_res,
        &format!("{full_datapath}/limits/results_{unique_name}.json"),
    ) {
        error!("failed writing to {full_datapath}/limits/results_{unique_name}.json: {e}");
    }

    info!("LAMA finished running on {db_name}");
}
fn save_results(result: DbResult, file_path: &str) -> Result<(), LAMAError> {
    // Create the file and wrap it in a BufWriter for better performance
    let file = File::create(file_path)?;
    let writer = BufWriter::new(file);

    serde_json::to_writer_pretty(writer, &result)?;
    Ok(())
}

fn save_reconstruction_data(
    responses: &HashMap<i64, Vec<i64>>,
    file_path: &str,
    unique_name: &str,
    loaded_db: &Box<dyn Searchable + Sync>,
) {
    // This will hold our "transposed" data.
    let mut data: Vec<Vec<ReconstructionDataPoint>> = Vec::new();

    // Iterate over the HashMap
    for (&encrypted_true, encrypted_recons) in responses {
        let true_vec = loaded_db.decrypt_point_f64(&encrypted_true);

        // Iterate over the reconstructions and get their index
        for (i, &encrypted_recon) in encrypted_recons.iter().enumerate() {
            let recon_vec = loaded_db.decrypt_point_f64(&encrypted_recon);

            let saved_point = ReconstructionDataPoint {
                true_points: true_vec.clone(),
                reconstructed_points: recon_vec,
                unscaled_points: None,
            };

            // If we have more reconstructions for this point than we have
            // outer arrays, we need to push a new empty Vec to hold them.
            if data.len() <= i {
                data.push(Vec::new());
            }

            // Push the saved point to the correct reconstruction index
            data[i].push(saved_point);
        }
    }

    fs::create_dir_all(file_path).unwrap();

    // Create the file and wrap it in a BufWriter for better performance
    let file = File::create(format!("{file_path}/{unique_name}_reconstruction.json")).unwrap();
    let writer = BufWriter::new(file);

    serde_json::to_writer_pretty(writer, &data).unwrap();
}
