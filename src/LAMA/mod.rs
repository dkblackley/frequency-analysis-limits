use crate::dataloader::datasets::TwoDMap;
use crate::dataloader::tester::testDB;
use crate::dataloader::{unflatten_nd, Searchable};
use crate::plotting::{DbResult, ReconstructionDataPoint};
use crate::LAMA::error::LAMAError;
use crate::LAMA::selector::Selector;
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
mod query;
pub mod selector;
pub mod solver;
pub mod translator;
pub mod utility;

pub fn lama_attack(
    db_name: &String,
    dir_path: &String,
    dist: &String,
    t: &u64,
    _save: &bool,
    eps: &f64,
    delt: &f64,
) {
    let loaded_db: Box<dyn Searchable + Sync>;
    let full_datapath = format!("{0}/{1}", dir_path, db_name);
    let unique_name = format!("{db_name}_{dist}_e{eps}_d{delt}");

    if db_name == "grid" {
        loaded_db = Box::new(testDB::new(15, 15, 70))
    } else if db_name == "nh" {
        todo!()
    } else {
        info!("Starting LAMA attack using {} dataset", db_name);
        debug!("Loading data from {full_datapath}/{db_name}.json");
        let loaded_locs =
            TwoDMap::load_array_locations_from_file(&format!("{full_datapath}/{db_name}.json"))
                .unwrap();

        loaded_db = Box::new(TwoDMap::new_unscaled(loaded_locs, db_name.as_str()).unwrap());
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

    let start = Instant::now();
    let (low_pair, high_pair) = loaded_db.get_dom_pair();
    let binding = loaded_db.get_universe();
    let largest_enc_val = binding.iter().max().unwrap();

    let selector = Selector::new(dist, &loaded_db, *eps, *delt);

    info!("Selector computing values");

    let _obs_t1: HashMap<u64, Vec<Vec<i64>>>;

    info!("Computing Observed frequencies -> 1 tuple");
    if *eps == 0.0 {
        //obs_t1 = selector.precompute_perfect_t_observed(1);
    } else {
        // empirical VC/calc required samples for this eps/delta
        todo!();
    }

    info!("Computing True Probabilities -> 1 tuple");
    let _query_dist_over_one = selector.build_theoretical_t_dict(1);

    let mut translator = Translator::new(*largest_enc_val, loaded_db.get_universe());

    let query_dist_ref = &selector.query_distribution;
    let high_pair_ref = &high_pair;
    let low_pair_ref = &low_pair;

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
    let active_eps = if *eps == 0.0 { 1e-3 } else { *eps };
    let validate_candidate = |enc_tuple: &[i64], proposed_plaintexts: &[i64]| -> bool {
        let obs_prob = get_observed_prob(enc_tuple);
        if obs_prob == 0.0 {
            return false;
        }

        let exp_prob = get_expected_prob(proposed_plaintexts);
        (obs_prob - exp_prob).abs() <= active_eps
    };

    let universe = loaded_db.get_universe();
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
    let end = Instant::now();

    let mut model = translator.get_proto_model();
    let responses = solver.solve(&mut model, false);

    if solver.solution_stat != CpSolverStatus::Optimal
        && solver.solution_stat != CpSolverStatus::Feasible
    {
        // minimum number of expected reconstructions
        let _freq_to_t_tuple: HashMap<(Value, Frequency), Vec<Vec<Value>>> =
            selector.get_freq_val_t_tup_dict(1).unwrap();
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
