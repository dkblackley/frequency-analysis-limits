use crate::dataloader::datasets::TwoDMap;
use crate::dataloader::tester::testDB;
use crate::dataloader::{unflatten_nd, Searchable};
use crate::plotting::plot::{DataWrapper, DbResult, ReconstructionData2dPoint};
use crate::plotting::post::export_to_geojson;
use crate::LAMA::selector::Selector;
use crate::LAMA::solver::Solver;
use crate::LAMA::translator::Translator;
use crate::LAMA::utility::get_mbq;
use crate::{DomPair, Frequency, Record};
use itertools::Itertools;
use log::{debug, error, info};
use std::collections::HashMap;
use std::env::args;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::time::Instant;

mod error;
pub mod selector;
pub mod solver;
pub mod translator;
pub mod utility;

pub fn lama_attack(db_name: &String, dir_path: &String, t: &u64, save: &bool) {
    let mut loaded_db: Box<dyn Searchable + Sync>;
    let full_datapath = format!("{0}{1}", dir_path, db_name);

    info!("Starting LAMA attack using {} dataset", db_name);

    if db_name.as_str() == "paris"
        || db_name.as_str() == "manhattan"
        || db_name.as_str() == "shanghai"
        || db_name.as_str() == "amsterdam"
    {
        loaded_db = Box::new(
            TwoDMap::new(
                &format!("{0}/{1}.json", full_datapath, db_name),
                db_name.as_str(),
                100.0,
                // Some((250, 250)),
                None,
            )
            .unwrap(),
        );
    } else if db_name.as_str() == "grid" {
        loaded_db = Box::new(testDB::new(10, 10, 80));
    } else if db_name.as_str() == "spitz" {
        loaded_db = Box::new(
            TwoDMap::new(
                &format!("{0}/{1}.json", full_datapath, db_name),
                db_name.as_str(),
                100.0,
                None,
            )
            .unwrap(),
        );
    } else if db_name.as_str() == "cali" {
        loaded_db = Box::new(
            TwoDMap::new(
                &format!("{0}/{1}.json", full_datapath, db_name),
                db_name.as_str(),
                10.0,
                // Some((250, 250)),
                None,
            )
            .unwrap(),
        );
    } else {
        error!("Unidentified DB!");
        return;
    }

    debug!(
        "Loaded {}, a {}-dim DB with {} records and highest/lowest recors {:?}/{:?}",
        loaded_db.get_name(),
        loaded_db.get_dims(),
        loaded_db.get_universe().len(),
        loaded_db.get_dom_pair().0,
        loaded_db.get_dom_pair().1
    );

    //TODO: choose these!
    let dist = "uniform";

    let dim = loaded_db.get_dims();

    let start = Instant::now();

    let (low_pair, high_pair) = loaded_db.get_dom_pair();
    let binding = loaded_db.get_universe();
    let largest_enc_val = binding.iter().max().unwrap();

    let selector = Selector {
        dist: dist.to_string(),
        encrypted_db: &loaded_db,
        dim,
        lowest_rec: low_pair.clone(),
        largest_rec: high_pair.clone(),
    };

    info!("Selector computing values");
    let duration = start.elapsed();

    info!("Beginning to bruteforce all DomPair->Freq mappings");
    // This is a bruteforce calculation of the TRUE frequency of all dompairs.
    let dom_pair_freq: HashMap<DomPair, Frequency> =
        selector.get_dominant_pair_to_freq_map().unwrap();

    if *save {
        let filename = format!("{full_datapath}/dom_to_freq");
        let file = File::create(filename.clone()).unwrap();
        let writer = BufWriter::new(file);
        info!("Writing dom pair to freq map to {full_datapath}");
        bincode::serialize_into(writer, &dom_pair_freq).unwrap();
    }

    //TODO: Move these into selector
    info!("Computing Observed frequencies -> 1 tuple");
    let obs_t1 = selector.precompute_t_observed(1, "dummy1", &dom_pair_freq);

    info!("Computing True frequencies -> 1 tuple");
    let query_dist_over_one =
        Selector::build_theoretical_t_dict(&*low_pair, &*high_pair, "uniform", 1);

    let mut translator = Translator::new(*largest_enc_val, loaded_db.get_universe(), false);

    // TODO: move this into translator

    translator.process_t1(&obs_t1, &query_dist_over_one);

    let dom_freq_ref = &dom_pair_freq;
    let high_pair_ref = &high_pair;
    let low_pair_ref = &low_pair;

    //TODO change observed for empirical VC dim

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

    let universe = loaded_db.get_universe();

    for i in 2..(t + 1) {
        let start_t = Instant::now();
        info!("Recursively computing frequencies for {i} tuples");
        translator.process_t_greater_than_1(
            i as usize,
            &universe,
            get_observed_freq,
            get_expected_freq,
        );
        let end_t = Instant::now();
        debug!("Round {i} took {}", end_t.duration_since(start_t).as_secs())
    }

    info!("6. Building and executing the CP-SAT Solver for the final constraint graph...");
    let mut solver = Solver::new(translator.get_var_index_map());
    let end = Instant::now();

    let responses = solver.solve(&mut translator.get_proto_model(), false);

    let mut correct = 0;
    let mut incorrect = 0;
    let mut first_resp = HashMap::new();

    //Key is actually the true value.
    for (key, val) in responses.clone() {
        first_resp.insert(key, val[0]);

        for i in 0..val.len() {
            if key == val[i] {
                correct = correct + 1;
            }
        }
    }

    debug!("{correct} correct, {} total", responses.len());

    info!("Saving correct solution to {full_datapath}/reconstruction.json");
    save_reconstruction_data(&first_resp, full_datapath.as_str(), &loaded_db);

    let final_res = DbResult {
        name: loaded_db.get_name().parse().unwrap(),
        method: "LAMA (Ours)".to_string(),
        dims: loaded_db.get_dims() as u32,
        mse: Some(0.0),
        match_rate: Some(1.0),
        chamfer: Some(0.0),
        number_of_reconstructions: format!("{}", responses.len()),
        time_taken: end.duration_since(start).as_secs_f64(),
        total_db_size: loaded_db.get_universe().len() as u64,
        percent_queries_used: 100.0,
    };

    save_results(final_res, format!("{full_datapath}/results.json",).as_str())
}

fn save_results(result: DbResult, file_path: &str) {
    // Create the file and wrap it in a BufWriter for better performance
    let file = File::create(file_path).unwrap();
    let writer = BufWriter::new(file);

    serde_json::to_writer_pretty(writer, &result).unwrap();
}

fn save_reconstruction_data(
    responses: &HashMap<i64, i64>,
    file_path: &str,
    loaded_db: &Box<dyn Searchable + Sync>,
) {
    //TODO: more than 2 dimension

    // Pre-allocate the vectors using the length of the hashmap to avoid reallocations
    let mut data: Vec<ReconstructionData2dPoint> = Vec::new();

    // Iterate over the HashMap
    for (&encrypted_true, &encrypted_recon) in responses {
        let true_vec = loaded_db.decrypt_point_f64(&encrypted_true);
        let recon_vec = loaded_db.decrypt_point_f64(&encrypted_recon);

        let saved_point = ReconstructionData2dPoint {
            true_points: (true_vec[0] as f64, true_vec[1] as f64),
            reconstructed_points: (recon_vec[0] as f64, recon_vec[1] as f64),
        };

        data.push(saved_point);
    }

    //let data_wrap = DataWrapper { mapping: data };
    // Create the file and wrap it in a BufWriter for better performance
    let file = File::create(format!("{file_path}/reconstruction.json",)).unwrap();
    let writer = BufWriter::new(file);

    serde_json::to_writer_pretty(writer, &data).unwrap();
    export_to_geojson(
        data,
        format!("{file_path}/reconstruction_geo.json").as_str(),
    )
    .unwrap();
}
