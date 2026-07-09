use crate::dataloader::tester::testDB;
use crate::dataloader::three_d::ThreeDMap;
use crate::dataloader::two_d::TwoDMap;
use crate::dataloader::{flatten_nd, unflatten_nd, Searchable};
use crate::plotting::two_d::spatial_plot::plot_spatial_reconstruction_all_items;
use crate::plotting::{DbResult, ReconstructionDataPoint};
use crate::LAMA::error::LAMAError;
use crate::LAMA::query::QueryDistribution;
use crate::LAMA::selector::Selector;
use crate::LAMA::solver::CpSolverStatus::{Feasible, Optimal};
use crate::LAMA::solver::Solver;
use crate::LAMA::translator::Translator;
use crate::LAMA::utility::{encloses, get_mbq};
use crate::{DomPair, Frequency, Record, Value};
use log::{debug, error, info, trace, warn};
use rand::distributions::Distribution;
use rayon::iter::ParallelIterator;
use rayon::prelude::IntoParallelRefIterator;
use rustc_hash::FxHashMap;
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::BufWriter;
use std::time::Instant;

mod error;
pub mod ortools_wrap;
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
    query_percent: &f64,
) {
    let loaded_db: Box<dyn Searchable + Sync>;
    let full_datapath = format!("{0}/{1}", dir_path, db_name);

    if db_name == "grid" {
        loaded_db = Box::new(testDB::new(*dim, 6, 35));
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

    let mut translator = Translator::new(
        largest_possible_val,
        loaded_db.get_universe(),
        &(t.clone() as usize),
    );

    let query_dist_ref = &selector.query_distribution;
    let high_pair_ref = &high_pair;
    let low_pair_ref = &low_pair;

    let get_expected_prob = |plaintexts: &[i64]| -> f64 {
        let pt_records: Vec<Record> = plaintexts
            .iter()
            .map(|&v| unflatten_nd(v, high_pair_ref, low_pair_ref))
            .collect();
        let pt_mbq = get_mbq(&pt_records);

        query_dist_ref.cumulative_prob_lookup(&pt_mbq)
    };

    let eps;
    let delta = 0.1; // fix to 90% confidence
    let num_queries;

    let get_observed_prob: Box<dyn Sync + Send + Fn(&[i64]) -> f64>;

    if *query_percent != 1.0 {
        // we're going to sample.
        let (observed_queries, responses, epsil) = selector.sample_percent_responses(
            *query_percent,
            delta,
            &selector.get_all_possible_responses(),
        );
        eps = epsil;
        num_queries = observed_queries.len();

        let all_dompairs = Selector::get_dom_pairs(&loaded_db);
        info!("Making observed mapping to frequency");

        let queries_len = observed_queries.len() as f64;

        let dom_to_enclosed_freq: FxHashMap<_, _> = all_dompairs
            .par_iter() // 1. Iterate over all_dompairs in parallel
            .map(|pair| {
                // 2. Count enclosures (safe to read observed_queries across threads)
                let enclose_count = observed_queries
                    .iter()
                    .filter(|que| encloses(que, pair))
                    .count();

                let freq = enclose_count as f64 / queries_len;

                // 3. Return the key-value pair tuple
                (pair.clone(), freq)
            })
            .collect();

        // 3. Box the closure and use 'move' to take ownership of the HashMap
        get_observed_prob = Box::new(move |enc_tuple: &[i64]| -> f64 {
            let underlying_vals: Vec<Record> = enc_tuple
                .iter()
                .map(|rec| unflatten_nd(*rec, high_pair_ref, low_pair_ref))
                .collect();
            let target_mbq = get_mbq(&underlying_vals);

            dom_to_enclosed_freq
                .get(&target_mbq)
                .copied() // Cleaner than .unwrap_or(&0.0).clone()
                .unwrap_or(0.0)
        });
    } else {
        eps = 0.0;
        // How many queries can we do? A query is a hyperrectangle, The total number of hyper rectangles
        // in a space is n*(n+1)/2 where n is the length of a dim, then mutliply for each dim.
        let mut total = 1;

        for item in high_pair.clone() {
            total *= ((item + 1) * (item + 2)) / 2;
        }
        num_queries = total as usize;

        // 4. Box the else closure as well to match types
        get_observed_prob = Box::new(move |enc_tuple: &[i64]| -> f64 {
            let true_point: Vec<Record> = enc_tuple
                .iter()
                .map(|rec| unflatten_nd(*rec, high_pair_ref, low_pair_ref))
                .collect();
            let dom_pair = get_mbq(&true_point);

            query_dist_ref.cumulative_prob_lookup(&dom_pair)
        });
    }

    // if in the 'perfect' world only use things within 0.001\% of the true
    let active_eps = if eps == 0.0 { 1e-6 } else { eps };
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
    let translater_met = translator.metadata.clone();

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
        trace!("'encrypted/encoded' universe of plaintexts: {universe:?}");
        trace!("Intvars workings: {:?}", solver);
        trace!("CPModel: {:?}", model);
        debug!("Solver Status: {:?}", solver.solution_stat);
        return;
    }

    let unique_name = format!("{db_name}_{dist}_p{query_percent}");

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
        percent_queries_used: *query_percent,
        num_queries_used: num_queries as u64,
        eps: Some(eps),
        delt: Some(delta),
        translator_meta: translater_met,
    };

    if let Err(e) = save_results(
        final_res,
        &format!("{full_datapath}/limits/results_{unique_name}.json"),
    ) {
        error!("failed writing to {full_datapath}/limits/results_{unique_name}.json: {e}");
    }

    info!("LAMA finished running on {db_name}");

    let data = into_recon_data(&responses, &loaded_db);
    let mut true_cords = Vec::new();
    let mut recon_coords = Vec::new();
    let mut true_done = false;

    for item in data {
        let mut current_recon = Vec::new();

        for i in item {
            if !true_done {
                true_cords.push(i.true_points);
            } else {
                current_recon.push(i.reconstructed_points);
            }
        }
        recon_coords.push(current_recon);
        true_done = true;
    }

    let _ = plot_spatial_reconstruction_all_items(
        &true_cords,
        &recon_coords,
        "figures/debug_last_run.svg",
        true,
        0.0,
        0.0,
    );
}
fn save_results(result: DbResult, file_path: &str) -> Result<(), LAMAError> {
    // Create the file and wrap it in a BufWriter for better performance
    let file = File::create(file_path)?;
    let writer = BufWriter::new(file);

    serde_json::to_writer_pretty(writer, &result)?;
    Ok(())
}

pub fn into_recon_data(
    responses: &HashMap<i64, Vec<i64>>,
    loaded_db: &Box<dyn Searchable + Sync>,
) -> Vec<Vec<ReconstructionDataPoint>> {
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
    data
}

fn save_reconstruction_data(
    responses: &HashMap<i64, Vec<i64>>,
    file_path: &str,
    unique_name: &str,
    loaded_db: &Box<dyn Searchable + Sync>,
) {
    let data: Vec<Vec<ReconstructionDataPoint>> = into_recon_data(responses, loaded_db);

    fs::create_dir_all(file_path).unwrap();

    // Create the file and wrap it in a BufWriter for better performance
    let file = File::create(format!("{file_path}/{unique_name}_reconstruction.json")).unwrap();
    let writer = BufWriter::new(file);

    serde_json::to_writer_pretty(writer, &data).unwrap();
}


#[cfg(test)]
mod metadata_tests {
    use super::*;
    use crate::dataloader::tester::testDB;
    use crate::dataloader::three_d::ThreeDMap;
    use crate::dataloader::two_d::TwoDMap;
    use crate::dataloader::Searchable;

    // Helper function to handle math and printing for any dimension
    fn print_stats(name: &str, size_str: &str, items: usize, high: &[i64]) {
        // Calculate total possible grid cells
        let capacity: i64 = high.iter().map(|&val| val + 1).product();
        let density = (items as f64 / capacity as f64) * 100.0;

        // Calculate total queries using f64 to avoid overflow on higher dimensions
        let mut total_queries: f64 = 1.0;
        for &item in high {
            total_queries *= ((item as f64 + 1.0) * (item as f64 + 2.0)) / 2.0;
        }

        let q_10 = (total_queries * 0.10).ceil() as i64;
        let q_20 = (total_queries * 0.20).ceil() as i64;
        let q_30 = (total_queries * 0.30).ceil() as i64;

        println!(
            "{:<12} | {:<15} | {:<10} | {:>6.2}%   | {:<12} | {:<12} | {:<12} | {:<15}",
            name, size_str, items, density, q_10, q_20, q_30, total_queries as i64
        );
    }

    #[test]
    fn print_all_database_metadata() {
        println!(
            "{:<12} | {:<15} | {:<10} | {:<10} | {:<12} | {:<12} | {:<12} | {:<15}",
            "Dataset", "Grid", "Items", "Density", "10% Queries", "20% Queries", "30% Queries", "100% Queries"
        );
        println!("{:-<115}", "");

        // 1. Process 2D Maps
        let grid_sizes = ["10x10", "20x20", "30x30", "40x40", "50x50"];
        let datasets = ["spitz", "cali", "shopparis", "highway", "busstop", "drink"];

        for name in datasets {
            for size in grid_sizes {
                let path = format!("databases/{}/{}/{}.json", size, name, name);

                if std::path::Path::new(&path).exists() {
                    let locs = TwoDMap::load_array_locations_from_file(&path).unwrap();
                    let db = TwoDMap::new_unscaled(locs, name, (0, 0), (0, 0)).unwrap();

                    let items = db.get_universe().len();
                    let (_low, high) = db.get_dom_pair();

                    print_stats(name, size, items, &high);
                }
            }
        }

        // 2. Process 3D Map (nh)
        let path_3d = "databases/16x16x14/nh/nh.json";
        if std::path::Path::new(&path_3d).exists() {
            let locs = ThreeDMap::load_array_locations_from_file(path_3d).unwrap();
            let db = ThreeDMap::new_unscaled(locs, "nh").unwrap();

            let items = db.get_universe().len();
            let (_low, high) = db.get_dom_pair();

            print_stats("nh", "16x16x14", items, &high);
        }

        // 3. Process N-Dimensional Synthetic Grids
        for dim in 1..=6 {
            let size_str = vec!["6"; dim].join("x");

            // Uses your exact test parameters: dim, size_per_dim=6, density=35
            let db = testDB::new(dim, 6, 35);
            let items = db.get_universe().len();
            let (_low, high) = db.get_dom_pair();

            print_stats("grid", &size_str, items, &high);
        }

        println!("Done!");
        panic!()
    }
}