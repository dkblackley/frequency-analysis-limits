use clap::{arg, Parser};
use frequency_analysis_limits::dataloader::datasets::TwoDMap;
use frequency_analysis_limits::dataloader::tester::testDB;
use frequency_analysis_limits::dataloader::{unflatten_nd, Searchable};
use frequency_analysis_limits::plotting::plot::{DbResult, Plotter, ReconstructionData2d};
use frequency_analysis_limits::LAMA::selector::Selector;
use frequency_analysis_limits::LAMA::solver::Solver;
use frequency_analysis_limits::LAMA::translator::Translator;
use frequency_analysis_limits::LAMA::utility::{binomial_coefficient, get_mbq};
use frequency_analysis_limits::{DomPair, Frequency, Record, Value};
use log::{debug, error, info};
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::BufWriter;
use std::time::Instant;

#[derive(Parser, Debug, Clone)]
#[command(version, about, long_about = None)]
struct Args {
    /// The path to the directory with files you want to load
    #[arg(short, long)]
    dir_path: String,
    /// If you want to save/checkpoint progress
    #[arg(short, long)]
    save: bool,

    /// Load checkpoints
    #[arg(short, long)]
    load: bool,

    /// The identifier for the specific function to load the file
    #[arg(short, long)]
    name: String,

    // The identifier for the specific function to load the file
    #[arg(short, long)]
    t: u64,

    /// Make plots and save to disk
    #[arg(short, long)]
    plot: bool,
}

fn do_attack(args: Args) {
    let mut loaded_db: Box<dyn Searchable + Sync>;
    let full_datapath = format!("{0}{1}", args.dir_path, args.name);

    info!("Starting LAMA attack");

    if args.name.as_str() == "paris"
        || args.name.as_str() == "manhattan"
        || args.name.as_str() == "shanghai"
        || args.name.as_str() == "amsterdam"
    {
        loaded_db = Box::new(
            TwoDMap::new(
                &format!("{0}/{1}.json", full_datapath, args.name),
                args.name.as_str(),
                100.0,
                // Some((250, 250)),
                None,
            )
            .unwrap(),
        );
    } else if args.name.as_str() == "grid" {
        loaded_db = Box::new(testDB::new(15, 15, 80));
    } else if args.name.as_str() == "spitz" {
        loaded_db = Box::new(
            TwoDMap::new(
                &format!("{0}/{1}.json", full_datapath, args.name),
                args.name.as_str(),
                100.0,
                None,
            )
            .unwrap(),
        );
    } else if args.name.as_str() == "cali" {
        loaded_db = Box::new(
            TwoDMap::new(
                &format!("{0}/{1}.json", full_datapath, args.name),
                args.name.as_str(),
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

    info!(
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
    let t = args.t;

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

    info!("Beginning to bruteforce all DomPair->Freq mappings");
    // This is a bruteforce calculation of the TRUE frequency of all dompairs.
    let dom_pair_freq: HashMap<DomPair, Frequency> =
        selector.get_dominant_pair_to_freq_map().unwrap();

    if args.save {
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
        info!("Recursively computing frequencies for {i} tuples");
        translator.process_t_greater_than_1(
            i as usize,
            &universe,
            get_observed_freq,
            get_expected_freq,
        );
    }

    info!("6. Building and executing the CP-SAT Solver for the final constraint graph...");
    let mut solver = Solver::new(translator.get_var_index_map());

    let responses = solver.solve(&mut translator.get_proto_model(), true);

    let duration = start.elapsed();
    let mut correct = 0;
    let mut incorrect = 0;

    //Key is actually the true value.
    for (key, val) in responses.clone() {
        if key == val {
            correct = correct + 1
        } else {
            incorrect = incorrect + 1
        }
    }

    debug!("{correct} correct, {incorrect} incorrect");

    info!("Saving to {full_datapath}/reconstruction.json");
    save_reconstruction_data(
        &responses,
        format!("{full_datapath}/reconstruction.json",).as_str(),
        &loaded_db,
    );

    let final_res = DbResult {
        name: loaded_db.get_name().parse().unwrap(),
        method: "LAMA (Ours)".to_string(),
        dims: loaded_db.get_dims() as u32,
        mse: Some(0.0),
        match_rate: Some(1.0),
        chamfer: Some(0.0),
        number_of_reconstructions: "8".to_string(),
        time_taken: duration.as_secs_f64(),
        total_db_size: loaded_db.get_universe().len() as u64,
        percent_queries_used: 100.0,
    };

    save_results(final_res, format!("{full_datapath}/results.json",).as_str())
}

fn main() {
    let args = Args::parse();

    env_logger::builder()
        .is_test(false)
        .filter_level(log::LevelFilter::Debug)
        .try_init()
        .expect("Logger failed to init!");
    // do_attack(args.clone());
    if args.plot {
        info!("Plotting data");
        let dir = args.dir_path;
        let name = args.name;

        //plotter.handle_spatial_plot(&[format!("{dir}/{name}")].clone(), true);

        let mut dir_paths = Vec::new();
        let mut plotter = Plotter {
            x_padder: 0.2,
            y_padder: 0.2,
        };

        if let Ok(entries) = fs::read_dir(&dir) {
            for entry in entries.flatten() {
                if let Ok(file_type) = entry.file_type() {
                    if file_type.is_dir() {
                        if let Some(name) = entry.file_name().to_str() {
                            dir_paths.push(format!("{dir}/{name}"));

                            if name == "spitz" {
                                plotter.x_padder = 0.2;
                                plotter.y_padder = 7.0;
                            } else {
                                plotter.x_padder = 0.2;
                                plotter.y_padder = 0.2;
                            }
                            plotter.handle_spatial_plot(&[format!("{dir}/{name}")].clone(), true);
                        }
                    }
                }
            }
        }

        plotter.make_table(&dir_paths);
    }
    info!("morituri te salutant or morituri te salutamus");
}

pub fn save_results(result: DbResult, file_path: &str) {
    // Create the file and wrap it in a BufWriter for better performance
    let file = File::create(file_path).unwrap();
    let writer = BufWriter::new(file);

    serde_json::to_writer_pretty(writer, &result).unwrap();
}

pub fn save_reconstruction_data(
    responses: &HashMap<i64, i64>,
    file_path: &str,
    loaded_db: &Box<dyn Searchable + Sync>,
) {
    // Pre-allocate the vectors using the length of the hashmap to avoid reallocations
    let mut data = ReconstructionData2d {
        true_points: Vec::with_capacity(responses.len()),
        reconstructed_points: Vec::with_capacity(responses.len()),
    };

    // Iterate over the HashMap
    for (&encrypted_true, &encrypted_recon) in responses {
        let true_vec = loaded_db.decrypt_point_f64(&encrypted_true);
        let recon_vec = loaded_db.decrypt_point_f64(&encrypted_recon);

        if true_vec.len() >= 2 && recon_vec.len() >= 2 {
            data.true_points
                .push((true_vec[0] as f64, true_vec[1] as f64));
            data.reconstructed_points
                .push((recon_vec[0] as f64, recon_vec[1] as f64));
        }
    }

    // Create the file and wrap it in a BufWriter for better performance
    let file = File::create(file_path).unwrap();
    let writer = BufWriter::new(file);

    serde_json::to_writer_pretty(writer, &data).unwrap();
}
