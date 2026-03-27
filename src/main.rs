use clap::Parser;
use frequency_analysis_limits::dataloader::datasets::CaliMap50;
use frequency_analysis_limits::dataloader::{unflatten_nd, Searchable};
use frequency_analysis_limits::LAMA::selector::Selector;
use frequency_analysis_limits::LAMA::solver::Solver;
use frequency_analysis_limits::LAMA::translator::Translator;
use frequency_analysis_limits::LAMA::utility::get_mbq;
use frequency_analysis_limits::{DomPair, Frequency, Record, Value};
use log::{error, info};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufWriter;

#[derive(Parser, Debug)]
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
}

fn main() {
    let args = Args::parse();
    let loaded_db: Box<dyn Searchable + Sync>;
    env_logger::builder()
        .is_test(false)
        .filter_level(log::LevelFilter::Debug)
        .try_init()
        .expect("Logger failed to init!");
    let full_datapath = format!("{0}{1}", args.dir_path, args.name);

    match args.name.as_str() {
        "cali_50" => {
            loaded_db = Box::new(
                CaliMap50::new(&format!("{0}/{1}.json", full_datapath, args.name)).unwrap(),
            );
        }
        _ => {
            error!("Unknown dataset name: {0}", args.name);
            std::process::exit(1);
        }
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
    // let t = loaded_db.get_dims() * 2;
    let t = 3;

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

    //     match file_res {
    //         Ok(mut file) => {
    //             let writer = BufWriter::new(file);
    //
    //             info!("Writing dom pair to freq map to {full_datapath}");
    //             bincode::serialize_into(writer, &dom_pair_freq)
    //                 .inspect_err(|e| {
    //                     error!(
    //                         "bincode failed to write dom pair freq map to {}: {}",
    //                         filename, e
    //                     )
    //                 })
    //                 .ok();
    //         }
    //         Err(e) => {
    //             error!(
    //                 "Error occurred when writing dom pair freq map to {}: {}",
    //                 filename, e
    //             );
    //         }
    //     }
    // }

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

    for i in 2..t {
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

    info!("morituri te salutant or morituri te salutamus");
}
