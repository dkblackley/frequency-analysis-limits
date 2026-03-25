use clap::Parser;
use frequency_analysis_limits::dataloader::datasets::CaliMap50;
use frequency_analysis_limits::dataloader::Searchable;
use frequency_analysis_limits::LAMA::selector::Selector;
use frequency_analysis_limits::LAMA::solver::Solver;
use frequency_analysis_limits::LAMA::translator::Translator;
use frequency_analysis_limits::{DomPair, Frequency, Value};
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
    env_logger::try_init().expect("Logger failed to init!");
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

    //TODO: choose these!
    let dist = "uniform";

    let dim = loaded_db.get_dims();
    let t = loaded_db.get_dims() * 2;

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

    info!("Beginning to bruteforce all DomPair->Freq mappings");
    // This is a bruteforce calculation of the TRUE frequency of all dompairs.
    let dom_pair_freq: HashMap<DomPair, Frequency> =
        selector.get_dominant_pair_to_freq_map().unwrap();

    if args.save {
        let filename = format!("{full_datapath}/dom_to_freq");
        let file_res = File::create(filename.clone());

        match file_res {
            Ok(mut file) => {
                let writer = BufWriter::new(file);

                info!("Writing dom pair to freq map to {full_datapath}");
                bincode::serialize_into(writer, &dom_pair_freq)
                    .inspect_err(|e| {
                        error!(
                            "bincode failed to write dom pair freq map to {}: {}",
                            filename, e
                        )
                    })
                    .ok();
            }
            Err(e) => {
                error!(
                    "Error occurred when writing dom pair freq map to {}: {}",
                    filename, e
                );
            }
        }
    }

    info!("Beginning to bruteforce all DomPair->Freq mappings");

    // Remember, value in this case means encoded ID, i.e a single point. We assume the 'ideal' case
    // that is: Both the true freq-plaintext and observed are the same.
    let mut freq_to_t_tuple: HashMap<(Value, Frequency), Vec<Vec<Value>>> = selector
        .get_freq_val_t_tup_dict(t as usize)
        .expect("Cannot calculate the frequency of t-tuples!");

    if args.save {
        let filename = format!("{full_datapath}/freq_to_tuple");

        let file_res = File::create(filename.clone());

        match file_res {
            Ok(mut file) => {
                let writer = BufWriter::new(file);

                info!("Writing freq to tuple map to {full_datapath}");
                bincode::serialize_into(writer, &freq_to_t_tuple)
                    .inspect_err(|e| {
                        error!(
                            "bincode failed to write freq to tuple map to {}: {}",
                            filename, e
                        )
                    })
                    .ok();
            }

            Err(e) => {
                error!(
                    "Error occurred when writing dom pair freq map to {}: {}",
                    filename, e
                );
            }
        }
    }

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

    info!("morituri te salutant or morituri te salutamus");
}
