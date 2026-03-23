use clap::Parser;
use frequency_analysis_limits::dataloader::datasets::{
    map_to_locations, save_locations_to_file, scale_points,
};
use frequency_analysis_limits::dataloader::Searchable;
use frequency_analysis_limits::CALI_ALL;
use log::{error, info};

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
    let loaded_db: &(dyn Searchable + Sync);
    env_logger::try_init().expect("Logger failed to init!");

    match args.name.as_str() {
        "hello" => {
            println!("Matched 'hello'")
        }
        "cali" => {}
        "cali_50" => {}
        _ => {
            error!("Unknown dataset name: {0}", args.name);
            std::process::exit(1);
        }
    }

    info!("morituri te salutant or morituri te salutamus")
}
