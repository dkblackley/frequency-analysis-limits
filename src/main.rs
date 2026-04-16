use clap::Parser;
use frequency_analysis_limits::plotting::post::{
    export_to_geo_and_align, export_to_geojson, process_and_map_points, procrustes_align,
};
use frequency_analysis_limits::plotting::{do_plotting, ReconstructionDataPoint};
use frequency_analysis_limits::LAMA::lama_attack;
use log::{debug, info};
use std::fs;
use std::fs::File;
use std::io::{BufReader, Write};

// Helps rayon when calling malloc
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Parser, Debug, Clone)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// The path to the directory with files you want to load
    #[arg(short, long)]
    dir_path: String,
    /// If you want to save/checkpoint progress
    #[arg(short, long)]
    save: bool,

    /// Load checkpoints
    #[arg(short, long)]
    load: bool,

    /// Run the LAMA attack or just do plotting
    #[arg(long)]
    skip_lama: bool,

    /// The identifier for the specific function to load the file
    #[arg(short, long)]
    name: String,

    // The identifier for the specific function to load the file
    #[arg(long)]
    t: u64,

    /// Make plots and save to disk
    #[arg(long)]
    plot: bool,

    // TODO: Maybe remove this? Or have it override eps, delta bounds/reverse engineer eps/delta
    #[arg(long, default_value = "100.0")]
    percent: f64,

    #[arg(long, default_value = "5")]
    dim: usize,

    #[arg(long, default_value = "uniform")]
    dist: String,

    #[arg(long)]
    post: bool,

    // 0.0 means full query dist
    #[arg(long, default_value = "0.0")]
    eps: f64,

    #[arg(long, default_value = "0.9")]
    delta: f64,
}

fn main() {
    let args = Args::parse();

    env_logger::builder()
        .is_test(false)
        .filter_level(log::LevelFilter::Debug)
        .try_init()
        .expect("Logger failed to init!");

    debug!("Args handed in: {:?}", args);

    debug!(
        "Rayon is operating with {} threads.",
        rayon::current_num_threads()
    );

    if !args.skip_lama {
        lama_attack(
            &args.name,
            &args.dir_path,
            &args.dist,
            &args.t,
            &args.dim,
            &args.save,
            &args.eps,
            &args.delta,
        );
    }

    // Do post-processing - export to GeoJSON and align
    if args.post {
        let dir = &args.dir_path;
        let remin_path = format!("{dir}/{0}/remin", args.name,);
        let less_path = format!("{dir}/{0}/even_less", args.name,);
        let unique_name = format!("{0}_prob{1}.0_{2}", args.name, args.percent, args.dist);

        export_to_geo_and_align(
            dir,
            &remin_path,
            &less_path,
            &unique_name,
            &args.dir_path,
            &args.name,
        );
    }

    if args.plot {
        info!("Plotting data");
        do_plotting();
    }

    info!("Moriturus te saluto");
}
