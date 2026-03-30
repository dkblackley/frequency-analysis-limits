use clap::{arg, Parser};
use frequency_analysis_limits::dataloader::Searchable;
use frequency_analysis_limits::plotting::plot::{DbResult, Plotter};
use frequency_analysis_limits::plotting::post::export_to_geojson;
use frequency_analysis_limits::LAMA::lama_attack;
use log::info;
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::BufWriter;

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
    #[arg(short, long)]
    t: u64,

    /// Make plots and save to disk
    #[arg(short, long)]
    plot: bool,
}

fn main() {
    let args = Args::parse();

    env_logger::builder()
        .is_test(false)
        .filter_level(log::LevelFilter::Debug)
        .try_init()
        .expect("Logger failed to init!");

    if !args.skip_lama {
        lama_attack(&args.name, &args.dir_path, &args.t, &args.save);
    }
    if args.plot {
        info!("Plotting data");
        let dir = args.dir_path;

        //plotter.handle_spatial_plot(&[format!("{dir}/{name}")].clone(), true);

        //let mut dir_paths = Vec::new();
        let mut plotter = Plotter {
            x_padder: 0.2,
            y_padder: 0.2,
        };

        plotter.handle_spatial_plot(&[format!("{dir}/{}", args.name)].clone(), true);

        // if let Ok(entries) = fs::read_dir(&dir) {
        //     for entry in entries.flatten() {
        //         if let Ok(file_type) = entry.file_type() {
        //             if file_type.is_dir() {
        //                 if let Some(name) = entry.file_name().to_str() {
        //                     dir_paths.push(format!("{dir}/{name}"));
        //
        //                     if name == "spitz" {
        //                         plotter.x_padder = 0.2;
        //                         plotter.y_padder = 7.0;
        //                     } else {
        //                         plotter.x_padder = 0.2;
        //                         plotter.y_padder = 0.2;
        //                     }
        //                     plotter.handle_spatial_plot(&[format!("{dir}/{name}")].clone(), true);
        //                 }
        //             }
        //         }
        //     }
        // }
        //
        // plotter.make_table(&dir_paths);
    }
}
