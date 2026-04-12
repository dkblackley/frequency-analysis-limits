use clap::Parser;
use frequency_analysis_limits::plotting::plot::{Plotter, ReconstructionData2dPoint};
use frequency_analysis_limits::plotting::post::{
    export_to_geojson, process_and_map_points, procrustes_align,
};
use frequency_analysis_limits::LAMA::lama_attack;
use log::{debug, info};
use std::fs;
use std::fs::File;
use std::io::{BufReader, Write};

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
            &args.save,
            &args.eps,
            &args.delta,
        );
    }

    if args.post {
        let dir = &args.dir_path;
        let remin_path = format!("{dir}{0}/remin", args.name,);
        let less_path = format!("{dir}{0}/even_less", args.name,);
        let unique_name = format!("{0}_prob{1}.0_{2}", args.name, args.percent, args.dist);

        let unique_rem_name = format!("{unique_name}_classic.json");
        let unique_less_name = format!("{unique_name}_even_less.json");

        let remin_res = format!("{remin_path}/{unique_rem_name}");
        let even_less_res = format!("{less_path}/{unique_less_name}");

        debug!("About to load data {remin_res}, {even_less_res}");

        let content = fs::read_to_string(&remin_res).unwrap();
        let remin_data: Vec<ReconstructionData2dPoint> = serde_json::from_str(&content).unwrap();

        let content = fs::read_to_string(&even_less_res).unwrap();
        let less_data: Vec<ReconstructionData2dPoint> = serde_json::from_str(&content).unwrap();

        let aligned = procrustes_align(&*remin_data, true, true, true);
        info!("MSE of Remin (Original): {:?}", aligned.1);

        let aligned = procrustes_align(&*less_data, true, true, true);
        info!("MSE of Even Less (Original): {:?}", aligned.1);

        if args.name == "spitz" {
            let spitz_orig =
                "/home/yelnat/Nextcloud/10TB-STHDD/datasets/freq_an/graw_drawing".to_string();
            let lat_long_truth = process_and_map_points(
                &format!("{spitz_orig}/metadata.json"),
                &format!("{spitz_orig}/Spitz.csv"),
                remin_data.clone(),
            )
            .unwrap();

            print_min_val(
                &lat_long_truth
                    .iter()
                    .map(|point| point.true_points)
                    .collect(),
            );

            let aligned = procrustes_align(&*lat_long_truth, true, true, true);

            info!("MSE of Remin (On map): {:?}", aligned.1);

            let out_path = format!("{remin_path}/{unique_rem_name}.geojson");
            let recon_points_vec: Vec<(f64, f64)> = aligned
                .0
                .iter()
                .map(|point| point.reconstructed_points)
                .collect();

            export_to_geojson(recon_points_vec, &out_path).unwrap();

            let true_points: Vec<(f64, f64)> =
                aligned.0.iter().map(|point| point.true_points).collect();
            let true_path = format!("{dir}{0}/true.geojson", args.name,);
            export_to_geojson(true_points, &true_path).unwrap();

            let lat_long_truth = process_and_map_points(
                &format!("{spitz_orig}/metadata.json"),
                &format!("{spitz_orig}/Spitz.csv"),
                less_data,
            )
            .unwrap();

            let aligned = procrustes_align(&*lat_long_truth, true, true, true);

            info!("MSE of even less (On map): {:?}", aligned.1);

            let out_path = format!("{less_path}/{unique_less_name}.geojson");

            let recon_points_vec: Vec<(f64, f64)> = aligned
                .0
                .iter()
                .map(|point| point.reconstructed_points)
                .collect();
            export_to_geojson(recon_points_vec, &out_path).unwrap();

            info!("Saved lili to geojson");
        }

        let out = procrustes_align(&*remin_data, true, true, true);
        let new_vec = out.0;
        let _mse = out.1;

        let mut output_file =
            File::create(format!("{0}{1}/reconstruction.json", args.dir_path, args.name).as_str())
                .unwrap();
        let procrus = serde_json::to_string_pretty(&new_vec).unwrap();
        output_file.write_all(procrus.as_bytes()).unwrap();

        quick_convert(
            format!("{0}{1}/reconstruction.json", args.dir_path, args.name).as_str(),
            format!(
                "{0}{1}/reconstruction_{2}.geojson",
                args.dir_path, args.name, args.name
            )
            .as_str(),
        )
    }

    if args.plot {
        info!("Plotting data");

        //plotter.handle_spatial_plot(&[format!("{dir}/{name}")].clone(), true);

        //let mut dir_paths = Vec::new();
        let _plotter = Plotter {
            x_padder: 0.2,
            y_padder: 0.2,
        };

        //plotter.handle_spatial_plot(&[format!("{dir}/{}", args.name)].clone(), true);

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

    info!("Moriturus te saluto");
}

fn quick_convert(file_path: &str, out_path: &str) {
    // Pre-allocate the vectors using the length of the hashmap to avoid reallocations
    let file = File::open(file_path).unwrap();
    let reader = BufReader::new(file);
    let data: Vec<ReconstructionData2dPoint> = serde_json::from_reader(reader).unwrap();

    let recon_points_vec: Vec<(f64, f64)> = data
        .iter()
        .map(|point| point.reconstructed_points)
        .collect();

    export_to_geojson(recon_points_vec, out_path).unwrap();
}

fn print_min_val(points: &Vec<(f64, f64)>) {
    let (min_x, max_x, min_y, max_y) = points.iter().fold(
        (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ),
        |(min_x, max_x, min_y, max_y), &(x, y)| {
            (min_x.min(x), max_x.max(x), min_y.min(y), max_y.max(y))
        },
    );

    println!("X: min={}, max={}", min_x, max_x);
    println!("Y: min={}, max={}", min_y, max_y);
}
