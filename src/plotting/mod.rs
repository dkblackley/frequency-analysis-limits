use crate::plotting::two_d::mse_by_all_reconstructions::plot_histograms_of_all_reconstructions;
use crate::plotting::two_d::mse_vs_grid_size::plot_grid_by_mse;
use crate::plotting::two_d::spatial_plot::run_spatial_plots;
use crate::plotting::two_d::worst_case_convex_hull::do_convex_hull_plots;
use log::warn;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs;

mod metrics;
pub mod post;
pub mod tables;
pub mod two_d;

#[derive(Debug, Deserialize, Serialize)]
pub struct DataWrapper {
    #[serde(rename = "mapping")]
    pub mapping: Vec<ReconstructionDataPoint>,
}

pub fn flip_coordinates(mut data_point: ReconstructionDataPoint) -> ReconstructionDataPoint {
    // Swap the x and y values for true_points
    if data_point.true_points.len() == 2 {
        data_point.true_points.swap(0, 1);
    }

    // Swap the x and y values for reconstructed_points
    if data_point.reconstructed_points.len() == 2 {
        data_point.reconstructed_points.swap(0, 1);
    }

    // Optional: Flipped the unscaled points as well if they happen to exist
    if let Some(unscaled) = &mut data_point.unscaled_points {
        if unscaled.len() == 2 {
            unscaled.swap(0, 1);
        }
    }

    data_point
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ReconstructionDataPoint {
    #[serde(rename = "true")]
    pub true_points: Vec<f64>,
    #[serde(rename = "reconstructed")]
    pub reconstructed_pointzs: Vec<f64>,
    #[serde(rename = "unscaled_true")]
    pub unscaled_points: Option<Vec<f64>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct DbResult {
    pub name: String,
    pub method: String,
    pub dims: u32,
    pub mse: Option<f64>,
    pub match_rate: Option<f64>,
    pub chamfer: Option<f64>,
    pub number_of_reconstructions: String,
    pub time_taken: f64,
    pub total_db_size: u64,
    pub percent_queries_used: f64,
    pub num_queries_used: u64,
    pub eps: Option<f64>,
    pub delt: Option<f64>,
}

/// Loads a standard JSON file containing a flat Vec of reconstruction points.
fn load_standard_method(path: &str) -> Result<Vec<ReconstructionDataPoint>, Box<dyn Error>> {
    let file_content = fs::read_to_string(path)?;
    let data: Vec<ReconstructionDataPoint> = serde_json::from_str(&file_content)?;
    Ok(data)
}

/// Loads the limits JSON file containing a Vec of Vecs of reconstruction points.
fn load_limits_method(path: &str) -> Result<Vec<Vec<ReconstructionDataPoint>>, Box<dyn Error>> {
    let file_content = fs::read_to_string(path)?;
    let data: Vec<Vec<ReconstructionDataPoint>> = serde_json::from_str(&file_content)?;
    Ok(data)
}

pub fn do_plotting() {
    // Hardcoded vectors for easy modification
    // let grid_sizes = vec![(20, "20x20"), (25, "25x25"), (50, "50x50")];
    // let grid_sizes = vec![(25, "25x25"), (50, "50x50")];
    let datasets = vec!["shopparis", "busstop", "cali", "drink", "highway", "spitz"];
    let methods = vec!["even_less", "remin", "limits"];
    let distributions = vec!["uniform", "gaussian", "beta"];

    let databases: Vec<String> = (20..=50)
        .step_by(5)
        .map(|n| format!("databases/{}x{}", n, n))
        .collect();

    let grid_sizes: Vec<(u32, String)> = (20..=50)
        .step_by(10)
        .map(|n| (n, format!("{}x{}", n, n)))
        .collect();

    for grid in &grid_sizes {
        for name in &datasets {
            let data = format!("databases/{}x{}", grid.0, grid.0);
            let res = run_spatial_plots(name, &*data, "uniform", grid.0);

            match res {
                Ok(_) => {}

                Err(e) => {
                    warn!("{name} failed when loaded from {data}:  {e}")
                }
            }
        }
    }

    for data in datasets.clone() {
        for dist in distributions.clone() {
            run_spatial_plots(data, "databases/50x50", dist, 50).expect("SPITZ DIRECT FAILED!");
        }
    }

    plot_grid_by_mse(&grid_sizes, &datasets, &methods, &distributions).unwrap();
    // do_table_plot(
    //     "databases",
    //     &grid_sizes.iter().map(|x| x.0).collect(),
    //     &datasets,
    //     &distributions,
    // );

    let dir = "databases/350x50";
    let name = "spitz";

    // do JSUT 350x50 spitz stuff
    let datasets = vec!["spitz"];
    let grid = (350, 50);
    run_spatial_plots("spitz", "databases/350x50", "uniform", 350).expect("SPITZ DIRECT FAILED!");
    // run_spatial_plots("spitz", &datasets, "databases/175x25", 175).expect("SPITZ DIRECT FAILED!");
    // do_convex_hull_plots("spitz", "databases/50x350");

    for dist in distributions {
        let res = plot_histograms_of_all_reconstructions("spitz", (350, 50), dist);
        match res {
            Ok(_) => {}
            Err(e) => {
                warn!("{name} failed when loaded from {dir}:  {e}")
            }
        }
        let res = plot_histograms_of_all_reconstructions("cali", (50, 50), dist);
        match res {
            Ok(_) => {}
            Err(e) => {
                warn!("{name} failed when loaded from {dir}:  {e}")
            }
        }
    }
}

fn get_remin_even_less(
    recon_path: &str,
    procrustes: bool,
) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>), Box<dyn Error>> {
    let content = fs::read_to_string(recon_path)?;
    let mut data: Vec<ReconstructionDataPoint> = serde_json::from_str(&content)?;

    if procrustes {
        let aligned = post::procrustes_align(&*data, true, true, true);
        data = aligned.0;
    }

    let mut true_point = Vec::new();
    let mut recon_point = Vec::new();

    for point in data {
        true_point.push(point.true_points);
        recon_point.push(point.reconstructed_points);
    }

    return Ok((true_point, recon_point));
}
