use crate::plotting::two_d::spatial_plot::run_spatial_plots;
use crate::plotting::two_d::worst_case::do_convex_hull_plots;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs;

mod metrics;
pub mod post;
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
    pub reconstructed_points: Vec<f64>,
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
    // let grid_sizes = vec![(25, "25x25"), (50, "50x50"), (75, "75x75")];
    // let grid_sizes = vec![(25, "25x25"), (50, "50x50")];
    // let datasets = vec!["shopparis", "busstop", "cali", "drink", "highway", "spitz"];
    //let datasets = vec!["shopparis", "busstop", "drink", "spitz"];
    let datasets = vec!["spitz"];
    let methods = vec!["even_less", "remin", "limits"];
    let distributions = vec!["uniform", "gaussian", "beta"];

    let databases: Vec<String> = (20..=50)
        .step_by(5)
        .map(|n| format!("databases/{}x{}", n, n))
        .collect();

    let grid_sizes: Vec<(u32, String)> = (20..=50)
        .step_by(5)
        .map(|n| (n, format!("{}x{}", n, n)))
        .collect();

    // Do spatial plots
    //plot_grid_by_mse(&grid_sizes, &datasets, &methods, &distributions).unwrap();
    run_spatial_plots(&datasets, "databases/50x350", 350);

    let dir = "databases/350x50";
    let name = "spitz";
    do_convex_hull_plots("spitz", "databases/350x50");

    // do JSUT 350x50 spitz stuff
    let grid = (350, 50);
    //plot_histograms_of_all_reconstructions();
}

fn get_remin_even_less(recon_path: &str, procrustes: bool) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let content = fs::read_to_string(recon_path).unwrap();
    let mut data: Vec<ReconstructionDataPoint> = serde_json::from_str(&content).unwrap();

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

    return (true_point, recon_point);
}
