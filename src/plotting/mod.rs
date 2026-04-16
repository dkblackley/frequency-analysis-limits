use crate::plotting::two_d::mse_vs_grid::plot_grid_by_mse;
use crate::plotting::two_d::spatial_plot::run_spatial_plots;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs;

pub mod post;
mod three_d;
pub mod two_d;

#[derive(Debug, Deserialize, Serialize)]
pub struct DataWrapper {
    #[serde(rename = "mapping")]
    pub mapping: Vec<ReconstructionDataPoint>,
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
    let grid_sizes = vec![(25, "25x25"), (50, "50x50"), (75, "75x75")];
    // let grid_sizes = vec![(25, "25x25"), (50, "50x50")];
    // let datasets = vec!["shopparis", "busstop", "cali", "drink", "highway", "spitz"];
    let datasets = vec!["shopparis", "busstop", "drink", "spitz"];
    // let datasets = vec!["spitz"];
    let methods = vec!["even_less", "remin", "limits"];
    let distributions = vec!["uniform", "gaussian", "beta"];
    let databases = vec!["databases/25x25", "databases/50x50", "databases/75x75"];

    // Do spatial plots

    let dir = "databases/50x50";

    run_spatial_plots(&datasets, dir);
    plot_grid_by_mse(&grid_sizes, &datasets, &methods, &distributions).unwrap();

    // do JSUT 350x50 spitz stuff
    let grid = (350, 50);
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
