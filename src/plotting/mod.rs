use crate::plotting::plot::ReconstructionDataPoint;
use crate::plotting::two_d::mse_vs_grid::plot_grid_by_mse;
use std::error::Error;
use std::fs;

mod approx;
mod error;
pub mod plot;
pub mod post;

pub mod two_d;

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
    let datasets = vec!["shopparis", "busstop", "cali", "drink", "highway", "spitz"];
    let methods = vec!["even_less", "remin", "limits"];
    let distributions = vec!["uniform", "gaussian", "beta"];

    plot_grid_by_mse(&grid_sizes, &datasets, &methods, &distributions).unwrap();
}
