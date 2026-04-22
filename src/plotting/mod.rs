use crate::plotting::post::{calculate_mse, scale_to_absolute_range};
use crate::plotting::two_d::debug_mse::plot_mse_frequency_histogram_split;
use crate::plotting::two_d::mse_by_all_reconstructions::plot_histograms_of_all_reconstructions;
use crate::plotting::two_d::mse_vs_grid_size::plot_grid_by_mse;
use crate::plotting::two_d::spatial_plot::run_spatial_plots;
use crate::LAMA::translator::TranslatorMeta;
use log::{info, warn};
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
    pub translator_meta: TranslatorMeta,
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
    // let datasets = vec!["highway", "spitz"];
    let methods = vec!["even_less", "remin", "limits"];
    let distributions = vec!["uniform", "gaussian", "beta"];

    for data in datasets.clone() {
        run_spatial_plots(data, "databases/50x50", "uniform", 50).unwrap();
    }

    let grid_sizes: Vec<(u32, String)> = (20..=50)
        .step_by(10)
        .map(|n| (n, format!("{}x{}", n, n)))
        .collect();

    plot_grid_by_mse(&grid_sizes, &datasets, &methods, &distributions).unwrap();

    // The numbers for the domains below don't exactly match the original domain. This is
    // because somethimes the best case/procrustes analysis sometimes actually falls outside the
    // domain. This is a little arbitrary, but just give these methods some more room to get the
    // truly best result.

    for name in datasets.clone() {
        let res = plot_histograms_of_all_reconstructions(name, (50, 50), (60, 60), "uniform");
        match res {
            Ok(_) => {}
            Err(e) => {
                warn!("{name} hustogram of all recons 50x50 failed:  {e}")
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
        let procruste = aligned.0;
        info!("MSE After procrustes: {}", aligned.1);

        // data = scale_to_absolute_range(&*procruste, (0.0, 50.0));
        // let new_mse = calculate_mse(&data);
        // info!("MSE After forced scaling: {}", new_mse);
        // plot_mse_frequency_histogram_split(
        //     procruste.clone(),
        //     data.clone(),
        //     "Even Less - Uniform Distribution on Amsterdam Dataset",
        //     "figures/debug.svg",
        // );
        data = procruste;
    }

    let mut true_point = Vec::new();
    let mut recon_point = Vec::new();

    for point in data {
        true_point.push(point.true_points);
        recon_point.push(point.reconstructed_points);
    }

    return Ok((true_point, recon_point));
}
