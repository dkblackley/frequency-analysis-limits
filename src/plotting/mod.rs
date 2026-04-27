use crate::plotting::convex_hull::get_per_point_convex_hulls;
use crate::plotting::post::{calculate_mse, do_averaging, scale_to_absolute_range};
use crate::plotting::two_d::box_plot_query::process_and_plot_boxplots;
use crate::plotting::two_d::debug_mse::plot_mse_frequency_histogram_split;
use crate::plotting::two_d::flat_dist_hist::plot_lama_distributions;
use crate::plotting::two_d::mse_by_all_reconstructions::plot_histograms_of_all_reconstructions;
use crate::plotting::two_d::mse_vs_grid_size::plot_grid_by_mse;
use crate::plotting::two_d::spatial_plot::run_spatial_plots;
use crate::plotting::two_d::three_dim::plot_nh_minimal_3d;
use crate::plotting::two_d::worst_case_convex_hull::{
    compute_dataset_metrics, process_and_plot_convex_hulls,
};
use crate::LAMA::translator::TranslatorMeta;
use log::{info, warn};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs;

mod convex_hull;
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
    // do_averaging().expect("TODO: panic message");
    // quick_and_dirty_analysis().unwrap();

    // Hardcoded vectors for easy modification
    // let grid_sizes = vec![(20, "20x20"), (25, "25x25"), (50, "50x50")];
    // let grid_sizes = vec![(25, "25x25"), (50, "50x50")];
    let datasets = vec!["busstop", "shopparis", "cali", "drink", "highway", "spitz"];
    // let datasets = vec!["highway", "spitz"];
    let methods = vec!["even_less", "remin", "limits"];
    let distributions = vec!["uniform", "gaussian", "beta"];

    // for name in datasets.clone() {
    //     let res = plot_lama_distributions(name);
    //     match res {
    //         Ok(_) => {}
    //         Err(e) => {
    //             warn!("{name} LAMa 15x15 flat distribution histogram failed: {e}")
    //         }
    //     }
    // }

    // for name in datasets.clone() {
    //     let res = plot_histograms_of_all_reconstructions(name, (50, 50), (60, 60), "uniform");
    //     match res {
    //         Ok(_) => {}
    //         Err(e) => {
    //             warn!("{name} histogram of all recons 50x50 failed:  {e}")
    //         }
    //     }
    // }

    // let grid_sizes: Vec<(u32, String)> = (20..=50)
    //     .step_by(10)
    //     .map(|n| (n, format!("{}x{}", n, n)))
    //     .collect();
    //
    // plot_grid_by_mse(&grid_sizes, &datasets, &methods, &distributions).unwrap();

    // for data in datasets.clone() {
    //     run_spatial_plots(data, "databases/50x50", "uniform", 50).unwrap();
    // }
    //
    // plot_nh_minimal_3d().expect("TODO: panic message");

    for name in datasets.clone() {
        // 2. Process and Plot Box Plots
        if let Err(e) = process_and_plot_boxplots(name, "databases/15x15", &distributions) {
            warn!("{} box plot failed: {}", name, e);
        }

        // 1. Process and Plot Convex Hulls
        if let Err(e) = process_and_plot_convex_hulls(name, "databases/15x15", &distributions) {
            warn!("{} convex hull plot failed: {}", name, e);
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

/// A quick and dirty function to print lengths of the 15x15 files
/// and compute the convex hull/centroid metrics for the 6x6* grids.
pub fn quick_and_dirty_analysis() -> Result<(), Box<dyn Error>> {
    // Helper closure to extract true points (reused from your code)
    let extract_true_points = |data: &[ReconstructionDataPoint]| -> Vec<Vec<f64>> {
        data.iter().map(|p| p.true_points.clone()).collect()
    };

    // ---------------------------------------------------------
    // 1. 15x15 Files: Load and print lengths
    // ---------------------------------------------------------
    let files_15x15 = vec![
        "databases/15x15/highway/limits/highway_flat_e0_d0.1_reconstruction.json",
        "databases/15x15/shopparis/limits/shopparis_flat_e0_d0.1_reconstruction.json",
        "databases/15x15/busstop/limits/busstop_flat_e0_d0.1_reconstruction.json",
    ];

    println!("--- 15x15 Dataset Lengths ---");
    for path in files_15x15 {
        match fs::read_to_string(path) {
            Ok(content) => {
                let data: Vec<Vec<ReconstructionDataPoint>> = serde_json::from_str(&content)?;
                println!("File: {} \n  -> Outer Vec Length: {}", path, data.len());
            }
            Err(e) => println!("Failed to read {}: {}", path, e),
        }
    }
    println!();

    build_hulls_and_calculate_mse()
}

pub fn build_hulls_and_calculate_mse() -> Result<(), Box<dyn Error>> {
    let grid_files = vec![
        "databases/6x6/grid/limits/grid_uniform_e0_d0.001_reconstruction.json",
        "databases/6x6x6/grid/limits/grid_uniform_e0_d0.001_reconstruction.json",
        "databases/6x6x6x6/grid/limits/grid_uniform_e0_d0.001_reconstruction.json",
    ];

    println!("--- Centroid-based Construction & MSE Analysis ---");

    for path in grid_files {
        match fs::read_to_string(path) {
            Ok(content) => {
                // Deserialize the multiple runs
                let multi_run_data: Vec<Vec<ReconstructionDataPoint>> =
                    serde_json::from_str(&content)?;

                if multi_run_data.is_empty() || multi_run_data[0].is_empty() {
                    println!("File is empty or invalid: {}", path);
                    continue;
                }

                // 1. Generate the convex hulls across all runs for each point
                let hulls = get_per_point_convex_hulls(&multi_run_data);

                // We use the ground truth from the first run (assuming it is constant across runs)
                let base_run = &multi_run_data[0];

                // 2. Build the new reconstructed dataset
                let mut centroid_reconstruction = Vec::with_capacity(base_run.len());

                for (i, original_point) in base_run.iter().enumerate() {
                    let hull = &hulls[i];
                    let dims = original_point.true_points.len();
                    let mut centroid = vec![0.0; dims];
                    let num_vertices = hull.vertices.len();

                    if num_vertices > 0 {
                        // Sum up all vertices
                        for v in &hull.vertices {
                            for d in 0..dims {
                                centroid[d] += v[d];
                            }
                        }
                        // Divide by the number of vertices to get the barycenter
                        for d in 0..dims {
                            centroid[d] /= num_vertices as f64;
                        }
                    } else {
                        // Fallback: if the hull is somehow empty, default to the standard reconstruction
                        centroid = original_point.reconstructed_points.clone();
                    }

                    // 3. Assemble the new Data Point mapping the centroid as the new reconstructed point
                    centroid_reconstruction.push(ReconstructionDataPoint {
                        true_points: original_point.true_points.clone(),
                        reconstructed_points: centroid,
                        unscaled_points: None, // Or clone original_point.unscaled_points if you need it later
                    });
                }

                // 4. Pass our newly built Vec to your calculation function
                let final_mse = calculate_mse(&centroid_reconstruction);

                println!("Processed: {}", path);
                println!("  -> Dataset Size: {}", centroid_reconstruction.len());
                println!("  -> Final Centroid Reconstruction MSE: {:.6}\n", final_mse);
            }
            Err(e) => println!("Failed to read {}: {}", path, e),
        }
    }

    Ok(())
}
