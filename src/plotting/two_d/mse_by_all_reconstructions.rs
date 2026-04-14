use std::fs;

use indicatif::ParallelProgressIterator;
use itertools::iproduct;
use log::debug;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::plotting::ReconstructionDataPoint;

pub fn plot_histograms_of_all_reconstructions() {
    let remin_mse = get_all_mse;
    let even_less_mse = get_all_mse((50, 50), 1.0, 10.0, (350, 50), 1.0, true);

    let name = "spitz";
    let path_to_root = "databases/350x50/{name}";

    let path = format!("{path_to_root}/remin/{name}_prob100.0_uniform_350x50_classic.json");

    let path = format!("{path_to_root}/even_less/{name}_prob100.0_uniform_350x50_even_less.json");

    debug!("About to load data from {}", &path);

    let content = fs::read_to_string(&path).unwrap();
    let all_data: Vec<ReconstructionDataPoint> = serde_json::from_str(&content).unwrap();

    combined_search(&all_data, (50, 50), 1.0, 10.0, (350, 50), 1.0);
}

pub fn get_all_mse(
    shift_domain: (usize, usize),
    shift_step: f64,
    rot_step_deg: f64,
    scale_domain: (usize, usize),
    scale_step: f64,
    even_less: bool,
) -> (Vec<ReconstructionDataPoint>, Vec<f64>) {
    let name = "spitz";
    let path_to_root = "databases/350x50/{name}";

    let path = "";

    if even_less {
        let path =
            format!("{path_to_root}/even_less/{name}_prob100.0_uniform_350x50_even_less.json");
    } else {
        let path = format!("{path_to_root}/remin/{name}_prob100.0_uniform_350x50_classic.json");
    }

    debug!("About to load data from {}", &path);

    let content = fs::read_to_string(&path).unwrap();
    let all_data: Vec<ReconstructionDataPoint> = serde_json::from_str(&content).unwrap();

    combined_search(
        &all_data,
        shift_domain,
        shift_step,
        rot_step_deg,
        scale_domain,
        scale_step,
    )
}

/// Applies a transformation and calculates MSE without allocating a new Vec
#[inline]
fn evaluate_transform(
    data: &[ReconstructionDataPoint],
    dx: f64,
    dy: f64,
    angle_deg: f64,
    sx: f64,
    sy: f64,
) -> f64 {
    let rad = angle_deg.to_radians();
    let (sin_t, cos_t) = rad.sin_cos();

    let sum_sq: f64 = data
        .iter()
        .map(|p| {
            // Assuming 2D data exists at the first two indices
            let tx = p.true_points[0];
            let ty = p.true_points[1];
            let rx = p.reconstructed_points[0];
            let ry = p.reconstructed_points[1];

            // Scale -> Rotate -> Translate
            let scaled_x = rx * sx;
            let scaled_y = ry * sy;

            let rot_x = scaled_x * cos_t - scaled_y * sin_t;
            let rot_y = scaled_x * sin_t + scaled_y * cos_t;

            let final_x = rot_x + dx;
            let final_y = rot_y + dy;

            let diff_x = tx - final_x;
            let diff_y = ty - final_y;
            diff_x * diff_x + diff_y * diff_y
        })
        .sum();

    sum_sq / data.len() as f64
}

/// Reconstructs the actual Vec for the best found parameters
fn apply_transform(
    data: &[ReconstructionDataPoint],
    dx: f64,
    dy: f64,
    angle_deg: f64,
    sx: f64,
    sy: f64,
) -> Vec<ReconstructionDataPoint> {
    let rad = angle_deg.to_radians();
    let (sin_t, cos_t) = rad.sin_cos();

    data.iter()
        .map(|p| {
            let rx = p.reconstructed_points[0];
            let ry = p.reconstructed_points[1];

            let scaled_x = rx * sx;
            let scaled_y = ry * sy;
            let rot_x = scaled_x * cos_t - scaled_y * sin_t;
            let rot_y = scaled_x * sin_t + scaled_y * cos_t;

            ReconstructionDataPoint {
                true_points: p.true_points.clone(),
                reconstructed_points: vec![rot_x + dx, rot_y + dy],
                unscaled_points: p.unscaled_points.clone(),
            }
        })
        .collect()
}

/// 1. Shift Search
pub fn search_shifts(
    data: &[ReconstructionDataPoint],
    domain_width: usize,
    domain_height: usize,
    step: f64,
) -> Vec<f64> {
    let x_steps = (0..=(domain_width as f64 / step) as usize).map(|i| i as f64 * step);
    let y_steps = (0..=(domain_height as f64 / step) as usize).map(|i| i as f64 * step);

    iproduct!(x_steps, y_steps)
        .par_bridge()
        .map(|(dx, dy)| evaluate_transform(data, dx, dy, 0.0, 1.0, 1.0))
        .collect()
}

/// 2. Rotation Search
pub fn search_rotations(data: &[ReconstructionDataPoint], step_deg: f64) -> Vec<f64> {
    let steps = (0..=(360.0 / step_deg) as usize).map(|i| i as f64 * step_deg);

    steps
        .par_bridge()
        .map(|angle| evaluate_transform(data, 0.0, 0.0, angle, 1.0, 1.0))
        .collect()
}

/// 3. Scale Search
pub fn search_scales(
    data: &[ReconstructionDataPoint],
    max_w: usize,
    max_h: usize,
    step: f64,
) -> Vec<f64> {
    let x_scales = (1..=(max_w as f64 / step) as usize).map(|i| i as f64 * step);
    let y_scales = (1..=(max_h as f64 / step) as usize).map(|i| i as f64 * step);

    iproduct!(x_scales, y_scales)
        .par_bridge()
        .map(|(sx, sy)| evaluate_transform(data, 0.0, 0.0, 0.0, sx, sy))
        .collect()
}

/// 4. Combined Multithreaded Grid Search
pub fn combined_search(
    data: &[ReconstructionDataPoint],
    shift_domain: (usize, usize),
    shift_step: f64,
    rot_step_deg: f64,
    scale_domain: (usize, usize),
    scale_step: f64,
) -> (Vec<ReconstructionDataPoint>, Vec<f64>) {
    let x_shifts: Vec<f64> = (0..=(shift_domain.0 as f64 / shift_step) as usize)
        .map(|i| i as f64 * shift_step)
        .collect();
    let y_shifts: Vec<f64> = (0..=(shift_domain.1 as f64 / shift_step) as usize)
        .map(|i| i as f64 * shift_step)
        .collect();
    let rotations: Vec<f64> = (0..=(360.0 / rot_step_deg) as usize)
        .map(|i| i as f64 * rot_step_deg)
        .collect();
    let x_scales: Vec<f64> = (1..=(scale_domain.0 as f64 / scale_step) as usize)
        .map(|i| i as f64 * scale_step)
        .collect();
    let y_scales: Vec<f64> = (1..=(scale_domain.1 as f64 / scale_step) as usize)
        .map(|i| i as f64 * scale_step)
        .collect();

    let grid = iproduct!(x_shifts, y_shifts, rotations, x_scales, y_scales);
    let progress = indicatif::ProgressBar::new(grid.size_hint().0 as u64);

    progress.set_style(
        indicatif::ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
            )
            .unwrap(),
    );

    // Optimized aggregation: Collect tuples instead of appending Vecs in reduce
    let evaluated: Vec<_> = grid
        .par_bridge()
        .progress_with(progress)
        .map(|(dx, dy, angle, sx, sy)| {
            let mse = evaluate_transform(data, dx, dy, angle, sx, sy);
            (mse, dx, dy, angle, sx, sy)
        })
        .collect();

    // Single linear pass to extract MSEs and find the optimal configuration
    let mut best_config = (f64::MAX, 0.0, 0.0, 0.0, 1.0, 1.0);
    let mut all_mses = Vec::with_capacity(evaluated.len());

    for &result in &evaluated {
        all_mses.push(result.0);
        if result.0 < best_config.0 {
            best_config = result;
        }
    }

    let (_, best_dx, best_dy, best_angle, best_sx, best_sy) = best_config;
    let best_vec = apply_transform(data, best_dx, best_dy, best_angle, best_sx, best_sy);

    (best_vec, all_mses)
}
