use crate::plotting::plot::ReconstructionData2dPoint;
use serde_json::json;
use std::error::Error;
use std::fs::File;
use std::io::Write;

/// Computes the optimal 2D Procrustes alignment (translation, rotation, uniform scaling)
/// analytically, returning the aligned dataset and the minimized global MSE.
pub fn procrustes_align(
    data: &[ReconstructionData2dPoint],
) -> (Vec<ReconstructionData2dPoint>, f64) {
    let n = data.len() as f64;
    if n == 0.0 {
        return (Vec::new(), 0.0);
    }

    // 1. Calculate centroids
    let (mut mean_true_x, mut mean_true_y) = (0.0, 0.0);
    let (mut mean_recon_x, mut mean_recon_y) = (0.0, 0.0);

    for p in data {
        mean_true_x += p.true_points.0;
        mean_true_y += p.true_points.1;
        mean_recon_x += p.reconstructed_points.0;
        mean_recon_y += p.reconstructed_points.1;
    }

    mean_true_x /= n;
    mean_true_y /= n;
    mean_recon_x /= n;
    mean_recon_y /= n;

    // 2. Center points and accumulate cross-covariance & variance
    let mut numerator_a = 0.0;
    let mut numerator_b = 0.0;
    let mut var_recon = 0.0;

    for p in data {
        let tx = p.true_points.0 - mean_true_x;
        let ty = p.true_points.1 - mean_true_y;
        let rx = p.reconstructed_points.0 - mean_recon_x;
        let ry = p.reconstructed_points.1 - mean_recon_y;

        numerator_a += tx * rx + ty * ry;
        numerator_b += ty * rx - tx * ry;
        var_recon += rx * rx + ry * ry;
    }

    // 3. Solve for combined scale and rotation factors (u = s*cos(theta), v = s*sin(theta))
    let (u, v) = if var_recon > 1e-12 {
        (numerator_a / var_recon, numerator_b / var_recon)
    } else {
        (1.0, 0.0)
    };

    // 4. Apply optimal transformation and compute final MSE
    let mut total_mse = 0.0;

    let aligned_data = data
        .iter()
        .map(|p| {
            let rx = p.reconstructed_points.0 - mean_recon_x;
            let ry = p.reconstructed_points.1 - mean_recon_y;

            let scaled_rotated_x = rx * u - ry * v;
            let scaled_rotated_y = rx * v + ry * u;

            let final_x = scaled_rotated_x + mean_true_x;
            let final_y = scaled_rotated_y + mean_true_y;

            let diff_x = p.true_points.0 - final_x;
            let diff_y = p.true_points.1 - final_y;
            total_mse += diff_x * diff_x + diff_y * diff_y;

            ReconstructionData2dPoint {
                true_points: p.true_points,
                reconstructed_points: (final_x, final_y),
            }
        })
        .collect();

    (aligned_data, total_mse / n)
}

use itertools::iproduct;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Calculates the standard Mean Squared Error (MSE)
pub fn calculate_mse(data: &[ReconstructionData2dPoint]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = data
        .iter()
        .map(|p| {
            let dx = p.true_points.0 - p.reconstructed_points.0;
            let dy = p.true_points.1 - p.reconstructed_points.1;
            dx * dx + dy * dy
        })
        .sum();
    sum_sq / data.len() as f64
}

/// Applies a transformation and calculates MSE without allocating a new Vec
#[inline]
fn evaluate_transform(
    data: &[ReconstructionData2dPoint],
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
            let (tx, ty) = p.true_points;
            let (rx, ry) = p.reconstructed_points;

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
    data: &[ReconstructionData2dPoint],
    dx: f64,
    dy: f64,
    angle_deg: f64,
    sx: f64,
    sy: f64,
) -> Vec<ReconstructionData2dPoint> {
    let rad = angle_deg.to_radians();
    let (sin_t, cos_t) = rad.sin_cos();

    data.iter()
        .map(|p| {
            let (rx, ry) = p.reconstructed_points;
            let scaled_x = rx * sx;
            let scaled_y = ry * sy;
            let rot_x = scaled_x * cos_t - scaled_y * sin_t;
            let rot_y = scaled_x * sin_t + scaled_y * cos_t;

            ReconstructionData2dPoint {
                true_points: p.true_points,
                reconstructed_points: (rot_x + dx, rot_y + dy),
            }
        })
        .collect()
}

/// 1. Shift Search
pub fn search_shifts(
    data: &[ReconstructionData2dPoint],
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
pub fn search_rotations(data: &[ReconstructionData2dPoint], step_deg: f64) -> Vec<f64> {
    let steps = (0..=(360.0 / step_deg) as usize).map(|i| i as f64 * step_deg);

    steps
        .par_bridge()
        .map(|angle| evaluate_transform(data, 0.0, 0.0, angle, 1.0, 1.0))
        .collect()
}

/// 3. Scale Search
pub fn search_scales(
    data: &[ReconstructionData2dPoint],
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
    data: &[ReconstructionData2dPoint],
    shift_domain: (usize, usize),
    shift_step: f64,
    rot_step_deg: f64,
    scale_domain: (usize, usize),
    scale_step: f64,
) -> (Vec<ReconstructionData2dPoint>, Vec<f64>) {
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

    // Create combinations. par_bridge() parallelizes the consumption of this massive iterator.
    let grid = iproduct!(x_shifts, y_shifts, rotations, x_scales, y_scales);

    // Map-Reduce to find all MSEs and track the best configuration simultaneously
    let (all_mses, best_config) = grid
        .par_bridge()
        .map(|(dx, dy, angle, sx, sy)| {
            let mse = evaluate_transform(data, dx, dy, angle, sx, sy);
            (vec![mse], (mse, dx, dy, angle, sx, sy))
        })
        .reduce(
            || (Vec::new(), (f64::MAX, 0.0, 0.0, 0.0, 1.0, 1.0)),
            |mut acc1, mut acc2| {
                acc1.0.append(&mut acc2.0);
                let best = if acc1.1.0 < acc2.1.0 { acc1.1 } else { acc2.1 };
                (acc1.0, best)
            },
        );

    let (_, best_dx, best_dy, best_angle, best_sx, best_sy) = best_config;
    let best_vec = apply_transform(data, best_dx, best_dy, best_angle, best_sx, best_sy);

    (best_vec, all_mses)
}

pub fn export_to_geojson(
    data: Vec<ReconstructionData2dPoint>,
    output_path: &str,
) -> Result<(), Box<dyn Error>> {
    let mut features = Vec::new();

    // 2. Map your data to standard GeoJSON features
    for (index, item) in data.iter().enumerate() {
        // GeoJSON strictly requires [longitude, latitude]
        // Ensure your f64 tuples are ordered correctly here!

        // Feature A: Ground Truth Point
        features.push(json!({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [item.true_points.0, item.true_points.1]
            },
            "properties": {
                "pair_id": index,
                "point_type": "ground_truth"
            }
        }));

        // Feature B: Reconstructed Point
        features.push(json!({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [item.reconstructed_points.0, item.reconstructed_points.1]
            },
            "properties": {
                "pair_id": index,
                "point_type": "reconstructed"
            }
        }));
    }

    // 3. Wrap it in a FeatureCollection
    let geojson = json!({
        "type": "FeatureCollection",
        "features": features
    });

    // 4. Write it out to the new file
    let mut output_file = File::create(output_path)?;
    let geojson_string = serde_json::to_string_pretty(&geojson)?;
    output_file.write_all(geojson_string.as_bytes())?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    // Assumes procrustes_align and the struct are in the parent module

    #[test]
    fn test_procrustes_alignment() {
        // True points: A simple 1x1 square
        let true_pts = vec![(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)];

        // Reconstructed points:
        // 1. Scaled by 2.0
        // 2. Rotated by 90 degrees counter-clockwise (x,y -> -y,x)
        // 3. Translated by x + 5.0, y + 10.0
        let recon_pts = vec![
            (5.0, 10.0), // from (0,0)
            (3.0, 10.0), // from (0,1)
            (3.0, 12.0), // from (1,1)
            (5.0, 12.0), // from (1,0)
        ];

        // Zip them into your struct
        let data: Vec<ReconstructionData2dPoint> = true_pts
            .into_iter()
            .zip(recon_pts.into_iter())
            .map(|(t, r)| ReconstructionData2dPoint {
                true_points: t,
                reconstructed_points: r,
            })
            .collect();

        // Run the alignment
        let (aligned_data, mse) = procrustes_align(&data);

        // Float comparison: MSE should be practically zero
        assert!(mse < 1e-10, "MSE is not zero: {}", mse);

        // Verify each aligned point matches the original true point perfectly
        for p in aligned_data {
            let dx = (p.true_points.0 - p.reconstructed_points.0).abs();
            let dy = (p.true_points.1 - p.reconstructed_points.1).abs();

            assert!(
                dx < 1e-10,
                "X mismatch: true {}, aligned {}",
                p.true_points.0,
                p.reconstructed_points.0
            );
            assert!(
                dy < 1e-10,
                "Y mismatch: true {}, aligned {}",
                p.true_points.1,
                p.reconstructed_points.1
            );
        }
    }
}
