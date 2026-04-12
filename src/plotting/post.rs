use crate::plotting::plot::ReconstructionData2dPoint;
use nalgebra::DMatrix;
use serde::Deserialize;
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fs;
use std::fs::File;
use std::io::Write;

/// Calculates the standard Mean Squared Error (MSE) across N dimensions
pub fn calculate_mse(data: &[ReconstructionData2dPoint]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = data
        .iter()
        .map(|p| {
            p.true_points
                .iter()
                .zip(p.reconstructed_points.iter())
                .map(|(t, r)| (t - r).powi(2))
                .sum::<f64>()
        })
        .sum();
    sum_sq / data.len() as f64
}

/// Computes the optimal N-Dimensional Procrustes alignment
/// using Singular Value Decomposition (SVD).
pub fn procrustes_align(
    data: &[ReconstructionData2dPoint],
    scale: bool,
    rotate: bool,
    shift: bool,
) -> (Vec<ReconstructionData2dPoint>, f64) {
    let n = data.len();
    if n == 0 {
        return (Vec::new(), 0.0);
    }

    // Determine dimensionality from the first point
    let d = data[0].true_points.len();

    // 1. Calculate centroids
    let mut mean_true = vec![0.0; d];
    let mut mean_recon = vec![0.0; d];

    for p in data {
        for i in 0..d {
            mean_true[i] += p.true_points[i];
            mean_recon[i] += p.reconstructed_points[i];
        }
    }

    for i in 0..d {
        mean_true[i] /= n as f64;
        mean_recon[i] /= n as f64;
    }

    // 2. Center points into dynamically sized matrices
    let mut var_recon = 0.0;
    let mut centered_true = DMatrix::<f64>::zeros(n, d);
    let mut centered_recon = DMatrix::<f64>::zeros(n, d);

    for (row, p) in data.iter().enumerate() {
        for col in 0..d {
            let t = p.true_points[col] - mean_true[col];
            let r = p.reconstructed_points[col] - mean_recon[col];
            centered_true[(row, col)] = t;
            centered_recon[(row, col)] = r;
            var_recon += r * r;
        }
    }

    // 3. Solve for N-D Rotation matrix and Scale factor
    let mut r_mat = DMatrix::<f64>::identity(d, d);
    let mut s_factor = 1.0;

    if var_recon > 1e-12 {
        if rotate {
            // Cross-covariance matrix H = Y^T * X
            let h = centered_recon.transpose() * &centered_true;

            // SVD of H
            let svd = h.svd(true, true);
            let u = svd.u.unwrap();
            let v_t = svd.v_t.unwrap();

            // Optimal rotation R = U * V^T
            let mut r_temp = &u * &v_t;
            let mut d_sign = 1.0;

            // Prevent reflection by enforcing a positive determinant
            if r_temp.determinant() < 0.0 {
                d_sign = -1.0;
                let mut modified_u = u.clone();
                for i in 0..d {
                    modified_u[(i, d - 1)] *= -1.0;
                }
                r_temp = modified_u * v_t;
            }
            r_mat = r_temp;

            // Optimal scale accounting for rotation
            if scale {
                let mut trace_sigma = 0.0;
                for i in 0..d {
                    let sign = if i == d - 1 { d_sign } else { 1.0 };
                    trace_sigma += svd.singular_values[i] * sign;
                }
                s_factor = trace_sigma / var_recon;
            }
        } else if scale {
            // Scale only (trace(Y^T * X) / variance)
            let h = centered_recon.transpose() * &centered_true;
            s_factor = h.trace() / var_recon;
        }
    }

    // 4. Apply transformations and calculate MSE
    let mut total_mse = 0.0;
    let mut aligned_data = Vec::with_capacity(n);

    for (row, p) in data.iter().enumerate() {
        // Extract 1xD row vector, apply rotation and scale: Y_transformed = Y * R * s
        let y_row = centered_recon.row(row);
        let transformed_y = y_row * &r_mat * s_factor;

        let mut final_true = vec![0.0; d];
        let mut final_recon = vec![0.0; d];
        let mut diff_sq_sum = 0.0;

        for col in 0..d {
            let tx = centered_true[(row, col)];
            let scaled_rotated_r = transformed_y[(0, col)];

            let (ft, fr) = if shift {
                (p.true_points[col], scaled_rotated_r + mean_true[col])
            } else {
                (tx, scaled_rotated_r)
            };

            final_true[col] = ft;
            final_recon[col] = fr;

            let diff = ft - fr;
            diff_sq_sum += diff * diff;
        }

        total_mse += diff_sq_sum;

        aligned_data.push(ReconstructionData2dPoint {
            true_points: final_true,
            reconstructed_points: final_recon,
            unscaled_points: None,
        });
    }

    (aligned_data, total_mse / n as f64)
}

pub fn export_to_geojson(data: Vec<Vec<f64>>, output_path: &str) -> Result<(), Box<dyn Error>> {
    let mut features = Vec::new();

    // 2. Map your data to standard GeoJSON features
    for (index, item) in data.iter().enumerate() {
        // GeoJSON strictly requires [longitude, latitude]
        // Ensure your f64 tuples are ordered correctly here!

        // Feature B: Reconstructed Point
        features.push(json!({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [item[0], item[1]]
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

#[derive(Deserialize)]
struct Metadata {
    offset_lon: i64,
    offset_lat: i64,
}

pub fn process_and_map_points(
    metadata_path: &str,
    csv_path: &str,
    input_data: Vec<ReconstructionData2dPoint>,
) -> Result<Vec<ReconstructionData2dPoint>, Box<dyn Error>> {
    // 1. Load offsets from metadata.json
    let meta_str = fs::read_to_string(metadata_path)?;
    let meta: Metadata = serde_json::from_str(&meta_str)?;
    let offset_lon = meta.offset_lon;
    let offset_lat = meta.offset_lat;

    // 2. Build lookup map: raw_map[(lon_int, lat_int)] -> HashSet<(lon_str, lat_str)>
    let mut raw_map: HashMap<(i64, i64), HashSet<(String, String)>> = HashMap::new();

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(csv_path)?;

    let headers = rdr.headers()?.clone();
    let laenge_idx = headers.iter().position(|h| h == "Laenge").unwrap_or(0);
    let breite_idx = headers.iter().position(|h| h == "Breite").unwrap_or(1);

    for result in rdr.records() {
        let record = result?;

        if let (Some(lon_str), Some(lat_str)) = (record.get(laenge_idx), record.get(breite_idx)) {
            let lon_trim = lon_str.trim();
            let lat_trim = lat_str.trim();

            if !lon_trim.is_empty() && !lat_trim.is_empty() {
                if let (Ok(lon_f), Ok(lat_f)) = (lon_trim.parse::<f64>(), lat_trim.parse::<f64>()) {
                    let lon_int = (lon_f * 100.0) as i64;
                    let lat_int = (lat_f * 100.0) as i64;

                    raw_map
                        .entry((lon_int, lat_int))
                        .or_insert_with(HashSet::new)
                        .insert((lon_trim.to_string(), lat_trim.to_string()));
                }
            }
        }
    }

    // 3. Map inputs, calculate averages, and build the new Struct Vec
    let mut processed_results = Vec::with_capacity(input_data.len());

    for entry in input_data {
        // Only process entries that have unscaled points
        if let Some(unscaled_x_unscaled_y ) = entry.unscaled_points {
            let unscaled_x = unscaled_x_unscaled_y[0];
            let unscaled_y = unscaled_x_unscaled_y[1];

            let orig_lon = (unscaled_x as i64) + offset_lon;
            let orig_lat = (unscaled_y as i64) + offset_lat;

            // Look up the matching strings
            if let Some(matched_strings) = raw_map.get(&(orig_lon, orig_lat)) {
                if matched_strings.is_empty() {
                    continue; // Skip if no original points map back (matches Python logic)
                }

                let mut sum_x = 0.0;
                let mut sum_y = 0.0;
                let mut count = 0;

                // Parse strings back to floats and sum them up
                for (lon_str, lat_str) in matched_strings {
                    if let (Ok(lon), Ok(lat)) = (lon_str.parse::<f64>(), lat_str.parse::<f64>()) {
                        sum_x += lon;
                        sum_y += lat;
                        count += 1;
                    }
                }

                // Calculate average and push new struct
                if count > 0 {
                    let avg_x = sum_x / (count as f64);
                    let avg_y = sum_y / (count as f64);

                    processed_results.push(ReconstructionData2dPoint {
                        true_points: vec![avg_x, avg_y],
                        reconstructed_points: entry.reconstructed_points,
                        unscaled_points: None, // Set to None as requested
                    });
                }
            }
        }
    }

    Ok(processed_results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_procrustes_alignment_nd() {
        // True points: A simple 1x1 square represented as N-D vectors
        let true_pts = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![1.0, 0.0],
        ];

        // Reconstructed points:
        // 1. Scaled by 2.0
        // 2. Rotated by 90 degrees counter-clockwise
        // 3. Translated by x + 5.0, y + 10.0
        let recon_pts = vec![
            vec![5.0, 10.0],
            vec![3.0, 10.0],
            vec![3.0, 12.0],
            vec![5.0, 12.0],
        ];

        let data: Vec<ReconstructionData2dPoint> = true_pts
            .into_iter()
            .zip(recon_pts.into_iter())
            .map(|(t, r)| ReconstructionData2dPoint {
                true_points: t,
                reconstructed_points: r,
                unscaled_points: None,
            })
            .collect();

        // Run the alignment
        let (aligned_data, mse) = procrustes_align(&data, true, true, true);

        assert!(mse < 1e-10, "MSE is not zero: {}", mse);

        // Verify each aligned point matches the original true point perfectly
        for p in aligned_data {
            let dx = (p.true_points[0] - p.reconstructed_points[0]).abs();
            let dy = (p.true_points[1] - p.reconstructed_points[1]).abs();

            assert!(
                dx < 1e-10,
                "X mismatch: true {}, aligned {}",
                p.true_points[0],
                p.reconstructed_points[0]
            );
            assert!(
                dy < 1e-10,
                "Y mismatch: true {}, aligned {}",
                p.true_points[1],
                p.reconstructed_points[1]
            );
        }
    }
}
