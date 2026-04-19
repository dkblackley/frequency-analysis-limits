use crate::plotting::ReconstructionDataPoint;
use log::{debug, info};
use nalgebra::DMatrix;
use serde::Deserialize;
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fs;
use std::fs::File;
use std::io::{BufReader, Write};

fn quick_convert(file_path: &str, out_path: &str) {
    // Pre-allocate the vectors using the length of the hashmap to avoid reallocations
    let file = File::open(file_path).unwrap();
    let reader = BufReader::new(file);
    let data: Vec<ReconstructionDataPoint> = serde_json::from_reader(reader).unwrap();

    let recon_points_vec: Vec<Vec<f64>> = data
        .iter()
        .map(|point| point.reconstructed_points.clone())
        .collect();

    export_to_geojson(recon_points_vec, out_path).unwrap();
}

pub fn export_to_geo_and_align(
    _dir: &str,
    remin_path: &str,
    less_path: &str,
    unique_name: &str,
    out_path: &str,
    db_name: &str,
) {
    // let dir = &args.dir_path;
    // let remin_path = format!("{dir}/{0}/remin", args.name,);
    // let less_path = format!("{dir}/{0}/even_less", args.name,);
    // let unique_name = format!("{0}_prob{1}.0_{2}", args.name, args.percent, args.dist);

    let unique_rem_name = format!("{unique_name}_classic.json");
    let unique_less_name = format!("{unique_name}_even_less.json");

    let remin_res = format!("{remin_path}/{unique_rem_name}");
    let even_less_res = format!("{less_path}/{unique_less_name}");

    debug!("About to load data {remin_res}, {even_less_res}");

    let content = fs::read_to_string(&remin_res).unwrap();
    let remin_data: Vec<ReconstructionDataPoint> = serde_json::from_str(&content).unwrap();

    let content = fs::read_to_string(&even_less_res).unwrap();
    let less_data: Vec<ReconstructionDataPoint> = serde_json::from_str(&content).unwrap();

    let aligned = procrustes_align(&*remin_data, true, true, true);
    info!("MSE of Remin (Original): {:?}", aligned.1);

    let aligned = procrustes_align(&*less_data, true, true, true);
    info!("MSE of Even Less (Original): {:?}", aligned.1);

    let out = procrustes_align(&*remin_data, true, true, true);
    let new_vec = out.0;
    let _mse = out.1;

    let mut output_file =
        File::create(format!("{0}{1}/reconstruction.json", out_path, db_name).as_str()).unwrap();
    let procrus = serde_json::to_string_pretty(&new_vec).unwrap();
    output_file.write_all(procrus.as_bytes()).unwrap();

    quick_convert(
        format!("{0}{1}/reconstruction.json", out_path, db_name).as_str(),
        format!(
            "{0}{1}/reconstruction_{2}.geojson",
            out_path, db_name, db_name
        )
        .as_str(),
    )
}

fn print_min_val_2d(points: &Vec<Vec<f64>>) {
    let (min_x, max_x, min_y, max_y) = points.iter().fold(
        (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ),
        |(min_x, max_x, min_y, max_y), x_y| {
            let (x, y) = (x_y[0], x_y[1]);
            (min_x.min(x), max_x.max(x), min_y.min(y), max_y.max(y))
        },
    );

    println!("X: min={}, max={}", min_x, max_x);
    println!("Y: min={}, max={}", min_y, max_y);
}

// "fix" the spitz DB to the original.
pub fn do_spitz_align(
    remin_data: Vec<ReconstructionDataPoint>,
    remin_path: &str,
    unique_rem_name: &str,
    less_data: Vec<ReconstructionDataPoint>,
    less_path: &str,
    unique_less_name: &str,
    dir: &str,
) {
    let spitz_orig = "/home/yelnat/Nextcloud/10TB-STHDD/datasets/freq_an/graw_drawing".to_string();
    let lat_long_truth = process_and_map_points(
        &format!("{spitz_orig}/metadata.json"),
        &format!("{spitz_orig}/Spitz.csv"),
        remin_data.clone(),
    )
    .unwrap();

    // For debug purposes/evgenios request
    print_min_val_2d(
        &lat_long_truth
            .iter()
            .map(|point| point.true_points.clone())
            .collect(),
    );

    let aligned = procrustes_align(&*lat_long_truth, true, true, true);

    info!("MSE of Remin (On map): {:?}", aligned.1);

    let out_path = format!("{remin_path}/{unique_rem_name}.geojson");
    let recon_points_vec: Vec<Vec<f64>> = aligned
        .0
        .iter()
        .map(|point| point.reconstructed_points.clone())
        .collect();

    export_to_geojson(recon_points_vec, &out_path).unwrap();

    let true_points: Vec<Vec<f64>> = aligned
        .0
        .iter()
        .map(|point| point.true_points.clone())
        .collect();
    let true_path = format!("{dir}spitz/true.geojson");
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

    let recon_points_vec: Vec<Vec<f64>> = aligned
        .0
        .iter()
        .map(|point| point.reconstructed_points.clone())
        .collect();
    export_to_geojson(recon_points_vec, &out_path).unwrap();

    info!("Saved lili to geojson");
}

/// Calculates the standard Mean Squared Error (MSE) across N dimensions
pub fn calculate_mse(data: &[ReconstructionDataPoint]) -> f64 {
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
    data: &[ReconstructionDataPoint],
    _: bool,
    _: bool,
    _: bool,
) -> (Vec<ReconstructionDataPoint>, f64) {
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

    let n_f64 = n as f64;
    for i in 0..d {
        mean_true[i] /= n_f64;
        mean_recon[i] /= n_f64;
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
        let mut trace_sigma = 0.0;
        for i in 0..d {
            let sign = if i == d - 1 { d_sign } else { 1.0 };
            trace_sigma += svd.singular_values[i] * sign;
        }
        s_factor = trace_sigma / var_recon;
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
            // Because we always shift, `ft` maps back to the original point,
            // and `fr` adds the true mean back to the scaled/rotated point.
            let ft = p.true_points[col];
            let fr = transformed_y[(0, col)] + mean_true[col];

            final_true[col] = ft;
            final_recon[col] = fr;

            let diff = ft - fr;
            diff_sq_sum += diff * diff;
        }

        total_mse += diff_sq_sum;

        aligned_data.push(ReconstructionDataPoint {
            true_points: final_true,
            reconstructed_points: final_recon,
            unscaled_points: None,
        });
    }

    (aligned_data, total_mse / n_f64)
}

pub fn scale_to_absolute_range(
    data: &[ReconstructionDataPoint],
    target_range: (f64, f64),
) -> Vec<ReconstructionDataPoint> {
    if data.is_empty() {
        return Vec::new();
    }

    let (target_min, target_max) = target_range;
    let target_spread = target_max - target_min;

    // 1. Find the min and max bounds for X and Y in the reconstructed data
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for p in data {
        if p.reconstructed_points.len() >= 2 {
            let x = p.reconstructed_points[0];
            let y = p.reconstructed_points[1];

            if x < min_x {
                min_x = x;
            }
            if x > max_x {
                max_x = x;
            }
            if y < min_y {
                min_y = y;
            }
            if y > max_y {
                max_y = y;
            }
        }
    }

    let range_x = max_x - min_x;
    let range_y = max_y - min_y;

    // 2. Map the points strictly into the [target_min, target_max] box
    let mut scaled_data = Vec::with_capacity(data.len());

    for p in data {
        let mut new_p = p.clone(); // Clone the entire struct to preserve other fields

        if new_p.reconstructed_points.len() >= 2 {
            let x = new_p.reconstructed_points[0];
            let y = new_p.reconstructed_points[1];

            // Scale X
            let scaled_x = if range_x > 0.0 {
                target_min + ((x - min_x) * target_spread) / range_x
            } else {
                // If there's no variance, center the point in the target range
                target_min + (target_spread / 2.0)
            };

            // Scale Y
            let scaled_y = if range_y > 0.0 {
                target_min + ((y - min_y) * target_spread) / range_y
            } else {
                target_min + (target_spread / 2.0)
            };

            new_p.reconstructed_points[0] = scaled_x;
            new_p.reconstructed_points[1] = scaled_y;
        }

        scaled_data.push(new_p);
    }

    scaled_data
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
    input_data: Vec<ReconstructionDataPoint>,
) -> Result<Vec<ReconstructionDataPoint>, Box<dyn Error>> {
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
        if let Some(unscaled_x_unscaled_y) = entry.unscaled_points {
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

                    processed_results.push(ReconstructionDataPoint {
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

        let data: Vec<ReconstructionDataPoint> = true_pts
            .into_iter()
            .zip(recon_pts.into_iter())
            .map(|(t, r)| ReconstructionDataPoint {
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
    #[test]
    fn test_scale_to_absolute_range() {
        // 1. Create mock data with values wildly outside the 0.0 - 50.0 range
        let mock_data = vec![
            ReconstructionDataPoint {
                true_points: vec![],
                reconstructed_points: vec![-999.0, 5000.0],
                unscaled_points: None,
            },
            ReconstructionDataPoint {
                true_points: vec![],
                reconstructed_points: vec![1234.5, -42.0],
                unscaled_points: None,
            },
            ReconstructionDataPoint {
                true_points: vec![],
                reconstructed_points: vec![0.0, 0.0],
                unscaled_points: None,
            },
        ];

        let target_min = 0.0;
        let target_max = 50.0;
        let target_range = (target_min, target_max);

        // 2. Scale the data
        let scaled_data = scale_to_absolute_range(&mock_data, target_range);

        // 3. Verify the output
        // We use a tiny epsilon to account for f64 floating-point inaccuracies
        let epsilon = 1e-10;

        for (i, p) in scaled_data.iter().enumerate() {
            let x = p.reconstructed_points[0];
            let y = p.reconstructed_points[1];

            // Print the points to the console when running `cargo test -- --nocapture`
            println!("Point {}: X = {}, Y = {}", i, x, y);

            assert!(
                x >= target_min - epsilon && x <= target_max + epsilon,
                "Point {} X value ({}) is out of bounds!",
                i,
                x
            );

            assert!(
                y >= target_min - epsilon && y <= target_max + epsilon,
                "Point {} Y value ({}) is out of bounds!",
                i,
                y
            );
        }
    }
}
