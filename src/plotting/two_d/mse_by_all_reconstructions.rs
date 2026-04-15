use crate::plotting::post::calculate_mse;
use crate::plotting::ReconstructionDataPoint;
use indicatif::ParallelProgressIterator;
use itertools::iproduct;
use log::debug;
use plotters::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::fmt::format;
use std::fs;

pub fn plot_histograms_of_all_reconstructions() {
    let name = "spitz";
    let path_to_root = format!("databases/350x50/{name}");

    let shift_step = 1.0;
    let scale_step = 10.0;
    let rotate = 30.0;

    let path = format!("{path_to_root}/even_less/{name}_prob100.0_uniform_350x50_even_less.json");
    debug!("About to load data from {}", &path);
    let content = fs::read_to_string(&path).unwrap();
    let all_data: Vec<ReconstructionDataPoint> = serde_json::from_str(&content).unwrap();
    let even_less_data = combined_search(
        &all_data,
        (350, 50),
        shift_step,
        rotate,
        (350, 50),
        scale_step,
    );

    let path = format!("{path_to_root}/remin/{name}_prob100.0_uniform_350x50_classic.json");
    debug!("About to load data from {}", &path);
    let content = fs::read_to_string(&path).unwrap();
    let all_data: Vec<ReconstructionDataPoint> = serde_json::from_str(&content).unwrap();
    let remin_data = combined_search(
        &all_data,
        (350, 50),
        shift_step,
        rotate,
        (350, 50),
        scale_step,
    );

    let path = format!("{path_to_root}/limits/{name}_uniform_e0_d0.9_reconstruction.json");
    debug!("About to load data from {}", &path);
    let content = fs::read_to_string(&path).unwrap();
    let all_data: Vec<Vec<ReconstructionDataPoint>> = serde_json::from_str(&content).unwrap();

    let mut limits_mse = Vec::new();
    let mut remin_mse = Vec::new();
    let mut even_less_mse = Vec::new();

    for reconstruction_data in all_data {
        let mse = calculate_mse(&*reconstruction_data);
        limits_mse.push(mse);
    }

    for tup in remin_data {
        let mse = tup.0;
        remin_mse.push(mse);
    }

    for tup in even_less_data {
        let mse = tup.0;
        even_less_mse.push(mse);
    }

    let len = even_less_mse.len();
    let best = even_less_mse[..10].to_vec();
    let worst = even_less_mse[len - 10..].to_vec();
    let middle_sampled = sample_uniformly(&even_less_mse, 10_000);

    let mut final_less = Vec::with_capacity(10 + 10_000 + 10);
    final_less.extend_from_slice(&best);
    final_less.extend(middle_sampled);
    final_less.extend_from_slice(&worst);

    let len = remin_mse.len();
    let best = remin_mse[..10].to_vec();
    let worst = remin_mse[len - 10..].to_vec();
    let middle_sampled = sample_uniformly(&remin_mse, 10_000);

    let mut final_remin = Vec::with_capacity(10 + 10_000 + 10);
    final_remin.extend_from_slice(&best);
    final_remin.extend(middle_sampled);
    final_remin.extend_from_slice(&worst);

    let mut methods_to_mse = HashMap::new();
    methods_to_mse.insert("Remin".to_string(), (final_remin, 1892.0));
    methods_to_mse.insert("Even Less".to_string(), (final_less, 2813.0));
    methods_to_mse.insert("LAMA".to_string(), (limits_mse, 0.0));

    // saving as a .svg can be very big.....
    debug!("About to plot lines");
    plot_normalized_ranked_line_with_circles(&methods_to_mse, "figures/mse_distributions_line.svg")
        .unwrap();
    debug!("About to plot step");
    plot_line_step_mse(&methods_to_mse, "figures/mse_distributions_step.svg").unwrap();
    debug!("About to plot step filled");
    plot_filled_step_mse(&methods_to_mse, "figures/mse_distributions_step_filled.svg").unwrap();
}

/// Uniformly samples a slice of f64 down to `num_samples` items.
pub fn sample_uniformly(data: &[f64], num_samples: usize) -> Vec<f64> {
    let len = data.len();

    // Handle edge cases
    if num_samples == 0 || len == 0 {
        return Vec::new();
    }
    if len <= num_samples {
        return data.to_vec();
    }
    if num_samples == 1 {
        return vec![data[0]];
    }

    // Calculate the floating-point step between indices
    let step = (len - 1) as f64 / (num_samples - 1) as f64;

    // Map each target sample to its nearest original index
    (0..num_samples)
        .map(|i| {
            let idx = (i as f64 * step).round() as usize;
            // .min() acts as a safety guard against floating point rounding overflow
            data[idx.min(len - 1)]
        })
        .collect()
}

#[inline]
fn evaluate_and_transform(
    data: &[ReconstructionDataPoint],
    dx: f64,
    dy: f64,
    angle_deg: f64,
    sx: f64,
    sy: f64,
) -> (f64, Vec<[f64; 2]>) {
    let rad = angle_deg.to_radians();
    let (sin_t, cos_t) = rad.sin_cos();

    // Pre-allocate the vector since we know the exact size needed
    let mut transformed_points = Vec::with_capacity(data.len());
    let mut sum_sq = 0.0;

    for p in data {
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

        // Store the transformed point
        transformed_points.push([final_x, final_y]);

        // Calculate the error
        let diff_x = tx - final_x;
        let diff_y = ty - final_y;
        sum_sq += diff_x * diff_x + diff_y * diff_y;
    }

    let mse = sum_sq / data.len() as f64;

    (mse, transformed_points)
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
) -> Vec<(f64, f64, f64, f64, f64, f64)> {
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
    let mut evaluated: Vec<_> = grid
        .par_bridge()
        .progress_with(progress)
        .map(|(dx, dy, angle, sx, sy)| {
            let mse = evaluate_transform(data, dx, dy, angle, sx, sy);
            (mse, dx, dy, angle, sx, sy)
        })
        .collect();

    evaluated.par_sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    return evaluated;

    // // Single linear pass to extract MSEs and find the optimal configuration
    // let mut best_config = (f64::MAX, 0.0, 0.0, 0.0, 1.0, 1.0);
    // let mut all_mses = Vec::with_capacity(evaluated.len());
    //
    // // for &result in &evaluated {
    //     all_mses.push(result.0);
    //     if result.0 < best_config.0 {
    //         best_config = result;
    //     }
    // }
    //
    // let (_, best_dx, best_dy, best_angle, best_sx, best_sy) = best_config;
    // let best_vec = apply_transform(data, best_dx, best_dy, best_angle, best_sx, best_sy);
    //
    // (best_vec, all_mses)
}

/// Generates a Ranked line plot with dynamic circles and baseline markers.
/// X-axis is normalized [0, 1], Y-axis is Logarithmic with scientific notation.
pub fn plot_normalized_ranked_line_with_circles(
    mse_data: &HashMap<String, (Vec<f64>, f64)>,
    output_path: &str,
) -> Result<(), Box<dyn Error>> {
    let root = SVGBackend::new(output_path, (1200, 800)).into_drawing_area();
    //let root = BitMapBackend::new(output_path, (1200, 800)).into_drawing_area();
    root.fill(&TRANSPARENT)?;

    if mse_data.is_empty() {
        return Ok(());
    }

    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    // Calculate Y-axis bounds based on both the MSE points and the baseline
    for (mse_list, baseline) in mse_data.values() {
        for &mse in mse_list.iter().chain(std::iter::once(baseline)) {
            let valid_mse = if mse <= 0.0 { 1e-9 } else { mse };
            min_y = min_y.min(valid_mse);
            max_y = max_y.max(valid_mse);
        }
    }

    if min_y == max_y {
        min_y *= 0.5;
        max_y *= 2.0;
    } else {
        min_y *= 0.8;
        max_y *= 1.2;
    }

    let mut chart = ChartBuilder::on(&root)
        .margin(40)
        .x_label_area_size(50)
        .y_label_area_size(100) // Space for scientific notation
        .build_cartesian_2d(0f64..1f64, (min_y..max_y).log_scale())?;

    let text_color = BLACK;
    chart
        .configure_mesh()
        .bold_line_style(RGBColor(220, 220, 220))
        .axis_style(&text_color)
        .x_desc("Normalized Reconstruction Rank (0 = Best, 1 = Worst)")
        .y_desc("Mean Squared Error (Log Scale)")
        .y_label_formatter(&|y| format_scientific(*y))
        .label_style(("sans-serif", 20).into_font().color(&text_color))
        .draw()?;

    let palette = [
        RGBColor(230, 159, 0),   // Orange
        RGBColor(86, 180, 233),  // Sky Blue
        RGBColor(0, 158, 115),   // Bluish Green
        RGBColor(240, 228, 66),  // Yellow
        RGBColor(0, 114, 178),   // Blue
        RGBColor(213, 94, 0),    // Vermilion
        RGBColor(204, 121, 167), // Reddish Purple
    ];

    // Keep color assignment alphabetically stable
    let mut alpha_methods: Vec<&String> = mse_data.keys().collect();
    alpha_methods.sort();
    let mut color_map = HashMap::new();
    for (i, method) in alpha_methods.iter().enumerate() {
        color_map.insert(method.to_string(), palette[i % palette.len()]);
    }

    // Sort render order by average MSE to keep layering consistent
    let mut plot_methods = alpha_methods.clone();
    plot_methods.sort_by(|a, b| {
        let mean_a = mse_data[*a].0.iter().sum::<f64>() / mse_data[*a].0.len() as f64;
        let mean_b = mse_data[*b].0.iter().sum::<f64>() / mse_data[*b].0.len() as f64;
        mean_b
            .partial_cmp(&mean_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for method_name in plot_methods {
        let color = color_map[method_name];
        let mut sorted_mse = mse_data[method_name].0.clone();
        let baseline_val = mse_data[method_name].1;

        sorted_mse.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let num_points = sorted_mse.len();

        // 1. Map all points to lines for a smooth curve
        let line_data: Vec<(f64, f64)> = sorted_mse
            .iter()
            .enumerate()
            .map(|(rank, &mse)| {
                let norm_rank = if num_points > 1 {
                    rank as f64 / (num_points - 1) as f64
                } else {
                    0.5
                };
                let valid_mse = if mse <= 0.0 { 1e-9 } else { mse };
                (norm_rank, valid_mse)
            })
            .collect();

        // Draw the smooth line
        chart
            .draw_series(LineSeries::new(line_data.clone(), color.stroke_width(3)))?
            .label(method_name)
            .legend(move |(x, y)| {
                // Plotters uses the `+` operator to compose multiple elements
                // into a single drawing coordinate.
                EmptyElement::at((x, y))
                    + PathElement::new(vec![(0, 0), (20, 0)], color.stroke_width(3))
                    + Circle::new((10, 0), 4, color.filled())
            });

        // 2. Dynamic Point Placement
        let stride = if num_points > 50 { num_points / 50 } else { 1 };
        let circle_data: Vec<(f64, f64)> = line_data.into_iter().step_by(stride).collect();

        // Draw the cleanly spaced circles over the line
        chart.draw_series(
            circle_data
                .into_iter()
                .map(|(x, y)| Circle::new((x, y), 5, color.filled())),
        )?;

        // 3. Draw horizontal dashed line for baseline
        let valid_baseline = if baseline_val <= 0.0 {
            1e-9
        } else {
            baseline_val
        };
        let dash_len = 1.0 / 80.0; // Break the width (1.0) into 80 segments

        chart.draw_series((0..40).map(|j| {
            let x0 = j as f64 * dash_len * 2.0;
            PathElement::new(
                vec![(x0, valid_baseline), (x0 + dash_len, valid_baseline)],
                color.stroke_width(3),
            )
        }))?;
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .background_style(WHITE.mix(0.9).filled())
        .border_style(BLACK)
        .label_font(("sans-serif", 18).into_font().color(&text_color))
        .draw()?;

    root.present()?;
    Ok(())
}

/// Formats axis ticks using proper academic scientific notation.
/// Transitions to m x 10^e notation for values >= 10^5 or < 10^-4.
fn format_scientific(val: f64) -> String {
    if val <= 0.0 {
        return "0".to_string();
    }
    if val < 1e-4 || val >= 1e5 {
        let e = val.log10().floor() as i32;
        let m = val / 10f64.powi(e);
        format!("{:.1}x10^{}", m, e)
    } else {
        let s = format!("{:.4}", val);
        let s = s.trim_end_matches('0');
        if s.ends_with('.') {
            format!("{}0", s)
        } else {
            s.to_string()
        }
    }
}

/// Generates the "Histogram" style Ranked Step Plot.
/// Includes semi-transparent area fills ordered to prevent burying the lowest curves.
pub fn plot_filled_step_mse(
    mse_data: &HashMap<String, (Vec<f64>, f64)>,
    output_path: &str,
) -> Result<(), Box<dyn Error>> {
    let root = SVGBackend::new(output_path, (1200, 800)).into_drawing_area();
    //let root = BitMapBackend::new(output_path, (1200, 800)).into_drawing_area();
    root.fill(&TRANSPARENT)?;

    if mse_data.is_empty() {
        return Ok(());
    }

    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for (mse_list, baseline) in mse_data.values() {
        for &mse in mse_list.iter().chain(std::iter::once(baseline)) {
            let valid_mse = if mse <= 0.0 { 1e-9 } else { mse };
            min_y = min_y.min(valid_mse);
            max_y = max_y.max(valid_mse);
        }
    }

    if min_y == max_y {
        min_y *= 0.5;
        max_y *= 2.0;
    } else {
        min_y *= 0.8;
        max_y *= 1.2;
    }

    let mut chart = ChartBuilder::on(&root)
        .margin(40)
        .x_label_area_size(50)
        .y_label_area_size(100)
        .build_cartesian_2d(0f64..1f64, (min_y..max_y).log_scale())?;

    let text_color = BLACK;
    chart
        .configure_mesh()
        .bold_line_style(RGBColor(220, 220, 220))
        .axis_style(&text_color)
        .x_desc("Normalized Reconstruction Rank (0 = Best, 1 = Worst)")
        .y_desc("Mean Squared Error (Log Scale)")
        .y_label_formatter(&|y| format_scientific(*y))
        .label_style(("sans-serif", 20).into_font().color(&text_color))
        .draw()?;

    let palette = [
        RGBColor(230, 159, 0),
        RGBColor(86, 180, 233),
        RGBColor(0, 158, 115),
        RGBColor(240, 228, 66),
        RGBColor(0, 114, 178),
        RGBColor(213, 94, 0),
        RGBColor(204, 121, 167),
    ];

    let mut alpha_methods: Vec<&String> = mse_data.keys().collect();
    alpha_methods.sort();
    let mut color_map = HashMap::new();
    for (i, method) in alpha_methods.iter().enumerate() {
        color_map.insert(method.to_string(), palette[i % palette.len()]);
    }

    // Sort rendering order by average MSE. Highest draws first (background), lowest draws last (foreground).
    let mut plot_methods = alpha_methods.clone();
    plot_methods.sort_by(|a, b| {
        let mean_a = mse_data[*a].0.iter().sum::<f64>() / mse_data[*a].0.len() as f64;
        let mean_b = mse_data[*b].0.iter().sum::<f64>() / mse_data[*b].0.len() as f64;
        mean_b
            .partial_cmp(&mean_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for method_name in plot_methods {
        let color = color_map[method_name];
        let mut sorted_mse = mse_data[method_name].0.clone();
        let baseline_val = mse_data[method_name].1;

        sorted_mse.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let num_points = sorted_mse.len();
        let step_width = 1.0 / num_points as f64;
        let mut step_data = Vec::with_capacity(num_points * 2);

        for (rank, &mse) in sorted_mse.iter().enumerate() {
            let valid_mse = if mse <= 0.0 { 1e-9 } else { mse };
            let x_start = rank as f64 * step_width;
            let x_end = (rank + 1) as f64 * step_width;

            step_data.push((x_start, valid_mse));
            step_data.push((x_end, valid_mse));
        }

        // 1. Draw the semi-transparent area fill underneath
        chart.draw_series(
            AreaSeries::new(step_data.clone(), min_y, color.mix(0.3).filled())
                .border_style(TRANSPARENT),
        )?;

        // 2. Draw the solid step line on top
        chart
            .draw_series(LineSeries::new(step_data, color.stroke_width(3)))?
            .label(method_name)
            .legend(move |(x, y)| {
                Rectangle::new([(x, y - 5), (x + 20, y + 5)], color.mix(0.6).filled())
            });

        // 3. Draw horizontal dashed line for baseline
        let valid_baseline = if baseline_val <= 0.0 {
            1e-9
        } else {
            baseline_val
        };
        let dash_len = 1.0 / 80.0;

        chart.draw_series((0..40).map(|j| {
            let x0 = j as f64 * dash_len * 2.0;
            PathElement::new(
                vec![(x0, valid_baseline), (x0 + dash_len, valid_baseline)],
                color.stroke_width(3),
            )
        }))?;
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .background_style(WHITE.mix(0.9).filled())
        .border_style(BLACK)
        .label_font(("sans-serif", 18).into_font().color(&text_color))
        .draw()?;

    root.present()?;
    Ok(())
}

/// Generates the standard Ranked Step Plot.
/// Axes match exactly, but removes all area fills for a clean lines-only look.
pub fn plot_line_step_mse(
    mse_data: &HashMap<String, (Vec<f64>, f64)>,
    output_path: &str,
) -> Result<(), Box<dyn Error>> {
    let root = SVGBackend::new(output_path, (1200, 800)).into_drawing_area();
    //let root = BitMapBackend::new(output_path, (1200, 800)).into_drawing_area();
    root.fill(&TRANSPARENT)?;

    if mse_data.is_empty() {
        return Ok(());
    }

    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for (mse_list, baseline) in mse_data.values() {
        for &mse in mse_list.iter().chain(std::iter::once(baseline)) {
            let valid_mse = if mse <= 0.0 { 1e-9 } else { mse };
            min_y = min_y.min(valid_mse);
            max_y = max_y.max(valid_mse);
        }
    }

    if min_y == max_y {
        min_y *= 0.5;
        max_y *= 2.0;
    } else {
        min_y *= 0.8;
        max_y *= 1.2;
    }

    let mut chart = ChartBuilder::on(&root)
        .margin(40)
        .x_label_area_size(50)
        .y_label_area_size(100)
        .build_cartesian_2d(0f64..1f64, (min_y..max_y).log_scale())?;

    let text_color = BLACK;
    chart
        .configure_mesh()
        .bold_line_style(RGBColor(220, 220, 220))
        .axis_style(&text_color)
        .x_desc("Normalized Reconstruction Rank (0 = Best, 1 = Worst)")
        .y_desc("Mean Squared Error (Log Scale)")
        .y_label_formatter(&|y| format_scientific(*y))
        .label_style(("sans-serif", 20).into_font().color(&text_color))
        .draw()?;

    let palette = [
        RGBColor(230, 159, 0),
        RGBColor(86, 180, 233),
        RGBColor(0, 158, 115),
        RGBColor(240, 228, 66),
        RGBColor(0, 114, 178),
        RGBColor(213, 94, 0),
        RGBColor(204, 121, 167),
    ];

    let mut alpha_methods: Vec<&String> = mse_data.keys().collect();
    alpha_methods.sort();
    let mut color_map = HashMap::new();
    for (i, method) in alpha_methods.iter().enumerate() {
        color_map.insert(method.to_string(), palette[i % palette.len()]);
    }

    // Sort order for lines is less critical without area fills, but keeps rendering consistent
    let mut plot_methods = alpha_methods.clone();
    plot_methods.sort_by(|a, b| {
        let mean_a = mse_data[*a].0.iter().sum::<f64>() / mse_data[*a].0.len() as f64;
        let mean_b = mse_data[*b].0.iter().sum::<f64>() / mse_data[*b].0.len() as f64;
        mean_b
            .partial_cmp(&mean_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for method_name in plot_methods {
        let color = color_map[method_name];
        let mut sorted_mse = mse_data[method_name].0.clone();
        let baseline_val = mse_data[method_name].1;

        sorted_mse.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let num_points = sorted_mse.len();
        let step_width = 1.0 / num_points as f64;
        let mut step_data = Vec::with_capacity(num_points * 2);

        for (rank, &mse) in sorted_mse.iter().enumerate() {
            let valid_mse = if mse <= 0.0 { 1e-9 } else { mse };
            let x_start = rank as f64 * step_width;
            let x_end = (rank + 1) as f64 * step_width;

            step_data.push((x_start, valid_mse));
            step_data.push((x_end, valid_mse));
        }

        // Draw ONLY the step line, no AreaSeries
        chart
            .draw_series(LineSeries::new(step_data, color.stroke_width(3)))?
            .label(method_name)
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(3))
            });

        // Draw horizontal dashed line for baseline
        let valid_baseline = if baseline_val <= 0.0 {
            1e-9
        } else {
            baseline_val
        };
        let dash_len = 1.0 / 80.0;

        chart.draw_series((0..40).map(|j| {
            let x0 = j as f64 * dash_len * 2.0;
            PathElement::new(
                vec![(x0, valid_baseline), (x0 + dash_len, valid_baseline)],
                color.stroke_width(3),
            )
        }))?;
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .background_style(WHITE.mix(0.9).filled())
        .border_style(BLACK)
        .label_font(("sans-serif", 18).into_font().color(&text_color))
        .draw()?;

    root.present()?;
    Ok(())
}
