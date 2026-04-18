use crate::plotting::post::{calculate_mse, procrustes_align};
use crate::plotting::two_d::{format_db_name, format_dist_name};
use crate::plotting::ReconstructionDataPoint;
use indicatif::ParallelProgressIterator;
use itertools::{iproduct, max};
use log::debug;
use plotters::prelude::*;
use plotters::style::FontStyle;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::fmt::format;
use std::fs;

pub fn plot_histograms_of_all_reconstructions(
    name: &str,
    domain: (usize, usize),
    search_domain: (usize, usize),
    dist: &str,
) -> Result<(), Box<dyn Error>> {
    let grid = format!("{}x{}", domain.0, domain.1);
    let path_to_root = format!("databases/{grid}/{name}");

    // let shift_step = 5.0;
    // let scale_step = 5.0;
    // let rotate = 5.0;

    let shift_step = 1.0;
    let scale_step = 1.0;
    let rotate = 10.0;

    let path = format!("{path_to_root}/even_less/{name}_prob100.0_{dist}_{grid}_even_less.json");
    debug!("About to load data from {}", &path);
    let content = fs::read_to_string(&path)?;
    let all_data: Vec<ReconstructionDataPoint> = serde_json::from_str(&content)?;
    let even_less_best = procrustes_align(&all_data, true, true, true).1;
    let even_less_data = combined_search(
        &all_data,
        search_domain,
        shift_step,
        rotate,
        search_domain,
        scale_step,
    );

    let path = format!("{path_to_root}/remin/{name}_prob100.0_{dist}_{grid}_classic.json");
    debug!("About to load data from {}", &path);
    let content = fs::read_to_string(&path)?;
    let all_data: Vec<ReconstructionDataPoint> = serde_json::from_str(&content)?;
    let remin_best = procrustes_align(&all_data, true, true, true).1;

    let remin_data = combined_search(
        &all_data,
        search_domain,
        shift_step,
        rotate,
        search_domain,
        scale_step,
    );

    let path = format!("{path_to_root}/limits/{name}_{dist}_e0_d0.9_reconstruction.json");
    debug!("About to load data from {}", &path);
    let content = fs::read_to_string(&path)?;
    let all_data: Vec<Vec<ReconstructionDataPoint>> = serde_json::from_str(&content)?;

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
    remin_mse.push(remin_best);

    for tup in even_less_data {
        let mse = tup.0;
        even_less_mse.push(mse);
    }
    even_less_mse.push(even_less_best);

    let len = even_less_mse.len();
    let best = even_less_mse[..10].to_vec();
    let worst = even_less_mse[len - 10..].to_vec();
    let middle_sampled = sample_uniformly(&even_less_mse, 10_000);

    let mut final_less = Vec::with_capacity(11 + 10_000 + 10);
    final_less.extend_from_slice(&best);
    final_less.extend(middle_sampled);
    final_less.extend_from_slice(&worst);

    let len = remin_mse.len();
    let best = remin_mse[..10].to_vec();
    let worst = remin_mse[len - 10..].to_vec();
    let middle_sampled = sample_uniformly(&remin_mse, 10_000);

    let mut final_remin = Vec::with_capacity(11 + 10_000 + 10);
    final_remin.extend_from_slice(&best);
    final_remin.extend(middle_sampled);
    final_remin.extend_from_slice(&worst);

    let mut methods_to_mse = HashMap::new();
    methods_to_mse.insert("Remin".to_string(), (remin_mse, remin_best));
    methods_to_mse.insert("Even Less".to_string(), (even_less_mse, even_less_best));
    methods_to_mse.insert("LAMa".to_string(), (limits_mse, 0.0));

    debug!("About to plot MSE frequency histogram");
    debug!(
        "Limits MSE for {dist}: {:?}",
        methods_to_mse.get("LAMa").unwrap()
    );
    plot_mse_frequency_histogram_split(
        &methods_to_mse,
        format_db_name(name),
        format_dist_name(dist),
        &format!("figures/mse_histogram_{name}_{dist}.svg"),
    )?;

    Ok(())
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
        .filter_map(|(dx, dy, angle, sx, sy)| {
            let x_wiggle = sx / shift_domain.0 as f64;
            let y_wiggle = sy / shift_domain.1 as f64;

            // if the amount we're trying to move along the x/y axis after a shift would throw us
            // into an invalid spot, skip this one.
            if dx > x_wiggle || dy > y_wiggle {
                return None;
            }

            let mse = evaluate_transform(data, dx, dy, angle, sx, sy);
            Some((mse, dx, dy, angle, sx, sy))
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

/// Helper function to format large numbers with 'k' or 'M' suffixes.
fn format_metric(val: f64) -> String {
    if val == 0.0 {
        return "0".to_string();
    }
    let abs_val = val.abs();
    if abs_val >= 1_000_000_000.0 {
        format!("{:.1}B", val / 1_000_000_000.0).replace(".0B", "B")
    } else if abs_val >= 1_000_000.0 {
        format!("{:.1}M", val / 1_000_000.0).replace(".0M", "M")
    } else if abs_val >= 1_000.0 {
        format!("{:.1}k", val / 1_000.0).replace(".0k", "K")
    } else {
        format!("{:.0}", val) // Standard whole number for anything < 1000
    }
}

/// Generates a Histogram split into individual side-by-side plots.
/// - Features separate bars with clean gaps.
/// - Formats large axis numbers with metric suffixes (k, M).
pub fn plot_mse_frequency_histogram_split(
    mse_data: &HashMap<String, (Vec<f64>, f64)>,
    database_name: &str,
    dist_name: &str,
    output_path: &str,
) -> Result<(), Box<dyn Error>> {
    let num_plots = mse_data.len().max(1);

    let root = SVGBackend::new(output_path, (600 * num_plots as u32, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    if mse_data.is_empty() {
        return Ok(());
    }

    // 1. Vertical Split for Super Title
    let (title_area, plot_area) = root.split_vertically(10);

    let text_color = BLACK;
    let font_family = "Linux Biolinum";
    let super_title_font = (font_family, 40, FontStyle::Bold)
        .into_font()
        .color(&text_color);

    ChartBuilder::on(&title_area)
        .caption(
            format!("{} {} Distribution", database_name, dist_name),
            super_title_font,
        )
        .build_cartesian_2d(0f32..1f32, 0f32..1f32)?;

    let num_bins = 8;

    // Typography
    let label_font = (font_family, 24).into_font().color(&text_color);
    let axis_font = (font_family, 26, FontStyle::Bold)
        .into_font()
        .color(&text_color);
    let title_font = (font_family, 32, FontStyle::Bold)
        .into_font()
        .color(&text_color);

    let palette = [
        RGBColor(230, 159, 0),   // Orange
        RGBColor(86, 180, 233),  // Sky Blue
        RGBColor(0, 158, 115),   // Bluish Green
        RGBColor(240, 228, 66),  // Yellow
        RGBColor(0, 114, 178),   // Blue
        RGBColor(213, 94, 0),    // Vermilion
        RGBColor(204, 121, 167), // Reddish Purple
    ];

    let mut alpha_methods: Vec<&String> = mse_data.keys().collect();
    alpha_methods.sort();

    let mut color_map = HashMap::new();
    for (i, method) in alpha_methods.iter().enumerate() {
        color_map.insert(method.to_string(), palette[i % palette.len()]);
    }

    let mut plot_methods = alpha_methods.clone();
    plot_methods.sort_by_key(|&m| match m.as_str() {
        "LAMa" => 0,
        "Even Less" => 1,
        "Remin" => 2,
        _ => 3,
    });

    let panels = plot_area.split_evenly((1, num_plots));

    for (i, method) in plot_methods.iter().enumerate() {
        let color = color_map[*method];
        let mses = &mse_data[*method].0;
        let panel = &panels[i];

        let min_mse = 0.0f64;
        let mut local_max_mse = 0.0f64;

        for &mse in mses {
            if mse > local_max_mse {
                local_max_mse = mse;
            }
        }

        let is_zero_mse = local_max_mse <= 0.0;

        if is_zero_mse {
            local_max_mse = 1.0;
        } else {
            local_max_mse *= 1.05;
        }

        let bin_width = (local_max_mse - min_mse) / num_bins as f64;

        // Bucketize Data
        let mut bins = vec![0; num_bins];
        for &mse in mses {
            let mut bin_idx = ((mse - min_mse) / bin_width).floor() as usize;
            if bin_idx >= num_bins {
                bin_idx = num_bins - 1;
            }
            bins[bin_idx] += 1;
        }

        let local_max_freq = *bins.iter().max().unwrap_or(&0);
        let max_y = ((local_max_freq as f64 * 1.1).ceil() as usize).max(1);

        // Dynamically scale the label area size to pull the title tighter on LAMa
        let y_label_area = if *method == "LAMa" { 50 } else { 70 };

        // Render Chart
        let mut chart = ChartBuilder::on(panel)
            .margin_top(40)
            .margin_bottom(40)
            .margin_left(40)
            // Centering fix: Expand the right margin to perfectly mirror the space taken up
            // by the left Y-axis labels. This perfectly centers the plot grid without drawing an axis.
            .margin_right(40 + y_label_area)
            .caption(*method, title_font.clone())
            .x_label_area_size(55)
            .y_label_area_size(y_label_area)
            .build_cartesian_2d(min_mse..local_max_mse, 0..max_y)?;

        chart
            .configure_mesh()
            .disable_x_mesh()
            .x_labels(8) // Restored normal label count
            .y_desc("Number of Solutions")
            .x_desc("Mean Squared Error (MSE)")
            .x_label_formatter(&|x| {
                // Specific IF statement for the 0-only scenario: forces all labels to be "0"
                if is_zero_mse {
                    "0".to_string()
                } else {
                    format_metric(*x)
                }
            })
            .y_label_formatter(&|y| format_metric(*y as f64))
            .label_style(label_font.clone())
            .axis_desc_style(axis_font.clone())
            .light_line_style(WHITE.mix(0.0))
            .bold_line_style(BLACK.mix(0.1))
            .draw()?;

        // Calculate a small gap width
        let gap = bin_width * 0.05;

        for (bin_idx, &count) in bins.iter().enumerate() {
            if count == 0 {
                continue;
            }

            let x_start = min_mse + (bin_idx as f64) * bin_width;
            let x_end = min_mse + ((bin_idx + 1) as f64) * bin_width;

            chart.draw_series(std::iter::once(Rectangle::new(
                [(x_start + gap, 0), (x_end - gap, count)],
                color.mix(0.5).filled(),
            )))?;

            chart.draw_series(std::iter::once(Rectangle::new(
                [(x_start + gap, 0), (x_end - gap, count)],
                color.stroke_width(2),
            )))?;
        }
    }

    root.present()?;
    Ok(())
}
