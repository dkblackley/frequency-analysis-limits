use crate::plotting::post::calculate_mse;
use crate::plotting::two_d::{format_db_name, format_dist_name};
use crate::plotting::ReconstructionDataPoint;
use log::debug;
use plotters::prelude::*;
use plotters::style::FontStyle;
use std::collections::HashMap;
use std::error::Error;
use std::fs;

fn format_metric(val: f64) -> String {
    if val == 0.0 {
        return "0".to_string();
    }

    let abs_val = val.abs();

    if abs_val >= 1_000_000_000.0 {
        let scaled = val / 1_000_000_000.0;
        if scaled.abs() < 10.0 {
            format!("{:.1}B", scaled).replace(".0B", "B")
        } else {
            format!("{:.0}B", scaled)
        }
    } else if abs_val >= 1_000_000.0 {
        let scaled = val / 1_000_000.0;
        if scaled.abs() < 10.0 {
            format!("{:.1}M", scaled).replace(".0M", "M")
        } else {
            format!("{:.0}M", scaled)
        }
    } else if abs_val >= 1_000.0 {
        let scaled = val / 1_000.0;
        if scaled.abs() < 10.0 {
            format!("{:.1}K", scaled).replace(".0K", "K")
        } else {
            format!("{:.0}K", scaled)
        }
    } else if abs_val >= 100.0 {
        // Round to the nearest 100
        format!("{:.0}", (val / 100.0).round() * 100.0)
    } else if abs_val >= 10.0 {
        // Round to the nearest 10
        format!("{:.0}", (val / 10.0).round() * 10.0)
    } else {
        // Round to the nearest whole number for anything under 10
        format!("{:.0}", val.round())
    }
}

pub fn plot_lama_distributions(name: &str) -> Result<(), Box<dyn Error>> {
    // Hardcoded to 15x15 as requested
    let grid = "15x15";
    let path_to_root = format!("databases/{grid}/{name}");

    // The three distributions we want to plot for LAMa
    let distributions = vec!["uniform", "gaussian", "flat"];

    let mut dist_to_mse: HashMap<String, Vec<f64>> = HashMap::new();

    // 1. Load the Data
    for dist in &distributions {
        // Formats to: databases/15x15/busstop/limits/busstop_flat_e0_d0.001_reconstruction.json
        let path = format!("{path_to_root}/limits/{name}_{dist}_e0_d0.001_reconstruction.json");
        debug!("About to load LAMa data from {}", &path);

        let content = fs::read_to_string(&path)?;
        let all_data: Vec<Vec<ReconstructionDataPoint>> = serde_json::from_str(&content)?;

        let mut limits_mse = Vec::new();
        for reconstruction_data in all_data {
            let mse = calculate_mse(&*reconstruction_data);
            limits_mse.push(mse);
        }

        dist_to_mse.insert(dist.to_string(), limits_mse);
    }

    // 2. Setup the Canvas
    let num_plots = distributions.len().max(1);
    let output_path = format!("figures/histogram_flat/histogram_{name}_lama_flat.svg");

    let root = SVGBackend::new(&output_path, (450 * num_plots as u32, 400)).into_drawing_area();
    root.fill(&WHITE)?;

    // 3. Vertical Split for Super Title
    let (title_area, plot_area) = root.split_vertically(0);

    let text_color = BLACK;
    let font_family = "Linux Biolinum";
    let super_title_font = (font_family, 50, FontStyle::Bold)
        .into_font()
        .color(&text_color);

    ChartBuilder::on(&title_area)
        .caption(format!("{}", format_db_name(name)), super_title_font)
        .build_cartesian_2d(0f32..1f32, 0f32..1f32)?;

    let num_bins = 8;

    // Typography
    let label_font = (font_family, 42).into_font().color(&text_color);
    let axis_font = (font_family, 50, FontStyle::Bold)
        .into_font()
        .color(&text_color);
    let title_font = (font_family, 50, FontStyle::Bold)
        .into_font()
        .color(&text_color);

    // Hardcode LAMa color (Sky Blue from your existing palette)
    let lama_color = RGBColor(86, 180, 233);

    let panels = plot_area.split_evenly((1, num_plots));

    // 4. Render Subplots
    for (i, dist) in distributions.iter().enumerate() {
        let mses = dist_to_mse.get(*dist).unwrap();
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

        let mut chart = ChartBuilder::on(panel)
            .margin_top(50)
            .margin_bottom(0)
            .margin_left(25)
            .margin_right(25)
            .caption(format_dist_name(dist), title_font.clone())
            .x_label_area_size(90)
            .y_label_area_size(100) // Standardized gap
            .build_cartesian_2d(0.0f64..(num_bins as f64), 0..max_y)?;

        chart
            .configure_mesh()
            .disable_x_mesh()
            .x_labels(5)
            .y_labels(6)
            .y_desc("# of Solutions")
            .x_desc("MSE")
            .x_label_formatter(&|x| {
                let real_x = min_mse + (*x * bin_width);
                if is_zero_mse {
                    "0".to_string()
                } else {
                    format_metric(real_x)
                }
            })
            .y_label_formatter(&|y| format_metric(*y as f64))
            .label_style(label_font.clone())
            .axis_desc_style(axis_font.clone())
            .light_line_style(WHITE.mix(0.0))
            .bold_line_style(BLACK.mix(0.1))
            .draw()?;

        let gap = 0.05;

        for (bin_idx, &count) in bins.iter().enumerate() {
            if count == 0 {
                continue;
            }

            let x_start = bin_idx as f64;
            let x_end = (bin_idx + 1) as f64;

            // Fill
            chart.draw_series(std::iter::once(Rectangle::new(
                [(x_start + gap, 0), (x_end - gap, count)],
                lama_color.mix(0.5).filled(),
            )))?;

            // Stroke
            chart.draw_series(std::iter::once(Rectangle::new(
                [(x_start + gap, 0), (x_end - gap, count)],
                lama_color.stroke_width(2),
            )))?;
        }
    }

    root.present()?;
    Ok(())
}
