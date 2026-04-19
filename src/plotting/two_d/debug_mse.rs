use crate::plotting::ReconstructionDataPoint;
use plotters::prelude::*;
use std::error::Error;

// Helper to calculate MSE from the vectors
fn calculate_squared_error(data: &[ReconstructionDataPoint]) -> Vec<f64> {
    data.iter()
        .map(|dp| {
            if dp.true_points.is_empty() {
                panic!();
            }
            let sum_sq: f64 = dp
                .true_points
                .iter()
                .zip(dp.reconstructed_points.iter())
                .map(|(t, r)| (t - r).powi(2))
                .sum();
            sum_sq
        })
        .collect()
}

pub fn plot_mse_frequency_histogram_split(
    pre_procrustes: Vec<ReconstructionDataPoint>,
    post_procrustes: Vec<ReconstructionDataPoint>,
    super_title: &str,
    output_path: &str,
) -> Result<(), Box<dyn Error>> {
    let num_plots = 2; // We strictly have Pre and Post

    // Setup backend
    let root = SVGBackend::new(output_path, (600 * num_plots as u32, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    if pre_procrustes.is_empty() && post_procrustes.is_empty() {
        return Ok(());
    }

    // Process the data
    let pre_errors = calculate_squared_error(&pre_procrustes);
    let post_errors = calculate_squared_error(&post_procrustes);

    // --- NEW: Calculate the overall MSE for each plot as an i64 ---
    let pre_mse_i64 = if pre_errors.is_empty() {
        0
    } else {
        (pre_errors.iter().sum::<f64>() / pre_errors.len() as f64).round() as i64
    };

    let post_mse_i64 = if post_errors.is_empty() {
        0
    } else {
        (post_errors.iter().sum::<f64>() / post_errors.len() as f64).round() as i64
    };
    // --------------------------------------------------------------

    // 1. Vertical Split for Super Title
    let (title_area, plot_area) = root.split_vertically(60);

    let text_color = BLACK;
    let font_family = "Linux Biolinum";
    let super_title_font = (font_family, 40, FontStyle::Bold)
        .into_font()
        .color(&text_color);

    // Super Title
    ChartBuilder::on(&title_area)
        .caption(super_title, super_title_font)
        .build_cartesian_2d(0f32..1f32, 0f32..1f32)?;

    let num_bins = 8;
    let min_err = 0.0f64;

    // Typography
    let label_font = (font_family, 24).into_font().color(&text_color);
    let axis_font = (font_family, 26, FontStyle::Bold)
        .into_font()
        .color(&text_color);
    let title_font = (font_family, 32, FontStyle::Bold)
        .into_font()
        .color(&text_color);

    // Split plot area evenly for side-by-side graphs
    let panels = plot_area.split_evenly((1, num_plots));

    // Bind methods to their respective raw errors (not pre-computed bins), colors, and titles
    let plot_data = vec![
        (
            format!("After Procrustes (MSE: {})", pre_mse_i64),
            &pre_errors,
            RGBColor(230, 159, 0),
        ),
        (
            format!("After Forced Scaling (MSE: {})", post_mse_i64),
            &post_errors,
            RGBColor(0, 114, 178),
        ),
    ];

    for (i, (method, errors, color)) in plot_data.into_iter().enumerate() {
        let panel = &panels[i];
        let y_label_area = 70;

        // Calculate Independent Scale for X-Axis
        let mut max_err = 0.0f64;
        for &err in errors {
            if err > max_err {
                max_err = err;
            }
        }
        if max_err <= 0.0 {
            max_err = 1.0;
        } else {
            max_err *= 1.05;
        }

        let bin_width = (max_err - min_err) / num_bins as f64;

        // Independent Bucketizing
        let mut bins = vec![0; num_bins];
        for &err in errors {
            let mut bin_idx = ((err - min_err) / bin_width).floor() as usize;
            if bin_idx >= num_bins {
                bin_idx = num_bins - 1;
            }
            bins[bin_idx] += 1;
        }

        // Calculate Independent Scale for Y-Axis
        let max_freq = *bins.iter().max().unwrap_or(&0);
        let max_y = ((max_freq as f64 * 1.1).ceil() as usize).max(1);

        let mut chart = ChartBuilder::on(panel)
            .margin_top(40)
            .margin_bottom(40)
            .margin_left(40)
            .margin_right(40 + y_label_area)
            .caption(&method, title_font.clone())
            .x_label_area_size(55)
            .y_label_area_size(y_label_area)
            .build_cartesian_2d(min_err..max_err, 0..max_y)?;

        chart
            .configure_mesh()
            .disable_x_mesh()
            .x_labels(num_bins)
            .y_desc("Number of points")
            .x_desc("Squared Error")
            .x_label_formatter(&|x| format!("{}", x.round() as i64))
            .y_label_formatter(&|y| format!("{}", y))
            .label_style(label_font.clone())
            .axis_desc_style(axis_font.clone())
            .light_line_style(WHITE.mix(0.0))
            .bold_line_style(BLACK.mix(0.1))
            .draw()?;

        let gap = bin_width * 0.05;

        for (bin_idx, &count) in bins.iter().enumerate() {
            if count == 0 {
                continue;
            }

            let x_start = min_err + (bin_idx as f64) * bin_width;
            let x_end = min_err + ((bin_idx + 1) as f64) * bin_width;

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
