use crate::plotting::post::do_spitz_align;
use crate::plotting::two_d::{format_db_name, format_dist_name};
use crate::plotting::{get_remin_even_less, ReconstructionDataPoint};
use log::debug;
use plotters::prelude::*;
use std::collections::HashMap;
use std::error::Error;
use std::fs;

pub fn run_spatial_plots(
    name: &str,
    dir: &str,
    dist: &str,
    grid: u32,
) -> Result<(), Box<dyn Error>> {
    let path_to_root = format!("{}/{}", dir, name);

    let mut even_less =
        format!("{path_to_root}/even_less/{name}_prob100.0_{dist}_{grid}x{grid}_even_less.json");
    let mut remin_path =
        format!("{path_to_root}/remin/{name}_prob100.0_{dist}_{grid}x{grid}_classic.json");
    let mut limits = format!("{path_to_root}/limits/{name}_{dist}_e0_d0.9_reconstruction.json");

    if grid == 350 {
        even_less =
            format!("{path_to_root}/even_less/{name}_prob100.0_{dist}_350x50_even_less.json");
        remin_path = format!("{path_to_root}/remin/{name}_prob100.0_{dist}_350x50_classic.json");
        limits = format!("{path_to_root}/limits/{name}_{dist}_e0_d0.9_reconstruction.json");
    }
    if grid == 175 {
        even_less =
            format!("{path_to_root}/even_less/{name}_prob100.0_{dist}_175x25_even_less.json");
        remin_path = format!("{path_to_root}/remin/{name}_prob100.0_{dist}_175x25_classic.json");
        limits = format!("{path_to_root}/limits/{name}_{dist}_e0_d0.9_reconstruction.json");
    }

    debug!(
        "About to load data from {}, {}, {}",
        &even_less, &remin_path, &limits
    );

    let mut even_less_data = get_remin_even_less(&even_less, true)?;
    let mut remin_data = get_remin_even_less(&remin_path, true)?;

    if name == "spitz" {
        let content = fs::read_to_string(remin_path)?;
        let mut raw_remin: Vec<ReconstructionDataPoint> = serde_json::from_str(&content)?;
        let content = fs::read_to_string(even_less)?;
        let mut raw_even_less: Vec<ReconstructionDataPoint> = serde_json::from_str(&content)?;

        (raw_remin, raw_even_less) = do_spitz_align(raw_remin, raw_even_less);

        let mut true_point = Vec::new();
        let mut recon_point = Vec::new();

        for point in raw_remin {
            true_point.push(point.true_points);
            recon_point.push(point.reconstructed_points);
        }
        remin_data = (true_point, recon_point);

        let mut true_point = Vec::new();
        let mut recon_point = Vec::new();

        for point in raw_even_less {
            true_point.push(point.true_points);
            recon_point.push(point.reconstructed_points);
        }
        even_less_data = (true_point, recon_point);
    }

    let content = fs::read_to_string(&limits)?;
    let all_data: Vec<Vec<ReconstructionDataPoint>> = serde_json::from_str(&content)?;
    let data = all_data[0].clone(); // jsut grab the first

    let mut true_point = Vec::new();
    let mut recon_point = Vec::new();

    for point in data {
        true_point.push(point.true_points);
        recon_point.push(point.reconstructed_points);
    }

    let _limits_data = (true_point, recon_point);

    let mut data_map = HashMap::new();

    data_map.insert("even_less".to_string(), even_less_data.1.clone());
    data_map.insert("remin".to_string(), remin_data.1);
    // TODO: Sample an item from limits and use procrustes?
    data_map.insert("limits".to_string(), even_less_data.0.clone());

    plot_spatial_reconstruction(
        name,
        dist,
        &*even_less_data.0,
        &data_map,
        &format!("figures/{grid}x{grid}_recon/{name}_{dist}_reconstruction.svg"),
        true,
    )?;

    Ok(())
}

pub fn plot_spatial_reconstruction(
    db: &str,
    dist: &str,
    true_coords: &[Vec<f64>],
    method_coords: &HashMap<String, Vec<Vec<f64>>>,
    output_path: &str,
    show_true_points: bool,
) -> Result<(), Box<dyn Error>> {
    let total_width = 3600;
    let total_height = 800;

    let root = SVGBackend::new(output_path, (total_width, total_height)).into_drawing_area();
    root.fill(&TRANSPARENT)?;

    // 1. Simple global bounds across ALL data
    let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);

    let mut update_bounds = |pts: &[Vec<f64>]| {
        for x_y in pts {
            if x_y.len() >= 2 {
                min_x = min_x.min(x_y[0]);
                max_x = max_x.max(x_y[0]);
                min_y = min_y.min(x_y[1]);
                max_y = max_y.max(x_y[1]);
            }
        }
    };

    update_bounds(true_coords);
    for recon in method_coords.values() {
        update_bounds(recon);
    }

    if min_x.is_infinite() || min_y.is_infinite() {
        return Ok(()); // Avoid crashing if all vectors are entirely empty
    }

    // 1% safety margin to ensure edge points aren't clipped by the SVG border
    let edge_margin_x = (max_x - min_x).max(1.0) * 0.01;
    let edge_margin_y = (max_y - min_y).max(1.0) * 0.01;

    // Master title formatting
    let master_title = format!(
        "{} - {} Distribution",
        format_db_name(db),
        format_dist_name(dist)
    );

    let (title_area, plot_area) = root.split_vertically(80);

    let title_font = ("Linux Biolinum", 78, FontStyle::Bold).into_font();
    let title_size = title_font
        .layout_box(&master_title)
        .unwrap_or(((0, 0), (0, 0)));
    let text_width = title_size.1 .0 - title_size.0 .0;

    title_area.draw_text(
        &master_title,
        &title_font.color(&BLACK),
        ((total_width as i32 - text_width) / 2, 20),
    )?;

    let sub_areas = plot_area.split_evenly((1, 3));

    let methods_to_plot = vec![
        ("limits", "LAMa", RGBColor(0, 114, 178)),
        ("even_less", "Even Less", RGBColor(230, 159, 0)),
        ("remin", "Remin", RGBColor(0, 158, 115)),
    ];

    let gt_color = RGBColor(150, 150, 150).mix(0.5);
    let text_color = BLACK;

    for (i, (method_key, method_name, recon_color)) in methods_to_plot.iter().enumerate() {
        let mut chart = ChartBuilder::on(&sub_areas[i])
            .margin(10)
            .margin_right(30)
            .caption(
                *method_name,
                ("Linux Biolinum", 68, FontStyle::Bold)
                    .into_font()
                    .color(&BLACK),
            )
            .build_cartesian_2d(
                (min_x - edge_margin_x)..(max_x + edge_margin_x),
                (min_y - edge_margin_y)..(max_y + edge_margin_y),
            )?;

        // REMOVED `chart.configure_mesh()...` entirely.
        // Without this block, Plotters will not draw any background grid, axes, or numbers.

        if show_true_points {
            chart
                .draw_series(true_coords.iter().filter_map(|x_y| {
                    if x_y.len() >= 2 {
                        Some(Circle::new((x_y[0], x_y[1]), 8, gt_color.filled()))
                    } else {
                        None
                    }
                }))?
                .label("Ground Truth")
                .legend(move |(x, y)| Circle::new((x, y), 6, gt_color.filled()));
        }

        if let Some(recon_points) = method_coords.get(*method_key) {
            chart
                .draw_series(recon_points.iter().filter_map(|x_y| {
                    if x_y.len() >= 2 {
                        Some(Circle::new((x_y[0], x_y[1]), 4, recon_color.filled()))
                    } else {
                        None
                    }
                }))?
                .label("Reconstructed")
                .legend(move |(x, y)| Circle::new((x, y), 4, recon_color.filled()));
        }

        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::UpperRight)
            .background_style(WHITE.mix(0.9).filled())
            .border_style(BLACK)
            .label_font(
                ("Linux Biolinum", 46, FontStyle::Bold)
                    .into_font()
                    .color(&text_color),
            )
            .margin(10)
            .draw()?;
    }

    root.present()?;

    println!(
        "Generated side-by-side spatial plot for {} -> saved to {}",
        db, output_path
    );

    Ok(())
}

pub fn just_points(
    true_coords: &[Vec<f64>],
    recon_coords: &[Vec<f64>],
    output_path: &str,
    show_true_points: bool,
    x_padder: f64,
    y_padder: f64,
) -> Result<(), Box<dyn Error>> {
    let root = SVGBackend::new(output_path, (1200, 800)).into_drawing_area();
    root.fill(&TRANSPARENT)?;

    if true_coords.is_empty() && recon_coords.is_empty() {
        return Ok(());
    }

    let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);
    for x_y in true_coords.iter().chain(recon_coords.iter()) {
        let (x, y) = (x_y[0], x_y[1]);
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }

    let x_pad = (max_x - min_x) * x_padder;
    let y_pad = (max_y - min_y) * y_padder;

    let mut chart = ChartBuilder::on(&root).build_cartesian_2d(
        (min_x - x_pad)..(max_x + x_pad),
        (min_y - y_pad)..(max_y + y_pad),
    )?;

    let sky_blue = RGBColor(86, 180, 233);

    if show_true_points {
        chart.draw_series(
            true_coords
                .iter()
                .map(|x_y| Circle::new((x_y[0], x_y[1]), 8, sky_blue.mix(0.8).filled())),
        )?;
    }

    chart.draw_series(recon_coords.iter().map(|x_y| {
        let (x, y) = (x_y[0], x_y[1]);
        let ratio = if max_y > min_y {
            (y - min_y) / (max_y - min_y)
        } else {
            0.5
        };
        let r = (86.0 + (213.0 - 86.0) * ratio) as u8;
        let g = (180.0 + (94.0 - 180.0) * ratio) as u8;
        let b = (233.0 + (0.0 - 233.0) * ratio) as u8;

        Circle::new((x, y), 4, RGBColor(r, g, b).mix(0.9).filled())
    }))?;

    root.present()?;
    Ok(())
}

// Plots arbitrary 2D points natively as an SVG with a transparent background.
// Uses procedural HSL color generation and dynamic, strictly decreasing point
// sizes to support visualizing 100+ overlapping reconstructed series.
pub fn plot_spatial_reconstruction_all_items(
    true_coords: &[Vec<f64>],
    recon_coords: &[Vec<Vec<f64>>],
    output_path: &str,
    show_true_points: bool,
    x_padder: f64,
    y_padder: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    // Switch to SVGBackend for lossless, scalable vector images
    let root = SVGBackend::new(output_path, (1200 * 2, 800)).into_drawing_area();
    root.fill(&TRANSPARENT)?;

    let recon_is_empty = recon_coords.is_empty() || recon_coords.iter().all(|r| r.is_empty());
    if true_coords.is_empty() && recon_is_empty {
        return Ok(());
    }

    let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);

    // Process bounds for ground truth
    for x_y in true_coords.iter() {
        let (x, y) = (x_y[0], x_y[1]);
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }

    // Process bounds for all reconstruction sets
    for recon_set in recon_coords.iter() {
        for x_y in recon_set.iter() {
            let (x, y) = (x_y[0], x_y[1]);
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
    }

    let x_pad = (max_x - min_x) * x_padder;
    let y_pad = (max_y - min_y) * y_padder;

    let mut chart = ChartBuilder::on(&root)
        .margin(30)
        .x_label_area_size(80)
        .y_label_area_size(90)
        .build_cartesian_2d(
            (min_x - x_pad)..(max_x + x_pad),
            (min_y - y_pad)..(max_y + y_pad),
        )?;

    let text_color = BLACK;
    let mesh_color = RGBColor(220, 220, 220);

    chart
        .configure_mesh()
        .bold_line_style(mesh_color)
        .axis_style(&text_color)
        .label_style(("sans-serif", 20).into_font().color(&text_color))
        .draw()?;

    // Keep ground truth prominent
    let sky_blue = RGBColor(86, 180, 233);
    let true_radius = 16; // Massively increased base size for ground truth

    // 1. True Points Series (Fully Opaque & Largest)
    if show_true_points && !true_coords.is_empty() {
        chart
            .draw_series(
                true_coords
                    .iter()
                    .map(|x_y| Circle::new((x_y[0], x_y[1]), true_radius, sky_blue.filled())),
            )?
            .label("Ground Truth")
            // Legend size capped so it doesn't blow out the layout
            .legend(move |(x, y)| Circle::new((x, y), 8, sky_blue.filled()));
    }

    // 2. Reconstructed Points Series Setup
    let num_recons = recon_coords.len();
    let max_recon_radius = 12.0; // Largest reconstruction size (sits just under truth)
    let min_recon_radius = 2.0; // Smallest size for the final iteration on top

    for (i, recon_set) in recon_coords.iter().enumerate() {
        if recon_set.is_empty() {
            continue;
        }

        // --- Decreasing Radius Logic ---
        // As `i` increases (drawn later, so they appear "on top" of the visual stack),
        // the radius shrinks continuously toward `min_recon_radius`.
        let current_radius = if num_recons > 1 {
            max_recon_radius
                - (i as f64 / (num_recons - 1) as f64) * (max_recon_radius - min_recon_radius)
        } else {
            max_recon_radius
        }
        .round() as u32;

        // --- Procedural Color Generation ---
        let hue = (i as f64 * 0.618033988749895) % 1.0;
        let lightness = match i % 3 {
            0 => 0.45,
            1 => 0.60,
            _ => 0.75,
        };
        let base_color = HSLColor(hue, 0.85, lightness);

        chart
            .draw_series(recon_set.iter().map(|x_y| {
                let (x, y) = (x_y[0], x_y[1]);
                Circle::new((x, y), current_radius, base_color.mix(0.85).filled())
            }))?
            .label(format!("Reconstructed {}", i + 1))
            // Apply a minimum bound to the legend icon size so it remains clickable/readable
            .legend(move |(x, y)| Circle::new((x, y), current_radius.max(3), base_color.filled()));
    }

    // 3. The Legend
    // chart
    //     .configure_series_labels()
    //     .position(SeriesLabelPosition::UpperRight)
    //     .background_style(WHITE.mix(0.9).filled())
    //     .border_style(BLACK)
    //     .label_font(("sans-serif", 20).into_font().color(&text_color))
    //     .draw()?;

    root.present()?;
    Ok(())
}
