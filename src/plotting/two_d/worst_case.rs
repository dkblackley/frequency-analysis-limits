use crate::plotting::ReconstructionDataPoint;
// Adjust to your crate's path
use geo::{ConvexHull, MultiPoint, Point};
use log::{debug, error};
use plotters::prelude::*;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs;

/// A simplified struct to hold only the essential connection info per cluster
#[derive(Debug, Clone)]
pub struct SleekPointConnection {
    pub true_point: [f64; 2],
    pub worst_case_point: [f64; 2],
}

pub fn do_convex_hull_plots(db_name: &str, data_dir: &str) {
    let path = format!(
        "{}/{}/limits/{}_uniform_e0_d0.9_reconstruction.json",
        data_dir, db_name, db_name
    );
    debug!("About to load data from {}", &path);

    let content = fs::read_to_string(&path).expect("Failed to read JSON file");
    let all_data: Vec<Vec<ReconstructionDataPoint>> =
        serde_json::from_str(&content).expect("Failed to parse JSON");

    let simplified_data = calculate_sleek_connections(&all_data);
    let output_path = format!("figures/{}_worst_case_sleek_modern.svg", db_name);

    if let Err(e) = plot_sleek_error_vectors(&simplified_data, &output_path, 0.1, 0.1) {
        error!("Failed to generate plot: {}", e);
    } else {
        debug!(
            "Successfully generated transparent SVG plot at {}",
            output_path
        );
    }
}

// -----------------------------------------------------------------------------
// Data Processing
// -----------------------------------------------------------------------------

fn squared_distance(p1: &[f64; 2], p2: &[f64; 2]) -> f64 {
    (p1[0] - p2[0]).powi(2) + (p1[1] - p2[1]).powi(2)
}

pub fn calculate_sleek_connections(
    data: &[Vec<ReconstructionDataPoint>],
) -> Vec<SleekPointConnection> {
    if data.is_empty() || data[0].is_empty() {
        return vec![];
    }

    let num_points = data[0].len();
    let num_runs = data.len();
    let mut results = Vec::with_capacity(num_points);

    for point_idx in 0..num_points {
        let true_coords = [
            data[0][point_idx].true_points[0],
            data[0][point_idx].true_points[1],
        ];

        let mut max_dist_sq = -1.0;
        let mut worst_case_point = [0.0, 0.0];

        // Only need to find the single worst-case point among all reconstructions
        for run_idx in 0..num_runs {
            let rx = data[run_idx][point_idx].reconstructed_points[0];
            let ry = data[run_idx][point_idx].reconstructed_points[1];
            let dist = squared_distance(&true_coords, &[rx, ry]);
            if dist > max_dist_sq {
                max_dist_sq = dist;
                worst_case_point = [rx, ry];
            }
        }

        results.push(SleekPointConnection {
            true_point: true_coords,
            worst_case_point,
        });
    }

    results
}

// -----------------------------------------------------------------------------
// Plotting
// -----------------------------------------------------------------------------

pub fn plot_sleek_error_vectors(
    data: &[SleekPointConnection],
    output_path: &str,
    x_padder: f64,
    y_padder: f64,
) -> Result<(), Box<dyn Error>> {
    let root = SVGBackend::new(output_path, (1200, 800)).into_drawing_area();
    root.fill(&TRANSPARENT)?;

    if data.is_empty() {
        return Ok(());
    }

    // Determine absolute boundaries based on simplified connections
    let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);

    for pb in data {
        min_x = min_x.min(pb.true_point[0]).min(pb.worst_case_point[0]);
        max_x = max_x.max(pb.true_point[0]).max(pb.worst_case_point[0]);
        min_y = min_y.min(pb.true_point[1]).min(pb.worst_case_point[1]);
        max_y = max_y.max(pb.true_point[1]).max(pb.worst_case_point[1]);
    }

    let x_pad = (max_x - min_x) * x_padder;
    let y_pad = (max_y - min_y) * y_padder;

    let mut chart = ChartBuilder::on(&root)
        .margin(40)
        .x_label_area_size(50)
        .y_label_area_size(80)
        .build_cartesian_2d(
            (min_x - x_pad)..(max_x + x_pad),
            (min_y - y_pad)..(max_y + y_pad),
        )?;

    let text_color = BLACK;
    let true_blue = RGBColor(0, 114, 178); // Blue for true, sleek/CB
    let error_vermilion = RGBColor(213, 94, 0); // Vermilion for worst, sleek/CB

    // Modern academic grid and labels matching your clean style
    chart
        .configure_mesh()
        .bold_line_style(RGBColor(220, 220, 220))
        .axis_style(&text_color)
        .x_desc("Component 1")
        .y_desc("Component 2")
        .label_style(("sans-serif", 20).into_font().color(&text_color))
        .draw()?;

    // LAYER 1: Dotted Error Vectors
    // Draw sleek, dotted, translucent connection lines behind points
    for pb in data {
        let p1 = (pb.true_point[0], pb.true_point[1]);
        let p2 = (pb.worst_case_point[0], pb.worst_case_point[1]);
        chart.draw_series(std::iter::once(PathElement::new(
            vec![p1, p2],
            error_vermilion.mix(0.2).stroke_width(2), //error_vermilion.mix(0.35).dash([5, 5]).stroke_width(2), // Sleek translucent dotted line
        )))?;
    }

    // LAYER 2: Impactful Anchor Points
    // Plot significantly larger points on top for maximum clarity

    // 2a. Worst Case Points (Vermilion, slightly larger, full opacity)
    chart
        .draw_series(data.iter().map(|pb| {
            Circle::new(
                (pb.worst_case_point[0], pb.worst_case_point[1]),
                10, // Larger dot!
                error_vermilion.filled(),
            )
        }))?
        .label("Worst-Case Bound")
        .legend(move |(x, y)| Circle::new((x, y), 8, error_vermilion.filled())); // Slightly smaller legend dot for balance

    // 2b. Ground Truth (Blue, large, full opacity)
    chart
        .draw_series(data.iter().map(|pb| {
            Circle::new((pb.true_point[0], pb.true_point[1]), 8, true_blue.filled())
            // Larger dot!
        }))?
        .label("Ground Truth")
        .legend(move |(x, y)| Circle::new((x, y), 8, true_blue.filled()));

    // Clean, modern Legend Box matching your academic style
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
