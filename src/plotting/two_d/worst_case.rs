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

    if let Err(e) = plot_sleek_convex_hull_worst_case(&simplified_data, &output_path, 0.1, 0.1) {
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

pub fn plot_sleek_convex_hull_worst_case(
    data: &[SleekPointConnection],
    output_path: &str,
    x_padder: f64,
    y_padder: f64,
) -> Result<(), Box<dyn Error>> {
    if data.is_empty() {
        return Ok(());
    }

    // 1. Determine absolute boundaries based on all connections
    let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);

    for pb in data {
        min_x = min_x.min(pb.true_point[0]).min(pb.worst_case_point[0]);
        max_x = max_x.max(pb.true_point[0]).max(pb.worst_case_point[0]);
        min_y = min_y.min(pb.true_point[1]).min(pb.worst_case_point[1]);
        max_y = max_y.max(pb.true_point[1]).max(pb.worst_case_point[1]);
    }

    // 2. Pad data ranges
    let x_pad = (max_x - min_x) * x_padder;
    let y_pad = (max_y - min_y) * y_padder;

    let final_min_x = min_x - x_pad;
    let final_max_x = max_x + x_pad;
    let final_min_y = min_y - y_pad;
    let final_max_y = max_y + y_pad;

    // -------------------------------------------------------------------------
    // ASPECT RATIO FIX: Force X and Y to share the same physical visual scale
    // -------------------------------------------------------------------------
    let range_x = final_max_x - final_min_x;
    let range_y = final_max_y - final_min_y;

    // Lock the width to a high-res wide format, calculate height dynamically
    let base_width: f64 = 1600.0;

    // If range_x is 350 and range_y is 50, the height will be 1/7th of the width
    // We add a max() buffer so the plot doesn't become too thin to render axes
    let calculated_height = (base_width * (range_y / range_x)).max(300.0);

    let root = SVGBackend::new(output_path, (base_width as u32, calculated_height as u32))
        .into_drawing_area();

    root.fill(&TRANSPARENT)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(40)
        .x_label_area_size(50)
        .y_label_area_size(80)
        .build_cartesian_2d(final_min_x..final_max_x, final_min_y..final_max_y)?;

    // Colorblind safe Okabe-Ito palette
    let text_color = BLACK;
    let true_blue = RGBColor(0, 114, 178);
    let error_vermilion = RGBColor(213, 94, 0);

    // Clean, flat academic grid
    chart
        .configure_mesh()
        .bold_line_style(RGBColor(235, 235, 235)) // Softer grid lines
        .light_line_style(TRANSPARENT) // Remove distracting sub-grids
        .axis_style(&text_color)
        .x_desc("Component 1")
        .y_desc("Component 2")
        .label_style(("sans-serif", 18).into_font().color(&text_color))
        .draw()?;

    // -------------------------------------------------------------------------
    // LAYER 1: Calculate and draw the Convex Hull
    // -------------------------------------------------------------------------
    // Map worst-case data into geo::Point structs
    let geo_points: Vec<Point<f64>> = data
        .iter()
        .map(|pb| Point::new(pb.worst_case_point[0], pb.worst_case_point[1]))
        .collect();

    // Wrap in a MultiPoint and calculate the hull
    let multi_point = MultiPoint::new(geo_points);
    let convex_hull_polygon = multi_point.convex_hull();

    // Extract the closed LineString exterior coordinates back into (f64, f64)
    let hull_coords: Vec<(f64, f64)> = convex_hull_polygon
        .exterior()
        .coords()
        .map(|coord| (coord.x, coord.y))
        .collect();

    if !hull_coords.is_empty() {
        // Draw the filled translucent interior of the hull
        chart.draw_series(std::iter::once(Polygon::new(
            hull_coords.clone(),
            error_vermilion.mix(0.15).filled(), // 15% opacity for a sleek fill
        )))?;

        // Draw the solid boundary line (PathElement will auto-close because geo guarantees it)
        chart
            .draw_series(std::iter::once(PathElement::new(
                hull_coords,
                error_vermilion.stroke_width(2),
            )))?
            .label("Worst-Case Boundary (Hull)")
            .legend(move |(x, y)| {
                // A small swatch representing a bordered area for the legend
                Rectangle::new(
                    [(x, y - 5), (x + 20, y + 5)],
                    error_vermilion.mix(0.15).filled(),
                )
            });
    }

    // -------------------------------------------------------------------------
    // LAYER 2: Plot Ground Truth Data on top
    // -------------------------------------------------------------------------
    chart
        .draw_series(data.iter().map(|pb| {
            Circle::new(
                (pb.true_point[0], pb.true_point[1]),
                5, // Slightly smaller point size to fit the wider/squished scale
                true_blue.filled(),
            )
        }))?
        .label("Ground Truth")
        .legend(move |(x, y)| Circle::new((x, y), 5, true_blue.filled()));

    // Clean, flat legend styling (removing heavy borders)
    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .background_style(WHITE.mix(0.95).filled())
        .border_style(TRANSPARENT) // Removed standard border for a modern aesthetic
        .label_font(("sans-serif", 16).into_font().color(&text_color))
        .draw()?;

    root.present()?;
    Ok(())
}
