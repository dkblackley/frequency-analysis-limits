use crate::plotting::ReconstructionDataPoint;
// Adjust to your crate's path
use geo::{ConvexHull, MultiPoint, Point};
use log::{debug, error};
use plotters::prelude::*;
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

    // let mut flipped: Vec<Vec<ReconstructionDataPoint>> = all_data
    //     .iter()
    //     .map(|cluster| {
    //         cluster
    //             .iter()
    //             .map(|point| flip_coordinates(point.clone()))
    //             .collect()
    //     })
    //     .collect();

    // let first = &flipped[0];
    // let mut truth = Vec::new();
    //
    // for recon in first {
    //     let tru = ReconstructionDataPoint {
    //         true_points: recon.true_points.clone(),
    //         reconstructed_points: recon.true_points.clone(),
    //         unscaled_points: None,
    //     };
    //
    //     truth.push(tru);
    // }
    //
    // flipped.push(truth);

    // Extract ALL points rather than just the worst-case connections
    let full_data = extract_all_points(&all_data);
    let output_path = format!("figures/{}_comprehensive_hull.svg", db_name);

    if let Err(e) = plot_comprehensive_convex_hull(&full_data, &output_path, 0.1, 3.0) {
        error!("Failed to generate plot: {}", e);
    } else {
        debug!(
            "Successfully generated transparent SVG plot at {}",
            output_path
        );
    }
}

pub struct FullPlotData {
    pub true_points: Vec<[f64; 2]>,
    pub all_reconstructed_points: Vec<[f64; 2]>,
}

// -----------------------------------------------------------------------------
// Data Processing
// -----------------------------------------------------------------------------

// We no longer need squared_distance since we aren't searching for the single worst point.

pub fn extract_all_points(data: &[Vec<ReconstructionDataPoint>]) -> FullPlotData {
    let mut true_points = Vec::new();
    let mut all_reconstructed_points = Vec::new();

    if data.is_empty() || data[0].is_empty() {
        return FullPlotData {
            true_points,
            all_reconstructed_points,
        };
    }

    let num_points = data[0].len();
    let num_runs = data.len();

    for point_idx in 0..num_points {
        // Collect ground truth point
        let true_coords = [
            data[0][point_idx].true_points[0],
            data[0][point_idx].true_points[1],
        ];
        true_points.push(true_coords);

        // Collect EVERY reconstructed point across all runs
        for run_idx in 0..num_runs {
            let rx = data[run_idx][point_idx].reconstructed_points[0];
            let ry = data[run_idx][point_idx].reconstructed_points[1];
            all_reconstructed_points.push([rx, ry]);
        }
    }

    FullPlotData {
        true_points,
        all_reconstructed_points,
    }
}

// -----------------------------------------------------------------------------
// Plotting
// -----------------------------------------------------------------------------

pub fn plot_comprehensive_convex_hull(
    data: &FullPlotData,
    output_path: &str,
    x_padder: f64,
    y_padder: f64,
) -> Result<(), Box<dyn Error>> {
    if data.true_points.is_empty() && data.all_reconstructed_points.is_empty() {
        return Ok(());
    }

    // 1. Determine absolute boundaries based on ALL points
    let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);

    let all_points_iter = data
        .true_points
        .iter()
        .chain(data.all_reconstructed_points.iter());

    for p in all_points_iter {
        min_x = min_x.min(p[0]);
        max_x = max_x.max(p[0]);
        min_y = min_y.min(p[1]);
        max_y = max_y.max(p[1]);
    }

    // 2. Pad data ranges
    let x_pad = (max_x - min_x) * x_padder;
    let y_pad = (max_y - min_y) * y_padder;

    let final_min_x = min_x - x_pad;
    let final_max_x = max_x + x_pad;
    let final_min_y = min_y - y_pad;
    let final_max_y = max_y + y_pad;

    // -------------------------------------------------------------------------
    // ASPECT RATIO FIX
    // -------------------------------------------------------------------------
    let range_x = final_max_x - final_min_x;
    let range_y = final_max_y - final_min_y;

    let base_width: f64 = 1600.0;
    let calculated_height = (base_width * (range_y / range_x)).max(300.0);

    let root = SVGBackend::new(output_path, (base_width as u32, calculated_height as u32))
        .into_drawing_area();

    root.fill(&TRANSPARENT)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(40)
        .x_label_area_size(50)
        .y_label_area_size(80)
        .build_cartesian_2d(final_min_x..final_max_x, final_min_y..final_max_y)?;

    let text_color = BLACK;
    let true_blue = RGBColor(0, 114, 178);
    let error_vermilion = RGBColor(213, 94, 0);

    chart
        .configure_mesh()
        .bold_line_style(RGBColor(235, 235, 235))
        .light_line_style(TRANSPARENT)
        .axis_style(&text_color)
        .x_desc("Component 1")
        .y_desc("Component 2")
        .label_style(("Linux Biolinum", 18).into_font().color(&text_color))
        .draw()?;

    // -------------------------------------------------------------------------
    // LAYER 1: Calculate and draw the Global Convex Hull
    // -------------------------------------------------------------------------
    // Map ALL data (ground truth + reconstructions) into geo::Point structs
    let geo_points: Vec<Point<f64>> = data
        .true_points
        .iter()
        .chain(data.all_reconstructed_points.iter())
        .map(|p| Point::new(p[0], p[1]))
        .collect();

    let multi_point = MultiPoint::new(geo_points);
    let convex_hull_polygon = multi_point.convex_hull();

    let hull_coords: Vec<(f64, f64)> = convex_hull_polygon
        .exterior()
        .coords()
        .map(|coord| (coord.x, coord.y))
        .collect();

    if !hull_coords.is_empty() {
        // Draw the filled translucent interior of the hull
        chart.draw_series(std::iter::once(Polygon::new(
            hull_coords.clone(),
            error_vermilion.mix(0.15).filled(),
        )))?;

        // Draw the solid boundary line
        chart
            .draw_series(std::iter::once(PathElement::new(
                hull_coords.clone(),
                error_vermilion.stroke_width(2),
            )))?
            .label("Global Boundary (Hull)")
            .legend(move |(x, y)| {
                Rectangle::new(
                    [(x, y - 5), (x + 20, y + 5)],
                    error_vermilion.mix(0.15).filled(),
                )
            });

        // NEW: Explicitly plot the vertices that define the convex hull
        chart
            .draw_series(
                hull_coords
                    .iter()
                    .map(|&(x, y)| Circle::new((x, y), 6, error_vermilion.filled())),
            )?
            .label("Hull Defining Points")
            .legend(move |(x, y)| Circle::new((x, y), 6, error_vermilion.filled()));
    }

    // -------------------------------------------------------------------------
    // LAYER 2: Plot Ground Truth Data on top
    // -------------------------------------------------------------------------
    chart
        .draw_series(
            data.true_points
                .iter()
                .map(|p| Circle::new((p[0], p[1]), 5, true_blue.filled())),
        )?
        .label("Ground Truth")
        .legend(move |(x, y)| Circle::new((x, y), 5, true_blue.filled()));

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .background_style(WHITE.mix(0.95).filled())
        .border_style(TRANSPARENT)
        .label_font(("Linux Biolinum", 16).into_font().color(&text_color))
        .draw()?;

    root.present()?;
    Ok(())
}
