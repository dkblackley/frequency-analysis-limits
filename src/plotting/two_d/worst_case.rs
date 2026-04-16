// Include your structs
// #[derive(Debug, Serialize, Deserialize, Clone)]
// pub struct ReconstructionData2dPoint { ... }
use crate::plotting::ReconstructionDataPoint;
use plotters::prelude::*;
use std::error::Error;

fn squared_distance(p1: &[f64], p2: &[f64]) -> f64 {
    p1.iter()
        .zip(p2.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum()
}

/// Calculates the midpoint between two N-dimensional points.
fn midpoint(p1: &[f64], p2: &[f64]) -> Vec<f64> {
    p1.iter()
        .zip(p2.iter())
        .map(|(a, b)| (a + b) / 2.0)
        .collect()
}

/// Finds the farthest reconstructed points across all model runs for each data point
/// and returns a single Vec of the newly centered points.
pub fn calculate_reconstructed_midpoints(
    data: &[Vec<ReconstructionDataPoint>],
) -> Vec<ReconstructionDataPoint> {
    // Return early if there is no data to process
    if data.is_empty() || data[0].is_empty() {
        return vec![];
    }

    let num_points = data[0].len();
    let num_runs = data.len();
    let mut results = Vec::with_capacity(num_points);

    for point_idx in 0..num_points {
        let mut max_dist_sq = -1.0;
        let mut farthest_pair = (0, 0);

        // Calculate pairwise distances to find the two farthest reconstructed points
        for i in 0..num_runs {
            for j in (i + 1)..num_runs {
                let p1 = &data[i][point_idx].reconstructed_points;
                let p2 = &data[j][point_idx].reconstructed_points;

                let dist_sq = squared_distance(p1, p2);
                if dist_sq > max_dist_sq {
                    max_dist_sq = dist_sq;
                    farthest_pair = (i, j);
                }
            }
        }

        // Retrieve the two farthest points found
        let p1 = &data[farthest_pair.0][point_idx].reconstructed_points;
        let p2 = &data[farthest_pair.1][point_idx].reconstructed_points;

        // Calculate the midpoint
        let new_reconstructed_midpoint = midpoint(p1, p2);

        // Create the new data point
        let new_point = ReconstructionDataPoint {
            // The true point is identical across runs, so we pull from index 0
            true_points: data[0][point_idx].true_points.clone(),
            reconstructed_points: new_reconstructed_midpoint,
            unscaled_points: None,
        };

        results.push(new_point);
    }

    results
}

pub fn plot_all_reconstructions_gradient(
    data: &[ReconstructionDataPoint],
    output_path: &str,
    x_padder: f64,
    y_padder: f64,
) -> Result<(), Box<dyn Error>> {
    let root = SVGBackend::new(output_path, (1200, 800)).into_drawing_area();
    root.fill(&TRANSPARENT)?;

    if data.is_empty() {
        return Ok(());
    }

    let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);
    let mut max_mse = 0.0_f64;

    for point_data in data {
        let tx = point_data.true_points[0];
        let ty = point_data.true_points[1];
        let rx = point_data.reconstructed_points[0];
        let ry = point_data.reconstructed_points[1];

        min_x = min_x.min(tx).min(rx);
        max_x = max_x.max(tx).max(rx);
        min_y = min_y.min(ty).min(ry);
        max_y = max_y.max(ty).max(ry);

        let mse = (tx - rx).powi(2) + (ty - ry).powi(2);
        if mse > max_mse {
            max_mse = mse;
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

    chart
        .configure_mesh()
        .bold_line_style(RGBColor(220, 220, 220))
        .axis_style(BLACK)
        .label_style(("sans-serif", 20).into_font().color(&BLACK))
        .draw()?;

    // 1. Plot Ground Truth (Black crosshairs act as visual anchors)
    chart
        .draw_series(data.iter().map(|p| {
            let (x, y) = (p.true_points[0], p.true_points[1]);
            Circle::new((x, y), 6, BLACK.filled())
        }))?
        .label("Ground Truth")
        .legend(move |(x, y)| Circle::new((x, y), 6, BLACK.filled()));

    // 2. Plot Reconstructed Points with calculated Error Gradient
    chart
        .draw_series(data.iter().map(|p| {
            let tx = p.true_points[0];
            let ty = p.true_points[1];
            let rx = p.reconstructed_points[0];
            let ry = p.reconstructed_points[1];

            let mse = (tx - rx).powi(2) + (ty - ry).powi(2);
            let ratio = if max_mse > 0.0 { mse / max_mse } else { 0.0 };

            // Gradient map: Bluish Green (0, 158, 115) to Vermilion (213, 94, 0)
            let r = (0.0 + (213.0 - 0.0) * ratio) as u8;
            let g = (158.0 + (94.0 - 158.0) * ratio) as u8;
            let b = (115.0 + (0.0 - 115.0) * ratio) as u8;

            let color = RGBColor(r, g, b);

            // Heavy transparency (0.4) allows overlapping areas to combine and form heatmaps visually
            Circle::new((rx, ry), 4, color.mix(0.4).filled())
        }))?
        .label("Reconstructed Heatmap")
        .legend(move |(x, y)| {
            // Representative legend color
            Circle::new((x, y), 4, RGBColor(213, 94, 0).mix(0.6).filled())
        });

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(WHITE.mix(0.9).filled())
        .border_style(BLACK)
        .label_font(("sans-serif", 20).into_font().color(&BLACK))
        .draw()?;

    root.present()?;
    Ok(())
}
