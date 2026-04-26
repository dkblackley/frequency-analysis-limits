use crate::plotting::convex_hull::{get_per_point_convex_hulls, PointConvexHull};
use crate::plotting::two_d::mse_by_all_reconstructions::combined_search;
use crate::plotting::ReconstructionDataPoint;
use log::{debug, warn};
use plotters::prelude::*;
use std::collections::HashMap;
use std::error::Error;
use std::fs;

use rand::seq::SliceRandom;
use rand::thread_rng;

/// Maps dataset method strings to their display names, rank, and color.
fn get_method_style(method: &str) -> (&'static str, usize, RGBColor) {
    match method {
        "limits" => ("LAMa", 0, RGBColor(0, 114, 178)), // Blue
        "even_less" => ("Even Less", 1, RGBColor(230, 159, 0)), // Orange
        "remin" => ("Remin", 2, RGBColor(0, 158, 115)), // Green
        _ => ("Unknown", 99, BLACK),
    }
}

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
            let (rx, ry) = (p.reconstructed_points[0], p.reconstructed_points[1]);

            let scaled_x = rx * sx;

            let scaled_y = ry * sy;

            let rot_x = scaled_x * cos_t - scaled_y * sin_t;

            let rot_y = scaled_x * sin_t + scaled_y * cos_t;

            ReconstructionDataPoint {
                true_points: p.true_points.clone(),

                reconstructed_points: vec![rot_x + dx, rot_y + dy],

                unscaled_points: None,
            }
        })
        .collect()
}

/// Runs the combined search, samples 10k transforms, and returns the generated point clouds.

pub fn generate_sampled_reconstructions(
    data: &[ReconstructionDataPoint],

    search_domain: (usize, usize),

    shift_step: f64,

    rot_step_deg: f64,

    scale_step: f64,
) -> Vec<Vec<ReconstructionDataPoint>> {
    // 1. Get all valid transform configurations

    let evaluated_transforms = combined_search(
        data,
        search_domain,
        shift_step,
        rot_step_deg,
        search_domain,
        scale_step,
    );

    // 2. Shuffle and take up to 10,000

    let mut rng = thread_rng();

    let mut sampled_transforms = evaluated_transforms;

    sampled_transforms.shuffle(&mut rng);

    let sample_size = sampled_transforms.len().min(10_000);

    let final_samples = &sampled_transforms[..sample_size];

    // 3. Apply the transform to only the chosen 10k to save memory

    let mut point_clouds = Vec::with_capacity(sample_size);

    for &(_, dx, dy, angle, sx, sy) in final_samples {
        // Assuming apply_transform is available in your scope as seen in your commented code

        let recon = apply_transform(data, dx, dy, angle, sx, sy);

        point_clouds.push(recon);
    }

    point_clouds
}

/// Helper function to format strings for the title
fn format_title_case(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + chars.as_str(),
    }
}

/// Helper to compute the squared Euclidean distance between two N-dimensional points

fn compute_distance_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Computes the GLOBAL MAXIMUM Centroid MSE and Worst-Case Midpoint MSE for a set of hulls.

/// Returns (Max Centroid MSE, Max Worst-Case MSE)

pub fn compute_dataset_metrics(
    hulls: &[PointConvexHull],

    true_points: &[Vec<f64>], // Ground truth points mapping 1:1 with the hulls
) -> (f64, f64) {
    let mut max_centroid_mse = 0.0_f64;
    let mut max_worst_case_mse = 0.0_f64;
    let mut has_valid_points = false;

    for (hull, true_pt) in hulls.iter().zip(true_points.iter()) {
        if hull.vertices.is_empty() {
            continue;
        }

        has_valid_points = true;
        let dims = true_pt.len();
        let num_vertices = hull.vertices.len() as f64;

        // ---------------------------------------------------------
        // 1. Calculate the Centroid (Barycenter)
        // ---------------------------------------------------------

        let mut centroid = vec![0.0; dims];

        for v in &hull.vertices {
            for i in 0..dims {
                centroid[i] += v[i];
            }
        }

        for i in 0..dims {
            centroid[i] /= num_vertices;
        }

        let centroid_mse = compute_distance_sq(&centroid, true_pt) / (dims as f64);

        // ---------------------------------------------------------

        // 2. Calculate the Worst-Case Midpoint (Longest Diameter)

        // ---------------------------------------------------------

        let mut max_dist_sq = -1.0;

        let mut best_v1 = &hull.vertices[0];

        let mut best_v2 = &hull.vertices[0];

        // O(N^2) search across vertices for the longest distance.

        for i in 0..hull.vertices.len() {
            for j in i..hull.vertices.len() {
                let d_sq = compute_distance_sq(&hull.vertices[i], &hull.vertices[j]);

                if d_sq > max_dist_sq {
                    max_dist_sq = d_sq;

                    best_v1 = &hull.vertices[i];

                    best_v2 = &hull.vertices[j];
                }
            }
        }

        let mut midpoint = vec![0.0; dims];

        for i in 0..dims {
            midpoint[i] = (best_v1[i] + best_v2[i]) / 2.0;
        }

        let worst_case_mse = compute_distance_sq(&midpoint, true_pt) / (dims as f64);

        // ---------------------------------------------------------

        // 3. Track the absolute maximums instead of averaging

        // ---------------------------------------------------------

        max_centroid_mse = max_centroid_mse.max(centroid_mse);

        max_worst_case_mse = max_worst_case_mse.max(worst_case_mse);
    }

    if !has_valid_points {
        return (0.0, 0.0);
    }

    (max_centroid_mse, max_worst_case_mse)
}

/// Orchestrates reading files for ALL distributions and calculating their convex hulls
pub fn process_and_plot_convex_hulls(
    db_name: &str,
    data_dir: &str,
    distributions: &[&str],
) -> Result<(), Box<dyn Error>> {
    // Structure: dist -> method -> vec of (query, cent_mse, worst_mse)
    let mut db_data: HashMap<&str, HashMap<String, Vec<(f64, f64, f64)>>> = HashMap::new();
    let search_domain = (60, 60);
    let query_percents = vec![10.0, 20.0, 30.0];

    // Helper closure to extract true points for the MSE calculator
    let extract_true_points = |data: &[ReconstructionDataPoint]| -> Vec<Vec<f64>> {
        data.iter().map(|p| p.true_points.clone()).collect()
    };

    for &dist in distributions {
        let mut plot_data: HashMap<String, Vec<(f64, f64, f64)>> = HashMap::new();

        for &query in &query_percents {
            // ---------------------------------------------------------
            // 1. EVEN LESS (Needs sampling)
            // ---------------------------------------------------------
            let el_path = format!(
                "{}/{}/even_less/{}_prob{}.0_{}_15x15_even_less.json",
                data_dir, db_name, db_name, query, dist
            );
            if let Ok(content) = fs::read_to_string(&el_path) {
                let data: Vec<ReconstructionDataPoint> = serde_json::from_str(&content)?;
                let true_points = extract_true_points(&data);

                let point_clouds =
                    generate_sampled_reconstructions(&data, search_domain, 5.0, 30.0, 5.0);
                let hulls = get_per_point_convex_hulls(&point_clouds);
                let (cent_mse, worst_mse) = compute_dataset_metrics(&hulls, &true_points);

                plot_data
                    .entry("even_less".to_string())
                    .or_default()
                    .push((query, cent_mse, worst_mse));
            }

            // ---------------------------------------------------------
            // 2. REMIN (Needs sampling)
            // ---------------------------------------------------------
            let remin_path = format!(
                "{}/{}/remin/{}_prob{}.0_{}_15x15_classic.json",
                data_dir, db_name, db_name, query, dist
            );
            if let Ok(content) = fs::read_to_string(&remin_path) {
                let data: Vec<ReconstructionDataPoint> = serde_json::from_str(&content)?;
                let true_points = extract_true_points(&data);

                let point_clouds =
                    generate_sampled_reconstructions(&data, search_domain, 5.0, 30.0, 5.0);
                let hulls = get_per_point_convex_hulls(&point_clouds);
                let (cent_mse, worst_mse) = compute_dataset_metrics(&hulls, &true_points);

                plot_data
                    .entry("remin".to_string())
                    .or_default()
                    .push((query, cent_mse, worst_mse));
            }

            // ---------------------------------------------------------
            // 3. LIMITS
            // ---------------------------------------------------------
            let limits_query = query / 100.0;
            let limits_path = format!(
                "{}/{}/limits/{}_{}_p{}.json",
                data_dir, db_name, db_name, dist, limits_query
            );

            if let Ok(content) = fs::read_to_string(&limits_path) {
                let multi_run_data: Vec<Vec<ReconstructionDataPoint>> =
                    serde_json::from_str(&content)?;

                if !multi_run_data.is_empty() {
                    let true_points = extract_true_points(&multi_run_data[0]);
                    let hulls = get_per_point_convex_hulls(&multi_run_data);
                    let (cent_mse, worst_mse) = compute_dataset_metrics(&hulls, &true_points);

                    plot_data
                        .entry("limits".to_string())
                        .or_default()
                        .push((query, cent_mse, worst_mse));
                }
            } else {
                warn!("Could not find limits file: {}", limits_path);
            }
        } // END QUERY LOOP

        db_data.insert(dist, plot_data);
    } // END DISTRIBUTION LOOP

    // ---------------------------------------------------------
    // 4. Generate the Combined Plot
    // ---------------------------------------------------------
    if !db_data.is_empty() {
        let output_svg = format!("figures/convex_lines/{}_convex_mse_combined.svg", db_name);
        plot_convex_side_by_side(db_name, &db_data, distributions, &output_svg)?;
        debug!(
            "Successfully generated combined convex hull plot at {}",
            output_svg
        );
    } else {
        warn!("No data found for {}, skipping plot.", db_name);
    }

    Ok(())
}

fn plot_convex_side_by_side(
    db_name: &str,
    db_data: &HashMap<&str, HashMap<String, Vec<(f64, f64, f64)>>>,
    distributions: &[&str],
    output_path: &str,
) -> Result<(), Box<dyn Error>> {
    let num_dists = distributions.len();
    if num_dists == 0 {
        return Ok(());
    }

    // REDUCED overall canvas size to make the plot area smaller
    let total_width = 850 * num_dists as u32;
    let total_height = 700;

    let root = SVGBackend::new(output_path, (total_width, total_height)).into_drawing_area();
    root.fill(&WHITE)?;

    let sub_areas = root.split_evenly((1, num_dists));
    let pretty_db = db_name.to_uppercase();

    for (i, &dist) in distributions.iter().enumerate() {
        let area = &sub_areas[i];

        let plot_data = match db_data.get(dist) {
            Some(data) if !data.is_empty() => data,
            _ => continue,
        };

        // 1. Find the Maximum Y value for THIS subplot
        let mut max_mse = 0.0_f64;
        for points in plot_data.values() {
            for &(_, cent_mse, worst_mse) in points {
                max_mse = max_mse.max(cent_mse).max(worst_mse);
            }
        }
        let y_max = if max_mse == 0.0 { 1.0 } else { max_mse * 1.15 };

        let pretty_dist = format_title_case(dist);
        let title = format!("{} - {} Dist.", pretty_db, pretty_dist);

        // Increased title space slightly to support larger font
        let (title_area, chart_area) = area.split_vertically(140);
        let centered_title_area = title_area.margin(0, 0, 180, 20);

        ChartBuilder::on(&centered_title_area)
            .caption(
                title,
                ("Linux Biolinum", 76, FontStyle::Bold) // MASSIVE TITLE
                    .into_font()
                    .color(&BLACK),
            )
            .build_cartesian_2d(0..1, 0..1)?;

        let mut chart = ChartBuilder::on(&chart_area)
            .margin_top(10)
            .margin_bottom(30)
            .margin_left(60)
            .margin_right(60)
            .x_label_area_size(140) // Increased area for massive x-axis text
            .y_label_area_size(200) // Increased area for massive y-axis text
            .build_cartesian_2d(0.0..40.0f64, 0.0..y_max)?;

        chart
            .configure_mesh()
            .bold_line_style(RGBColor(230, 230, 230))
            .light_line_style(TRANSPARENT)
            .axis_style(RGBColor(100, 100, 100))
            .x_desc("Query Percentage (%)")
            .y_desc("Mean Squared Error (MSE)")
            .x_labels(5) // Hits 0, 10, 20, 30, 40
            .y_labels(6)
            .axis_desc_style(("Linux Biolinum", 86, FontStyle::Bold).into_font()) // MASSIVE AXIS DESC
            .label_style(("Linux Biolinum", 68).into_font()) // MASSIVE LABELS
            .x_label_formatter(&|x| format!("{:.0}%", x))
            .y_label_formatter(&|y| format!("{:.2}", y))
            .draw()?;

        // Grab methods and sort by Rank so LAMa (Blue) is always first
        let mut mapped_methods: Vec<(&String, &str, usize, RGBColor)> = plot_data
            .keys()
            .map(|method| {
                let (pretty, rank, color) = get_method_style(method);
                (method, pretty, rank, color)
            })
            .collect();
        mapped_methods.sort_by_key(|&(_, _, rank, _)| rank);

        for (raw_method, pretty_name, _, color) in mapped_methods {
            let mut sorted_data = plot_data[raw_method].clone();
            sorted_data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            let centroid_points: Vec<_> = sorted_data.iter().map(|(q, c, _)| (*q, *c)).collect();
            let worst_points: Vec<_> = sorted_data.iter().map(|(q, _, w)| (*q, *w)).collect();

            // --- A: SOLID LINE (Worst-Case Midpoint) ---
            chart
                .draw_series(LineSeries::new(worst_points.clone(), color.stroke_width(6)))?
                .label(pretty_name) // SIMPLIFIED: Just the method name
                .legend(move |(x, y)| {
                    // SIMPLIFIED: Massive colored rectangle
                    Rectangle::new([(x, y - 18), (x + 50, y + 18)], color.filled())
                });

            chart.draw_series(
                worst_points
                    .iter()
                    .map(|(x, y)| Circle::new((*x, *y), 15, color.filled())),
            )?;

            // --- B: DASHED LINE (Centroid Barycenter) ---
            // Notice: We omit `.label()` and `.legend()` entirely here
            chart.draw_series(DashedLineSeries::new(
                centroid_points.clone(),
                15, // Dash length
                10, // Space length
                color.stroke_width(5),
            ))?;

            chart.draw_series(
                centroid_points
                    .iter()
                    .map(|(x, y)| Circle::new((*x, *y), 12, color.stroke_width(4))),
            )?;
        }

        // Apply simplified, ultra-accessible legend styling
        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::MiddleRight)
            .background_style(RGBColor(255, 255, 255).mix(0.95))
            .border_style(TRANSPARENT) // Removed border for cleaner look
            .label_font(("Linux Biolinum", 56, FontStyle::Bold).into_font()) // MASSIVE LEGEND FONT
            .margin(20)
            .draw()?;
    }

    root.present()?;
    Ok(())
}
