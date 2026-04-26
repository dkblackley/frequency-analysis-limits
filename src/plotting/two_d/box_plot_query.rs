use crate::plotting::post::calculate_mse;
use crate::plotting::two_d::worst_case_convex_hull::{
    compute_distance_sq, format_title_case, generate_sampled_reconstructions, get_method_style,
};
use crate::plotting::two_d::{format_db_name, format_dist_name};
use crate::plotting::ReconstructionDataPoint;
use log::{debug, warn};
use plotters::data::Quartiles;
use plotters::prelude::*;
use std::collections::HashMap;
use std::error::Error;
use std::fs;
// (Keep your existing `get_method_style`, `apply_transform`, `generate_sampled_reconstructions`,
// `format_title_case`, and `compute_distance_sq` functions exactly as they are).

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

/// The Wrapper: Runs your function 10,000 times to get the distribution for the Box Plot
pub fn compute_reconstruction_mses(point_clouds: &[Vec<ReconstructionDataPoint>]) -> Vec<f64> {
    point_clouds
        .iter()
        .map(|single_point_cloud| calculate_mse(single_point_cloud))
        .collect() // Returns a Vec of 10,000 MSEs!
}
/// Orchestrates reading files for ALL distributions and calculating their MSE distributions
pub fn process_and_plot_boxplots(
    db_name: &str,
    data_dir: &str,
    distributions: &[&str],
) -> Result<(), Box<dyn Error>> {
    // Structure: dist -> method -> vec of (query, vec_of_mses)
    let mut db_data: HashMap<&str, HashMap<String, Vec<(f64, Vec<f64>)>>> = HashMap::new();
    let search_domain = (20, 20);
    let query_percents = vec![10.0, 20.0, 30.0];

    for &dist in distributions {
        let mut plot_data: HashMap<String, Vec<(f64, Vec<f64>)>> = HashMap::new();

        for &query in &query_percents {
            // 1. EVEN LESS (Needs sampling)
            let el_path = format!(
                "{}/{}/even_less/{}_prob{}.0_{}_15x15_even_less.json",
                data_dir, db_name, db_name, query, dist
            );
            if let Ok(content) = fs::read_to_string(&el_path) {
                let data: Vec<ReconstructionDataPoint> = serde_json::from_str(&content)?;
                let point_clouds =
                    generate_sampled_reconstructions(&data, search_domain, 5.0, 30.0, 5.0);
                let mses = compute_reconstruction_mses(&point_clouds);

                plot_data
                    .entry("even_less".to_string())
                    .or_default()
                    .push((query, mses));
            }

            // 2. REMIN (Needs sampling)
            let remin_path = format!(
                "{}/{}/remin/{}_prob{}.0_{}_15x15_classic.json",
                data_dir, db_name, db_name, query, dist
            );
            if let Ok(content) = fs::read_to_string(&remin_path) {
                let data: Vec<ReconstructionDataPoint> = serde_json::from_str(&content)?;
                let point_clouds =
                    generate_sampled_reconstructions(&data, search_domain, 5.0, 30.0, 5.0);
                let mses = compute_reconstruction_mses(&point_clouds);

                plot_data
                    .entry("remin".to_string())
                    .or_default()
                    .push((query, mses));
            }

            // 3. LIMITS (Pre-sampled 10k)
            let limits_query = query / 100.0;
            let limits_path = format!(
                "{}/{}/limits/{}_{}_p{}_reconstruction.json",
                data_dir, db_name, db_name, dist, limits_query
            );
            if let Ok(content) = fs::read_to_string(&limits_path) {
                let multi_run_data: Vec<Vec<ReconstructionDataPoint>> =
                    serde_json::from_str(&content)?;
                if !multi_run_data.is_empty() {
                    let mses = compute_reconstruction_mses(&multi_run_data);
                    plot_data
                        .entry("limits".to_string())
                        .or_default()
                        .push((query, mses));
                }
            } else {
                warn!("Could not find limits file: {}", limits_path);
            }
        }

        db_data.insert(dist, plot_data);
    }

    if !db_data.is_empty() {
        let output_svg = format!("figures/box/{}_mse_distribution_boxplot.svg", db_name);
        plot_boxplot_side_by_side(db_name, &db_data, distributions, &output_svg)?;
        debug!("Successfully generated boxplot at {}", output_svg);
    }

    Ok(())
}

fn plot_boxplot_side_by_side(
    db_name: &str,
    db_data: &HashMap<&str, HashMap<String, Vec<(f64, Vec<f64>)>>>,
    distributions: &[&str],
    output_path: &str,
) -> Result<(), Box<dyn Error>> {
    let num_dists = distributions.len();
    if num_dists == 0 {
        return Ok(());
    }

    let total_width = 850 * num_dists as u32;
    // 1. INCREASED overall height to accommodate the super title
    let total_height = 850;

    let root = SVGBackend::new(output_path, (total_width, total_height)).into_drawing_area();
    root.fill(&WHITE)?;

    // 2. CREATE THE SUPER TITLE AREA
    // Slice off the top 140 pixels across the entire width for our main title
    let (title_area, body_area) = root.split_vertically(80);
    let pretty_db = format_db_name(db_name);
    let super_title = format!("{}", pretty_db);

    ChartBuilder::on(&title_area)
        .caption(
            super_title,
            ("Linux Biolinum", 96, FontStyle::Bold)
                .into_font()
                .color(&BLACK),
        )
        .build_cartesian_2d(0..1, 0..1)?;

    // 3. SPLIT THE REMAINING BODY AREA FOR THE SUBPLOTS
    let sub_areas = body_area.split_evenly((1, num_dists));

    for (i, &dist) in distributions.iter().enumerate() {
        let area = &sub_areas[i];
        let plot_data = match db_data.get(dist) {
            Some(data) if !data.is_empty() => data,
            _ => continue,
        };

        // Check for crazy outliers to see if we need a scale larger than 10,000
        let mut max_mse = 10000.0_f32;
        for points in plot_data.values() {
            for &(_, ref mses) in points {
                for &val in mses {
                    if val as f32 > max_mse {
                        max_mse = val as f32;
                    }
                }
            }
        }

        // Dynamically scale up the power of 10 if needed, but baseline is 10,000
        let y_max = 10f32.powf(max_mse.log10().ceil());

        let pretty_dist = format_dist_name(dist);
        // Sub-title now only contains the distribution to avoid text collision
        let title = format!("{} Dist.", pretty_dist);

        let (sub_title_area, chart_area) = area.split_vertically(100);
        let centered_sub_title_area = sub_title_area.margin(0, 0, 0, 0);

        ChartBuilder::on(&centered_sub_title_area)
            .caption(
                title,
                ("Linux Biolinum", 76, FontStyle::Bold)
                    .into_font()
                    .color(&BLACK),
            )
            .build_cartesian_2d(0..1, 0..1)?;

        let mut chart = ChartBuilder::on(&chart_area)
            .margin_top(20)
            .margin_bottom(30)
            .margin_left(55)
            .margin_right(55)
            .x_label_area_size(250)
            // INCREASED Y label area size so '10000' fits nicely without clipping
            .y_label_area_size(270)
            // 4. APPLY LOG SCALE (Base 10) FROM 10 TO y_max
            .build_cartesian_2d(5.0..35.0f64, (10.0f32..y_max).log_scale())?;

        chart
            .configure_mesh()
            .bold_line_style(RGBColor(230, 230, 230))
            .light_line_style(TRANSPARENT)
            .axis_style(RGBColor(100, 100, 100))
            .x_desc("Query Percent")
            .y_desc("MSE")
            .x_labels(4)
            // A hint to Plotters to try and generate 5 major ticks (powers of 10)
            .y_labels(5)
            .axis_desc_style(("Linux Biolinum", 86, FontStyle::Bold).into_font())
            .label_style(("Linux Biolinum", 68).into_font())
            .x_label_formatter(&|x| format!("{:.0}%", x))
            // Format labels normally (will naturally render 10, 100, 1000, 10000)
            .y_label_formatter(&|y| format_metric(*y as f64))
            .draw()?;

        let mut mapped_methods: Vec<(&String, &str, usize, RGBColor, i32)> = plot_data
            .keys()
            .map(|method| {
                let (pretty, rank, color) = get_method_style(method);
                let offset = match rank {
                    0 => -30, // Shift LAMa left
                    1 => 0,   // Keep Even Less centered
                    2 => 30,  // Shift Remin right
                    _ => 0,
                };
                (method, pretty, rank, color, offset)
            })
            .collect();
        mapped_methods.sort_by_key(|&(_, _, rank, _, _)| rank);

        for (raw_method, pretty_name, _, color, offset_px) in mapped_methods {
            let mut sorted_data = plot_data[raw_method].clone();
            sorted_data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            let mut boxplots = Vec::new();
            for (q, mses) in sorted_data {
                if mses.is_empty() {
                    continue;
                }
                let quartiles = plotters::data::Quartiles::new(&mses);

                // LAYER 1: The Shading (Flat UI Light Fill)
                // We mix the color heavily with white (e.g., 20% color, 80% white)
                boxplots.push(
                    Boxplot::new_vertical(q, &quartiles)
                        .width(35) // Made wider (from 25 to 35) for better visibility
                        .whisker_width(0.6)
                        .style(color.mix(0.65).filled())
                        .offset(offset_px),
                );

                // LAYER 2: The Lines (Thick, bold borders and whiskers)
                // We use the pure, solid color and apply a stroke width
                boxplots.push(
                    Boxplot::new_vertical(q, &quartiles)
                        .width(35) // Must match the width of Layer 1
                        .whisker_width(0.6)
                        .style(color.stroke_width(4)) // Thick flat lines!
                        .offset(offset_px),
                );
            }

            // Draw both layers to the chart
            chart
                .draw_series(boxplots)?
                .label(pretty_name)
                .legend(move |(x, y)| {
                    // Update the legend to reflect the Flat UI style:
                    // Draw a solid, slightly shorter block so it looks like a thick UI element
                    Rectangle::new([(x, y - 10), (x + 30, y + 10)], color.filled())
                });
        }

        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::UpperRight)
            .background_style(RGBColor(255, 255, 255).mix(0.15))
            .border_style(TRANSPARENT)
            .label_font(("Linux Biolinum", 44, FontStyle::Bold).into_font())
            .margin(11)
            .draw()?;
    }

    root.present()?;
    Ok(())
}
