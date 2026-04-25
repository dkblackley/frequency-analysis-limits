use crate::plotting::post::calculate_mse;
use crate::plotting::two_d::format_db_name;
use crate::plotting::{load_limits_method, load_standard_method};
use log::error;
use plotters::prelude::*;
use std::collections::HashMap;
use std::error::Error;

pub fn plot_grid_by_mse(
    grid_sizes: &[(u32, String)],
    datasets: &[&str],
    methods: &[&str],
    distributions: &[&str],
) -> Result<(), Box<dyn Error>> {
    let base_dir = "databases";

    for &db in datasets {
        // Data structure: dist -> method -> vec of (grid_size, mse)
        let mut db_data: HashMap<&str, HashMap<String, Vec<(u32, f64)>>> = HashMap::new();
        let mut path = "".to_string();

        for &dist in distributions {
            let mut plot_data: HashMap<String, Vec<(u32, f64)>> = HashMap::new();

            for &method in methods {
                for (grid_val, grid_str) in grid_sizes {
                    let mse_result = match method {
                        "even_less" => {
                            path = format!(
                                "{}/{}/{}/even_less/{}_prob100.0_{}_{}_even_less.json",
                                base_dir, grid_str, db, db, dist, grid_str
                            );
                            load_standard_method(&path).map(|data| calculate_mse(&data))
                        }
                        "remin" => {
                            path = format!(
                                "{}/{}/{}/remin/{}_prob100.0_{}_{}_classic.json",
                                base_dir, grid_str, db, db, dist, grid_str
                            );
                            load_standard_method(&path).map(|data| calculate_mse(&data))
                        }
                        "limits" => {
                            path = format!(
                                "{}/{}/{}/limits/{}_{}_e0_d0.9_reconstruction.json",
                                base_dir, grid_str, db, db, dist
                            );
                            load_limits_method(&path).map(|reconstructions| {
                                reconstructions
                                    .iter()
                                    .map(|recon| calculate_mse(recon))
                                    .filter(|m| m.is_finite()) // Prevents propagating NaNs
                                    .fold(f64::INFINITY, f64::min)
                            })
                        }
                        _ => continue,
                    };

                    match mse_result {
                        // The is_finite check prevents freezing if limits folded to Infinity
                        Ok(mse) if mse.is_finite() => {
                            plot_data
                                .entry(method.to_string())
                                .or_default()
                                .push((grid_val.clone(), mse));
                        }
                        Ok(_) => {
                            // Because our method is perfect, MSE sometimes accidentally becomes NAN
                            // due to floating point/division errors (I think?).
                            plot_data
                                .entry(method.to_string())
                                .or_default()
                                .push((grid_val.clone(), 0.0));
                        }
                        Err(e) => {
                            // Replace with log::error! or your preferred macro
                            error!("failed to plot for {method}, {dist}, {db} at {grid_str}. Path was {path}: {e}")
                        }
                    }
                }
            }
            db_data.insert(dist, plot_data);
        }

        if !db_data.is_empty() {
            let output_path = format!("figures/mse_grid_size/mse_grid_{}.svg", db);
            plot_db_side_by_side(db, &db_data, distributions, &output_path)?;
            println!(
                "Generated combined plot for {} -> saved to {}",
                db, output_path
            );
        }
    }

    Ok(())
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

fn plot_db_side_by_side(
    db: &str,
    db_data: &HashMap<&str, HashMap<String, Vec<(u32, f64)>>>,
    distributions: &[&str],
    output_path: &str,
) -> Result<(), Box<dyn Error>> {
    let num_dists = distributions.len();
    if num_dists == 0 {
        return Ok(());
    }

    let total_width = 900 * num_dists as u32;
    let total_height = 600;

    let root = SVGBackend::new(output_path, (total_width, total_height)).into_drawing_area();
    root.fill(&TRANSPARENT)?;

    let sub_areas = root.split_evenly((1, num_dists));
    let pretty_db = format_db_name(db);

    for (i, &dist) in distributions.iter().enumerate() {
        let area = &sub_areas[i];

        let plot_data = match db_data.get(dist) {
            Some(data) if !data.is_empty() => data,
            _ => continue,
        };

        // Determine min/max boundaries manually
        let mut min_grid = f64::INFINITY;
        let mut max_grid = f64::NEG_INFINITY;
        let mut min_mse = f64::INFINITY;
        let mut max_mse = f64::NEG_INFINITY;

        for points in plot_data.values() {
            for &(g, m) in points {
                let g_f = g as f64;
                if g_f < min_grid {
                    min_grid = g_f;
                }
                if g_f > max_grid {
                    max_grid = g_f;
                }
                if m < min_mse {
                    min_mse = m;
                }
                if m > max_mse {
                    max_mse = m;
                }
            }
        }

        if min_grid.is_infinite() || min_mse.is_infinite() {
            continue;
        }

        let mut x_pad = (max_grid - min_grid) * 0.1;
        if x_pad == 0.0 {
            x_pad = 1.0;
        }

        x_pad = 0.0;

        // Ensure Y-max is at least 10.0 so the scale makes sense even with tiny numbers
        let y_max = if max_mse <= 1.0 { 10.0 } else { max_mse };

        let mut chars = dist.chars();
        let pretty_dist = match chars.next() {
            None => String::new(),
            Some(f) => f.to_uppercase().collect::<String>() + chars.as_str(),
        };

        // THE FIX: Changed "Distribution" to "Dist."
        let title = format!("{} - {} Dist.", pretty_db, pretty_dist);

        // THE FIX: Increased from 90 to 120 pixels to prevent intersection with the chart
        let (title_area, chart_area) = area.split_vertically(120);

        // THE FIX: Pushed left margin from 110 to 160 to center the title better
        // over the newly expanded chart area beneath it
        let centered_title_area = title_area.margin(0, 0, 160, 40);

        // Draw the title safely in its dedicated space
        ChartBuilder::on(&centered_title_area)
            .caption(
                title,
                ("Linux Biolinum", 64, FontStyle::Bold)
                    .into_font()
                    .color(&BLACK),
            )
            .build_cartesian_2d(0..1, 0..1)?;

        let mut chart = ChartBuilder::on(&chart_area)
            .margin_top(10) // Top spacing is naturally handled by the title_area above it
            .margin_bottom(0)
            .margin_left(50)
            .margin_right(50)
            .x_label_area_size(120)
            // THE FIX: Increased to 180 to give the Y-axis numbers and text plenty of room
            .y_label_area_size(180)
            .build_cartesian_2d(
                (min_grid - x_pad)..(max_grid + x_pad),
                (1.0f64..y_max).log_scale(), // Implement Log scale starting at 1.0
            )?;

        chart
            .configure_mesh()
            .bold_line_style(RGBColor(230, 230, 230))
            .light_line_style(TRANSPARENT)
            .axis_style(RGBColor(100, 100, 100))
            .x_desc("Grid Size")
            // THE FIX: Changed "Mean Squared Error" to "MSE"
            .y_desc("MSE")
            // THE FIX: Bumped axis text size up from 60 to 72
            .x_labels(6)
            .axis_desc_style(("Linux Biolinum", 80, FontStyle::Bold).into_font())
            .x_label_formatter(&|x| format_metric(*x))
            .y_label_formatter(&|y| {
                if *y <= 1.001 {
                    // Intercept the 1.0 tick and visually label it as "0"
                    "0".to_string()
                } else {
                    format_metric(*y)
                }
            })
            // THE FIX: Bumped axis tick numbers up from 54 to 60
            .label_style(("Linux Biolinum", 60).into_font())
            .draw()?;

        // Grab the methods and explicitly map them to their formatted name and rank order
        let mut mapped_methods: Vec<(&String, &str, usize, RGBColor)> = plot_data
            .keys()
            .map(|method| match method.as_str() {
                "limits" => (method, "LAMa", 0, RGBColor(0, 114, 178)), // Blue
                "even_less" => (method, "Even Less", 1, RGBColor(230, 159, 0)), // Orange/Yellow
                "remin" => (method, "Remin", 2, RGBColor(0, 158, 115)), // Green
                _ => (method, method.as_str(), 99, RGBColor(0, 0, 0)),  // Fallback
            })
            .collect();

        // Sort by the rank we just assigned so LAMa renders first (top of legend)
        mapped_methods.sort_by_key(|&(_, _, rank, _)| rank);

        for (raw_method, pretty_name, _, color) in mapped_methods {
            let mut sorted_data = plot_data[raw_method].clone();
            sorted_data.sort_by(|a, b| a.0.cmp(&b.0));

            let continuous_data: Vec<_> = sorted_data
                .iter()
                .map(|(g, m)| {
                    // Clamp values to a minimum of 1.0 so they cleanly sit on the "0" axis line
                    (*g as f64, m.max(1.0))
                })
                .collect();

            // Draw line
            chart
                .draw_series(LineSeries::new(
                    continuous_data.clone(),
                    color.stroke_width(6),
                ))?
                .label(pretty_name) // Use the nice label
                .legend(move |(x, y)| {
                    PathElement::new(vec![(x, y), (x + 25, y)], color.stroke_width(6))
                });

            // Draw smooth solid dots
            chart.draw_series(
                continuous_data
                    .iter()
                    .map(|(x, y)| Circle::new((*x, *y), 15, color.filled())),
            )?;
        }

        chart
            .configure_series_labels()
            // Change UpperRight to MiddleRight
            .position(SeriesLabelPosition::LowerRight)
            // I've added the solid background back here, but you can leave it TRANSPARENT if you prefer!
            .background_style(RGBColor(255, 255, 255).mix(0.9))
            .border_style(RGBColor(200, 200, 200))
            .label_font(("Linux Biolinum", 40, FontStyle::Bold).into_font())
            .margin(10)
            .draw()?;
    }

    root.present()?;
    Ok(())
}
