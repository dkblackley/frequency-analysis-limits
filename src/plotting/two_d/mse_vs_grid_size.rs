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
            let output_path = format!("figures/{}_mse_vs_grid_combined.svg", db);
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

    // Allocate 800px width per distribution subplot
    let total_width = 800 * num_dists as u32;
    let total_height = 600;

    let root = SVGBackend::new(output_path, (total_width, total_height)).into_drawing_area();

    // Explicitly enforce transparency over the entire canvas block
    root.fill(&TRANSPARENT)?;

    // Horizontally split our parent drawing area by the number of distributions
    let sub_areas = root.split_evenly((1, num_dists));
    let pretty_db = format_db_name(db);

    // High-contrast, colorblind-friendly palette (Wong)
    let palette = [
        RGBColor(0, 114, 178),   // Blue
        RGBColor(213, 94, 0),    // Vermilion
        RGBColor(0, 158, 115),   // Bluish Green
        RGBColor(204, 121, 167), // Reddish Purple
        RGBColor(230, 159, 0),   // Orange
        RGBColor(86, 180, 233),  // Sky Blue
    ];

    for (i, &dist) in distributions.iter().enumerate() {
        let area = &sub_areas[i];

        let plot_data = match db_data.get(dist) {
            Some(data) if !data.is_empty() => data,
            _ => continue, // Skip entirely if no lines populated this subplot
        };

        // Determine min/max boundaries manually to frame out the graph space
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
            continue; // No valid finite points to build the mesh
        }

        // Prevent infinite tick-generation loops if min == max
        let mut x_pad = (max_grid - min_grid) * 0.1;
        if x_pad == 0.0 {
            x_pad = 1.0;
        }

        let mut y_pad = (max_mse - min_mse) * 0.1;
        if y_pad == 0.0 {
            y_pad = 0.1;
        }

        // Formatting the title: Capitalize the distribution strings nicely
        let mut chars = dist.chars();
        let pretty_dist = match chars.next() {
            None => String::new(),
            Some(f) => f.to_uppercase().collect::<String>() + chars.as_str(),
        };

        // Minimalist Title Update
        let title = format!("{} {} Query Distribution", pretty_db, pretty_dist);

        let mut chart = ChartBuilder::on(area)
            .margin(40)
            // Matched font family and weight to your previous graphs
            .caption(
                title,
                ("Linux Biolinum", 32, FontStyle::Bold)
                    .into_font()
                    .color(&BLACK),
            )
            .x_label_area_size(60)
            .y_label_area_size(70) // Accommodates wider 'k' / 'M' labels
            .build_cartesian_2d(
                (min_grid - x_pad)..(max_grid + x_pad),
                (min_mse - y_pad)..(max_mse + y_pad),
            )?;

        // Sleek configuration: lighter grid lines and distinct axis lines
        chart
            .configure_mesh()
            .bold_line_style(RGBColor(230, 230, 230))
            .light_line_style(TRANSPARENT)
            .axis_style(RGBColor(100, 100, 100))
            .x_desc("Grid Size")
            .y_desc("Mean Squared Error")
            // Added formatting metrics to X and Y axes
            .x_label_formatter(&|x| format_metric(*x))
            .y_label_formatter(&|y| format_metric(*y))
            // Matched font family to your previous graphs
            .label_style(("Linux Biolinum", 18).into_font())
            .draw()?;

        // Stabilize legend rendering sequence by alphabetically sorting methods
        let mut methods: Vec<_> = plot_data.keys().collect();
        methods.sort();

        for (color_idx, &method) in methods.iter().enumerate() {
            let color = palette[color_idx % palette.len()];

            // Unbroken lines demand strict X-axis sorting
            let mut sorted_data = plot_data[method].clone();
            sorted_data.sort_by(|a, b| a.0.cmp(&b.0));

            let continuous_data: Vec<_> =
                sorted_data.iter().map(|(g, m)| (*g as f64, *m)).collect();

            // Draw line
            chart
                .draw_series(LineSeries::new(
                    continuous_data.clone(),
                    color.stroke_width(4),
                ))?
                .label(method.clone())
                .legend(move |(x, y)| {
                    PathElement::new(vec![(x, y), (x + 25, y)], color.stroke_width(4))
                });

            // Draw smooth solid dots
            chart.draw_series(
                continuous_data
                    .iter()
                    .map(|(x, y)| Circle::new((*x, *y), 8, color.filled())),
            )?;
        }

        // Beautiful academic legend placement
        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::UpperRight)
            .background_style(RGBColor(255, 255, 255).mix(0.9))
            .border_style(RGBColor(200, 200, 200))
            .label_font(("Linux Biolinum", 16)) // Matched font family here as well
            .margin(10)
            .draw()?;
    }

    // Flush and save the SVG
    root.present()?;
    Ok(())
}
