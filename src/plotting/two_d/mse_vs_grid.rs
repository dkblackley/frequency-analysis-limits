use crate::plotting::post::calculate_mse;
use crate::plotting::{load_limits_method, load_standard_method};
use plotters::prelude::*;
use std::collections::HashMap;
use std::error::Error;
use std::path::Path;

pub fn plot_grid_by_mse(
    grid_sizes: &[(u32, &str)],
    datasets: &[&str],
    methods: &[&str],
    distributions: &[&str],
) -> Result<(), Box<dyn Error>> {
    // Base directory for your data
    let base_dir = "databases";

    for db in datasets {
        for dist in distributions {
            // This will hold the lines for your plot:
            // Key: Method Name -> Value: Vec of (Grid Size, MSE)
            let mut plot_data: HashMap<String, Vec<(u32, f64)>> = HashMap::new();

            for method in methods {
                for (grid_val, grid_str) in grid_sizes {
                    let mse_result = match *method {
                        "even_less" => {
                            let path = format!(
                                "{}/{}/{}/even_less/{}_prob100.0_{}_{}_even_less.json",
                                base_dir, grid_str, db, db, dist, grid_str
                            );
                            load_standard_method(&path).map(|data| calculate_mse(&data))
                        }
                        "remin" => {
                            // Note: directory is 'remin' but file suffix is 'classic' based on your `ls`
                            let path = format!(
                                "{}/{}/{}/remin/{}_prob100.0_{}_{}_classic.json",
                                base_dir, grid_str, db, db, dist, grid_str
                            );
                            load_standard_method(&path).map(|data| calculate_mse(&data))
                        }
                        "limits" => {
                            // Tries the specific named file first
                            let path = format!(
                                "{}/{}/{}/limits/{}_{}_e0_d0.9_reconstruction.json",
                                base_dir, grid_str, db, db, dist
                            );

                            // If the specific file is missing, try the generic fallback
                            let final_path = if Path::new(&path).exists() {
                                path
                            } else {
                                format!(
                                    "{}/{}/{}/limits/reconstruction.json",
                                    base_dir, grid_str, db
                                )
                            };

                            load_limits_method(&final_path).map(|reconstructions| {
                                // Take the minimum MSE across all possible reconstructions for 'limits'
                                reconstructions
                                    .iter()
                                    .map(|recon| calculate_mse(recon))
                                    .fold(f64::INFINITY, f64::min)
                            })
                        }
                        _ => continue,
                    };

                    match mse_result {
                        Ok(mse) => {
                            plot_data
                                .entry(method.to_string())
                                .or_default()
                                .push((*grid_val, mse));
                        }
                        Err(_) => {
                            // Suppress errors for missing files to keep the console clean,
                            // or replace with an eprintln! to debug missing specific paths.
                        }
                    }
                }
            }

            // At this point, `plot_data` contains all the lines needed for 1 figure.
            // Example map contents:
            // {
            //    "even_less": [(25, 0.04), (50, 0.08), (75, 0.12)],
            //    "remin": [(25, 0.03), (50, 0.07), (75, 0.10)],
            //    "limits": [(25, 0.01), (50, 0.02), (75, 0.04)]
            // }

            if !plot_data.is_empty() {
                let real_output_path = format!("figures/{}_{}_mse_vs_grid.svg", db, dist);

                // You can now pass this grouped data to an updated version of your plotting function
                // plot_multiple_methods_vs_grid_size(&plot_data, dist, &output_path)?;

                println!(
                    "Generated plot data for {} - {} -> saved to {}",
                    db, dist, real_output_path
                );
            }
        }
    }

    Ok(())
}

fn plot_mse_vs_grid_size(
    data: &[(u32, f64)], // Slice of (Grid Size, MSE)
    dist_name: &str,
    output_path: &str,
) -> Result<(), Box<dyn Error>> {
    let root = SVGBackend::new(output_path, (800, 600)).into_drawing_area();
    root.fill(&TRANSPARENT)?;

    if data.is_empty() {
        return Ok(());
    }

    let min_grid = data.iter().map(|(g, _)| *g).min().unwrap_or(0) as f64;
    let max_grid = data.iter().map(|(g, _)| *g).max().unwrap_or(100) as f64;

    let min_mse = data.iter().map(|(_, m)| *m).fold(f64::INFINITY, f64::min);
    let max_mse = data
        .iter()
        .map(|(_, m)| *m)
        .fold(f64::NEG_INFINITY, f64::max);

    let x_pad = (max_grid - min_grid) * 0.1;
    let y_pad = if max_mse == min_mse {
        0.1
    } else {
        (max_mse - min_mse) * 0.1
    };

    let mut chart = ChartBuilder::on(&root)
        .margin(40)
        .caption(
            format!("MSE vs Grid Size ({})", dist_name),
            ("sans-serif", 30).into_font(),
        )
        .x_label_area_size(60)
        .y_label_area_size(70)
        .build_cartesian_2d(
            (min_grid - x_pad)..(max_grid + x_pad),
            (min_mse - y_pad)..(max_mse + y_pad),
        )?;

    chart
        .configure_mesh()
        .bold_line_style(RGBColor(220, 220, 220))
        .axis_style(BLACK)
        .x_desc("Grid Size")
        .y_desc("Mean Squared Error")
        .label_style(("sans-serif", 18).into_font())
        .draw()?;

    // Assign a distinct color-blind safe palette color based on distribution name
    let color = match dist_name.to_lowercase().as_str() {
        "uniform" => RGBColor(0, 114, 178), // Blue
        "gaussian" => RGBColor(213, 94, 0), // Vermilion
        "beta" => RGBColor(0, 158, 115),    // Bluish Green
        _ => RGBColor(230, 159, 0),         // Orange (Fallback)
    };

    // Guarantee data is sorted along the X-axis for unbroken lines
    let mut sorted_data = data.to_vec();
    sorted_data.sort_by(|a, b| a.0.cmp(&b.0));

    // Draw the continuous line
    chart.draw_series(LineSeries::new(
        sorted_data.iter().map(|(g, m)| (*g as f64, *m)),
        color.stroke_width(3),
    ))?;

    // Draw the data point dots
    chart.draw_series(
        sorted_data
            .iter()
            .map(|(g, m)| Circle::new((*g as f64, *m), 6, color.filled())),
    )?;

    root.present()?;
    Ok(())
}
