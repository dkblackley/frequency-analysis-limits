use comfy_table::{presets::UTF8_FULL, Cell, Color as TableColor, Table};
use plotters::prelude::*;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize, Serialize)]
pub struct DbResult {
    pub name: String,
    pub method: String,
    pub dims: u32,
    pub mse: Option<f64>,
    pub match_rate: Option<f64>,
    pub chamfer: Option<f64>,
    pub number_of_reconstructions: String,
    pub time_taken: f64,
    pub total_db_size: u64,
    pub percent_queries_used: f64,
}

// #[derive(Debug, Serialize, Deserialize)]
// pub struct ReconstructionData2d {
//     #[serde(rename = "true")]
//     pub true_points: Vec<(f64, f64)>,
//     #[serde(rename = "reconstructed")]
//     pub reconstructed_points: Vec<(f64, f64)>,
// }

#[derive(Debug, Deserialize, Serialize)]
pub struct DataWrapper {
    #[serde(rename = "mapping")]
    pub mapping: Vec<ReconstructionData2dPoint>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ReconstructionData2dPoint {
    #[serde(rename = "true")]
    pub true_points: (f64, f64),
    #[serde(rename = "reconstructed")]
    pub reconstructed_points: (f64, f64),
}

pub struct Plotter {
    pub x_padder: f64,
    pub y_padder: f64,
}

impl Plotter {
    /// Generates a sleek CLI table for a vector of DbResult structs.
    pub fn make_table(&self, dir_paths: &[String]) {
        let mut all_results = Vec::new();

        for dir_path in dir_paths {
            let dir = Path::new(dir_path);

            // 1. Handle Results Table
            let results_path = dir.join("results.json");
            if results_path.exists() {
                let content = fs::read_to_string(&results_path).unwrap();
                all_results.push(serde_json::from_str::<DbResult>(&content).unwrap());
            }
        }

        if !all_results.is_empty() {
            self.print_results_table(&all_results);
        }
    }

    pub fn handle_spatial_plot(&self, dir_paths: &[String], show_true_points: bool) {
        for dir_path in dir_paths {
            let dir = Path::new(dir_path);

            // 2. Handle Spatial Plotting
            let recon_path = dir.join("reconstruction.json");
            if recon_path.exists() {
                let content = fs::read_to_string(&recon_path).unwrap();
                let data: Vec<ReconstructionData2dPoint> = serde_json::from_str(&content).unwrap();

                let (true_points, mut reconstructed): (Vec<_>, Vec<_>) = data
                    .into_iter()
                    .map(|p| (p.true_points, p.reconstructed_points))
                    .unzip();

                //TODO: Move this into 'post' crate.

                // if dir_path.contains("lili") {
                //     // 1. Find the current min and max for X and Y
                //     let min_x = reconstructed
                //         .iter()
                //         .map(|rec| rec.0 as f64)
                //         .fold(f64::INFINITY, |a, b| a.min(b));
                //     let max_x = reconstructed
                //         .iter()
                //         .map(|rec| rec.0 as f64)
                //         .fold(f64::NEG_INFINITY, |a, b| a.max(b));
                //
                //     let min_y = reconstructed
                //         .iter()
                //         .map(|rec| rec.1 as f64)
                //         .fold(f64::INFINITY, |a, b| a.min(b));
                //     let max_y = reconstructed
                //         .iter()
                //         .map(|rec| rec.1 as f64)
                //         .fold(f64::NEG_INFINITY, |a, b| a.max(b));
                //
                //     // Calculate the current range.
                //     // We use max(1e-6) to prevent dividing by zero if all points share the exact same axis.
                //     let range_x = (max_x - min_x).max(1e-6);
                //     let range_y = (max_y - min_y).max(1e-6);
                //     let target_max = 50.0;
                //
                //     // 2. Scale the points using the Min-Max formula
                //     let mut scaled: Vec<(f64, f64)> = Vec::with_capacity(reconstructed.len());
                //
                //     for rec in &reconstructed {
                //         let x = rec.0 as f64;
                //         let y = rec.1 as f64;
                //
                //         let shifted_x = (((x - min_x) / range_x) * 10.0).round();
                //         let shifted_y = (((y - min_y) / range_y) * 53.0).round();
                //
                //         scaled.push((shifted_x, shifted_y));
                //     }

                // debug!("Name: {0}", dir_path);
                // debug!("Scaled: {:?}", scaled);
                // debug!("Ground truth: {:?}", &data.true_points);
                //     reconstructed = scaled;
                // }

                let mut output_img = dir.join("reconstruction_plot.png");
                self.plot_spatial_reconstruction(
                    &true_points,
                    &reconstructed,
                    output_img.to_str().unwrap(),
                    show_true_points,
                )
                .unwrap();
                output_img = dir.join("original_plot.png");
                self.plot_spatial_reconstruction(
                    &true_points,
                    &true_points,
                    output_img.to_str().unwrap(),
                    false,
                )
                .unwrap();

                output_img = dir.join("just_points.png");
                self.just_points(
                    &true_points,
                    &reconstructed,
                    output_img.to_str().unwrap(),
                    false,
                )
                .unwrap();
                output_img = dir.join("just_true.png");
                self.just_points(
                    &true_points,
                    &true_points,
                    output_img.to_str().unwrap(),
                    false,
                )
                .unwrap();
            }
        }
    }

    pub fn print_results_table(&self, results: &[DbResult]) {
        let mut table = Table::new();
        table.load_preset(UTF8_FULL).set_header(vec![
            Cell::new("Name").fg(TableColor::Cyan),
            Cell::new("Method").fg(TableColor::Cyan),
            Cell::new("Dims").fg(TableColor::Cyan),
            Cell::new("MSE").fg(TableColor::Cyan),
            Cell::new("Match Rate").fg(TableColor::Cyan),
            Cell::new("Chamfer").fg(TableColor::Cyan),
            Cell::new("Reconstructions").fg(TableColor::Cyan),
            Cell::new("Time (s)").fg(TableColor::Cyan),
            Cell::new("DB Size").fg(TableColor::Cyan),
            Cell::new("Queries used (%)").fg(TableColor::Cyan),
        ]);

        for res in results {
            let acc_str = res
                .mse
                .map(|a| format!("{:.2}", a))
                .unwrap_or_else(|| "N/A".to_string());

            let acc_color = if res.mse == Some(0.0) {
                TableColor::Green
            } else {
                TableColor::Reset
            };

            table.add_row(vec![
                Cell::new(&res.name),
                Cell::new(&res.method),
                Cell::new(res.dims),
                Cell::new(acc_str).fg(acc_color),
                Cell::new(format!("{:.4}", res.match_rate.unwrap_or(0.0))).fg(acc_color),
                Cell::new(format!("{:.4}", res.chamfer.unwrap_or(0.0))).fg(acc_color),
                Cell::new(res.number_of_reconstructions.clone()),
                Cell::new(format!("{:.4}", res.time_taken)),
                Cell::new(res.total_db_size),
                Cell::new(res.percent_queries_used),
            ]);
        }

        println!("{table}");
    }

    /// Plots arbitrary 2D points with a modern dark theme.
    /// Optionally overlays true coordinates (x, x) as a glowing backdrop.
    pub fn plot_spatial_reconstruction(
        &self,
        true_coords: &[(f64, f64)],
        recon_coords: &[(f64, f64)],
        output_path: &str,
        show_true_points: bool,
    ) -> Result<(), Box<dyn Error>> {
        let root = BitMapBackend::new(output_path, (1200, 800)).into_drawing_area();
        let background_color = RGBColor(15, 16, 20);
        root.fill(&background_color)?;

        // let root = BitMapBackend::new(output_path, (1200, 800)).into_drawing_area();
        // let background_color = &TRANSPARENT;
        // root.fill(&background_color)?; // Replaces the dark background

        if true_coords.is_empty() && recon_coords.is_empty() {
            return Ok(());
        }

        let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);
        for &(x, y) in true_coords.iter().chain(recon_coords.iter()) {
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }

        let x_pad = (max_x - min_x) * self.x_padder;
        let y_pad = (max_y - min_y) * self.y_padder;

        let mut chart = ChartBuilder::on(&root)
            .margin(30)
            .x_label_area_size(80)
            .y_label_area_size(90)
            .build_cartesian_2d(
                (min_x - x_pad)..(max_x + x_pad),
                (min_y - y_pad)..(max_y + y_pad),
            )?;

        let text_color = RGBColor(220, 225, 240);
        chart
            .configure_mesh()
            .bold_line_style(RGBColor(50, 52, 65))
            .axis_style(&text_color)
            .label_style(("sans-serif", 20).into_font().color(&text_color))
            .draw()?;

        // --- Colors ---
        let cyber_cyan = RGBColor(0, 255, 210);
        let warm_magenta = HSLColor(0.9, 0.9, 0.6);

        // --- 1. True Points Series ---
        if show_true_points {
            chart
                .draw_series(
                    true_coords
                        .iter()
                        .map(|&(x, y)| Circle::new((x, y), 8, cyber_cyan.mix(0.3).filled())),
                )?
                .label("Ground Truth")
                .legend(move |(x, y)| Circle::new((x, y), 6, cyber_cyan.mix(0.4).filled()));
        }

        // --- 2. Reconstructed Points Series ---
        chart
            .draw_series(recon_coords.iter().map(|&(x, y)| {
                let ratio = if max_y > min_y {
                    (y - min_y) / (max_y - min_y)
                } else {
                    0.5
                };
                let hue = 0.15 - (0.15 * ratio);
                let color = HSLColor(if hue < 0.0 { hue + 1.0 } else { hue }, 1.0, 0.6);
                Circle::new((x, y), 4, color.mix(0.9).filled())
            }))?
            .label("Reconstructed")
            .legend(move |(x, y)| Circle::new((x, y), 4, warm_magenta.filled()));

        // --- 3. The Key (Legend) ---
        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::UpperRight)
            .background_style(background_color.mix(0.8).filled())
            .border_style(RGBColor(70, 72, 85))
            .label_font(("sans-serif", 20).into_font().color(&text_color))
            .draw()?;

        root.present()?;
        Ok(())
    }

    pub fn just_points(
        &self,
        true_coords: &[(f64, f64)],
        recon_coords: &[(f64, f64)],
        output_path: &str,
        show_true_points: bool,
    ) -> Result<(), Box<dyn Error>> {
        // 1. Setup drawing area with a transparent background
        let root = SVGBackend::new(output_path, (1200, 800)).into_drawing_area();
        root.fill(&TRANSPARENT)?;

        if true_coords.is_empty() && recon_coords.is_empty() {
            return Ok(());
        }

        // 2. Calculate coordinate bounds
        let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);
        for &(x, y) in true_coords.iter().chain(recon_coords.iter()) {
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }

        let x_pad = (max_x - min_x) * self.x_padder;
        let y_pad = (max_y - min_y) * self.y_padder;

        // 3. Build the chart explicitly without margins or label areas
        let mut chart = ChartBuilder::on(&root)
            // Notice: margin(), x_label_area_size(), and y_label_area_size() are gone
            .build_cartesian_2d(
                (min_x - x_pad)..(max_x + x_pad),
                (min_y - y_pad)..(max_y + y_pad),
            )?;

        // Notice: chart.configure_mesh()...draw()? has been completely removed.
        // This stops Plotters from generating grid lines, axes, and axis text.

        // --- Colors ---
        let cyber_cyan = RGBColor(0, 255, 210);

        // --- 4. Draw True Points Series ---
        if show_true_points {
            chart.draw_series(
                true_coords
                    .iter()
                    .map(|&(x, y)| Circle::new((x, y), 8, cyber_cyan.mix(0.3).filled())),
            )?;
            // Notice: .label() and .legend() are removed from here
        }

        // --- 5. Draw Reconstructed Points Series ---
        chart.draw_series(recon_coords.iter().map(|&(x, y)| {
            let ratio = if max_y > min_y {
                (y - min_y) / (max_y - min_y)
            } else {
                0.5
            };
            let hue = 0.15 - (0.15 * ratio);
            let color = HSLColor(if hue < 0.0 { hue + 1.0 } else { hue }, 1.0, 0.6);
            Circle::new((x, y), 4, color.mix(0.9).filled())
        }))?;
        // Notice: .label() and .legend() are removed from here as well

        // Notice: chart.configure_series_labels()...draw()? has been completely removed.
        // This stops the legend box from rendering entirely.

        root.present()?;
        Ok(())
    }
}
