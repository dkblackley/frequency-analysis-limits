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

#[derive(Debug, Serialize, Deserialize)]
pub struct ReconstructionData2d {
    #[serde(rename = "true")]
    pub true_points: Vec<(f64, f64)>,
    #[serde(rename = "reconstructed")]
    pub reconstructed_points: Vec<(f64, f64)>,
}

pub struct Plotter {
    pub x_padder: f64,
    pub y_padder: f64,
}

impl Plotter {
    /// Generates a sleek CLI table for a vector of DbResult structs.
    pub fn process_data_directories(&self, dir_paths: &[String], show_true_points: bool) {
        let mut all_results = Vec::new();

        for dir_path in dir_paths {
            let dir = Path::new(dir_path);

            // 1. Handle Results Table
            let results_path = dir.join("results.json");
            if results_path.exists() {
                let content = fs::read_to_string(&results_path).unwrap();
                all_results.push(serde_json::from_str::<DbResult>(&content).unwrap());
            }

            // 2. Handle Spatial Plotting
            let recon_path = dir.join("reconstruction.json");
            if recon_path.exists() {
                let content = fs::read_to_string(&recon_path).unwrap();
                let data: ReconstructionData2d = serde_json::from_str(&content).unwrap();

                let mut output_img = dir.join("reconstruction_plot.png");
                self.plot_spatial_reconstruction(
                    &data.true_points,
                    &data.reconstructed_points,
                    output_img.to_str().unwrap(),
                    show_true_points,
                )
                .unwrap();
                output_img = dir.join("original_plot.png");
                self.plot_spatial_reconstruction(
                    &data.true_points,
                    &data.true_points,
                    output_img.to_str().unwrap(),
                    false,
                )
                .unwrap();
            }
        }

        if !all_results.is_empty() {
            self.print_results_table(&all_results);
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::f64::consts::PI;
    use tempfile::tempdir;

    #[test]
    fn test_spatial_visuals() -> Result<(), Box<dyn Error>> {
        let dir = tempdir()?;
        let mut paths = Vec::new();

        // Helper to generate a "Shape" (a circle for this test)
        let generate_circle = |count: i32| -> Vec<(f64, f64)> {
            (0..count)
                .map(|i| {
                    let angle = (i as f64) * (2.0 * PI / count as f64);
                    (angle.cos() * 100.0, angle.sin() * 100.0)
                })
                .collect()
        };

        // 1. Setup OUR Exact Attack (Perfect Overlap)
        let our_path = dir.path().join("our_attack");
        fs::create_dir(&our_path)?;
        let true_points = generate_circle(100);

        fs::write(our_path.join("results.json"), json!({
            "name": "CA_Database_Exact", "method": "Leakage Abuse (Ours)", "dims": 2,
            "accuracy": 100.0, "number_of_reconstructions": "1", "time_taken": 0.02, "total_db_size": 100
        }).to_string())?;

        fs::write(
            our_path.join("reconstruction.json"),
            json!({
                "true": true_points,
                "reconstructed": true_points // Perfectly identical
            })
            .to_string(),
        )?;
        paths.push(our_path.to_str().unwrap().to_string());

        // 2. Setup SOTA Noisy Attack (Visible Drift)
        let sota_path = dir.path().join("sota_attack");
        fs::create_dir(&sota_path)?;
        let mut noisy_recon = true_points.clone();
        for p in &mut noisy_recon {
            p.0 += 5.0; // Shift it slightly to the right to see the "miss"
            p.1 += 2.0;
        }

        fs::write(sota_path.join("results.json"), json!({
            "name": "CA_Database_SOTA", "method": "VolAn", "dims": 2,
            "accuracy": 82.1, "number_of_reconstructions": "5", "time_taken": 10.5, "total_db_size": 100
        }).to_string())?;

        fs::write(
            sota_path.join("reconstruction.json"),
            json!({
                "true": true_points,
                "reconstructed": noisy_recon
            })
            .to_string(),
        )?;
        paths.push(sota_path.to_str().unwrap().to_string());

        // Run logic - Output visible in console with `cargo test -- --nocapture`
        process_data_directories(&paths, true)?;

        // Assertions to ensure PNGs were generated in the temp dir
        assert!(our_path.join("reconstruction_plot.png").exists());
        assert!(sota_path.join("reconstruction_plot.png").exists());

        Ok(())
    }
}
