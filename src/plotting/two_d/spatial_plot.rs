use crate::plotting::{get_remin_even_less, ReconstructionDataPoint};
use log::debug;
use plotters::prelude::*;
use std::error::Error;
use std::fs;

pub fn run_spatial_plots(datasets: &Vec<&str>, dir: &str) {
    for name in datasets {
        let path_to_root = format!("{}/{}", dir, name);

        let even_less =
            format!("{path_to_root}/even_less/{name}_prob100.0_uniform_50x50_even_less.json");
        let remin_path =
            format!("{path_to_root}/remin/{name}_prob100.0_uniform_50x50_classic.json");
        let limits = format!("{path_to_root}/limits/{name}_uniform_e0_d0.9_reconstruction.json");

        debug!(
            "About to load data from {}, {}, {}",
            &even_less, &remin_path, &limits
        );

        let even_less_data = get_remin_even_less(&even_less, true);
        let remin_data = get_remin_even_less(&remin_path, true);

        let content = fs::read_to_string(limits).unwrap();
        let all_data: Vec<Vec<ReconstructionDataPoint>> = serde_json::from_str(&content).unwrap();
        let data = all_data[0].clone(); // jsut grab the first

        let mut true_point = Vec::new();
        let mut recon_point = Vec::new();

        for point in data {
            true_point.push(point.true_points);
            recon_point.push(point.reconstructed_points);
        }

        let limits_data = (true_point, recon_point);

        plot_spatial_reconstruction(
            &*even_less_data.0,
            &*even_less_data.1,
            &format!("{path_to_root}/{name}_even_less.svg"),
            true,
            0.5,
            0.5,
        )
        .unwrap();
        plot_spatial_reconstruction(
            &*remin_data.0,
            &*remin_data.1,
            &format!("{path_to_root}/{name}_remin.svg"),
            true,
            0.0,
            0.0,
        )
        .unwrap();
        plot_spatial_reconstruction(
            &*limits_data.0,
            &*limits_data.1,
            &format!("{path_to_root}/{name}_limits.svg"),
            true,
            0.0,
            0.0,
        )
        .unwrap();
    }
}

/// Plots arbitrary 2D points natively as an SVG with a transparent background.
/// Colors are updated to use a colorblind-friendly gradient (Okabe-Ito Palette).
pub fn plot_spatial_reconstruction(
    true_coords: &[Vec<f64>],
    recon_coords: &[Vec<f64>],
    output_path: &str,
    show_true_points: bool,
    x_padder: f64,
    y_padder: f64,
) -> Result<(), Box<dyn Error>> {
    // Switch to SVGBackend for lossless, scalable vector images
    let root = SVGBackend::new(output_path, (1200, 800)).into_drawing_area();
    root.fill(&TRANSPARENT)?;

    if true_coords.is_empty() && recon_coords.is_empty() {
        return Ok(());
    }

    let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);
    for x_y in true_coords.iter().chain(recon_coords.iter()) {
        let (x, y) = (x_y[0], x_y[1]);
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
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

    let text_color = BLACK;
    let mesh_color = RGBColor(220, 220, 220);

    chart
        .configure_mesh()
        .bold_line_style(mesh_color)
        .axis_style(&text_color)
        .label_style(("sans-serif", 20).into_font().color(&text_color))
        .draw()?;

    // --- Color-Blind Friendly Okabe-Ito Palette ---
    let sky_blue = RGBColor(86, 180, 233);

    // 1. True Points Series
    if show_true_points {
        chart
            .draw_series(
                true_coords
                    .iter()
                    .map(|x_y| Circle::new((x_y[0], x_y[1]), 8, sky_blue.mix(0.8).filled())),
            )?
            .label("Ground Truth")
            .legend(move |(x, y)| Circle::new((x, y), 6, sky_blue.filled()));
    }

    // 2. Reconstructed Points Series with Light-Theme Gradient
    chart
        .draw_series(recon_coords.iter().map(|x_y| {
            let (x, y) = (x_y[0], x_y[1]);
            let ratio = if max_y > min_y {
                (y - min_y) / (max_y - min_y)
            } else {
                0.5
            };

            // // Gradient: Sky Blue (86, 180, 233) to Vermilion (213, 94, 0)
            // let r = (86.0 + (213.0 - 86.0) * ratio) as u8;
            // let g = (180.0 + (94.0 - 180.0) * ratio) as u8;
            // let b = (233.0 + (0.0 - 233.0) * ratio) as u8;
            //
            // let color = RGBColor(r, g, b);
            // remove gradient
            let color = RGBColor(213, 94, 0);
            Circle::new((x, y), 4, color.mix(0.9).filled())
        }))?
        .label("Reconstructed")
        .legend(move |(x, y)| Circle::new((x, y), 4, RGBColor(213, 94, 0).filled()));

    // 3. The Legend
    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(WHITE.mix(0.9).filled())
        .border_style(BLACK)
        .label_font(("sans-serif", 20).into_font().color(&text_color))
        .draw()?;

    root.present()?;
    Ok(())
}

pub fn just_points(
    true_coords: &[Vec<f64>],
    recon_coords: &[Vec<f64>],
    output_path: &str,
    show_true_points: bool,
    x_padder: f64,
    y_padder: f64,
) -> Result<(), Box<dyn Error>> {
    let root = SVGBackend::new(output_path, (1200, 800)).into_drawing_area();
    root.fill(&TRANSPARENT)?;

    if true_coords.is_empty() && recon_coords.is_empty() {
        return Ok(());
    }

    let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);
    for x_y in true_coords.iter().chain(recon_coords.iter()) {
        let (x, y) = (x_y[0], x_y[1]);
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }

    let x_pad = (max_x - min_x) * x_padder;
    let y_pad = (max_y - min_y) * y_padder;

    let mut chart = ChartBuilder::on(&root).build_cartesian_2d(
        (min_x - x_pad)..(max_x + x_pad),
        (min_y - y_pad)..(max_y + y_pad),
    )?;

    let sky_blue = RGBColor(86, 180, 233);

    if show_true_points {
        chart.draw_series(
            true_coords
                .iter()
                .map(|x_y| Circle::new((x_y[0], x_y[1]), 8, sky_blue.mix(0.8).filled())),
        )?;
    }

    chart.draw_series(recon_coords.iter().map(|x_y| {
        let (x, y) = (x_y[0], x_y[1]);
        let ratio = if max_y > min_y {
            (y - min_y) / (max_y - min_y)
        } else {
            0.5
        };
        let r = (86.0 + (213.0 - 86.0) * ratio) as u8;
        let g = (180.0 + (94.0 - 180.0) * ratio) as u8;
        let b = (233.0 + (0.0 - 233.0) * ratio) as u8;

        Circle::new((x, y), 4, RGBColor(r, g, b).mix(0.9).filled())
    }))?;

    root.present()?;
    Ok(())
}
