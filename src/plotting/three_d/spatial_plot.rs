use crate::plotting::two_d::spatial_plot::plot_spatial_reconstruction;
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

        plot_spatial_reconstruction_3d(
            &*even_less_data.0,
            &*even_less_data.1,
            &format!("{path_to_root}/{name}_even_less.svg"),
            true,
            0.5,
        )
        .unwrap();
        plot_spatial_reconstruction_3d(
            &*remin_data.0,
            &*remin_data.1,
            &format!("{path_to_root}/{name}_remin.svg"),
            true,
            0.0,
        )
        .unwrap();
        plot_spatial_reconstruction_3d(
            &*limits_data.0,
            &*limits_data.1,
            &format!("{path_to_root}/{name}_limits.svg"),
            true,
            0.0,
        )
        .unwrap();
    }
}

/// Plots arbitrary 3D points natively as an SVG with a transparent background.
/// Uses a colorblind-friendly Okabe-Ito palette.
pub fn plot_spatial_reconstruction_3d(
    true_coords: &[Vec<f64>],
    recon_coords: &[Vec<f64>],
    output_path: &str,
    show_true_points: bool,
    padder: f64,
) -> Result<(), Box<dyn Error>> {
    let root = SVGBackend::new(output_path, (1200, 800)).into_drawing_area();
    root.fill(&TRANSPARENT)?;

    if true_coords.is_empty() && recon_coords.is_empty() {
        return Ok(());
    }

    // Find the bounding box for X, Y, and Z
    let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut min_z, mut max_z) = (f64::INFINITY, f64::NEG_INFINITY);

    for xyz in true_coords.iter().chain(recon_coords.iter()) {
        let (x, y, z) = (xyz[0], xyz[1], xyz[2]);
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
        min_z = min_z.min(z);
        max_z = max_z.max(z);
    }

    let x_pad = (max_x - min_x) * padder;
    let y_pad = (max_y - min_y) * padder;
    let z_pad = (max_z - min_z) * padder;

    let mut chart = ChartBuilder::on(&root).margin(30).build_cartesian_3d(
        (min_x - x_pad)..(max_x + x_pad),
        (min_y - y_pad)..(max_y + y_pad),
        (min_z - z_pad)..(max_z + z_pad),
    )?;

    // Adjust the camera/viewing angle here
    chart.with_projection(|mut pb| {
        pb.pitch = 0.5; // Adjust up/down rotation (radians)
        pb.yaw = 0.6; // Adjust left/right rotation (radians)
        pb.scale = 0.9; // Adjust zoom
        pb.into_matrix()
    });

    let text_color = BLACK;
    let mesh_color = RGBColor(220, 220, 220);

    // In 3D, we configure axes rather than a 2D mesh
    chart
        .configure_axes()
        .label_style(("sans-serif", 18).into_font().color(&text_color))
        .light_grid_style(mesh_color)
        .axis_panel_style(WHITE.mix(0.1)) // Subtle backing panel for depth
        .draw()?;

    // --- Color-Blind Friendly Okabe-Ito Palette ---
    let sky_blue = RGBColor(86, 180, 233);
    let vermilion = RGBColor(213, 94, 0);

    // 1. True Points Series
    if show_true_points {
        chart
            .draw_series(true_coords.iter().map(|xyz| {
                let (x, y, z) = (xyz[0], xyz[1], xyz[2]);
                // Anchor a 2D circle to the 3D coordinate space
                EmptyElement::at((x, y, z)) + Circle::new((0, 0), 5, sky_blue.mix(0.8).filled())
            }))?
            .label("Ground Truth")
            .legend(move |(x, y)| Circle::new((x, y), 5, sky_blue.filled()));
    }

    // 2. Reconstructed Points Series
    chart
        .draw_series(recon_coords.iter().map(|xyz| {
            let (x, y, z) = (xyz[0], xyz[1], xyz[2]);
            // Draw slightly smaller to allow truth to pop if they overlap heavily
            EmptyElement::at((x, y, z)) + Circle::new((0, 0), 3, vermilion.mix(0.9).filled())
        }))?
        .label("Reconstructed")
        .legend(move |(x, y)| Circle::new((x, y), 4, vermilion.filled()));

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
