use crate::plotting::convex_hull::get_per_point_convex_hulls;
use crate::plotting::ReconstructionDataPoint;
use plotters::prelude::*;
use std::error::Error;
use std::fs;

pub fn plot_nh_minimal_3d() -> Result<(), Box<dyn Error>> {
    let filepath = "databases/16x16x14/nh/limits/nh_uniform_e0_d0.001_reconstruction.json";
    let output_svg = "figures/3d_recon/nh_side_by_side_minimal.svg";

    // 1. Load data
    let content = fs::read_to_string(filepath)?;
    let all_data: Vec<Vec<ReconstructionDataPoint>> = serde_json::from_str(&content)?;

    if all_data.is_empty() || all_data[0].is_empty() {
        return Ok(());
    }

    // 2. Compute convex hulls & extract points
    let hulls = get_per_point_convex_hulls(&all_data);

    let mut true_points = Vec::with_capacity(hulls.len());
    let mut centroid_points = Vec::with_capacity(hulls.len());

    let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut min_z, mut max_z) = (f64::INFINITY, f64::NEG_INFINITY);

    let mut update_bounds = |x: f64, y: f64, z: f64| {
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
        min_z = min_z.min(z);
        max_z = max_z.max(z);
    };

    for (i, hull) in hulls.iter().enumerate() {
        let true_pt = &all_data[0][i].true_points;
        if true_pt.len() >= 3 {
            let (tx, ty, tz) = (true_pt[0], true_pt[1], true_pt[2]);
            true_points.push((tx, ty, tz));
            update_bounds(tx, ty, tz);
        }

        if hull.vertices.is_empty() {
            continue;
        }

        let dims = true_pt.len();
        let mut centroid = vec![0.0; dims];
        for v in &hull.vertices {
            for j in 0..dims {
                centroid[j] += v[j];
            }
        }

        let num_vertices = hull.vertices.len() as f64;
        for j in 0..dims {
            centroid[j] /= num_vertices;
        }

        if dims >= 3 {
            let (cx, cy, cz) = (centroid[0], centroid[1], centroid[2]);
            centroid_points.push((cx, cy, cz));
            update_bounds(cx, cy, cz);
        }
    }

    let x_pad = (max_x - min_x).max(1.0) * 0.15;
    let y_pad = (max_y - min_y).max(1.0) * 0.15;
    let z_pad = (max_z - min_z).max(1.0) * 0.15;

    // 3. Setup a Wide Canvas (2400x1200) for Side-By-Side
    let root = SVGBackend::new(output_svg, (2400, 1200)).into_drawing_area();
    // Using a very subtle off-white background to make the black outlines pop even more
    let bg_color = RGBColor(248, 249, 250);
    root.fill(&bg_color)?;

    // 4. Create the Master Title Space
    let (title_area, plot_area) = root.split_vertically(120);

    let title_font = ("Linux Biolinum", 160, FontStyle::Bold).into_font();
    let text = "New Hampshire Mountains";
    let text_size = title_font.layout_box(text).unwrap_or(((0, 0), (0, 0)));
    let text_width = text_size.1 .0 - text_size.0 .0;

    // Centering the master title
    title_area.draw_text(
        text,
        &title_font.color(&BLACK),
        (((2400 - text_width) / 2) + 200, 40),
    )?;

    // 5. Split horizontally for the two totally independent 3D plots
    let (left_area, right_area) = plot_area.split_horizontally(1200);

    // Okabe-Ito Color Palette
    let sky_blue = RGBColor(86, 180, 233);
    let vermillion = RGBColor(213, 94, 0);
    let subtitle_font = ("Linux Biolinum", 120, FontStyle::Bold).into_font();

    let mut chart_left = ChartBuilder::on(&left_area)
        .caption("Ground Truth", subtitle_font.clone().color(&BLACK))
        .margin(60)
        .build_cartesian_3d(
            (min_x - x_pad)..(max_x + x_pad),
            (min_y - y_pad)..(max_y + y_pad),
            (min_z - z_pad)..(max_z + z_pad),
        )?;

    chart_left.with_projection(|mut pb| {
        // FIX 2: Add ~1.57 radians (90 degrees) to the pitch to make it "fall over".
        // Use std::f64::consts::FRAC_PI_2 for exact math.
        pb.pitch = -0.2 + std::f64::consts::FRAC_PI_2;
        pb.yaw = -0.2 + std::f64::consts::PI;
        pb.scale = 0.9;
        pb.into_matrix()
    });

    // Restore subtle grid lines so the brain can perceive 3D space
    chart_left
        .configure_axes()
        .axis_panel_style(TRANSPARENT)
        .bold_grid_style(BLACK.mix(0.2)) // <-- Changed from TRANSPARENT
        .light_grid_style(TRANSPARENT)
        .x_formatter(&|_| String::new())
        .y_formatter(&|_| String::new())
        .z_formatter(&|_| String::new())
        .draw()?;

    chart_left.draw_series(true_points.iter().map(|&(x, y, z)| {
        EmptyElement::at((x, y, z))
            // Semi-transparent fill creates depth via density when points overlap
            + Circle::new((0, 0), 8, sky_blue.mix(0.8).filled())
            // Crisp, thin white rim provides a sleek, modern separation
            + Circle::new((0, 0), 8, WHITE.mix(0.9).stroke_width(1))
    }))?;

    let mut chart_right = ChartBuilder::on(&right_area)
        .caption("'Worst case' Reconstruction", subtitle_font.color(&BLACK))
        .margin(60)
        .build_cartesian_3d(
            (min_x - x_pad)..(max_x + x_pad),
            (min_y - y_pad)..(max_y + y_pad),
            (min_z - z_pad)..(max_z + z_pad),
        )?;

    chart_right.with_projection(|mut pb| {
        // FIX 2: Add ~1.57 radians (90 degrees) to the pitch to make it "fall over".
        // Use std::f64::consts::FRAC_PI_2 for exact math.
        pb.pitch = 0.2 + std::f64::consts::FRAC_PI_2;
        pb.yaw = std::f64::consts::PI;
        pb.scale = 1.3;
        pb.into_matrix()
    });

    // Apply the same subtle grid for consistency
    chart_right
        .configure_axes()
        .axis_panel_style(TRANSPARENT)
        .bold_grid_style(BLACK.mix(0.2)) // <-- Changed from TRANSPARENT
        .light_grid_style(TRANSPARENT)
        .x_formatter(&|_| String::new())
        .y_formatter(&|_| String::new())
        .z_formatter(&|_| String::new())
        .draw()?;

    chart_right.draw_series(centroid_points.iter().map(|&(x, y, z)| {
        EmptyElement::at((x, y, z))
            + Circle::new((0, 0), 8, vermillion.mix(0.7).filled())
            + Circle::new((0, 0), 8, WHITE.mix(0.9).stroke_width(1))
    }))?;

    root.present()?;
    Ok(())
}
