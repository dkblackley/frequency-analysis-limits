pub mod mse_by_all_reconstructions;
pub(crate) mod mse_vs_grid_size;
pub mod spatial_plot;
pub mod worst_case_convex_hull;

pub fn format_db_name(db: &str) -> &str {
    match db {
        "shopparis" => "Paris",
        "busstop" => "Shanghai",
        "cali" => "Cali",
        "drink" => "Amsterdam",
        "highway" => "Manhattan",
        "spitz" => "Spitz",
        _ => db,
    }
}

pub fn format_dist_name(dist: &str) -> &str {
    match dist {
        "uniform" => "Uniform",
        "gaussian" => "Gaussian",
        "beta" => "Beta",
        "flat" => "Flattened",
        _ => dist,
    }
}
