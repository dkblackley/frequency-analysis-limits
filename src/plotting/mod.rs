use crate::LAMA::translator::TranslatorMeta;
use serde::{Deserialize, Serialize};

pub mod post;

#[derive(Debug, Deserialize, Serialize)]
pub struct DataWrapper {
    #[serde(rename = "mapping")]
    pub mapping: Vec<ReconstructionDataPoint>,
}

pub fn flip_coordinates(mut data_point: ReconstructionDataPoint) -> ReconstructionDataPoint {
    // Swap the x and y values for true_points
    if data_point.true_points.len() == 2 {
        data_point.true_points.swap(0, 1);
    }

    // Swap the x and y values for reconstructed_points
    if data_point.reconstructed_points.len() == 2 {
        data_point.reconstructed_points.swap(0, 1);
    }

    // Optional: Flipped the unscaled points as well if they happen to exist
    if let Some(unscaled) = &mut data_point.unscaled_points {
        if unscaled.len() == 2 {
            unscaled.swap(0, 1);
        }
    }

    data_point
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ReconstructionDataPoint {
    #[serde(rename = "true")]
    pub true_points: Vec<f64>,
    #[serde(rename = "reconstructed")]
    pub reconstructed_points: Vec<f64>,
    #[serde(rename = "unscaled_true")]
    pub unscaled_points: Option<Vec<f64>>,
}

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
    pub num_queries_used: u64,
    pub eps: Option<f64>,
    pub delt: Option<f64>,
    pub translator_meta: TranslatorMeta,
    /// Number of reconstructed databases ('number_of_reconstructions' is the number of records)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub number_of_databases: Option<usize>,
    /// Sampled setting only: what the pair LNS found, and how the true database scores
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lns: Option<crate::LAMA::lazy_lns::LnsSummary>,
}
