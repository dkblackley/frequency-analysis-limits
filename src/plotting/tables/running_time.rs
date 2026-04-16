use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::BufReader;

// We only specify the field we care about.
// Serde will automatically ignore everything else (like mse, total_queries, etc.)
#[derive(Deserialize, Debug)]
struct ResultJson {
    time_taken: f64,
}

// Holds the times for a single dataset row
#[derive(Default, Debug)]
pub struct MethodTimes {
    pub even_less: Option<f64>,
    pub remin: Option<f64>,
    pub lama: Option<f64>,
}

pub fn do_table_plot(
    path_to_root: &str,
    grids: &Vec<u32>,
    datasets: &Vec<&str>,
    distributions: &Vec<&str>,
) {
    let mut aggregator = ResultsAggregator::new();

    for grid in grids {
        for dataset in datasets {
            for dist in distributions {
                // Note: I updated the filenames here based on the exact examples you provided
                // e.g. "drink_prob100.0_beta_25x25_results_even_less.json"

                let even_less_path = format!(
                    "{path_to_root}/even_less/{dataset}_prob100.0_{dist}_{grid}x{grid}_results_even_less.json"
                );

                let remin_path = format!(
                    "{path_to_root}/remin/{dataset}_prob100.0_{dist}_{grid}x{grid}_results_classic.json"
                );

                let limits_path =
                    format!("{path_to_root}/limits/results_{dataset}_{dist}_e0_d0.9.json");

                // Load the data into our struct
                aggregator.load_file("even_less", dist, dataset, &even_less_path);
                aggregator.load_file("remin", dist, dataset, &remin_path);
                aggregator.load_file("lama", dist, dataset, &limits_path);
            }
        }
    }
    aggregator.print_tables();
}

// The main class/struct for aggregating results
pub struct ResultsAggregator {
    // Map of Distribution -> (Map of Dataset -> MethodTimes)
    // Using BTreeMap for datasets so they print in alphabetical order
    data: HashMap<String, BTreeMap<String, MethodTimes>>,
}

impl ResultsAggregator {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    /// Loads the JSON from the given path and inserts the time_taken into the correct slot
    pub fn load_file(&mut self, method: &str, dist: &str, dataset: &str, path: &str) {
        // Attempt to open and parse the file
        let file = match File::open(path) {
            Ok(f) => f,
            Err(_) => {
                // Silently skip if file doesn't exist, or you could print a warning:
                // eprintln!("Warning: File not found: {}", path);
                return;
            }
        };

        let reader = BufReader::new(file);
        let result: ResultJson = match serde_json::from_reader(reader) {
            Ok(data) => data,
            Err(_) => {
                // eprintln!("Warning: Could not parse JSON at {}", path);
                return;
            }
        };

        // Navigate to the correct distribution and dataset, creating them if they don't exist
        let dataset_times = self
            .data
            .entry(dist.to_string())
            .or_insert_with(BTreeMap::new)
            .entry(dataset.to_string())
            .or_insert_with(MethodTimes::default);

        // Map the parsed time to the correct method
        match method {
            "even_less" => dataset_times.even_less = Some(result.time_taken),
            "remin" => dataset_times.remin = Some(result.time_taken),
            "limits" | "lama" => dataset_times.lama = Some(result.time_taken),
            _ => {}
        }
    }

    /// Prints a copy-pastable Markdown table for each distribution
    pub fn print_tables(&self) {
        for (dist, datasets) in &self.data {
            println!("### Distribution: {}", dist.to_uppercase());
            println!("| Dataset | Even Less | Remin | LAMA |");
            println!("|---|---|---|---|");

            for (dataset, times) in datasets {
                // Format the numbers to 4 decimal places, or output "N/A" if the file was missing
                let el_str = times
                    .even_less
                    .map_or("N/A".to_string(), |v| format!("{:.4}", v));
                let rm_str = times
                    .remin
                    .map_or("N/A".to_string(), |v| format!("{:.4}", v));
                let lama_str = times
                    .lama
                    .map_or("N/A".to_string(), |v| format!("{:.4}", v));

                println!("| {} | {} | {} | {} |", dataset, el_str, rm_str, lama_str);
            }
            println!("\n"); // Spacing between tables
        }
    }
}
