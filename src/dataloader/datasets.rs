use crate::dataloader::error::DataLoadingError;
use crate::dataloader::{flatten_nd, get_bounding_box, Searchable};
use crate::{Record, Value};
use log::info;
use ndarray::{s, Array2};
use serde::{Deserialize, Serialize};
use std::cmp::{max, min};
use std::collections::HashSet;
use std::fs::File;
use std::io::BufReader;
// From lilika's paper

// Database 1: Users
#[derive(Debug)]
pub struct CaliMap50 {
    //Map of 'node_id' to the two lat and longs (multiplied by 100 and cast to u64) - inner vec should
    // always be of size 2
    plaintext: Vec<(Value, Value)>,
    encrypted_db: Vec<Value>,
    grid: Array2<Value>,
    dimensions: Value, // should always be two
    name: String,
    upper: Vec<Value>,
    lower: Vec<Value>,
    offset: Vec<Value>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Location {
    id: u32,
    longitude: f64,
    latitude: f64,
}

impl CaliMap50 {
    fn load_locations_from_file(filepath: &str) -> Result<Vec<Location>, DataLoadingError> {
        let file = File::open(filepath)?;
        let reader = BufReader::new(file);
        let locations = serde_json::from_reader(reader)?;
        Ok(locations)
    }

    // Add the offset field to your struct definition
    // pub offset: Vec<Value>,

    pub fn new(file_path: &str) -> Result<Self, DataLoadingError> {
        info!("About to load {}", file_path);

        let locations = Self::load_locations_from_file(file_path)?;

        let points_iter = locations
            .clone()
            .into_iter()
            .map(|loc| [loc.longitude as Value, loc.latitude as Value]);

        let (true_lower, true_upper) =
            get_bounding_box(points_iter).unwrap_or_else(|| ([0 as Value; 2], [0 as Value; 2]));

        // 1. Define the offset as the true minimums
        let offset = vec![true_lower[0], true_lower[1]];

        // 2. Normalize upper and lower bounds to start at 0
        let lower = [0, 0];
        let upper = [true_upper[0] - offset[0], true_upper[1] - offset[1]];

        let mut plaintext = Vec::new();
        let mut encrypted_db = Vec::new();
        let mut my_set = HashSet::new();

        for location in locations {
            // 3. Shift every point by subtracting the offset
            let point1 = location.longitude as Value - offset[0];
            let point2 = location.latitude as Value - offset[1];

            plaintext.push((point1, point2));
            let flat_point = flatten_nd(&[point1, point2], &upper, &lower);
            encrypted_db.push(flat_point);
            my_set.insert(flat_point);
        }

        let dim_x = (upper[0] - lower[0]) as usize + 1;
        let dim_y = (upper[1] - lower[1]) as usize + 1;

        let grid = Array2::from_shape_fn((dim_x, dim_y), |(x, y)| {
            let flat_point = flatten_nd(&[x as i64, y as i64], &upper, &lower);

            if my_set.contains(&flat_point) {
                flat_point
            } else {
                i64::MIN
            }
        });

        encrypted_db.sort();
        encrypted_db.dedup();

        Ok(CaliMap50 {
            plaintext,
            encrypted_db,
            grid,
            dimensions: 2,
            name: "CaliMap".to_string(),
            upper: upper.to_vec(),
            lower: lower.to_vec(),
            offset, // 4. Store for later use
        })
    }
}

impl Searchable for CaliMap50 {
    fn get_dims(&self) -> Value {
        2
    }

    fn get_name(&self) -> &str {
        self.name.as_str()
    }

    fn do_search(&self, lower: &Record, upper: &Record) -> Vec<Value> {
        let bounding_box = self.grid.slice(s![
            lower[0] as usize..=upper[0] as usize,
            lower[1] as usize..=upper[1] as usize
        ]);

        bounding_box
            .iter()
            .copied()
            .filter(|&v| v != i64::MIN)
            .collect()
    }

    fn get_dom_pair(&self) -> (Record, Record) {
        (self.lower.clone(), self.upper.clone())
    }

    fn get_universe(&self) -> Vec<Value> {
        self.encrypted_db.clone()
    }
}

#[test]
fn test_search_covers_entire_universe() {
    // Initialize the map
    let path = "/home/yelnat/Nextcloud/10TB-STHDD/Sync-Folder-STHDD/programmin/frequency_analysis_limits/databases/cali_50/cali_50.json";
    let map = CaliMap50::new(path).expect("Failed to initialize CaliMap50");

    // Get the domain boundaries
    let (lower, upper) = map.get_dom_pair();

    // Perform the search across the full range
    let mut search_results = map.do_search(&lower, &upper);

    // Get the expected full universe of values
    let mut universe_values = map.get_universe();

    // Sort both to ensure the comparison is order-independent
    search_results.sort();
    universe_values.sort();

    // Assert that every value in the universe is present in the full-range search
    assert_eq!(
        search_results, universe_values,
        "The search results using domain pairs do not match the expected universe."
    );
}
