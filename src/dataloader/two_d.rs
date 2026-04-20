use crate::dataloader::{flatten_nd, unflatten_nd, Searchable};
use crate::{Record, Value};
use log::info;
use ndarray::{s, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::File;
use std::io::BufReader;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Location {
    pub longitude: f64, // this is X axis
    pub latitude: f64,  // this is y-axis
}

#[derive(Debug)]
pub struct TwoDMap {
    encrypted_db: Vec<Value>,
    grid: Array2<Value>,
    dimensions: Value,
    name: String,
    scale: f64,
    upper: Vec<Value>,
    lower: Vec<Value>,
    offset: Vec<Value>,
    pub pad: Option<(Value, Value)>, // NEW: Explicitly track the lower padding
}

impl TwoDMap {
    fn load_locations_from_file(filepath: &str) -> Result<Vec<Location>, std::io::Error> {
        let file = File::open(filepath)?;
        let reader = BufReader::new(file);
        let locations = serde_json::from_reader(reader)?;
        Ok(locations)
    }

    /// NEW: Loads locations from a JSON file formatted as an array of arrays: [[lat, long], ...]
    pub fn load_array_locations_from_file(filepath: &str) -> Result<Vec<Location>, std::io::Error> {
        let file = File::open(filepath)?;
        let reader = BufReader::new(file);

        // Deserialize into a temporary Vec of [f64; 2] arrays
        let raw_data: Vec<[u64; 2]> = serde_json::from_reader(reader)?;

        // Map the arrays into the Location struct (index 0 is lat, index 1 is long)
        let locations = raw_data
            .into_iter()
            .map(|arr| Location {
                latitude: arr[1] as f64,
                longitude: arr[0] as f64,
            })
            .collect();

        Ok(locations)
    }

    /// NEW: Loads locations from a statically embedded Rust array: [[Value; 2]; N]
    pub fn load_from_embedded(data: &[[Value; 2]]) -> Vec<Location> {
        data.iter()
            .map(|arr| Location {
                latitude: arr[1] as f64,
                longitude: arr[0] as f64,
            })
            .collect()
    }

    /// `scale_factor` preserves decimal places before integer cast (e.g., 100.0).
    /// `target_grid` allows squishing the map into a bounded space, e.g., Some((50, 50)).
    pub fn new(
        locations: Vec<Location>,
        name: &str,
        scale_factor: f64,
        target_grid: Option<(Value, Value)>,
    ) -> Result<Self, std::io::Error> {
        info!("Loaded DB {name}");

        // 1. Scale floats and cast to Value (i64)
        let scaled_points: Vec<[Value; 2]> = locations
            .into_iter()
            .map(|loc| {
                [
                    (loc.longitude * scale_factor).round() as Value,
                    (loc.latitude * scale_factor).round() as Value,
                ]
            })
            .collect();

        // 2. Find bounding box
        let mut true_lower = [Value::MAX, Value::MAX];
        let mut true_upper = [Value::MIN, Value::MIN];

        for p in &scaled_points {
            if p[0] < true_lower[0] {
                true_lower[0] = p[0];
            }
            if p[1] < true_lower[1] {
                true_lower[1] = p[1];
            }
            if p[0] > true_upper[0] {
                true_upper[0] = p[0];
            }
            if p[1] > true_upper[1] {
                true_upper[1] = p[1];
            }
        }

        if true_lower[0] == Value::MAX {
            true_lower = [0, 0];
            true_upper = [0, 0];
        }

        let offset = vec![true_lower[0], true_lower[1]];
        let lower = [0, 0];
        let upper = [true_upper[0] - offset[0], true_upper[1] - offset[1]];

        // Calculate the maximum ranges to act as our denominator for scaling
        let mut max_x_range = true_upper[0] - true_lower[0];
        let mut max_y_range = true_upper[1] - true_lower[1];

        // Prevent divide-by-zero if all points are identical
        if max_x_range == 0 {
            max_x_range = 1;
        }
        if max_y_range == 0 {
            max_y_range = 1;
        }

        let mut encrypted_db = Vec::new();
        let mut unique_points = HashSet::new();

        // 3. Shift, optionally scale, and deduplicate
        for p in scaled_points {
            let mut shifted_x = p[0] - offset[0];
            let mut shifted_y = p[1] - offset[1];

            // Here is the Rust equivalent of your friend's Python code!
            if let Some((grid_x, grid_y)) = target_grid {
                // (current_val / max_range) * target_dimension
                shifted_x =
                    ((shifted_x as f64 / max_x_range as f64) * grid_x as f64).round() as Value;
                shifted_y =
                    ((shifted_y as f64 / max_y_range as f64) * grid_y as f64).round() as Value;

                // Clamping to 0 instead of 1 to keep things 0-indexed
                shifted_x = shifted_x.max(0);
                shifted_y = shifted_y.max(0);
            }

            let flat_point = flatten_nd(&[shifted_x, shifted_y], &upper, &lower);
            if unique_points.insert(flat_point) {
                encrypted_db.push(flat_point);
            }
        }

        // 4. Adjust the 'upper' bound tracked by the struct depending on scaling
        let upper = if let Some((grid_x, grid_y)) = target_grid {
            [grid_x, grid_y]
        } else {
            [max_x_range, max_y_range]
        };

        // 5. Build the grid efficiently
        let dim_x = (upper[0] - lower[0]) as usize + 1;
        let dim_y = (upper[1] - lower[1]) as usize + 1;

        let mut grid = Array2::from_elem((dim_x, dim_y), i64::MIN);
        for &flat_point in &encrypted_db {
            let grid_point = unflatten_nd(flat_point, &upper, &lower);
            grid[[grid_point[0] as usize, grid_point[1] as usize]] = flat_point;
        }

        encrypted_db.sort_unstable();

        Ok(TwoDMap {
            encrypted_db,
            grid,
            dimensions: 2,
            name: name.to_string(),
            scale: scale_factor,
            upper: upper.to_vec(),
            lower: lower.to_vec(),
            offset,
            pad: None,
        })
    }

    /// Initializes a map from exact integer locations, preserving the offset to [0,0]
    /// and deduplicating, but applying no scaling factors.
    /// Flags the `scale` field as -1.0 for downstream decryption handlers.
    pub fn new_unscaled(
        locations: Vec<Location>,
        name: &str,
        x_pad: (Value, Value),
        y_pad: (Value, Value),
    ) -> Result<Self, std::io::Error> {
        info!("Loaded Unscaled DB {name}");

        // 1. Extract values directly without scaling
        let raw_points: Vec<[Value; 2]> = locations
            .into_iter()
            .map(|loc| [loc.longitude as Value, loc.latitude as Value])
            .collect();

        // 2. Find the bounding box to calculate the offset
        let mut true_lower = [Value::MAX, Value::MAX];
        let mut true_upper = [Value::MIN, Value::MIN];

        for p in &raw_points {
            if p[0] < true_lower[0] {
                true_lower[0] = p[0];
            }
            if p[1] < true_lower[1] {
                true_lower[1] = p[1];
            }
            if p[0] > true_upper[0] {
                true_upper[0] = p[0];
            }
            if p[1] > true_upper[1] {
                true_upper[1] = p[1];
            }
        }

        if true_lower[0] == Value::MAX {
            true_lower = [0, 0];
            true_upper = [0, 0];
        }

        // 3. Subtract the first tuple values (.0) from the offset.
        // This shifts the real "0,0" point forward in the grid, leaving empty indices
        // from 0 up to x_pad.0 / y_pad.0
        let offset = vec![true_lower[0] - x_pad.0, true_lower[1] - y_pad.0];
        let lower = [0, 0];

        // 4. Calculate shifted upper bounds relative to 0 AND apply the second tuple values (.1).
        // `offset` already accounts for the lower padding, so we just add the upper padding
        // to extend the far edges of the grid.
        let upper = [
            (true_upper[0] - offset[0]) + x_pad.1,
            (true_upper[1] - offset[1]) + y_pad.1,
        ];

        let mut encrypted_db = Vec::new();
        let mut unique_points = HashSet::new();

        // 5. Shift the points and deduplicate
        for p in raw_points {
            let shifted_x = p[0] - offset[0];
            let shifted_y = p[1] - offset[1];

            // The resulting flat_point sits comfortably inside the new asymmetrically padded bounds
            let flat_point = flatten_nd(&[shifted_x, shifted_y], &upper, &lower);
            if unique_points.insert(flat_point) {
                encrypted_db.push(flat_point);
            }
        }

        // 6. Build the grid efficiently
        let dim_x = (upper[0] - lower[0]) as usize + 1;
        let dim_y = (upper[1] - lower[1]) as usize + 1;

        // The grid is initialized entirely with empty pads (i64::MIN)
        let mut grid = Array2::from_elem((dim_x, dim_y), i64::MIN);
        for &flat_point in &encrypted_db {
            let grid_point = unflatten_nd(flat_point, &upper, &lower);
            grid[[grid_point[0] as usize, grid_point[1] as usize]] = flat_point;
        }

        encrypted_db.sort_unstable();

        Ok(TwoDMap {
            encrypted_db,
            grid,
            dimensions: 2,
            name: name.to_string(),
            scale: -1.0, // Flag for your decrypt_point_f64 method
            upper: upper.to_vec(),
            lower: lower.to_vec(),
            offset,
            pad: Some((x_pad.0, y_pad.0)),
        })
    }
}

impl Searchable for TwoDMap {
    fn get_dims(&self) -> Value {
        self.dimensions
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

    fn decrypt_point(&self, val: &Value) -> Record {
        // 1. Unflatten using the normalized 0-based bounds
        let grid_point = unflatten_nd(*val, &self.upper, &self.lower);

        // 2. Add the offset back. We DO NOT divide by scale here because
        // Record is a Vec<i64> and integer division will destroy the decimal data.
        // Use `decrypt_to_f64` if you need the real-world floats.
        vec![
            grid_point[0] + self.offset[0],
            grid_point[1] + self.offset[1],
        ]
    }

    fn get_universe(&self) -> Vec<Value> {
        self.encrypted_db.clone()
    }

    fn decrypt_point_f64(&self, val: &Value) -> Vec<f64> {
        let grid_point = unflatten_nd(*val, &self.upper, &self.lower);

        if self.scale == -1.0 {
            // Undo the explicit padding, forcing grid origin [0,0] to [-pad_x, -pad_y]
            if let Some((pad_x, pad_y)) = self.pad {
                return vec![
                    (grid_point[0] - pad_x) as f64,
                    (grid_point[1] - pad_y) as f64,
                ];
            }

            // Fallback just in case
            return vec![
                (grid_point[0] + self.offset[0]) as f64,
                (grid_point[1] + self.offset[1]) as f64,
            ];
        }

        vec![
            (grid_point[0] + self.offset[0]) as f64 / self.scale,
            (grid_point[1] + self.offset[1]) as f64 / self.scale,
        ]
    }
}

#[test]
fn test_search_covers_entire_universe() {
    // Initialize the map
    let path = "/home/yelnat/Nextcloud/10TB-STHDD/Sync-Folder-STHDD/programmin/frequency_analysis_limits/databases/cali_50/cali_50.json";
    let map = TwoDMap::new(
        TwoDMap::load_array_locations_from_file(path).unwrap(),
        "temp",
        10.0,
        None,
    )
    .expect("Failed to initialize CaliMap50");

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
