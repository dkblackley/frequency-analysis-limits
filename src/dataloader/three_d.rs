use crate::dataloader::{flatten_nd, unflatten_nd, Searchable};
use crate::{Record, Value};
use log::info;
use ndarray::{s, Array3};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::File;
use std::io::BufReader;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Point3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(Debug)]
pub struct ThreeDMap {
    encrypted_db: Vec<Value>,
    grid: Array3<Value>,
    dimensions: Value,
    name: String,
    scale: f64,
    upper: Vec<Value>,
    lower: Vec<Value>,
    offset: Vec<Value>,
}

impl ThreeDMap {
    /// Loads locations from a JSON file formatted as an array of arrays: [[x, y, z], ...]
    pub fn load_array_locations_from_file(filepath: &str) -> Result<Vec<Point3D>, std::io::Error> {
        let file = File::open(filepath)?;
        let reader = BufReader::new(file);

        // Deserialize into a temporary Vec of [f64; 3] arrays
        let raw_data: Vec<[f64; 3]> = serde_json::from_reader(reader)?;

        // Map the arrays into the Point3D struct
        let locations = raw_data
            .into_iter()
            .map(|arr| Point3D {
                x: arr[0],
                y: arr[1],
                z: arr[2],
            })
            .collect();

        Ok(locations)
    }

    /// Loads locations from a statically embedded Rust array: [[Value; 3]; N]
    pub fn load_from_embedded(data: &[[Value; 3]]) -> Vec<Point3D> {
        data.iter()
            .map(|arr| Point3D {
                x: arr[0] as f64,
                y: arr[1] as f64,
                z: arr[2] as f64,
            })
            .collect()
    }

    /// `scale_factor` preserves decimal places before integer cast (e.g., 100.0).
    /// `target_grid` allows squishing the map into a bounded space, e.g., Some((50, 50, 50)).
    pub fn new(
        locations: Vec<Point3D>,
        name: &str,
        scale_factor: f64,
        target_grid: Option<(Value, Value, Value)>,
    ) -> Result<Self, std::io::Error> {
        info!("Loaded DB {name}");

        // 1. Scale floats and cast to Value (i64)
        let scaled_points: Vec<[Value; 3]> = locations
            .into_iter()
            .map(|loc| {
                [
                    (loc.x * scale_factor).round() as Value,
                    (loc.y * scale_factor).round() as Value,
                    (loc.z * scale_factor).round() as Value,
                ]
            })
            .collect();

        // 2. Find bounding box
        let mut true_lower = [Value::MAX, Value::MAX, Value::MAX];
        let mut true_upper = [Value::MIN, Value::MIN, Value::MIN];

        for p in &scaled_points {
            for i in 0..3 {
                if p[i] < true_lower[i] {
                    true_lower[i] = p[i];
                }
                if p[i] > true_upper[i] {
                    true_upper[i] = p[i];
                }
            }
        }

        if true_lower[0] == Value::MAX {
            true_lower = [0, 0, 0];
            true_upper = [0, 0, 0];
        }

        let offset = vec![true_lower[0], true_lower[1], true_lower[2]];
        let lower = [0, 0, 0];
        let upper = [
            true_upper[0] - offset[0],
            true_upper[1] - offset[1],
            true_upper[2] - offset[2],
        ];

        // Calculate the maximum ranges to act as our denominator for scaling
        let mut max_x_range = true_upper[0] - true_lower[0];
        let mut max_y_range = true_upper[1] - true_lower[1];
        let mut max_z_range = true_upper[2] - true_lower[2];

        // Prevent divide-by-zero if all points are identical
        if max_x_range == 0 {
            max_x_range = 1;
        }
        if max_y_range == 0 {
            max_y_range = 1;
        }
        if max_z_range == 0 {
            max_z_range = 1;
        }

        let mut encrypted_db = Vec::new();
        let mut unique_points = HashSet::new();

        // 3. Shift, optionally scale, and deduplicate
        for p in scaled_points {
            let mut shifted_x = p[0] - offset[0];
            let mut shifted_y = p[1] - offset[1];
            let mut shifted_z = p[2] - offset[2];

            if let Some((grid_x, grid_y, grid_z)) = target_grid {
                shifted_x =
                    ((shifted_x as f64 / max_x_range as f64) * grid_x as f64).round() as Value;
                shifted_y =
                    ((shifted_y as f64 / max_y_range as f64) * grid_y as f64).round() as Value;
                shifted_z =
                    ((shifted_z as f64 / max_z_range as f64) * grid_z as f64).round() as Value;

                shifted_x = shifted_x.max(0);
                shifted_y = shifted_y.max(0);
                shifted_z = shifted_z.max(0);
            }

            let flat_point = flatten_nd(&[shifted_x, shifted_y, shifted_z], &upper, &lower);
            if unique_points.insert(flat_point) {
                encrypted_db.push(flat_point);
            }
        }

        // 4. Adjust the 'upper' bound tracked by the struct depending on scaling
        let upper = if let Some((grid_x, grid_y, grid_z)) = target_grid {
            [grid_x, grid_y, grid_z]
        } else {
            [max_x_range, max_y_range, max_z_range]
        };

        // 5. Build the grid efficiently
        let dim_x = (upper[0] - lower[0]) as usize + 1;
        let dim_y = (upper[1] - lower[1]) as usize + 1;
        let dim_z = (upper[2] - lower[2]) as usize + 1;

        let mut grid = Array3::from_elem((dim_x, dim_y, dim_z), i64::MIN);
        for &flat_point in &encrypted_db {
            let grid_point = unflatten_nd(flat_point, &upper, &lower);
            grid[[
                grid_point[0] as usize,
                grid_point[1] as usize,
                grid_point[2] as usize,
            ]] = flat_point;
        }

        encrypted_db.sort_unstable();

        Ok(ThreeDMap {
            encrypted_db,
            grid,
            dimensions: 3,
            name: name.to_string(),
            scale: scale_factor,
            upper: upper.to_vec(),
            lower: lower.to_vec(),
            offset,
        })
    }

    /// Initializes a map from exact integer locations, preserving the offset to [0,0,0]
    /// and deduplicating, but applying no scaling factors.
    pub fn new_unscaled(locations: Vec<Point3D>, name: &str) -> Result<Self, std::io::Error> {
        info!("Loaded Unscaled DB {name}");

        // 1. Extract values directly without scaling
        let raw_points: Vec<[Value; 3]> = locations
            .into_iter()
            .map(|loc| [loc.x as Value, loc.y as Value, loc.z as Value])
            .collect();

        // 2. Find the bounding box to calculate the offset
        let mut true_lower = [Value::MAX, Value::MAX, Value::MAX];
        let mut true_upper = [Value::MIN, Value::MIN, Value::MIN];

        for p in &raw_points {
            for i in 0..3 {
                if p[i] < true_lower[i] {
                    true_lower[i] = p[i];
                }
                if p[i] > true_upper[i] {
                    true_upper[i] = p[i];
                }
            }
        }

        if true_lower[0] == Value::MAX {
            true_lower = [0, 0, 0];
            true_upper = [0, 0, 0];
        }

        let offset = vec![true_lower[0], true_lower[1], true_lower[2]];
        let lower = [0, 0, 0];
        let upper = [
            true_upper[0] - offset[0],
            true_upper[1] - offset[1],
            true_upper[2] - offset[2],
        ];

        let mut encrypted_db = Vec::new();
        let mut unique_points = HashSet::new();

        // 3. Shift the points to 0,0,0 and deduplicate
        for p in raw_points {
            let shifted_x = p[0] - offset[0];
            let shifted_y = p[1] - offset[1];
            let shifted_z = p[2] - offset[2];

            let flat_point = flatten_nd(&[shifted_x, shifted_y, shifted_z], &upper, &lower);
            if unique_points.insert(flat_point) {
                encrypted_db.push(flat_point);
            }
        }

        // 4. Build the grid efficiently
        let dim_x = (upper[0] - lower[0]) as usize + 1;
        let dim_y = (upper[1] - lower[1]) as usize + 1;
        let dim_z = (upper[2] - lower[2]) as usize + 1;

        let mut grid = Array3::from_elem((dim_x, dim_y, dim_z), i64::MIN);
        for &flat_point in &encrypted_db {
            let grid_point = unflatten_nd(flat_point, &upper, &lower);
            grid[[
                grid_point[0] as usize,
                grid_point[1] as usize,
                grid_point[2] as usize,
            ]] = flat_point;
        }

        encrypted_db.sort_unstable();

        Ok(ThreeDMap {
            encrypted_db,
            grid,
            dimensions: 3,
            name: name.to_string(),
            scale: -1.0, // Flag for your decrypt_point_f64 method
            upper: upper.to_vec(),
            lower: lower.to_vec(),
            offset,
        })
    }
}

impl Searchable for ThreeDMap {
    fn get_dims(&self) -> Value {
        self.dimensions
    }

    fn get_name(&self) -> &str {
        self.name.as_str()
    }

    fn do_search(&self, lower: &Record, upper: &Record) -> Vec<Value> {
        let bounding_box = self.grid.slice(s![
            lower[0] as usize..=upper[0] as usize,
            lower[1] as usize..=upper[1] as usize,
            lower[2] as usize..=upper[2] as usize
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

        // 2. Add the offset back.
        vec![
            grid_point[0] + self.offset[0],
            grid_point[1] + self.offset[1],
            grid_point[2] + self.offset[2],
        ]
    }

    fn get_universe(&self) -> Vec<Value> {
        self.encrypted_db.clone()
    }

    fn decrypt_point_f64(&self, val: &Value) -> Vec<f64> {
        let grid_point = unflatten_nd(*val, &self.upper, &self.lower);

        if self.scale == -1.0 {
            return vec![
                (grid_point[0] + self.offset[0]) as f64,
                (grid_point[1] + self.offset[1]) as f64,
                (grid_point[2] + self.offset[2]) as f64,
            ];
        }

        vec![
            (grid_point[0] + self.offset[0]) as f64 / self.scale,
            (grid_point[1] + self.offset[1]) as f64 / self.scale,
            (grid_point[2] + self.offset[2]) as f64 / self.scale,
        ]
    }
}
