use crate::dataloader::error::DataLoadingError;
use crate::dataloader::{flatten_nd, get_bounding_box, Searchable};
use crate::{Record, Value};
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

    pub fn new(file_path: &str) -> Result<Self, DataLoadingError> {
        let locations = Self::load_locations_from_file(file_path)?;

        let points_iter = locations.clone().into_iter().map(|loc| {
            [
                (loc.longitude * 100.0) as Value,
                (loc.latitude * 100.0) as Value,
            ]
        });

        let (lower, upper) =
            get_bounding_box(points_iter).unwrap_or_else(|| ([0 as Value; 2], [0 as Value; 2]));

        let mut plaintext = Vec::new();
        let mut encrypted_db = Vec::new();
        let mut my_set = HashSet::new();

        for location in locations {
            // only store 2 decimal points and then cut off.
            let point1 = (location.longitude * 100.0) as Value;
            let point2 = (location.latitude * 100.0) as Value;

            plaintext.push((point1, point2));
            let flat_point = flatten_nd(&[point1, point2], &upper, &lower);
            encrypted_db.push(flat_point);
            my_set.insert(flat_point);
        }

        let lower_u = min(lower[0] as usize, lower[1] as usize);
        let upper_u = max(upper[0] as usize, upper[1] as usize);

        // from_shape_fn passes a tuple of the current coordinates to the closure.
        let grid = Array2::from_shape_fn((lower_u, upper_u), |(x, y)| {
            let flat_point = flatten_nd(&[x as i64, y as i64], &upper, &lower);

            if my_set.contains(&flat_point) {
                flat_point
            } else {
                // It's a "miss". Place magic number.
                i64::MIN
            }
        });

        Ok(CaliMap50 {
            plaintext,
            encrypted_db,
            grid,
            dimensions: 2,
            name: "CaliMap".to_string(),
            upper: upper.to_vec(),
            lower: lower.to_vec(),
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

    fn do_search(&self, lower: Record, upper: Record) -> Vec<Value> {
        let bounding_box = self.grid.slice(s![
            lower[0] as usize..=upper[0] as usize,
            lower[1] as usize..=upper[1] as usize
        ]);

        bounding_box
            .iter()
            .copied()
            //.filter(|&v| v != i64::MIN)
            .collect()
    }

    fn get_dom_pair(&self) -> (Record, Record) {
        (self.lower.clone(), self.upper.clone())
    }

    fn get_universe(&self) -> Vec<Value> {
        self.encrypted_db.clone()
    }
}
