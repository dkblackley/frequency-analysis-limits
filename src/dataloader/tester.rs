use crate::dataloader::{flatten_nd, unflatten_nd, Searchable};
use crate::{Coord, Record, Value};
use ndarray::{s, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

#[derive(Debug)]
pub struct testDB {
    //Map of 'node_id' to the two lat and longs (multiplied by 100 and cast to u64)
    idMap: HashMap<Value, Vec<Value>>,
    dimensions: Value, // should always be two
    name: String,
    grid: Array2<Value>,
    lowest_val: Record,
    upper_val: Record,
}

impl Default for testDB {
    fn default() -> Self {
        Self::new(50, 50, 50)
    }
}

impl testDB {
    pub fn new(rows: usize, cols: usize, density_pct: u8) -> Self {
        let mut id_map = HashMap::new();

        Self {
            idMap: id_map,
            dimensions: 2,
            name: "testDB".to_string(),
            grid: Self::generate_encoded_grid(rows, cols, density_pct),
            lowest_val: vec![0, 0],
            upper_val: vec![(rows - 1) as Coord, (cols - 1) as Coord],
        }
    }

    fn generate_encoded_grid(rows: usize, cols: usize, density_pct: u8) -> Array2<i64> {
        let mut rng = StdRng::seed_from_u64(42);

        // Define the boundaries for your flatten_nd function
        let lower = [0, 0];
        let upper = [(rows - 1) as i64, (cols - 1) as i64];

        // Notice the |(y, x)| here!
        // from_shape_fn passes a tuple of the current coordinates to the closure.
        Array2::from_shape_fn((rows, cols), |(y, x)| {
            let roll = rng.gen_range(1..=100);

            if roll <= density_pct {
                // It's a "hit". Encode the current (y, x) position!
                // We cast the usize indices to i64 to match your function signature.
                let point = [y as i64, x as i64];
                flatten_nd(&point, &upper, &lower)
            } else {
                // It's a "miss". Place your magic number.
                i64::MIN
            }
        })
    }
}

impl Searchable for testDB {
    fn get_dims(&self) -> Value {
        self.dimensions
    }

    // fn get_id_map(&self) -> &HashMap<Self::Key, Self::Value> {
    //     &self.idMap
    // }

    fn get_name(&self) -> &str {
        &self.name
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

    /// Remember, to compute the dominating vals/prob-freq pairs we don't really need the DB, we just
    /// Need the domain, specifically the lowest and highest possible value (not flatten, highest
    /// x/y/z/whatever)> expects first item to be lowest and second to be largest.
    fn get_dom_pair(&self) -> (Record, Record) {
        (self.lowest_val.clone(), self.upper_val.clone())
    }

    fn decrypt_point(&self, val: &Value) -> Record {
        let grid_point = unflatten_nd(*val, &self.upper_val, &self.lowest_val);

        // 2. Add the offset back to restore real-world coordinates
        vec![grid_point[0], grid_point[1]]
    }

    fn decrypt_point_f64(&self, val: &Value) -> Vec<f64> {
        let grid_point = unflatten_nd(*val, &self.upper_val, &self.lowest_val);

        vec![grid_point[0] as f64, grid_point[1] as f64]
    }

    fn get_universe(&self) -> Vec<Value> {
        let (lower_bound, upper_bound) = self.get_dom_pair();

        self.do_search(&lower_bound, &upper_bound)
            .into_iter()
            .filter(|v| v != &i64::MIN)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::dataloader::flatten_nd;
        use std::collections::HashSet;
        // Import everything from the parent module

        #[test]
        fn test_full_grid_search_and_encoding() {
            // 1. Setup DB with a 100x100 grid at 100% density
            let rows = 100;
            let cols = 100;
            let db = testDB {
                idMap: HashMap::new(),
                dimensions: 2,
                name: "testDB".to_string(),
                grid: testDB::generate_encoded_grid(rows, cols, 100),
                lowest_val: vec![0, 0],
                upper_val: vec![99, 99],
            };

            // 2. Perform the search across the entire grid
            // 0-indexed bounds: (0, 0) to (99, 99)
            let lower_bound = vec![0, 0];
            let upper_bound = vec![99, 99];
            let results = db.do_search(&lower_bound, &upper_bound);

            // 3. Manually calculate what the expected values SHOULD be
            let mut expected_values = Vec::new();
            let lower_params = [0, 0];
            let upper_params = [99, 99];

            for y in 0..rows {
                for x in 0..cols {
                    let point = [y as i64, x as i64];
                    let encoded_val = flatten_nd(&point, &upper_params, &lower_params);
                    expected_values.push(encoded_val);
                }
            }

            // 4. Verify using HashSets for O(1) unordered comparisons
            let results_set: HashSet<i64> = results.into_iter().collect();
            let expected_set: HashSet<i64> = expected_values.into_iter().collect();

            // Check 1: Did we get exactly 10,000 items?
            // Docs: https://doc.rust-lang.org/std/macro.assert_eq.html
            assert_eq!(
                results_set.len(),
                10_000,
                "The search should return exactly 10,000 values."
            );

            // Check 2: Are the sets of items identical?
            assert_eq!(
                results_set, expected_set,
                "The returned values do not perfectly match the expected flattened values!"
            );
        }
    }
}
