use crate::dataloader::{flatten_nd, unflatten_nd, Searchable};
use crate::{Coord, Record, Value};
use ndarray::{ArrayD, Dimension, IxDyn, Slice};
// Swap s! and Array2 for ArrayD, IxDyn, and Slice
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

#[derive(Debug)]
pub struct testDB {
    id_map: HashMap<Value, Vec<Value>>,
    dimensions: Value, // Now dynamically tracks the 'dim' argument
    name: String,
    grid: ArrayD<Value>, // Upgraded to support N dimensions
    lowest_val: Record,
    upper_val: Record,
}

impl Default for testDB {
    fn default() -> Self {
        // Default to a 2-dimensional 50x50 grid at 50% density to preserve previous behavior
        Self::new(2, 50, 50)
    }
}

impl testDB {
    pub fn new(dim: usize, size_per_dim: usize, density_pct: u8) -> Self {
        let id_map = HashMap::new();

        // Dynamically build the lower and upper bounds based on the dim count
        let lowest_val = vec![0; dim];
        let upper_val = vec![(size_per_dim - 1) as Coord; dim];

        Self {
            id_map,
            dimensions: dim as Value,
            name: "testDB".to_string(),
            grid: Self::generate_encoded_grid(dim, size_per_dim, density_pct),
            lowest_val,
            upper_val,
        }
    }

    fn generate_encoded_grid(dim: usize, size_per_dim: usize, density_pct: u8) -> ArrayD<i64> {
        let mut rng = StdRng::seed_from_u64(42);

        let lower = vec![0; dim];
        let upper = vec![(size_per_dim - 1) as i64; dim];

        // Create a shape array where each dimension is 'size_per_dim' long
        let shape = vec![size_per_dim; dim];

        // from_shape_fn for dynamic arrays passes an `IxDyn` index to the closure
        ArrayD::from_shape_fn(IxDyn(&shape), |idx| {
            let roll = rng.gen_range(1..=100);

            if roll <= density_pct {
                // Extract the N-dimensional coordinate from idx
                let point: Vec<i64> = idx.slice().iter().map(|&v| v as i64).collect();
                flatten_nd(&point, &upper, &lower)
            } else {
                i64::MIN
            }
        })
    }
}

impl Searchable for testDB {
    fn get_dims(&self) -> Value {
        self.dimensions
    }

    fn get_name(&self) -> &str {
        &self.name
    }

    fn do_search(&self, lower: &Record, upper: &Record) -> Vec<Value> {
        // Because the s![] macro requires a statically known number of dimensions,
        // we use `slice_each_axis` to dynamically construct the slice bounds for N axes.
        let bounding_box = self.grid.slice_each_axis(|ax| {
            let ax_idx = ax.axis.0;
            Slice::from(lower[ax_idx] as isize..=upper[ax_idx] as isize)
        });

        bounding_box
            .iter()
            .copied()
            .filter(|&v| v != i64::MIN)
            .collect()
    }

    fn get_dom_pair(&self) -> (Record, Record) {
        (self.lowest_val.clone(), self.upper_val.clone())
    }

    fn decrypt_point(&self, val: &Value) -> Record {
        let grid_point = unflatten_nd(*val, &self.upper_val, &self.lowest_val);

        // Dynamically map all axes back instead of hardcoding [0] and [1]
        grid_point.into_iter().map(|x| x as Coord).collect()
    }

    fn decrypt_point_f64(&self, val: &Value) -> Vec<f64> {
        let grid_point = unflatten_nd(*val, &self.upper_val, &self.lowest_val);

        grid_point.into_iter().map(|x| x as f64).collect()
    }

    fn get_universe(&self) -> Vec<Value> {
        let (lower_bound, upper_bound) = self.get_dom_pair();

        self.do_search(&lower_bound, &upper_bound)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataloader::flatten_nd;
    use std::collections::HashSet;

    #[test]
    fn test_full_grid_search_and_encoding() {
        // 1. Setup DB with a 2D 100x100 grid at 100% density
        let dim = 2;
        let size_per_dim = 100;
        let db = testDB {
            id_map: HashMap::new(),
            dimensions: dim as Value,
            name: "testDB".to_string(),
            grid: testDB::generate_encoded_grid(dim, size_per_dim, 100),
            lowest_val: vec![0, 0],
            upper_val: vec![99, 99],
        };

        // 2. Perform the search across the entire grid
        let lower_bound = vec![0, 0];
        let upper_bound = vec![99, 99];
        let results = db.do_search(&lower_bound, &upper_bound);

        // 3. Manually calculate what the expected values SHOULD be
        let mut expected_values = Vec::new();
        let lower_params = [0, 0];
        let upper_params = [99, 99];

        for y in 0..size_per_dim {
            for x in 0..size_per_dim {
                let point = [y as i64, x as i64];
                let encoded_val = flatten_nd(&point, &upper_params, &lower_params);
                expected_values.push(encoded_val);
            }
        }

        // 4. Verify using HashSets for O(1) unordered comparisons
        let results_set: HashSet<i64> = results.into_iter().collect();
        let expected_set: HashSet<i64> = expected_values.into_iter().collect();

        assert_eq!(
            results_set.len(),
            10_000,
            "The search should return exactly 10,000 values."
        );

        assert_eq!(
            results_set, expected_set,
            "The returned values do not perfectly match the expected flattened values!"
        );
    }
}
