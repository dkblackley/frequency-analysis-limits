use crate::{Record, Value};

pub mod datasets;
mod error;
mod raw_data;
pub mod tester;

// Your shared trait
pub trait Searchable: Sync {
    fn get_dims(&self) -> Value;
    // fn get_id_map(&self) -> &HashMap<Self::Key, Self::Value>;
    fn get_name(&self) -> &str;
    fn do_search(&self, lower: &Record, upper: &Record) -> Vec<Value>;

    /// Remember, to compute the dominating vals/prob-freq pairs we don't really need the DB, we just
    /// Need the domain, specifically the lowest and highest possible value on x/y/z/whatever.
    /// you can think of this as the 'largest dominating pair value'
    fn get_dom_pair(&self) -> (Record, Record);

    fn decrypt_point(&self, enc_point: &Value) -> Record;
    fn decrypt_point_f64(&self, enc_point: &Value) -> Vec<f64>;

    /// Returns all individual encrypted records.
    fn get_universe(&self) -> Vec<Value>;
}

/// Flattens an N-dimensional point with arbitrary upper and lower bounds into a 1D index.
/// Assumes `upper` bounds are inclusive (e.g., bounds 1 to 10 means 10 elements (0-9).
pub fn flatten_nd(point: &[i64], upper: &[i64], lower: &[i64]) -> i64 {
    let mut index = 0;
    let mut multiplier = 1;

    for i in (0..point.len()).rev() {
        let point_scaled = point[i] - lower[i];

        index += point_scaled * multiplier;

        let dimension_size = upper[i] - lower[i] + 1;
        multiplier *= dimension_size;
    }

    index
}

pub fn unflatten_nd(mut index: Value, upper: &[Value], lower: &[Value]) -> Record {
    let len = upper.len();
    let mut point = vec![0; len];

    // We iterate backwards, mirroring the flattening function
    for i in (0..len).rev() {
        let dimension_size = upper[i] - lower[i] + 1;

        // Modulo extracts the 0-based coordinate for this specific dimension
        let point_scaled = index % dimension_size;

        // Add the lower bound back to restore the original coordinate space
        point[i] = point_scaled + lower[i];

        // Integer division peels off the current dimension so the next loop
        // can evaluate the preceding dimension
        index /= dimension_size;
    }

    point
}

fn get_bounding_box<T, const D: usize>(
    points: impl Iterator<Item = [T; D]>,
) -> Option<([T; D], [T; D])>
where
    T: PartialOrd + Copy,
{
    let mut iter = points;
    // Initialize lower and upper bounds with the first point.
    // If the iterator is empty, it returns None.
    let first = iter.next()?;

    Some(iter.fold((first, first), |(mut lower, mut upper), point| {
        for i in 0..D {
            if point[i] < lower[i] {
                lower[i] = point[i];
            }
            if point[i] > upper[i] {
                upper[i] = point[i];
            }
        }
        (lower, upper)
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    // Imports your flatten_nd and unflatten_nd functions

    #[test]
    fn test_flatten_unflatten_roundtrip() {
        let lower = &[-1, 0, 2];
        let upper = &[5, 5, 10];

        // Test Case 1: A coordinate somewhere in the middle
        let original_point = &[0, 2, 5];
        let index = flatten_nd(original_point, upper, lower);
        let restored_point = unflatten_nd(index, upper, lower);
        assert_eq!(
            original_point,
            &restored_point[..],
            "Failed on middle coordinate"
        );

        // Test Case 2: The absolute lower bounds
        let min_point = &[-1, 0, 2];
        let min_index = flatten_nd(min_point, upper, lower);
        let restored_min = unflatten_nd(min_index, upper, lower);
        assert_eq!(min_point, &restored_min[..], "Failed on lower bounds");
        assert_eq!(
            min_index, 0,
            "Lower bounds should always flatten to index 0"
        );

        // Test Case 3: The absolute upper bounds
        let max_point = &[5, 5, 10];
        let max_index = flatten_nd(max_point, upper, lower);
        let restored_max = unflatten_nd(max_index, upper, lower);
        assert_eq!(max_point, &restored_max[..], "Failed on upper bounds");
    }
}
