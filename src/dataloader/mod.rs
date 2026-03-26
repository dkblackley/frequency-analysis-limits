use crate::{Record, Value};

pub mod datasets;
mod error;
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
