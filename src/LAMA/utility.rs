use crate::{Coord, DomPair, Record};
use itertools::Itertools;
use log::{debug, info};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::str::FromStr;

// Helper to determine if point u dominates point v (u_i >= v_i for all i)
pub fn dominates(u: &[Coord], v: &[Coord]) -> bool {
    u.iter().zip(v.iter()).all(|(u_val, v_val)| u_val >= v_val)
}

/// Returns true if the 'outer' DomPair spatially encloses the 'inner' DomPair.
pub fn encloses(outer: &DomPair, inner: &DomPair) -> bool {
    // 1. Check Lower Bounds: outer.0 <= inner.0
    // This is equivalent to: dominates(&inner.0, &outer.0)
    let lower_check = dominates(&inner.0, &outer.0);

    // 2. Check Upper Bounds: outer.1 >= inner.1
    // This is equivalent to: dominates(&outer.1, &inner.1)
    let upper_check = dominates(&outer.1, &inner.1);

    lower_check && upper_check
}

// Helper for L1 distance calculation
pub fn _l1_distance(p1: &[Coord], p2: &[Coord]) -> u64 {
    p1.iter().zip(p2.iter()).map(|(a, b)| a.abs_diff(*b)).sum()
}

pub fn get_all_dominating_values(v: &[Coord], largest_rec: &[Coord]) -> Vec<Record> {
    v.iter()
        .zip(largest_rec.iter()) // Pair each v_val with its specific dimension's max
        .map(|(&v_val, &max_val)| v_val..=max_val)
        .multi_cartesian_product()
        .collect()
}

// Define an Enum to avoid string comparisons in the hot loop
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DistributionType {
    Uniform,
    Gaussian,
    Beta,
    Flat,
}

impl FromStr for DistributionType {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "uniform" => Ok(DistributionType::Uniform),
            "gaussian" => Ok(DistributionType::Gaussian),
            "beta" => Ok(DistributionType::Beta),
            "flat" => Ok(DistributionType::Flat),
            _ => Err(()),
        }
    }
}

impl std::fmt::Display for DistributionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DistributionType::Uniform => write!(f, "uniform"),
            DistributionType::Gaussian => write!(f, "gaussian"),
            DistributionType::Beta => write!(f, "beta"),
            DistributionType::Flat => write!(f, "flat"),
        }
    }
}

pub fn get_mbq(t_tup: &[Record]) -> DomPair {
    // Avoid allocating empty vectors with Value::MAX/MIN.
    // Clone the first record to use as our baseline bounds.
    let mut minima = t_tup[0].clone();
    let mut maxima = t_tup[0].clone();

    // Iterate through the remaining records to find true min/max
    for p in t_tup.iter().skip(1) {
        for (d, &val) in p.iter().enumerate() {
            if val < minima[d] {
                minima[d] = val;
            }
            if val > maxima[d] {
                maxima[d] = val;
            }
        }
    }
    (minima, maxima)
}

pub fn find_valid_solution(
    responses: &HashMap<i64, Vec<i64>>,
    total_responses: &i32,
) -> (i32, usize) {
    let mut best_universe = -1;
    let mut max_matches = 0;

    // The maximum possible score is having a match for every key in the HashMap
    let target_matches = responses.len();

    for i in 0..*total_responses {
        let mut current_matches = 0;

        for (key, val) in responses.iter() {
            if val.get(i as usize) == Some(key) {
                current_matches += 1;
            }
        }

        // Update our leaderboard if this universe scored higher
        if current_matches > max_matches {
            max_matches = current_matches;
            best_universe = i;
        }

        // Early exit: if we hit a perfect match, no need to check the remaining universes
        if max_matches == target_matches {
            break;
        }
    }

    if max_matches == target_matches && target_matches > 0 {
        debug!("Found a consistent universe at index: {}", best_universe);
    } else if best_universe != -1 {
        debug!(
            "No perfectly consistent universe found. Closest was index: {} with {} matches",
            best_universe, max_matches
        );
    } else {
        debug!("No consistent universe found.");
    }

    (best_universe, max_matches)
}

pub fn check_isomorphism(
    responses: &HashMap<i64, i64>,
    rows: i64,
    cols: i64,
) -> Option<&'static str> {
    // 2. Build the solver's actual mapping: True_Plaintext_ID -> Guessed_Plaintext_ID
    let mut true_to_guessed = HashMap::new();
    for (encrypted_alias, guessed_id) in responses {
        true_to_guessed.insert(encrypted_alias, *guessed_id);
    }

    let max_r = rows - 1;
    let max_c = cols - 1;

    // Helper closures to translate between 1D IDs and 2D Coordinates
    // This matches the math from your `flatten_nd` function
    let to_coord = |id: i64| -> (i64, i64) { (id / cols, id % cols) };
    let to_id = |r: i64, c: i64| -> i64 { r * cols + c };

    // 3. Define the 8 valid geometric transformations for a 2D grid
    let transformations: Vec<(&str, Box<dyn Fn(i64, i64) -> (i64, i64)>)> = vec![
        ("Identity (Perfect Match)", Box::new(|r, c| (r, c))),
        ("Rotated 90°", Box::new(move |r, c| (c, max_r - r))),
        ("Rotated 180°", Box::new(move |r, c| (max_r - r, max_c - c))),
        ("Rotated 270°", Box::new(move |r, c| (max_c - c, r))),
        (
            "Reflected Horizontal (Flip Y)",
            Box::new(move |r, c| (r, max_c - c)),
        ),
        (
            "Reflected Vertical (Flip X)",
            Box::new(move |r, c| (max_r - r, c)),
        ),
        ("Reflected Main Diagonal", Box::new(|r, c| (c, r))),
        (
            "Reflected Anti-Diagonal",
            Box::new(move |r, c| (max_c - c, max_r - r)),
        ),
    ];

    // 4. Test the solver's mapping against each transformation
    for (name, transform) in transformations {
        let mut is_match = true;

        for (&true_id, &guessed_id) in &true_to_guessed {
            let (r, c) = to_coord(*true_id);
            let (trans_r, trans_c) = transform(r, c);
            let expected_guessed_id = to_id(trans_r, trans_c);

            if guessed_id != expected_guessed_id {
                is_match = false;
                break;
            }
        }

        // If all points conform to this specific transformation, we cracked it
        if is_match {
            return Some(name);
        }
    }

    None
}

pub fn binomial_coefficient(n: usize, t: usize) -> u64 {
    if t > n {
        return 0;
    }
    // Take advantage of symmetry: C(n, t) == C(n, n-t)
    let t = std::cmp::min(t, n - t);
    let mut result = 1u64;
    for i in 1..=t {
        result = result * (n as u64 - i as u64 + 1) / (i as u64);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dominates() {
        let p1 = vec![1, 2, 3];
        let p2 = vec![1, 1, 1];
        let p3 = vec![2, 2, 2];
        assert!(dominates(&p1, &p2));
        assert!(!dominates(&p2, &p1));
        assert!(!dominates(&p1, &p3));
        assert!(!dominates(&p2, &p3));
    }

    #[test]
    fn test_get_mbq() {
        let points = vec![vec![1, 10, 5], vec![5, 2, 8], vec![3, 5, 1]];
        let (minima, maxima) = get_mbq(&points);
        assert_eq!(minima, vec![1, 2, 1]);
        assert_eq!(maxima, vec![5, 10, 8]);
    }
}
