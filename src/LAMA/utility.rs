use crate::{Coord, DomPair, Record};
use itertools::Itertools;
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
pub enum DistributionType {
    Uniform,
    Gaussian,
    Beta,
}

impl FromStr for DistributionType {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "uniform" => Ok(DistributionType::Uniform),
            "gaussian" => Ok(DistributionType::Gaussian),
            "beta" => Ok(DistributionType::Beta),
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
