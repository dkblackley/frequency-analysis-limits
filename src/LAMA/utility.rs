use crate::{Coord, DomPair, Frequency, Record, Value};
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
pub enum Distribution {
    Uniform,
    Other,
}

impl FromStr for Distribution {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "uniform" => Ok(Distribution::Uniform),
            _ => Ok(Distribution::Other),
        }
    }
}

pub fn compute_pair_weight(
    pair: &DomPair,
    dist: &Distribution,
    lowest_rec: &[Value],
    largest_rec: &[Value],
) -> Frequency {
    let (lower, upper) = pair;

    match dist {
        Distribution::Uniform => {
            let mut dominating_vals: u64 = 1;
            for (&u_val, &max_val) in upper.iter().zip(largest_rec.iter()) {
                dominating_vals *= ((max_val + 1) - u_val) as u64;
            }

            let mut dominated_vals: u64 = 1;
            for (&l_val, &min_val) in lower.iter().zip(lowest_rec.iter()) {
                dominated_vals *= ((l_val + 1) - min_val) as u64;
            }

            dominated_vals * dominating_vals
        }
        Distribution::Other => {
            // Fallback logic
            0
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

    #[test]
    fn test_compute_pair_weight_uniform_1d() {
        // In 1D, for range [l, u] in domain [1, n], weight is l * (n + 1 - u)
        let lower = vec![2];
        let upper = vec![4];
        let domain_min = vec![1];
        let domain_max = vec![10];
        let n = 10;
        let weight = compute_pair_weight(
            &(lower, upper),
            &"uniform".parse::<Distribution>().unwrap(),
            &domain_min,
            &domain_max,
        );
        // 2 * (10 + 1 - 4) = 2 * 7 = 14
        assert_eq!(weight, 14);
    }

    #[test]
    fn test_compute_pair_weight_2d_exhaustive() {
        // 2D Domain where N=2.
        let n = 2;
        let dist = "uniform";

        // Point definitions for N=2
        let p_11 = vec![1, 1];
        let p_12 = vec![1, 2];
        let p_21 = vec![2, 1];
        let p_22 = vec![2, 2];

        // Format: (v, dv, expected_weight)
        let test_cases = vec![
            // What are all the possible 'rectangles' that can be made over these points?
            // for any one point there's pretty much always going to be 4. For a 'rectanlge'
            // it's always gonna be 2 and then there is one that covers all points.
            (&p_11, &p_11, 4), // (1*1) * (2*2) = 4
            (&p_11, &p_12, 2), // (1*1) * (2*1) = 2
            (&p_11, &p_21, 2), // (1*1) * (1*2) = 2
            (&p_11, &p_22, 1), // (1*1) * (1*1) = 1
            (&p_12, &p_12, 4), // (1*2) * (2*1) = 4
            (&p_12, &p_22, 2), // (1*2) * (1*1) = 2
            (&p_21, &p_21, 4), // (2*1) * (1*2) = 4
            (&p_21, &p_22, 2), // (2*1) * (1*1) = 2
            (&p_22, &p_22, 4), // (2*2) * (1*1) = 4
        ];

        for (v, dv, expected) in test_cases {
            let pair = (v.clone(), dv.clone());
            let weight = compute_pair_weight(&pair, &dist.parse().unwrap(), &p_11, &p_22);
            assert_eq!(weight, expected, "Failed for pair: v={:?}, dv={:?}", v, dv);
        }
    }
}
