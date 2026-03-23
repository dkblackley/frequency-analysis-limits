use crate::{Coord, DomPair, Frequency, Record, Value};
use itertools::Itertools;
use log::warn;

// Helper to determine if point u dominates point v (u_i >= v_i for all i)
pub fn dominates(u: &[Coord], v: &[Coord]) -> bool {
    u.iter().zip(v.iter()).all(|(u_val, v_val)| u_val >= v_val)
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

pub fn compute_pair_weight(
    pair: &DomPair,
    dist: &String,
    lowest_rec: &[Value],
    largest_rec: &[Value],
) -> Frequency {
    let (lower, upper) = pair;

    // Under uniform, simply count the queries covering the pair
    if dist == "uniform" {
        let mut dominating_vals: u64 = 1;
        // Pair each upper value with its dimension's max bound
        for (&u_val, &max_val) in upper.iter().zip(largest_rec.iter()) {
            dominating_vals *= ((max_val + 1) - u_val) as u64;
        }

        let mut dominated_vals: u64 = 1;
        // Pair each lower value with its dimension's min bound
        for (&l_val, &min_val) in lower.iter().zip(lowest_rec.iter()) {
            dominated_vals *= ((l_val + 1) - min_val) as u64;
        }

        dominated_vals * dominating_vals
    }
    // Fallback for unimplemented distributions ('random', 'flattened', etc.)
    //TODO: Cartesian prodect of all possible queries over any given 'rectangle'
    else {
        warn!(
            "Distribution '{}' is not fully implemented. Returning weight 0.",
            dist
        );
        return 0;
    }
}

/// Return the minimum bounding query (MBQ) of a t-tuple (i.e. dominating vals)
pub fn get_mbq(t_tup: &[Record]) -> DomPair {
    let dim = t_tup[0].len();
    let mut minima = vec![Value::MAX; dim];
    let mut maxima = vec![Value::MIN; dim];

    for p in t_tup {
        for d in 0..dim {
            if p[d] < minima[d] {
                minima[d] = p[d];
            }
            if p[d] > maxima[d] {
                maxima[d] = p[d];
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
        let n = 10;
        let weight = compute_pair_weight(
            &(lower.clone(), upper.clone()),
            "uniform".into(),
            &lower,
            &upper,
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
            let weight = compute_pair_weight(&pair, dist, &p_11, &p_22);
            assert_eq!(weight, expected, "Failed for pair: v={:?}, dv={:?}", v, dv);
        }
    }
}
