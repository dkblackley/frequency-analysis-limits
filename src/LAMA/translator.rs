/// Translator: One Formula from All Matching Pairs[cite: 186].
/// Finds matching pairs and translates them into a logical formula constraining
/// value-to-record assignments[cite: 186, 187].
pub struct Translator;

impl Translator {
    /// Computes the Manhattan distance (L1) between two values in the domain[cite: 117].
    pub fn l1_distance(u: &[u32], v: &[u32]) -> u32 {
        u.iter().zip(v.iter()).map(|(a, b)| a.abs_diff(*b)).sum()
    }

    /// Determines if value `u` dominates value `v` across all dimensions[cite: 116].
    pub fn dominates(u: &[u32], v: &[u32]) -> bool {
        u.iter().zip(v.iter()).all(|(a, b)| a >= b)
    }

    /// Finds the Minimum Bounding Query (MBQ) for a given tuple of values.
    /// A range query is essentially a hyperrectangle defined by two vertices[cite: 125].
    pub fn get_mbq(t_tuple: &[Vec<u32>]) -> (Vec<u32>, Vec<u32>) {
        if t_tuple.is_empty() {
            return (vec![], vec![]);
        }
        let dim = t_tuple[0].len();
        let mut minima = vec![u32::MAX; dim];
        let mut maxima = vec![u32::MIN; dim];

        for v in t_tuple {
            for d in 0..dim {
                if v[d] < minima[d] {
                    minima[d] = v[d];
                }
                if v[d] > maxima[d] {
                    maxima[d] = v[d];
                }
            }
        }
        (minima, maxima)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l1_distance() {
        let u = vec![9, 7, 3];
        let v = vec![17, 8, 5];
        assert_eq!(Translator::l1_distance(&u, &v), 11);
    }

    #[test]
    fn test_get_mbq() {
        let t_tuple = vec![vec![1, 5], vec![3, 2], vec![2, 4]];
        let (minima, maxima) = Translator::get_mbq(&t_tuple);
        assert_eq!(minima, vec![1, 2]);
        assert_eq!(maxima, vec![3, 5]);
    }
}
