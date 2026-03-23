use crate::dataloader::datasets::Searchable;
use itertools::Itertools;
use rayon::prelude::*;

// /// Precomputes and serializes TRUE frequencies for dominant pairs and t-tuples of values.
// ///
// /// This function iterates through the domain to calculate how many queries cover specific
// /// point pairs (dominant pairs) and groups of $t$ points (t-tuples). The results are
// /// saved as binary files using `bincode` for later use in frequency analysis.
// ///
// /// # Arguments
// /// * `t` - The size of the value tuples to analyze. Should always be 2 * dimension
// /// * `dim` - The dimensionality of the data.
// /// * `dist` - The distribution type (e.g., "uniform").
// ///
// /// # Returns
// ///
// pub fn get_freq_to_dominant_pair_map(
//     dim: Value,
//     lowest_rec: Value,
//     largest_rec: Value,
//     dist: &str,
// ) -> Result<HashMap<Frequency, DomPair>, DataProcessingError> {
//     info!("Task 1: Computing dominant pair frequencies...");
//     let timer = Instant::now();
//
//     // THis just assumes we start at 1
//     let total_pairs = largest_rec.pow(dim as u32) as Value;
//
//     // Assuming every query can occur, what is the frequency of each dominant pair?
//     let mut true_pair_frequency_dict: HashMap<Frequency, DomPair> = HashMap::new();
//     let domain_iter = (0..dim).map(|_| 1..=largest_rec).multi_cartesian_product();
//
//     // Set up indicatif progress bar
//     let pb = ProgressBar::new(total_pairs as u64);
//     pb.set_style(
//         ProgressStyle::default_bar()
//             .template(
//                 "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})",
//             )?
//             .progress_chars("#>-"),
//     );
//
//     for v in domain_iter {
//         for dv in get_all_dominating_values(&v, largest_rec) {
//             let pair = (v.clone(), dv.clone());
//             let frequency = compute_pair_weight(&pair, dist, largest_rec);
//             true_pair_frequency_dict.insert(frequency, pair);
//         }
//         pb.inc(1); // Increment the progress bar silently
//     }
//     pb.finish_with_message("Done computing dominant pair frequencies");
//
//     // let file = File::create(&path)?;
//     // bincode::serialize_into(BufWriter::new(file), &true_pair_frequency_dict)?;
//
//     info!("Finished DP frequencies in {:?}", timer.elapsed());
//     Ok(true_pair_frequency_dict)
// }

// Test functions
#[cfg(test)]
mod tests {
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
        let weight =
            compute_pair_weight(&(lower.clone(), upper.clone()), "uniform", &lower, &upper);
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
