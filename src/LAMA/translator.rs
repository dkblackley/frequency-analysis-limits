use crate::Value;
use crate::LAMA::utility::binomial_coefficient;
use cp_sat::builder::{CpModelBuilder, IntVar};
use cp_sat::proto::constraint_proto::Constraint;
use cp_sat::proto::sat_parameters::Polarity::True;
use cp_sat::proto::CpSolverStatus;
use cp_sat::proto::IntegerVariableProto;
use cp_sat::proto::LinearExpressionProto;
use cp_sat::proto::SatParameters;
use cp_sat::proto::{ConstraintProto, CpModelProto, TableConstraintProto};
use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use log::{debug, warn};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use std::collections::HashMap;
use std::io::Read;
use std::sync::atomic::{AtomicU64, Ordering};

// Reads sequential bincode items and groups them by frequency on the fly
struct GroupReader<R: Read> {
    reader: R,
    peeked: Option<(u64, Vec<i64>)>,
}

impl<R: Read> GroupReader<R> {
    fn new(mut reader: R) -> Self {
        let peeked = bincode::deserialize_from(&mut reader).ok();
        Self { reader, peeked }
    }

    fn next_group(&mut self) -> Option<(u64, Vec<Vec<i64>>)> {
        let (current_freq, first_tuple) = self.peeked.take()?;
        let mut group = vec![first_tuple];

        loop {
            match bincode::deserialize_from::<_, (u64, Vec<i64>)>(&mut self.reader) {
                Ok((next_freq, next_tuple)) => {
                    if next_freq == current_freq {
                        group.push(next_tuple);
                    } else {
                        self.peeked = Some((next_freq, next_tuple));
                        return Some((current_freq, group));
                    }
                }
                Err(_) => {
                    return Some((current_freq, group));
                }
            }
        }
    }
}

/// Translator: One Formula from All Matching Pairs.
pub struct Translator {
    upper: i64,
    enc_id_to_intvar: HashMap<i64, IntVar>,
    var_index_map: HashMap<IntVar, (i32, i64)>,
    proto_model: CpModelProto,
    pub candidate_cache: HashMap<usize, HashMap<Vec<i64>, Vec<Vec<i64>>>>,
}

impl Translator {
    pub fn new(largest_val: i64, mut encrypted_records: Vec<i64>) -> Self {
        let mut rng = StdRng::seed_from_u64(42);

        // If we don't shuffle the encrypted records, the first reconstruction found is always the
        // correct one (if there are multiple to be found)
        encrypted_records.shuffle(&mut rng);

        let mut cp_model = CpModelBuilder::default();
        let (var_index_map, enc_id_to_intvar) =
            Self::set_all_vars(&mut cp_model, largest_val, encrypted_records);

        Self {
            proto_model: cp_model.proto().clone(),
            upper: largest_val,
            enc_id_to_intvar,
            var_index_map,
            candidate_cache: HashMap::new(),
        }
    }

    pub fn get_var_index_map(&self) -> HashMap<IntVar, (i32, i64)> {
        return self.var_index_map.clone();
    }

    pub fn get_enc_id_to_intvar(&self) -> HashMap<i64, IntVar> {
        return self.enc_id_to_intvar.clone();
    }

    pub fn get_proto_model(self) -> CpModelProto {
        return self.proto_model;
    }

    pub fn set_proto_model(&mut self, new_model: CpModelProto) {
        self.proto_model = new_model;
    }

    /// Initialise all unknown variables. Each encrypted record is the id of the unknown var,
    /// but this 'encrypted id' is actually just an encoded point. This allows for much easier
    /// tracked of which var was which point (as opposed to actual encryption/decryption, which is
    /// functionally the same).
    fn set_all_vars(
        cp_model: &mut CpModelBuilder,
        upper: Value,
        encrypted_records: Vec<i64>,
    ) -> (HashMap<IntVar, (i32, i64)>, HashMap<i64, IntVar>) {
        let mut count = 0;
        let mut all_vars = Vec::new();

        let mut var_index_map = HashMap::new();
        let mut enc_id_to_intvar = HashMap::new();

        let pb = ProgressBar::new(encrypted_records.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        for encrypted_id in encrypted_records {
            let var = cp_model.new_int_var_with_name([(0, upper)], format!("rec_{encrypted_id}"));

            enc_id_to_intvar.insert(encrypted_id, var);
            var_index_map.insert(var, (count, encrypted_id));
            all_vars.push(var);
            count += 1;
            pb.inc(1);
        }

        pb.finish();
        cp_model.add_all_different(all_vars);

        return (var_index_map, enc_id_to_intvar);
    }

    fn add_allowed_assignments(
        model: &mut CpModelProto,
        vars: &Vec<IntVar>,
        allowed_plaintexts: &Vec<Vec<i64>>,
        var_index_map: &HashMap<IntVar, (i32, i64)>,
    ) {
        let mut table_proto = TableConstraintProto::default();

        for var in vars {
            let var_index = var_index_map.get(var).expect("Unknown variable");

            table_proto
                .exprs
                .push(cp_sat::proto::LinearExpressionProto {
                    vars: vec![var_index.0],
                    coeffs: vec![1],
                    offset: 0,
                });
        }

        for tuple in allowed_plaintexts {
            for plaintext in tuple {
                table_proto.values.push(*plaintext);
            }
        }

        let mut constraint_proto = ConstraintProto::default();
        constraint_proto.constraint = Some(Constraint::Table(table_proto));
        model.constraints.push(constraint_proto);
    }

    /// Base Case: t=1. Direct dictionary lookup. TODO: Replace these with vecs of probs (f64) and records.
    pub fn process_t1(
        &mut self,
        observed_t1: &HashMap<u64, Vec<Vec<i64>>>,
        freq_to_all_pts: &HashMap<u64, Vec<Vec<i64>>>,
    ) {
        let mut t1_cache = HashMap::new();

        for (freq, enc_tuples_list) in observed_t1 {
            // Only proceed if this observed frequency exists in the theoretical perfect map
            if let Some(valid_plaintexts) = freq_to_all_pts.get(freq) {
                // Iterate over every specific encrypted 1-tuple observed at this frequency
                for enc_tuple in enc_tuples_list {
                    // Extract the literal i64 record alias (since t=1, it's at index 0)
                    let rec = enc_tuple[0];

                    // Fetch its corresponding CP-SAT variable
                    let var = *self.enc_id_to_intvar.get(&rec).unwrap_or_else(|| {
                        panic!(
                            "FATAL: Record {} observed but not in the variable map!",
                            rec
                        )
                    });

                    // Constrain this variable to the valid theoretical plaintexts
                    Self::add_allowed_assignments(
                        &mut self.proto_model,
                        &vec![var],
                        valid_plaintexts,
                        &self.var_index_map,
                    );

                    // Insert the specific 1-tuple into the cache for Apriori round t=2
                    t1_cache.insert(enc_tuple.clone(), valid_plaintexts.clone());
                }
            } else {
                // If the map-reduce finds a frequency that doesn't exist mathematically,
                // log it so you know there's a mismatch in the query distribution closure.
                warn!("Observed frequency {} has no theoretical match!", freq);
            }
        }
        self.candidate_cache.insert(1, t1_cache);
    }

    fn generate_apriori_candidates(prev_keys: &mut [Vec<i64>], t: usize) -> Vec<Vec<i64>> {
        let mut candidates = Vec::new();

        // Sort keys lexically so matching prefixes are adjacent
        prev_keys.sort_unstable();

        debug!("Attempting to find candidates using previous round candidates");
        let pb = ProgressBar::new(prev_keys.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
                )
                .unwrap()
                .progress_chars("#>-"),
        );

        for i in 0..prev_keys.len() {
            for j in i + 1..prev_keys.len() {
                // Take the first 'outer key'
                let k1 = &prev_keys[i];
                // Check for ALL possible inner keys.
                let k2 = &prev_keys[j];

                // If the first (t-2) elements match, we can join the last elements
                if k1[..t - 2] == k2[..t - 2] {
                    let mut candidate = Vec::with_capacity(t);
                    candidate.extend_from_slice(&k1[..t - 2]);

                    let a = k1[t - 2];
                    let b = k2[t - 2];

                    if a < b {
                        candidate.push(a);
                        candidate.push(b);
                    } else {
                        candidate.push(b);
                        candidate.push(a);
                    }
                    candidates.push(candidate);
                } else {
                    // Because they are sorted, once the prefix stops matching,
                    // no further keys in the inner loop will match. Break early!
                    break;
                }
            }
            pb.inc(1);
        }
        pb.finish_with_message("found candidates");

        // Remove identical values from solver. I don't think this 'removes reflections' as this
        // is only looking at the encrypted values
        candidates.sort_unstable();
        candidates.dedup();
        candidates
    }

    /// Helper for processing chunks in parallel using bare-metal CP-SAT Protobufs
    fn process_chunk_cpsat<V>(
        chunk: &[Vec<i64>],
        upper_bound: i64,
        prev_t_cache: &HashMap<Vec<i64>, Vec<Vec<i64>>>,
        validate_candidate: &V,
    ) -> Vec<(Vec<i64>, Vec<Vec<i64>>)>
    where
        V: Fn(&[i64], &[i64]) -> bool + Sync + Send,
    {
        let pb = ProgressBar::new(chunk.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        let atom_count = AtomicU64::new(0);

        chunk
            .into_par_iter()
            .filter_map(|enc_t_tuple| {
                let t = enc_t_tuple.len();
                let current = atom_count.fetch_add(1, Ordering::Relaxed);
                if current % 1000 == 0 {
                    pb.set_message(format!("Processed {} tuples...", current));
                }

                // 1. APPLY T-1 PRUNING FIRST! (Made when running apriori candidates?)
                for sub_tuple in enc_t_tuple.iter().copied().combinations(t - 1) {
                    if !prev_t_cache.contains_key(&sub_tuple) {
                        return None;
                    }
                }

                // 3. BUILD RAW PROTOBUF MODEL (Zero Strings, Zero Clones)
                let mut raw_model = CpModelProto::default();

                // Add variables 0 to t-1
                for _ in 0..t {
                    let mut var_proto = IntegerVariableProto::default();
                    var_proto.domain.push(0);
                    var_proto.domain.push(upper_bound);
                    raw_model.variables.push(var_proto);
                }

                let mut all_diff = cp_sat::proto::AllDifferentConstraintProto::default();
                for i in 0..t {
                    all_diff.exprs.push(LinearExpressionProto {
                        vars: vec![i as i32],
                        coeffs: vec![1],
                        offset: 0,
                    });
                }
                let mut constraint_proto = ConstraintProto::default();
                constraint_proto.constraint = Some(Constraint::AllDiff(all_diff));
                raw_model.constraints.push(constraint_proto);

                // 4. INJECT TABLE CONSTRAINTS
                for sub_indices in (0..t).combinations(t - 1) {
                    let mut sub_tuple = Vec::with_capacity(t - 1);
                    for &idx in &sub_indices {
                        sub_tuple.push(enc_t_tuple[idx]);
                    }

                    // We already verified the cache contains this subset in step 1
                    let allowed_sub_assignments = prev_t_cache.get(&sub_tuple).unwrap();

                    let mut table_proto = TableConstraintProto::default();

                    // Bind the local variable indices (0 to t-1) to the table
                    for &idx in &sub_indices {
                        table_proto.exprs.push(LinearExpressionProto {
                            vars: vec![idx as i32],
                            coeffs: vec![1],
                            offset: 0,
                        });
                    }

                    // Flatten the valid assignments into the table constraint
                    for pt_tuple in allowed_sub_assignments {
                        table_proto.values.extend(pt_tuple);
                    }

                    let mut constraint_proto = ConstraintProto::default();
                    constraint_proto.constraint = Some(Constraint::Table(table_proto));
                    raw_model.constraints.push(constraint_proto);
                }

                // 5. SOLVE THE SANDBOX
                let mut params = SatParameters::default();
                params.enumerate_all_solutions = Some(true);
                params.fill_additional_solutions_in_response = Some(true);
                params.solution_pool_size = Some(50_000);
                params.num_search_workers = Some(1);

                let response = cp_sat::ffi::solve_with_parameters(&raw_model, &params);

                if response.status() != CpSolverStatus::Optimal
                    && response.status() != CpSolverStatus::Feasible
                {
                    return None;
                }

                let mut valid_t_assignments = Vec::new();

                // Extract Primary Solution
                let primary_vals: Vec<i64> = (0..t).map(|i| response.solution[i]).collect();
                if validate_candidate(enc_t_tuple, &primary_vals) {
                    valid_t_assignments.push(primary_vals);
                }
                for add_sol in &response.additional_solutions {
                    let add_vals: Vec<i64> = (0..t).map(|i| add_sol.values[i]).collect();

                    if validate_candidate(enc_t_tuple, &add_vals) {
                        valid_t_assignments.push(add_vals);
                    }
                }

                // Canonicalize: CP-SAT might return duplicates in the pool, deduplicate them
                valid_t_assignments.sort_unstable();
                valid_t_assignments.dedup();

                if valid_t_assignments.is_empty() {
                    None
                } else {
                    Some((enc_t_tuple.clone(), valid_t_assignments))
                }
            })
            .collect()
    }

    /// Recursive Case: t > 1.
    pub fn process_t_greater_than_1<V>(
        &mut self,
        t: usize,
        encrypted_records: &[i64],
        validate_candidate: V,
    ) where
        V: Fn(&[i64], &[i64]) -> bool + Sync + Send,
    {
        // 1. Fetch the cache from the previous round
        let prev_t_cache = self
            .candidate_cache
            .get(&(t - 1))
            .expect("Missing previous round cache!")
            .clone();

        // 2. Extract the keys (the valid (t-1)-tuples of ENCRYPTED records)
        let mut prev_valid_tuples: Vec<Vec<i64>> = prev_t_cache.keys().cloned().collect();

        // 3. Generate the Apriori candidates for round t
        let candidate_combinations = Self::generate_apriori_candidates(&mut prev_valid_tuples, t);

        let chunk_size = 20_000;
        let mut all_results = Vec::new();

        // let total_combinations = binomial_coefficient(encrypted_records.len() as usize, t);
        let pb = ProgressBar::new(candidate_combinations.len() as u64);
        pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})")
                    .unwrap()
                    .progress_chars("#>-"),
            );

        //let mut chunk = Vec::with_capacity(chunk_size);

        // Map Phase: Batch combinations to avoid eager memory bombs
        for chunk in candidate_combinations.chunks(chunk_size) {
            let results = Self::process_chunk_cpsat(
                chunk, // Pass it as a slice directly
                self.upper,
                &prev_t_cache,
                &validate_candidate,
            );
            all_results.extend(results);
            pb.inc(chunk.len() as u64); // Safe for partial chunks!
        }

        pb.finish_with_message(format!("Finished finding tuples for t={t}"));

        // Reduce Phase: Sequential Injection
        let mut current_t_cache = HashMap::new();
        for (enc_t_tuple, valid_assignments) in all_results {
            let main_vars: Vec<_> = enc_t_tuple
                .iter()
                .map(|rec| *self.enc_id_to_intvar.get(rec).unwrap())
                .collect();

            Self::add_allowed_assignments(
                &mut self.proto_model,
                &main_vars,
                &valid_assignments,
                &self.var_index_map,
            );
            current_t_cache.insert(enc_t_tuple, valid_assignments);
        }

        self.candidate_cache.insert(t, current_t_cache);
    }
}

// #[cfg(test)]
// mod equivalence_tests {
//     use super::*;
//     // Pulls in Translator, process_chunk, process_chunk_cpsat, etc.
//     use std::collections::HashMap;
//
//     #[test]
//     fn test_dfs_vs_cpsat_equivalence() {
//         // 1. Setup Mock Variables for a t=3 scenario
//         let t = 3;
//         let upper_bound = 15; // Max plaintext value
//         let enc_t_tuple = vec![10i64, 20i64, 30i64]; // Our target 3-tuple of encrypted records
//
//         // 2. Build a mock prev_t_cache (t-1 = 2)
//         // We are saying: "In round 2, these were the only valid plaintext assignments for these pairs"
//         let mut prev_t_cache: HashMap<Vec<i64>, Vec<Vec<i64>>> = HashMap::new();
//
//         // Valid plaintexts for (10, 20) are (1, 2) and (8, 9) and a distractor (5, 5)
//         prev_t_cache.insert(vec![10, 20], vec![vec![1, 2], vec![8, 9], vec![5, 5]]);
//
//         // Valid plaintexts for (10, 30) are (1, 3) and (8, 10)
//         prev_t_cache.insert(vec![10, 30], vec![vec![1, 3], vec![8, 10]]);
//
//         // Valid plaintexts for (20, 30) are (2, 3) and (9, 10)
//         prev_t_cache.insert(vec![20, 30], vec![vec![2, 3], vec![9, 10]]);
//
//         // Mathematically, the natural join of these tables should ONLY yield:
//         // [1, 2, 3] and [8, 9, 10]. The distractor [5, 5] has no matching subsets.
//
//         // 3. Mock Frequency Closures
//         // Only return a target frequency of '42' if the exact target tuple is queried
//         let get_observed_freq = |tuple: &[i64]| -> u64 {
//             if tuple == enc_t_tuple.as_slice() {
//                 42
//             } else {
//                 0
//             }
//         };
//
//         // Only return '42' if the solver guesses one of our valid joined plaintexts
//         let get_expected_freq = |vals: &[i64]| -> u64 {
//             if vals == [1, 2, 3] || vals == [8, 9, 10] {
//                 42
//             } else {
//                 0
//             }
//         };
//
//         // 4. Create the chunk
//         let chunk = vec![enc_t_tuple.clone()];
//
//         // 5. Run the pure Rust DFS method
//         let mut dfs_results = Translator::process_chunk(
//             &chunk,
//             &prev_t_cache,
//             &get_observed_freq,
//             &get_expected_freq,
//         );
//
//         // 6. Run the CP-SAT Protobuf method
//         let mut cpsat_results = Translator::process_chunk_cpsat(
//             &chunk,
//             upper_bound,
//             &prev_t_cache,
//             &get_observed_freq,
//             &get_expected_freq,
//         );
//
//         // 7. Canonicalize sorting to ensure equality isn't tripped up by vector order
//         if let Some((_, dfs_assignments)) = dfs_results.get_mut(0) {
//             dfs_assignments.sort_unstable();
//         }
//         if let Some((_, cpsat_assignments)) = cpsat_results.get_mut(0) {
//             cpsat_assignments.sort_unstable();
//         }
//
//         // 8. Print Results for visual confirmation
//         println!("DFS Results:   {:?}", dfs_results);
//         println!("CP-SAT Results: {:?}", cpsat_results);
//
//         // 9. Assert absolute mathematical equivalence
//         assert_eq!(
//             dfs_results, cpsat_results,
//             "FATAL: DFS and CP-SAT produced different candidate sets!"
//         );
//
//         // 10. Verify they both successfully filtered out the distractor and found the true joins
//         let expected_assignments = vec![vec![1, 2, 3], vec![8, 9, 10]];
//         assert_eq!(dfs_results[0].1, expected_assignments);
//     }
// }
