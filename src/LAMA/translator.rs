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
use log::{debug, error, info, warn};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::io::Read;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Translator: One Formula from All Matching Pairs.
pub struct Translator {
    upper: i64,
    enc_id_to_intvar: HashMap<i64, IntVar>,
    var_index_map: HashMap<IntVar, (i32, i64)>,
    proto_model: CpModelProto,
    // A mapping of t -> (t_tuples -> valid_assignments). prev_t_assignments is specifically for
    // the last round that was run.
    pub t_assignment_archive: HashMap<usize, HashMap<Vec<i64>, Vec<Vec<i64>>>>,
    pub prev_t_assignments: HashMap<i64, Vec<Vec<i64>>>, // map from t-> candidates used in that round.
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
            t_assignment_archive: HashMap::new(),
            prev_t_assignments: HashMap::new(),
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
        self.t_assignment_archive.insert(1, t1_cache);
    }

    /// Given some possible (t-tuple) -> (Valid assignments for that tuple) returns the 'flat' map
    /// that is: Get a direct 'what are the possible values for this single encrypted id'
    fn genereate_direct_mapping(
        t_assignments: &HashMap<Vec<i64>, Vec<Vec<i64>>>,
    ) -> HashMap<i64, HashSet<i64>> {
        let mut direct_map: HashMap<i64, HashSet<i64>> = HashMap::new();

        for (encrypted_tuple, valid_assignments) in t_assignments {
            for (idx, &encrypted_val) in encrypted_tuple.iter().enumerate() {
                // For this specific encrypted value, collect every plaintext
                // the solver allowed it to be in this tuple
                for assignment in valid_assignments {
                    let allowed_plaintext = assignment[idx];

                    direct_map
                        .entry(encrypted_val)
                        .or_default()
                        .insert(allowed_plaintext);
                }
            }
        }

        direct_map
    }

    /// Generates valid t-tuples by taking combinations of previously seen items
    /// and ensuring they have at least one valid, non-overlapping plaintext assignment.
    pub fn generate_candidates_with_assignments(
        prev_t_tuples: &[Vec<i64>], // the 'prev t tuples' the mini solver was ran on. So each item in this vec will be of len 2 if t=3 here.
        t: usize,
        found_assignments: &HashMap<i64, HashSet<i64>>, // A direct 'enc id'-> 'set of possible found values'
    ) -> Vec<Vec<i64>> {
        // 1. Extract all unique encrypted elements from the previous round
        let unique_elements: HashSet<i64> = prev_t_tuples.iter().flatten().copied().collect();
        let unique_elements_sorted: Vec<i64> = unique_elements.into_iter().sorted().collect();

        debug!("Attempting to find candidates for t={t} using assignment mapping");

        // Calculate total combinations for the progress bar (n choose t)
        // Note: for large n, this exact calculation might overflow, so just use a generic spinner
        // or calculate an approximate bound if needed.
        let total_combinations = binomial_coefficient(unique_elements_sorted.len(), t);
        let pb = ProgressBar::new(total_combinations);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} (ETA: {eta_precise}) - Checking candidates...")
                .unwrap()
                .progress_chars("#>-")
        );

        let counter = AtomicUsize::new(0);

        // For O(1) lookups.
        let prev_t_tuples_hashset: HashSet<Vec<i64>> = prev_t_tuples.iter().cloned().collect();

        // 2. Generate all combinations of size 't'
        // From the proof of optimal selection, we know we don't need to consider both [A, B] and [B, A],
        // it's just n choose t. This means that we can only consider the proba of seeing [A, B]
        let valid_candidates: Vec<Vec<i64>> = unique_elements_sorted
            .into_iter()
            .combinations(t)
            .par_bridge() // Converts the sequential iterator to parallel
            .filter(|candidate| {
                let current = counter.fetch_add(1, Ordering::Relaxed);

                // Only pay the Mutex cost of updating the progress bar every 10,000 iterations
                if current % 10_000 == 0 {
                    pb.set_position(current as u64);
                }

                // A trick called the 'apriori trick' says that any valid set t must be made from
                // valid t-1 subsets. For t=2, any t=1 tuple could be any encrypted record, but at the
                // end of this we update to the new 'possible' sub-tuples. Ensure all (t-1) subsets of
                // 'candidate' exist in `prev_t_tuples`.
                let mut all_subsets_exist = true;

                // Generate every possible (t-1) length subset from our current candidate
                // e.g. if candidate is [1, 2, 3], subsets are [1, 2], [1, 3], [2, 3]
                for subset in candidate.clone().into_iter().combinations(t - 1) {
                    // If the previous round didn't find this subset to be valid,
                    // then this current 't' candidate CANNOT be valid.
                    if !prev_t_tuples_hashset.contains(&subset) {
                        all_subsets_exist = false;
                        break; // Break out of the subset loop early
                    }
                }

                if !all_subsets_exist {
                    return false;
                }

                // Trim the space using the assignment map
                return Self::has_valid_injective_assignment(&candidate, found_assignments);
            })
            .collect();

        pb.finish_with_message(format!("Found {} valid candidates", valid_candidates.len()));

        debug!(
            "Found {} valid candidates in round {}",
            valid_candidates.len(),
            t
        );

        valid_candidates
    }

    /// You could think of this function as being equivalent to getting a t-tuple we might be
    /// considering into the SAT solver with the simple 'all different' clause. That is, if we are
    /// considering this t-tuple, is there even t unique plaintexts we can assign (based on
    /// previous results)? (Note we don't actually use a SAT solver as this is simple enough)
    fn has_valid_injective_assignment(
        encrypted_tuple: &[i64],
        possible_assignments: &HashMap<i64, HashSet<i64>>,
    ) -> bool {
        // RUNNING EXAMPLE:
        // Let's say we are checking a t=3 tuple: encrypted_tuple = [10, 20, 30]
        //
        // Our possible_assignments map says:
        // 10 -> [1, 2]
        // 20 -> [2, 3]
        // 30 -> [2]

        // The `stack` holds "paths". A path is just a list of plaintext assignments
        // we are currently testing.
        // We start by pushing an empty path `[]` onto the stack.
        let mut stack: Vec<Vec<i64>> = Vec::new();
        stack.push(Vec::new());

        // Loop until we run out of paths to try.
        while let Some(current_path) = stack.pop() {
            // The length of our path tells us which encrypted item we are trying to map next.
            // Example: If current_path is [1], its len is 1.
            // This means we successfully mapped index 0 (item 10).
            // Now we need to map index 1 (item 20).
            let current_index = current_path.len();

            // If our path is as long as our target tuple, we found a full, valid mapping!
            if current_index == encrypted_tuple.len() {
                return true;
            }

            // Grab the encrypted value we need to find an assignment for.
            let target_encrypted_val = encrypted_tuple[current_index];

            // EXPLORE OPTIONS
            // Look up the possible plaintexts for this encrypted value.
            if let Some(plaintexts) = possible_assignments.get(&target_encrypted_val) {
                // Loop through every possible plaintext option
                for &candidate_plaintext in plaintexts {
                    // The crucial 'Injective' check: Are we already using this plaintext in our path?
                    // Because 't' is small (e.g., 3 or 4), a simple .contains() on a Vec is
                    // actually faster and cleaner than maintaining a HashSet.
                    if !current_path.contains(&candidate_plaintext) {
                        // This plaintext is free!
                        // Clone our current path, add this new choice to it, and push it to the stack
                        // so we can evaluate the next index on the next loop iteration.
                        let mut new_path = current_path.clone();
                        new_path.push(candidate_plaintext);
                        stack.push(new_path);
                    }
                }
            }
        }

        // If the while loop finishes and the stack is empty, it means every single path
        // hit a collision or a dead end. No valid mapping exists.
        false
    }

    /// Takes in a mapping from 'encrypted values' to their possible 'plaintext values'. Takes in
    /// the 't' that we are currently at. If t=2 then we have all the valid 1-tuple assignments
    /// We then want to return a vec of 2-tuples (that are valid 2-tuples). We can do this as follows:
    /// Get a 1 tuple and all 'm' valid 1-tuple assignments. Then for 2-tuples, instead of generating
    /// every possible n choose 2 assignment we
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
            // iterate over all previous t-tuple of encrypted records that had possible assignments
            let k1 = &prev_keys[i];

            for j in i + 1..prev_keys.len() {
                // Check for ALL possible other t-tuples that have possible assignments
                let k2 = &prev_keys[j];

                // Check if there is an overlap in the first (t-2) elements. If this is the case,
                // then the previous model decided that these two t-tuples can be joined and there is a valid
                // assignment. Hence, we must work out the remaining (up to t) possible assignments.
                // (so this t-tuple is a candidate). We then take whatever the remaining elements are
                // from each and add them to the candidate.
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

    /// Helper for processing chunks in parallel using bare-metal CP-SAT Protobufs. Returns A vec of
    /// tuples, the first inner item is the vec of size t we attempted to solve and the second
    /// item is the vec of all found solutions for that t-tuple.
    fn process_chunk_cpsat<V>(
        chunk: &[Vec<i64>],
        upper_bound: i64,
        tuples_to_assignments: &HashMap<Vec<i64>, Vec<Vec<i64>>>,
        validate_candidate: &V,
    ) -> Vec<(Vec<i64>, Vec<Vec<i64>>)>
    where
        V: Fn(&[i64], &[i64]) -> bool + Sync + Send,
    {
        chunk
            .into_par_iter()
            .filter_map(|enc_t_tuple| {
                let t = enc_t_tuple.len();

                let mut raw_model = CpModelProto::default();

                // Add variables 0 to t-1
                for _ in 0..t {
                    // Don't given them a name, we just know we're considering t-items just now
                    // We can always 'find' the right value with a linear loop.
                    let mut var_proto = IntegerVariableProto::default();
                    var_proto.domain.push(0);
                    var_proto.domain.push(upper_bound);
                    raw_model.variables.push(var_proto);
                }

                // force each var to be unique
                let mut all_diff = cp_sat::proto::AllDifferentConstraintProto::default();
                for i in 0..t {
                    // endure the model commits to having one value per variable
                    all_diff.exprs.push(LinearExpressionProto {
                        vars: vec![i as i32],
                        coeffs: vec![1],
                        offset: 0,
                    });
                }

                let mut constraint_proto = ConstraintProto::default();
                constraint_proto.constraint = Some(Constraint::AllDiff(all_diff));
                raw_model.constraints.push(constraint_proto);

                let mut total_possible_solutions = 0;

                // Now we do a 'OR' operation for this t-round. We take the results from the
                // previous round and say "from the previous round, we know that only these specific
                // tuples from the t combinations were possible" and then for all these combinations,
                // we pass in a closure that checks if we're within eps distance of the known prob
                // for this tuple. If we are, add this as a possible assignment. Notice that the
                // frequency-probability check actually comes AFTER.
                for sub_indices in (0..t).combinations(t - 1) {
                    let mut sub_tuple = Vec::with_capacity(t - 1);
                    for &idx in &sub_indices {
                        sub_tuple.push(enc_t_tuple[idx]);
                    }

                    // What plaintexts did we find for this tuple in t-1?
                    let previously_found_plaintexts: &Vec<Vec<i64>> =
                        tuples_to_assignments.get(&sub_tuple).unwrap();

                    // This table does a CP SAT OR. If this t-tuple (found in the previous round)
                    // has a possible assignment (do a lookup in the prob table) then add it here
                    // as a possible assignment
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
                    for pt_tuple in previously_found_plaintexts {
                        total_possible_solutions += pt_tuple.len();
                        table_proto.values.extend(pt_tuple);
                    }

                    let mut constraint_proto = ConstraintProto::default();
                    constraint_proto.constraint = Some(Constraint::Table(table_proto));
                    raw_model.constraints.push(constraint_proto);
                }

                // Run the solver, which now tells ALL us valid assignments for this specific
                // t-tuple given the constraints found from the previous rounds. Notice that we
                // don't use the probability table YET.
                let mut params = SatParameters::default();
                params.enumerate_all_solutions = Some(true);
                params.fill_additional_solutions_in_response = Some(true);
                params.solution_pool_size = Some(total_possible_solutions as i32);
                params.num_search_workers = Some(1);

                let response = cp_sat::ffi::solve_with_parameters(&raw_model, &params);

                if response.status() != CpSolverStatus::Optimal
                    && response.status() != CpSolverStatus::Feasible
                {
                    return None;
                }

                let mut valid_t_assignments = Vec::new();

                // CP Sat has now given us all POSSIBLE t=3 assignments given our t=2 constraints.
                // However, we don't actually know if these t=3 possible assignments make sense
                // in regards to our known plaintext/prob pairs for this 3-tuple. We now
                // 'validate' the candidate by using it's observed frequency vs the known plaintext
                // match. If frequency of the possible 3-tuple matches the known probability of
                // observing it then. If it matches, keep it as valid, else discard it.
                let primary_vals: Vec<i64> = (0..t).map(|i| response.solution[i]).collect();
                if validate_candidate(enc_t_tuple, &primary_vals) {
                    valid_t_assignments.push(primary_vals);
                }
                // extract all the other possible solutions
                for add_sol in &response.additional_solutions {
                    let add_vals: Vec<i64> = (0..t).map(|i| add_sol.values[i]).collect();

                    if validate_candidate(enc_t_tuple, &add_vals) {
                        valid_t_assignments.push(add_vals);
                    }
                }

                if valid_t_assignments.is_empty() {
                    None
                } else {
                    Some((enc_t_tuple.clone(), valid_t_assignments))
                }
            })
            .collect()
    }

    /// The translator in the paper and the translator in code work differently but functionally the same.
    /// Specifically: We compute a solution at each 't' round, considering the previous round result
    /// That is: For, say, t=2, we solve all the 'OR' constraints at t=2 given the constraints at t=1
    /// Then put all these results in a big 'AND'. This is equivalent to how translator usually works
    /// but we just solve the ORs as we see them, and use the results to quickly prune the possible
    /// remaining n choose t items.
    pub fn process_t_greater_than_1<V>(
        &mut self,
        t: usize,
        encrypted_records: &[i64],
        validate_candidate: V,
    ) where
        V: Fn(&[i64], &[i64]) -> bool + Sync + Send,
    {
        // This should be a mapping of (t-1)-tuples to their found values
        let prev_t_cache = self
            .t_assignment_archive
            .get(&(t - 1))
            .expect("Missing previous round cache!")
            .clone();

        let direct_map = Self::genereate_direct_mapping(&prev_t_cache);

        // Extract just the keys (the valid (t-1)-tuples of ENCRYPTED records)
        let prev_valid_tuples: Vec<Vec<i64>> = prev_t_cache.keys().cloned().collect();

        // Generate the Apriori candidates for round t
        // let candidate_combinations = Self::generate_apriori_candidates(&mut prev_valid_tuples, t);
        let candidate_combinations =
            Self::generate_candidates_with_assignments(&prev_valid_tuples, t, &direct_map);

        let chunk_size = 5_000;
        let mut all_results = Vec::new();

        // let total_combinations = binomial_coefficient(encrypted_records.len() as usize, t);
        let pb = ProgressBar::new(candidate_combinations.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        //  Batch combinations to avoid eager memory bombs
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

        let mut current_t_cache = HashMap::new();
        for (enc_t_tuple, valid_assignments) in all_results {
            let main_vars: Vec<_> = enc_t_tuple
                .iter()
                .map(|rec| *self.enc_id_to_intvar.get(rec).unwrap())
                .collect();

            // Finally, we do all this optimisation work just to trim down out t-tuple assignment.
            // This is now the OR seen in the Paper. I.e. in main_vars might be [id_a, id_b] and
            // assignments might be [[3, 4], [4, 3]] to get (id_a = 3 AND id_b=4) OR (id_a = 4 AND id_b=3)
            // (And then ortools explicitly puts and AND between any future calls)
            Self::add_allowed_assignments(
                &mut self.proto_model,
                &main_vars,
                &valid_assignments,
                &self.var_index_map,
            );
            current_t_cache.insert(enc_t_tuple, valid_assignments);
        }

        self.t_assignment_archive.insert(t, current_t_cache);
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
