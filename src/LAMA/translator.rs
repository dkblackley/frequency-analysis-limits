use crate::Value;
use crate::LAMA::utility::binomial_coefficient;
use cp_sat::builder::{CpModelBuilder, IntVar};
use cp_sat::proto::constraint_proto::Constraint;
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
use rustc_hash::FxHashMap;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};

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

    /// Base Case: t=1. Direct dictionary lookup. TODO: Add in closure?
    /// Base Case: t=1. Evaluates all 1-tuples using the probability closure.
    pub fn process_t1<V>(&mut self, encrypted_records: &[i64], validate_candidate: V)
    where
        V: Fn(&[i64], &[i64]) -> bool,
    {
        let mut t1_cache = HashMap::new();

        // Optional: Add a progress bar to match the styling of your other rounds
        let pb = ProgressBar::new(encrypted_records.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        for &enc_id in encrypted_records {
            let enc_tuple = vec![enc_id];
            let mut valid_plaintexts = Vec::new();

            // Test this encrypted record against every possible plaintext in the domain.
            // Since we initialized the CP-SAT vars with domain [(0, self.upper)],
            // we check all values from 0 up to self.upper.
            for pt in 0..=self.upper {
                let pt_tuple = vec![pt];

                // 1. Run the unified closure!
                if validate_candidate(&enc_tuple, &pt_tuple) {
                    valid_plaintexts.push(pt_tuple);
                }
            }

            // 2. If the closure found valid matches, add them to the model and cache
            if !valid_plaintexts.is_empty() {
                // Fetch its corresponding CP-SAT variable
                let var = *self.enc_id_to_intvar.get(&enc_id).unwrap_or_else(|| {
                    panic!(
                        "FATAL: Record {} observed but not in the variable map!",
                        enc_id
                    )
                });

                // Constrain this variable to the valid theoretical plaintexts
                Self::add_allowed_assignments(
                    &mut self.proto_model,
                    &vec![var],
                    &valid_plaintexts,
                    &self.var_index_map,
                );

                // Insert the specific 1-tuple into the cache for Apriori round t=2
                t1_cache.insert(enc_tuple, valid_plaintexts);
            } else {
                // If it maps to nothing, it's good to log a warning so you know
                // something is mathematically mismatched in the epsilon bounds
                warn!(
                    "Encrypted record {} has no valid plaintext assignments!",
                    enc_id
                );
            }

            pb.inc(1);
        }

        pb.finish_with_message("Finished finding tuples for t=1");

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

        // for better multi-threading
        let all_combinations: Vec<Vec<i64>> =
            unique_elements_sorted.into_iter().combinations(t).collect();

        // 2. Generate all combinations of size 't'
        // From the proof of optimal selection, we know we don't need to consider both [A, B] and [B, A],
        // it's just n choose t. This means that we can only consider the proba of seeing [A, B]
        let valid_candidates: Vec<Vec<i64>> = all_combinations
            .into_par_iter()
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
    /// considering this t-tuple, is there even t unique plaintexts we can assign - based on
    /// previous results? (Note we don't actually use a SAT solver as this is simple enough)
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

    /// Helper for processing chunks in parallel using bare-metal CP-SAT Protobufs. Returns A vec of
    /// tuples, the first inner item is the vec of size t we attempted to solve and the second
    /// item is the vec of all found solutions for that t-tuple.
    fn process_cpsat<V>(
        candidates: &[Vec<i64>],
        upper_bound: i64,
        tuples_to_assignments: &HashMap<Vec<i64>, Vec<Vec<i64>>>,
        domain_sizes: &FxHashMap<i64, usize>,
        validate_candidate: &V,
    ) -> Vec<(Vec<i64>, Vec<Vec<i64>>)>
    where
        V: Fn(&[i64], &[i64]) -> bool + Sync + Send,
    {
        let pb = ProgressBar::new(candidates.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        let counter = AtomicUsize::new(0);

        candidates
            .into_par_iter()
            .filter_map(|enc_t_tuple| {
                counter.fetch_add(1, Ordering::Relaxed);

                if counter.load(Ordering::Relaxed) % 10_000 == 0 {
                    pb.set_position(counter.load(Ordering::Relaxed) as u64);
                }

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

                // We need to know the potential number of solutions in the output. We work this
                // out dynamically.
                let mut lowest_solution_bound: usize = usize::MAX;

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

                    // 1. Find which index is missing
                    let missing_idx = (0..t).find(|x| !sub_indices.contains(x)).unwrap();
                    let missing_enc_val = enc_t_tuple[missing_idx];

                    // 2. Lightning fast lookup of the precomputed size
                    let missing_domain_size = domain_sizes
                        .get(&missing_enc_val)
                        .copied() // FxHashMap yields a reference, so we copy the usize
                        .unwrap_or((upper_bound + 1) as usize);

                    // 3. Calculate max solutions bounded by THIS specific table
                    let max_from_this_table =
                        previously_found_plaintexts.len() * missing_domain_size;

                    if max_from_this_table < lowest_solution_bound {
                        lowest_solution_bound = max_from_this_table;
                    }

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
                        table_proto.values.extend(pt_tuple);
                    }

                    let mut constraint_proto = ConstraintProto::default();
                    constraint_proto.constraint = Some(Constraint::Table(table_proto));
                    raw_model.constraints.push(constraint_proto);
                }

                // Because we're trying to solve many many small
                // SAT problems, the 'precompute' ends up taking more
                // time than the actual solution. We turn all that off with these params
                let mut params = SatParameters::default();
                params.enumerate_all_solutions = Some(true);
                params.fill_additional_solutions_in_response = Some(true);
                params.solution_pool_size = Some(lowest_solution_bound as i32);
                params.num_search_workers = Some(1);

                // params.cp_model_presolve = Some(false);
                // params.linearization_level = Some(0);
                // params.log_search_progress = Some(false);
                // params.catch_sigint_signal = Some(false);

                params.catch_sigint_signal = Some(false);
                params.log_search_progress = Some(false);

                // 2. Kill the "Smart" Heuristics (Overhead reducers)
                params.cp_model_presolve = Some(false);
                params.cp_model_probing_level = Some(0);
                params.symmetry_level = Some(0);
                params.use_probing_search = Some(false);

                // 3. Kill the LP Engine
                params.linearization_level = Some(0);
                params.add_cg_cuts = Some(false);
                params.add_mir_cuts = Some(false);
                params.add_lin_max_cuts = Some(false);

                // 4. Force a simple search strategy
                params.random_branches_ratio = Some(0.0);

                // Run the solver, which now tells ALL us valid assignments for this specific
                // t-tuple given the constraints found from the previous rounds. Notice that we
                // don't use the probability table YET.
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
        _encrypted_records: &[i64],
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

        let mut domain_sizes: FxHashMap<i64, usize> = FxHashMap::default();
        for (&enc_val, plaintexts) in &direct_map {
            domain_sizes.insert(enc_val, plaintexts.len());
        }

        // Extract just the keys (the valid (t-1)-tuples of ENCRYPTED records)
        let prev_valid_tuples: Vec<Vec<i64>> = prev_t_cache.keys().cloned().collect();

        // Generate the Apriori candidates for round t
        // let candidate_combinations = Self::generate_apriori_candidates(&mut prev_valid_tuples, t);
        let candidate_combinations =
            Self::generate_candidates_with_assignments(&prev_valid_tuples, t, &direct_map);

        let chunk_size = 25_000;
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

        let results = Self::process_cpsat(
            &*candidate_combinations, // Pass it as a slice directly
            self.upper,
            &prev_t_cache,
            &domain_sizes,
            &validate_candidate,
        );
        all_results.extend(results);

        pb.finish_with_message(format!("Finished finding tuples for t={t}"));
        debug!("Found {} results for t={}", all_results.len(), t);

        debug!("Updating t-cache for next round...");
        let pb = ProgressBar::new(all_results.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        // OPTIMIZATION: Pre-allocate the exact capacity so the HashMap doesn't have to
        // constantly re-allocate and move memory as it grows to 700k items.
        let mut current_t_cache = HashMap::with_capacity(all_results.len());

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

            // FIX: Increment the progress bar INSIDE the loop!
            // (Indicatif handles internal throttling automatically, so calling this
            // 700k times will not slow down your loop).
            pb.inc(1);
        }

        pb.finish_with_message("Finished updating t-cache");

        self.t_assignment_archive.insert(t, current_t_cache);
    }
}
