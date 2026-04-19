use crate::Value;
use cp_sat::builder::{CpModelBuilder, IntVar};
use cp_sat::proto::constraint_proto::Constraint;
use cp_sat::proto::CpSolverStatus;
use cp_sat::proto::LinearExpressionProto;
use cp_sat::proto::SatParameters;
use cp_sat::proto::{ConstraintProto, CpModelProto, TableConstraintProto};
use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use log::{debug, info, warn};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

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
        info!("Starting");
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

            table_proto.exprs.push(LinearExpressionProto {
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

        info!(
            "Starting t=1, processing {} records",
            encrypted_records.len()
        );

        for &enc_id in encrypted_records {
            pb.inc(1);
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
                // Might happen in EVC but will never in perfect...
                warn!(
                    "Encrypted record {} has no valid plaintext assignments!",
                    enc_id
                );
            }
        }

        pb.finish_with_message("Finished finding tuples for t=1");

        self.t_assignment_archive.insert(1, t1_cache);

        debug!("Done with t1");
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

    /// Pure brute-force n-choose-t evaluation.
    /// Does not use previous round caches. Tests every possible plaintext combination.
    pub fn process_t_brute_force<V>(
        t: usize,
        largest_val: i64,
        encrypted_records: &[i64],
        validate_candidate: V,
    ) -> (CpModelProto, HashMap<IntVar, (i32, i64)>)
    where
        V: Fn(&[i64], &[i64]) -> bool + Sync + Send,
    {
        info!("Processing t={} directly", t);
        let mut cp_model = CpModelBuilder::default();
        let (var_index_map, enc_id_to_intvar) =
            Self::set_all_vars(&mut cp_model, largest_val, Vec::from(encrypted_records));

        let mut raw_model = cp_model.proto().clone();
        // 1. Generate ALL n choose t combinations of the encrypted records
        let enc_combinations: Vec<Vec<i64>> =
            encrypted_records.iter().copied().combinations(t).collect();

        let pb = ProgressBar::new(enc_combinations.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        // 2. Brute-force evaluate all possible plaintexts for every tuple
        // (Using par_iter because this search space is huge)
        let t_table_constraints: Vec<(Vec<i64>, Vec<Vec<i64>>)> = enc_combinations
            .into_par_iter()
            .filter_map(|enc_tuple| {
                let mut valid_plaintexts = Vec::new();

                // Create a cartesian product of 0..=upper for 't' dimensions
                let domains: Vec<_> = (0..t).map(|_| 0..=largest_val).collect();

                for pt_tuple in domains.into_iter().multi_cartesian_product() {
                    if validate_candidate(&enc_tuple, &pt_tuple) {
                        valid_plaintexts.push(pt_tuple);
                    }
                }

                pb.inc(1);

                // 2. If the closure found valid matches, keep them
                if valid_plaintexts.is_empty() {
                    None
                } else {
                    Some((enc_tuple, valid_plaintexts))
                }
            })
            .collect();

        for (enc_t_tuple, valid_assignments) in &t_table_constraints {
            let main_vars: Vec<_> = enc_t_tuple
                .iter()
                .map(|rec| *enc_id_to_intvar.get(rec).unwrap())
                .collect();

            Self::add_allowed_assignments(
                &mut raw_model,
                &main_vars,
                &valid_assignments,
                &var_index_map,
            );
        }

        pb.finish_with_message(format!("Finished finding tuples for t={}", t));

        debug!("Done with pure brute-force for t={}", t);

        (raw_model, var_index_map)
    }

    /// Helper for building allowed candidate tables in parallel and running ONE global CP-SAT solve.
    /// Returns an Option containing a tuple:
    /// 1. The ordered list of the unique global variables used in the model.
    /// 2. A Vec of all found global solutions for those variables.
    fn process_cpsat_global<V>(
        &mut self,
        t_minus_1_assignments: &HashMap<Vec<i64>, Vec<Vec<i64>>>,
        single_assignments: &HashMap<i64, HashSet<i64>>, // Mapping of 1-tuple (var) -> [val_1, val_2, ...]
        validate_candidate: &V,
    ) -> Option<HashMap<Vec<i64>, Vec<Vec<i64>>>>
    where
        V: Fn(&[i64], &[i64]) -> bool + Sync + Send,
    {
        // ------------------------------------------------------------------------
        // PHASE 1: Parallel Dynamic Candidate Generation
        // ------------------------------------------------------------------------

        let pb = ProgressBar::new(t_minus_1_assignments.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {eta}")
                .unwrap()
                .progress_chars("##-"),
        );

        // We parallelize over the outer t-1 mapping.
        // flat_map lets us yield an arbitrary number of valid t-tuples from each t-1 tuple.
        let table_constraints: Vec<(Vec<i64>, Vec<Vec<i64>>)> = t_minus_1_assignments
            .par_iter()
            .flat_map(|(t_minus_1_tuple, sub_assigns)| {
                // A local buffer to collect the valid constraints generated by this t-1 tuple
                let mut local_constraints = Vec::new();

                // let current = counter.fetch_add(1, Ordering::Relaxed);
                //
                // if current.is_multiple_of(10_000) {
                //     pb.set_position(current as u64);
                // }
                pb.inc(1);

                for (&single_var, single_assigns) in single_assignments.iter() {
                    // 1. Avoid self-intersection: make sure the single variable isn't already in our tuple
                    if t_minus_1_tuple.contains(&single_var) {
                        continue;
                    }

                    // 2. Construct the new t-tuple
                    let mut t_tuple = Vec::with_capacity(t_minus_1_tuple.len() + 1);
                    // add the 'one-tuple' s unkown var to the vec of vars
                    t_tuple.extend_from_slice(t_minus_1_tuple);
                    t_tuple.push(single_var);

                    let mut pre_validated_assignments = Vec::new();

                    // These unknown vars should map to: For each unknown in the t-1 tuple,
                    // the cartesian product over the possible plaintexts.
                    for sub_assign in sub_assigns {
                        for single_assign in single_assigns {
                            let mut candidate = Vec::with_capacity(sub_assign.len() + 1);

                            // candidate.extend_from_slice(sub_assign);
                            // candidate.push(*single_assign);
                            //
                            // // Get the total number of items to generate full-length permutations
                            // let len = candidate.len();
                            //
                            // // .into_iter() consumes the original 'candidate' vector and
                            // // .permutations(len) yields a new Vec<T> for each permutation.
                            // for perm in candidate.into_iter().permutations(len) {
                            //     // 4. Validate each permutation BEFORE adding it as a constraint
                            //     if validate_candidate(&t_tuple, &perm) {
                            //         pre_validated_assignments.push(perm);
                            //     }
                            // }

                            candidate.extend_from_slice(sub_assign);
                            // Note: Assuming `single_assigns` is Vec<Vec<i64>>.
                            // If it's just Vec<i64>, change this to `candidate.push(*single_assign);`
                            candidate.push(*single_assign);

                            // 4. Validate assignment BEFORE adding it as a constraint
                            if validate_candidate(&t_tuple, &candidate) {
                                pre_validated_assignments.push(candidate);
                            }
                        }
                    }

                    // If we found valid candidates for this specific t-tuple, keep them for the solver
                    if !pre_validated_assignments.is_empty() {
                        local_constraints.push((t_tuple, pre_validated_assignments));
                    }
                }

                local_constraints
            })
            .collect();

        if table_constraints.is_empty() {
            return None; // No valid configurations found to even build a model
        }

        // ------------------------------------------------------------------------
        // PHASE 2: Single Global CP-SAT Model
        // ------------------------------------------------------------------------

        // Gather all unique variables to build the global model.
        // The solver needs variables to be strictly bounded (0..N), so we create a mapping
        // from your internal IDs (i64) to the CP-SAT local indices.

        let mut raw_model = &mut self.proto_model.clone();
        let mut current_t_cache = HashMap::with_capacity(table_constraints.len());

        for (enc_t_tuple, valid_assignments) in &table_constraints {
            let main_vars: Vec<_> = enc_t_tuple
                .iter()
                .map(|rec| *self.enc_id_to_intvar.get(rec).unwrap())
                .collect();

            Self::add_allowed_assignments(
                &mut raw_model,
                &main_vars,
                &valid_assignments,
                &self.var_index_map,
            );

            current_t_cache.insert(enc_t_tuple, valid_assignments);
        }

        // ------------------------------------------------------------------------
        // PHASE 3: Solve
        // ------------------------------------------------------------------------

        let mut params = SatParameters::default();
        params.enumerate_all_solutions = Some(true);
        params.fill_additional_solutions_in_response = Some(true);
        params.solution_pool_size = Some(self.upper as i32);
        params.keep_all_feasible_solutions_in_presolve = Some(true);
        // Since this is a global solve, the solution pool needs to be reasonably bounded or omitted.
        // We'll leave it to the defaults or handle it purely through the 'enumerate_all_solutions' flag.

        // Kill the LP Engine
        params.linearization_level = Some(0);
        params.add_cg_cuts = Some(false);
        params.add_mir_cuts = Some(false);
        params.add_lin_max_cuts = Some(false);

        // Run the solver.
        debug!("Running mini-solve");
        let pb = ProgressBar::new_spinner();
        pb.enable_steady_tick(std::time::Duration::from_millis(500));
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.blue} [{elapsed_precise}] Mini-Solver thinking (no strict ETA for SAT problems)...")
                .unwrap()
        );
        let response = cp_sat::ffi::solve_with_parameters(&raw_model.clone(), &params);
        pb.finish_with_message(format!("Mini-Solver finished in {:?}", pb.elapsed()));
        if response.status() != CpSolverStatus::Optimal
            && response.status() != CpSolverStatus::Feasible
        {
            warn!("Solver returned error: {:?}", response.status());
            return None;
        }
        debug!(
            "Finished mini-solve with {} possible reconstructions in {:?} seconds",
            response.additional_solutions.len(),
            pb.elapsed().as_secs()
        );

        let mut all_global_solutions = Vec::new();

        // Extract the primary solution
        let primary_vals: Vec<i64> = (0..self.enc_id_to_intvar.len())
            .map(|i| response.solution[i])
            .collect();
        all_global_solutions.push(primary_vals);

        // Extract all additional solutions
        for add_sol in &response.additional_solutions {
            let add_vals: Vec<i64> = (0..self.enc_id_to_intvar.len())
                .map(|i| add_sol.values[i])
                .collect();
            all_global_solutions.push(add_vals);
        }

        // Create the final, filtered cache mapping valid t-tuples to their surviving assignments.
        let mut updated_t_cache: HashMap<Vec<i64>, Vec<Vec<i64>>> =
            HashMap::with_capacity(table_constraints.len());

        for (enc_t_tuple, _old_valid_assignments) in table_constraints {
            // 1. Get the indices for the variables in this specific tuple.
            let solver_indices: Vec<usize> = enc_t_tuple
                .iter()
                .map(|rec| {
                    let intvar = self.enc_id_to_intvar.get(rec).unwrap();
                    let (index, _) = self.var_index_map.get(intvar).unwrap();
                    *index as usize
                })
                .collect();

            // 2. Project every global solution down to just this tuple's variables.
            let surviving_assignments: Vec<Vec<i64>> = all_global_solutions
                .iter()
                .map(|global_sol| solver_indices.iter().map(|&idx| global_sol[idx]).collect())
                .collect();

            // let len = surviving_assignments.len();
            // for survivor in surviving_assignments.into_iter().permutations(len) {
            //     for survived in &survivor {
            //         if validate_candidate(&enc_t_tuple, &*survived) {
            //             let main_vars: Vec<_> = enc_t_tuple
            //                 .iter()
            //                 .map(|rec| *self.enc_id_to_intvar.get(rec).unwrap())
            //                 .collect();
            //
            //             Self::add_allowed_assignments(
            //                 &mut self.proto_model,
            //                 &main_vars,
            //                 &survivor,
            //                 &self.var_index_map,
            //             );
            //
            //             // // 3. Deduplicate!
            //             // // Multiple distinct global solutions often share the exact same local sub-assignment.
            //             // surviving_assignments.sort_unstable();
            //             // surviving_assignments.dedup();
            //
            //             // 4. Store in the new cache
            //             if !survivor.is_empty() {
            //                 updated_t_cache.insert(enc_t_tuple.clone(), survivor.clone());
            //             }
            //         }
            //     }
            // }

            // as a sanity check, the surviving assignments MUST match the probs
            for survived in &surviving_assignments {
                if !validate_candidate(&enc_t_tuple, survived) {
                    panic!("Solver broken!")
                }
            }

            let main_vars: Vec<_> = enc_t_tuple
                .iter()
                .map(|rec| *self.enc_id_to_intvar.get(rec).unwrap())
                .collect();

            Self::add_allowed_assignments(
                &mut self.proto_model,
                &main_vars,
                &surviving_assignments,
                &self.var_index_map,
            );

            // // 3. Deduplicate!
            // // Multiple distinct global solutions often share the exact same local sub-assignment.
            // surviving_assignments.sort_unstable();
            // surviving_assignments.dedup();

            // 4. Store in the new cache
            if !surviving_assignments.is_empty() {
                updated_t_cache.insert(enc_t_tuple, surviving_assignments);
            }
        }

        Some(updated_t_cache)
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
        let prev_t_cache = &self
            .t_assignment_archive
            .get(&(t - 1))
            .expect("Missing previous round cache!")
            .clone();

        let direct_map = Self::genereate_direct_mapping(prev_t_cache);
        // Extract just the keys (the valid (t-1)-tuples of ENCRYPTED records)

        let current_t_cache =
            self.process_cpsat_global(prev_t_cache, &direct_map, &validate_candidate);
        self.t_assignment_archive.insert(
            t,
            current_t_cache.expect("WHAT IN THE GOOD GOD DAMN IS GOING ON"),
        );
    }
}
