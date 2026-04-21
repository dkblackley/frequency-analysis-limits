use crate::LAMA::ortools_wrap::{IntVar, PythonCpModel, TableConstraint};
use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use log::{debug, info, trace, warn};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

const TRUNC_AMOUNT: [i32; 4] = [10, 500, 1000, 1500];

/// Translator: One Formula from All Matching Pairs.
pub struct Translator {
    upper: i64,
    enc_id_to_intvar: HashMap<i64, IntVar>,
    var_index_map: HashMap<IntVar, (i32, i64)>,
    proto_model: PythonCpModel, // Replaced CpModelProto
    pub t_assignment_archive: HashMap<usize, HashMap<Vec<i64>, Vec<Vec<i64>>>>,
    pub prev_t_assignments: HashMap<i64, Vec<Vec<i64>>>,
}

impl Translator {
    pub fn new(largest_val: i64, mut encrypted_records: Vec<i64>) -> Self {
        info!("Starting");
        let mut rng = StdRng::seed_from_u64(42);
        encrypted_records.shuffle(&mut rng);

        let mut cp_model = PythonCpModel::new();
        let (var_index_map, enc_id_to_intvar) =
            Self::set_all_vars(&mut cp_model, largest_val, encrypted_records);

        Self {
            proto_model: cp_model,
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

    pub fn get_proto_model(self) -> PythonCpModel {
        return self.proto_model;
    }

    pub fn set_proto_model(&mut self, new_model: PythonCpModel) {
        self.proto_model = new_model;
    }

    fn set_all_vars(
        cp_model: &mut PythonCpModel,
        _upper: i64,
        encrypted_records: Vec<i64>,
    ) -> (HashMap<IntVar, (i32, i64)>, HashMap<i64, IntVar>) {
        let mut count = 0;
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
            let var = IntVar(count as usize);

            enc_id_to_intvar.insert(encrypted_id, var);
            var_index_map.insert(var, (count, encrypted_id));
            count += 1;
            pb.inc(1);
        }

        pb.finish();

        // Signal to the wrapper that all variables must be distinct
        cp_model.num_vars = count as usize;
        cp_model.all_different = true;

        return (var_index_map, enc_id_to_intvar);
    }

    fn add_allowed_assignments(
        model: &mut PythonCpModel,
        vars: &Vec<IntVar>,
        allowed_plaintexts: &Vec<Vec<i64>>,
        var_index_map: &HashMap<IntVar, (i32, i64)>,
    ) {
        let mut var_indices = Vec::new();
        for var in vars {
            let var_index = var_index_map.get(var).expect("Unknown variable");
            var_indices.push(var_index.0 as usize);
        }

        model.table_constraints.push(TableConstraint {
            vars: var_indices,
            values: allowed_plaintexts.clone(),
        });
    }

    pub fn process_t1<V>(&mut self, encrypted_records: &[i64], validate_candidate: V)
    where
        V: Fn(&[i64], &[i64]) -> (bool, f64),
    {
        let mut t1_cache = HashMap::new();

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

            for pt in 0..=self.upper {
                let pt_tuple = vec![pt];
                let (valid, prob) = validate_candidate(&enc_tuple, &pt_tuple);
                if valid {
                    valid_plaintexts.push((pt_tuple, prob));
                }
            }

            if !valid_plaintexts.is_empty() {
                let var = *self.enc_id_to_intvar.get(&enc_id).unwrap();

                valid_plaintexts.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                valid_plaintexts.truncate(TRUNC_AMOUNT[0] as usize);
                let just_plaintexts: Vec<Vec<i64>> =
                    valid_plaintexts.iter().map(|(pt, _)| pt.clone()).collect();

                // Exactly matches your logic: modifying global proto_model
                Self::add_allowed_assignments(
                    &mut self.proto_model,
                    &vec![var],
                    &just_plaintexts,
                    &self.var_index_map,
                );

                t1_cache.insert(enc_tuple, just_plaintexts);
            } else {
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

    fn genereate_direct_mapping(
        t_assignments: &HashMap<Vec<i64>, Vec<Vec<i64>>>,
    ) -> HashMap<i64, HashSet<i64>> {
        let mut direct_map: HashMap<i64, HashSet<i64>> = HashMap::new();

        for (encrypted_tuple, valid_assignments) in t_assignments {
            for (idx, &encrypted_val) in encrypted_tuple.iter().enumerate() {
                for assignment in valid_assignments {
                    direct_map
                        .entry(encrypted_val)
                        .or_default()
                        .insert(assignment[idx]);
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
        mut cp_model: PythonCpModel,
        validate_candidate: V,
    ) -> (PythonCpModel, HashMap<IntVar, (i32, i64)>)
    where
        V: Fn(&[i64], &[i64]) -> (bool, f64) + Sync + Send,
    {
        info!("Processing t={} directly", t);

        let (var_index_map, enc_id_to_intvar) =
            Self::set_all_vars(&mut cp_model, largest_val, Vec::from(encrypted_records));

        // 2. Generate ALL n choose t combinations of the encrypted records
        let enc_combinations: Vec<Vec<i64>> =
            encrypted_records.iter().copied().combinations(t).collect();

        let pb = ProgressBar::new(enc_combinations.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        // 3. Brute-force evaluate all possible plaintexts for every tuple
        // (Using par_iter because this search space is huge)
        let t_table_constraints: Vec<(Vec<i64>, Vec<(Vec<i64>, f64)>)> = enc_combinations
            .into_par_iter()
            .filter_map(|enc_tuple| {
                let mut valid_plaintexts = Vec::new();

                // Create a cartesian product of 0..=upper for 't' dimensions
                let domains: Vec<_> = (0..t).map(|_| 0..=largest_val).collect();

                for pt_tuple in domains.into_iter().multi_cartesian_product() {
                    let (valid, prob) = validate_candidate(&enc_tuple, &pt_tuple);
                    if valid {
                        valid_plaintexts.push((pt_tuple, prob));
                    }
                }

                pb.inc(1);

                // If the closure found valid matches, keep them
                if valid_plaintexts.is_empty() {
                    None
                } else {
                    Some((enc_tuple, valid_plaintexts))
                }
            })
            .collect();

        // 4. Append all surviving valid permutations into the model state
        for (enc_t_tuple, mut valid_assignments) in t_table_constraints {
            let main_vars: Vec<_> = enc_t_tuple
                .iter()
                .map(|rec| *enc_id_to_intvar.get(rec).unwrap())
                .collect();

            valid_assignments.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            valid_assignments.truncate(*TRUNC_AMOUNT.get(t).unwrap_or(&4000) as usize);
            let just_plaintexts: Vec<Vec<i64>> =
                valid_assignments.iter().map(|(pt, _)| pt.clone()).collect();

            // Mutate our custom PythonCpModel directly
            Self::add_allowed_assignments(
                &mut cp_model,
                &main_vars,
                &just_plaintexts,
                &var_index_map,
            );
        }

        pb.finish_with_message(format!("Finished finding tuples for t={}", t));

        debug!("Done with pure brute-force for t={}", t);

        (cp_model, var_index_map)
    }

    fn process_cpsat_global<V>(
        &mut self,
        t_minus_1_assignments: &HashMap<Vec<i64>, Vec<Vec<i64>>>,
        single_assignments: &HashMap<i64, HashSet<i64>>,
        validate_candidate: &V,
    ) -> Option<HashMap<Vec<i64>, Vec<Vec<i64>>>>
    where
        V: Fn(&[i64], &[i64]) -> (bool, f64) + Sync + Send,
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
        let mut t = 0;
        if let Some(random_key) = t_minus_1_assignments.keys().next() {
            t = random_key.len();
        }

        let table_constraints: Vec<(Vec<i64>, Vec<(Vec<i64>, f64)>)> = t_minus_1_assignments
            .clone()
            .par_iter()
            .flat_map(|(t_minus_1_tuple, sub_assigns)| {
                let mut local_constraints = Vec::new();
                pb.inc(1);

                for (&single_var, single_assigns) in single_assignments.iter() {
                    if t_minus_1_tuple.contains(&single_var) {
                        continue;
                    }

                    let mut t_tuple = Vec::with_capacity(t_minus_1_tuple.len() + 1);
                    t_tuple.extend_from_slice(t_minus_1_tuple);
                    t_tuple.push(single_var);

                    let mut pre_validated_assignments = Vec::new();

                    for sub_assign in sub_assigns {
                        for single_assign in single_assigns {
                            let mut candidate = Vec::with_capacity(sub_assign.len() + 1);
                            candidate.extend_from_slice(sub_assign);
                            candidate.push(*single_assign);
                            let (valid, prob) = validate_candidate(&t_tuple, &candidate);

                            if valid {
                                pre_validated_assignments.push((candidate, prob));
                            }
                        }
                    }

                    if !pre_validated_assignments.is_empty() {
                        local_constraints.push((t_tuple, pre_validated_assignments));
                    }
                }
                local_constraints
            })
            .collect();

        if table_constraints.is_empty() {
            return None;
        }

        // ------------------------------------------------------------------------
        // PHASE 2: Single Global CP-SAT Model
        // ------------------------------------------------------------------------
        let mut current_t_cache = HashMap::with_capacity(table_constraints.len());

        for (enc_t_tuple, mut valid_assignments) in table_constraints.clone() {
            let main_vars: Vec<_> = enc_t_tuple
                .iter()
                .map(|rec| *self.enc_id_to_intvar.get(rec).unwrap())
                .collect();

            valid_assignments.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            valid_assignments.truncate(*TRUNC_AMOUNT.get(t).unwrap_or(&4000) as usize);
            let just_plaintexts = valid_assignments.iter().map(|(pt, _)| pt.clone()).collect();

            // Pushing constraints into global proto_model
            Self::add_allowed_assignments(
                &mut self.proto_model,
                &main_vars,
                &just_plaintexts,
                &self.var_index_map,
            );

            current_t_cache.insert(enc_t_tuple, valid_assignments);
        }

        // ------------------------------------------------------------------------
        // PHASE 3: Solve
        // ------------------------------------------------------------------------
        debug!("Running mini-solve via Python");
        let solve_pb = ProgressBar::new_spinner();
        solve_pb.enable_steady_tick(std::time::Duration::from_millis(500));
        solve_pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.blue} [{elapsed_precise}] Mini-Solver thinking (no strict ETA for SAT problems)...")
                .unwrap()
        );

        // This calls your wrapper instead of ffi::solve_with_parameters
        let response = self.proto_model.solve(self.upper, false);
        solve_pb.finish_with_message(format!("Mini-Solver finished in {:?}", solve_pb.elapsed()));

        if response.is_none() {
            warn!("Solver returned error or Infeasible");
            return None;
        }

        let all_global_solutions = response.unwrap();
        debug!(
            "Finished mini-solve with {} possible reconstructions",
            all_global_solutions.len()
        );

        // ------------------------------------------------------------------------
        // PHASE 4: Extract and Filter
        // ------------------------------------------------------------------------
        let mut updated_t_cache: HashMap<Vec<i64>, Vec<Vec<i64>>> =
            HashMap::with_capacity(table_constraints.len());

        for (enc_t_tuple, _old_valid_assignments) in table_constraints {
            let solver_indices: Vec<usize> = enc_t_tuple
                .iter()
                .map(|rec| {
                    let intvar = self.enc_id_to_intvar.get(rec).unwrap();
                    self.var_index_map.get(intvar).unwrap().0 as usize
                })
                .collect();

            let mut surviving_assignments: Vec<Vec<i64>> = all_global_solutions
                .iter()
                .map(|global_sol| solver_indices.iter().map(|&idx| global_sol[idx]).collect())
                .collect();

            // Match your sanity check
            for survived in &surviving_assignments {
                if !validate_candidate(&enc_t_tuple, survived).0 {
                    panic!("Solver broken!")
                }
            }

            let main_vars: Vec<_> = enc_t_tuple
                .iter()
                .map(|rec| *self.enc_id_to_intvar.get(rec).unwrap())
                .collect();

            // Match your double-addition
            Self::add_allowed_assignments(
                &mut self.proto_model,
                &main_vars,
                &surviving_assignments,
                &self.var_index_map,
            );

            // Deduplicate
            surviving_assignments.sort_unstable();
            surviving_assignments.dedup();

            if !surviving_assignments.is_empty() {
                updated_t_cache.insert(enc_t_tuple, surviving_assignments);
            }
        }

        Some(updated_t_cache)
    }

    pub fn process_t_greater_than_1<V>(
        &mut self,
        t: usize,
        _encrypted_records: &[i64],
        validate_candidate: V,
    ) where
        V: Fn(&[i64], &[i64]) -> (bool, f64) + Sync + Send,
    {
        let prev_t_cache = &self
            .t_assignment_archive
            .get(&(t - 1))
            .expect("Missing previous round cache!")
            .clone();

        let direct_map = Self::genereate_direct_mapping(prev_t_cache);

        let current_t_cache =
            self.process_cpsat_global(prev_t_cache, &direct_map, &validate_candidate);

        self.t_assignment_archive
            .insert(t, current_t_cache.expect("Failed..."));
    }
}
