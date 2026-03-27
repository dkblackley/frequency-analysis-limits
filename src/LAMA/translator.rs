use crate::Value;
use crate::LAMA::error::LAMAError;
use crate::LAMA::utility::binomial_coefficient;
use cp_sat::builder::{CpModelBuilder, IntVar};
use cp_sat::proto::constraint_proto::Constraint;
use cp_sat::proto::{ConstraintProto, CpModelProto, TableConstraintProto};
use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};

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

    fn find_matching_pairs(
        &mut self,
        plaintext_filepath: &str,
        observed_filepath: &str,
        t: u64,
    ) -> Result<(), LAMAError> {
        println!("Co-iterating over sorted plaintext and observed files...");
        let mut model = self.proto_model.clone();

        let pt_file = File::open(plaintext_filepath).expect("Could not open plaintext file");
        let obs_file = File::open(observed_filepath).expect("Could not open observed file");

        let pt_reader = BufReader::with_capacity(50 * 1024 * 1024, pt_file);
        let obs_reader = BufReader::with_capacity(50 * 1024 * 1024, obs_file);

        let mut pt_grouper = GroupReader::new(pt_reader);
        let mut obs_grouper = GroupReader::new(obs_reader);

        let mut pt_current = pt_grouper.next_group();
        let mut obs_current = obs_grouper.next_group();

        let total_freqs = binomial_coefficient((self.upper + 1) as usize, t as usize);
        let pb = ProgressBar::new(total_freqs);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})",
                )?
                .progress_chars("#>-"),
        );
        let mut counter: usize = 0;
        let batch_size = 10_000;

        let mut matched_frequencies = 0;

        while let (Some((f_pt, pt_tuples)), Some((f_obs, obs_tuples))) = (&pt_current, &obs_current)
        {
            counter += pt_tuples.len();
            if counter % batch_size == 0 {
                pb.inc(batch_size as u64);
            }

            if f_pt < f_obs {
                pt_current = pt_grouper.next_group();
            } else if f_obs < f_pt {
                obs_current = obs_grouper.next_group();
            } else {
                matched_frequencies += 1;

                for encrypted_tuple in obs_tuples {
                    let mut vars = Vec::with_capacity(encrypted_tuple.len());

                    for &encrypted_rec in encrypted_tuple {
                        if let Some(var) = self.enc_id_to_intvar.get(&encrypted_rec) {
                            vars.push(*var);
                        } else {
                            panic!(
                                "FATAL: The ID {} was found in the observed file, but it does not exist in the variable map!",
                                encrypted_rec
                            );
                        }
                    }

                    Self::add_allowed_assignments(
                        &mut model,
                        &vars,
                        pt_tuples,
                        &self.var_index_map,
                    );
                }

                pt_current = pt_grouper.next_group();
                obs_current = obs_grouper.next_group();
            }
        }

        pb.finish();

        println!(
            "Successfully generated constraints for {} unique frequencies.",
            matched_frequencies
        );
        self.proto_model = model.clone();
        Ok(())
    }

    pub fn translate(
        &mut self,
        freq_record_match_filepath: &str,
        observed_freq_record_match_filepath: &str,
        t: u64,
    ) -> Result<(), LAMAError> {
        self.find_matching_pairs(
            freq_record_match_filepath,
            observed_freq_record_match_filepath,
            t,
        )
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

    /// Base Case: t=1. Direct dictionary lookup.
    pub fn process_t1(
        &mut self,
        observed_t1: &[(u64, Vec<i64>)],
        pt_dict: &HashMap<u64, Vec<Vec<i64>>>,
    ) {
        let mut t1_cache = HashMap::new();

        for (freq, enc_tuple) in observed_t1 {
            if let Some(valid_plaintexts) = pt_dict.get(freq) {
                let rec = enc_tuple[0];
                let var = *self.enc_id_to_intvar.get(&rec).unwrap();

                Self::add_allowed_assignments(
                    &mut self.proto_model,
                    &vec![var],
                    valid_plaintexts,
                    &self.var_index_map,
                );

                t1_cache.insert(enc_tuple.clone(), valid_plaintexts.clone());
            }
        }
        self.candidate_cache.insert(1, t1_cache);
    }

    /// Pure Rust DFS Backtracking Search
    /// Iteratively prunes domain subsets using the mathematical definitions of combinations
    fn dfs_find_valid_assignments<E>(
        pos: usize,
        current_vals: &mut Vec<i64>,
        enc_t_tuple: &[i64],
        domains: &[Vec<i64>],
        prev_t_cache: &HashMap<Vec<i64>, Vec<Vec<i64>>>,
        get_expected_freq: &E,
        target_freq: u64,
        valid_assignments: &mut Vec<Vec<i64>>,
    ) where
        E: Fn(&[i64]) -> u64 + Sync + Send,
    {
        let t = enc_t_tuple.len();

        // Base case: Valid combination found
        if pos == t {
            if get_expected_freq(current_vals) == target_freq {
                valid_assignments.push(current_vals.clone());
            }
            return;
        }

        for &val in &domains[pos] {
            let mut ok = true;

            // Only check constraints when we have enough variables to form a t-1 subset
            if pos >= t - 1 {
                for subset_indices in (0..pos).combinations(t - 2) {
                    let mut check_rec_subset = Vec::with_capacity(t - 1);
                    let mut check_val_subset = Vec::with_capacity(t - 1);

                    for &idx in &subset_indices {
                        check_rec_subset.push(enc_t_tuple[idx]);
                        check_val_subset.push(current_vals[idx]);
                    }
                    check_rec_subset.push(enc_t_tuple[pos]);
                    check_val_subset.push(val);

                    if let Some(allowed) = prev_t_cache.get(&check_rec_subset) {
                        if !allowed.contains(&check_val_subset) {
                            ok = false;
                            break;
                        }
                    } else {
                        ok = false;
                        break;
                    }
                }
            }

            if ok {
                current_vals.push(val);
                Self::dfs_find_valid_assignments(
                    pos + 1,
                    current_vals,
                    enc_t_tuple,
                    domains,
                    prev_t_cache,
                    get_expected_freq,
                    target_freq,
                    valid_assignments,
                );
                current_vals.pop();
            }
        }
    }

    /// Helper for processing chunks of combinations in parallel without eager allocations
    fn process_chunk<O, E>(
        chunk: &[Vec<i64>],
        prev_t_cache: &HashMap<Vec<i64>, Vec<Vec<i64>>>,
        get_observed_freq: &O,
        get_expected_freq: &E,
    ) -> Vec<(Vec<i64>, Vec<Vec<i64>>)>
    where
        O: Fn(&[i64]) -> u64 + Sync + Send,
        E: Fn(&[i64]) -> u64 + Sync + Send,
    {
        chunk
            .into_par_iter()
            .filter_map(|enc_t_tuple| {
                let t = enc_t_tuple.len();

                // 1. APPLY T-1 PRUNING FIRST
                for sub_tuple in enc_t_tuple.iter().combinations(t - 1) {
                    let sub_tuple_cloned: Vec<i64> = sub_tuple.into_iter().copied().collect();
                    if !prev_t_cache.contains_key(&sub_tuple_cloned) {
                        return None;
                    }
                }

                // 2. DYNAMIC LOOKUP
                let target_freq = get_observed_freq(enc_t_tuple);
                if target_freq == 0 {
                    return None;
                }

                // 3. EXTRACT DOMAINS FOR DFS
                let mut domains: Vec<Vec<i64>> = Vec::with_capacity(t);
                for i in 0..t {
                    let rec = enc_t_tuple[i];
                    let mut subset = enc_t_tuple.clone();

                    // Drop an element that isn't `rec` to fetch valid assignments for `rec`
                    let remove_idx = if i == 0 { 1 } else { 0 };
                    subset.remove(remove_idx);

                    let mut valid_vals = std::collections::HashSet::new();
                    if let Some(tuples) = prev_t_cache.get(&subset) {
                        let rec_idx_in_subset = subset.iter().position(|&r| r == rec).unwrap();
                        for tup in tuples {
                            valid_vals.insert(tup[rec_idx_in_subset]);
                        }
                    }

                    if valid_vals.is_empty() {
                        return None; // No valid domain for this variable, abort
                    }
                    domains.push(valid_vals.into_iter().collect());
                }

                // 4. PURE RUST DFS
                let mut valid_t_assignments = Vec::new();
                let mut current_vals = Vec::with_capacity(t);

                Self::dfs_find_valid_assignments(
                    0,
                    &mut current_vals,
                    enc_t_tuple,
                    &domains,
                    prev_t_cache,
                    get_expected_freq,
                    target_freq,
                    &mut valid_t_assignments,
                );

                if valid_t_assignments.is_empty() {
                    None
                } else {
                    Some((enc_t_tuple.clone(), valid_t_assignments))
                }
            })
            .collect()
    }

    /// Recursive Case: t > 1. Iterative Pruning via Batched DFS.
    pub fn process_t_greater_than_1<O, E>(
        &mut self,
        t: usize,
        encrypted_records: &[i64],
        get_observed_freq: O,
        get_expected_freq: E,
    ) where
        O: Fn(&[i64]) -> u64 + Sync + Send,
        E: Fn(&[i64]) -> u64 + Sync + Send,
    {
        let prev_t_cache = self
            .candidate_cache
            .get(&(t - 1))
            .expect("Missing previous round cache!")
            .clone();

        let total_combinations = binomial_coefficient(encrypted_records.len() as usize, t);
        let pb = ProgressBar::new(total_combinations);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        let chunk_size = 10_000;
        let mut all_results = Vec::new();
        let mut chunk = Vec::with_capacity(chunk_size);

        // Map Phase: Batch combinations to avoid eager `.collect()` memory bombs
        for enc_t_tuple in encrypted_records.iter().copied().combinations(t) {
            chunk.push(enc_t_tuple);
            if chunk.len() == chunk_size {
                let results = Self::process_chunk(
                    &chunk,
                    &prev_t_cache,
                    &get_observed_freq,
                    &get_expected_freq,
                );
                all_results.extend(results);
                pb.inc(chunk_size as u64);
                chunk.clear();
            }
        }

        // Process final partial chunk
        if !chunk.is_empty() {
            let remainder = chunk.len() as u64;
            let results = Self::process_chunk(
                &chunk,
                &prev_t_cache,
                &get_observed_freq,
                &get_expected_freq,
            );
            all_results.extend(results);
            pb.inc(remainder);
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
