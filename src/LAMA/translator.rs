use crate::Value;
use crate::LAMA::error::LAMAError;
use crate::LAMA::utility::binomial_coefficient;
use cp_sat::builder::{CpModelBuilder, IntVar};
use cp_sat::proto::constraint_proto::Constraint;
use cp_sat::proto::{ConstraintProto, CpModelProto, TableConstraintProto};
use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use rand::rngs::StdRng;
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
        // Prime the pump by reading the first item
        let peeked = bincode::deserialize_from(&mut reader).ok();
        Self { reader, peeked }
    }

    // Returns (Frequency, Vec<Tuples>) and advances the stream to the next frequency
    fn next_group(&mut self) -> Option<(u64, Vec<Vec<i64>>)> {
        let (current_freq, first_tuple) = self.peeked.take()?;
        let mut group = vec![first_tuple];

        loop {
            match bincode::deserialize_from::<_, (u64, Vec<i64>)>(&mut self.reader) {
                Ok((next_freq, next_tuple)) => {
                    if next_freq == current_freq {
                        group.push(next_tuple);
                    } else {
                        // The frequency changed. Save this item for the next call and yield the group.
                        self.peeked = Some((next_freq, next_tuple));
                        return Some((current_freq, group));
                    }
                }
                Err(_) => {
                    // EOF reached. We have no peeked item for next time.
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
}

impl Translator {
    pub fn new(largest_val: i64, mut encrypted_records: Vec<i64>) -> Self {
        let mut rng = StdRng::seed_from_u64(42);

        //encrypted_records.shuffle(&mut rng);

        let mut cp_model = CpModelBuilder::default();
        let (var_index_map, enc_id_to_intvar) =
            Self::set_all_vars(&mut cp_model, largest_val, encrypted_records);

        Self {
            proto_model: cp_model.proto().clone(),
            upper: largest_val,
            enc_id_to_intvar,
            var_index_map,
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
        mut cp_model: &mut CpModelBuilder,
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
        // Set up indicatif progress bar
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

        // The Merge Join Logic
        while let (Some((f_pt, pt_tuples)), Some((f_obs, obs_tuples))) = (&pt_current, &obs_current)
        {
            counter += pt_tuples.len();
            if counter % batch_size == 0 {
                pb.inc(batch_size as u64);
            }

            if f_pt < f_obs {
                // Plaintext is behind, catch up
                pt_current = pt_grouper.next_group();
            } else if f_obs < f_pt {
                // Encrypted is behind, catch up
                // (This handles the case where an observed freq is completely missing from plaintexts)
                obs_current = obs_grouper.next_group();
            } else {
                // MATCH! Both files are looking at the exact same frequency.
                matched_frequencies += 1;

                // For every single encrypted tuple we observed at this frequency...
                for encrypted_tuple in obs_tuples {
                    let mut vars = Vec::with_capacity(encrypted_tuple.len());

                    for &encrypted_rec in encrypted_tuple {
                        if let Some(var) = self.enc_id_to_intvar.get(&encrypted_rec) {
                            vars.push(*var);
                        } else {
                            // STOP SILENT FAILURES
                            panic!(
                                "FATAL: The ID {} was found in the observed file, but it does not exist in the variable map! Did you pass the wrong universe to set_all_vars?",
                                encrypted_rec
                            );
                        }
                    }

                    // ...constrain it to be ONE OF the plaintext tuples at this frequency
                    self.add_allowed_assignments(&mut model, vars, pt_tuples);
                }

                // Advance both files
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
        &self,
        model: &mut CpModelProto,
        vars: Vec<IntVar>,
        allowed_plaintexts: &Vec<Vec<i64>>,
    ) {
        let mut table_proto = TableConstraintProto::default();

        for var in &vars {
            let var_index = self.var_index_map.get(var).expect("Unknown variable");

            // THE FIX: Use `exprs` instead of the deprecated `vars`
            // We wrap the variable in a basic expression: (1 * var) + 0
            table_proto
                .exprs
                .push(cp_sat::proto::LinearExpressionProto {
                    vars: vec![var_index.0],
                    coeffs: vec![1],
                    offset: 0,
                });
        }

        // The rest of your logic remains exactly the same
        for tuple in allowed_plaintexts {
            for perm in tuple.iter().cloned().permutations(tuple.len()) {
                for plaintext in perm {
                    table_proto.values.push(plaintext);
                }
            }
        }

        let mut constraint_proto = ConstraintProto::default();
        constraint_proto.constraint = Some(Constraint::Table(table_proto));
        model.constraints.push(constraint_proto);
    }
}
