use cp_sat::builder::{CpModelBuilder, IntVar};
use cp_sat::proto::constraint_proto::Constraint;
use cp_sat::proto::{ConstraintProto, CpModelProto, TableConstraintProto};
use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
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
    cp_model: CpModelBuilder,
    upper: i64,
    enc_id_to_intvar: Vec<Option<IntVar>>,
    var_index_map: HashMap<IntVar, (i32, i64)>,
}

impl Translator {
    pub fn new(largest_val: i64, max_encrypted_id: usize) -> Self {
        Self {
            cp_model: CpModelBuilder::default(),
            upper: largest_val,
            enc_id_to_intvar: vec![None; max_encrypted_id + 1],
            var_index_map: HashMap::new(),
        }
    }

    pub fn set_all_vars(&mut self, encrypted_records: Vec<i64>) {
        let mut count = 0;
        let mut all_vars = Vec::new();

        let pb = ProgressBar::new(encrypted_records.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        for encrypted_id in encrypted_records {
            let var = self
                .cp_model
                .new_int_var_with_name([(0, self.upper)], format!("rec_{encrypted_id}"));

            // FAST PATH: Array indexing instead of HashMap insertion
            self.enc_id_to_intvar[encrypted_id as usize] = Some(var);

            self.var_index_map.insert(var, (count, encrypted_id));
            all_vars.push(var);
            count += 1;
            pb.inc(1);
        }

        pb.finish();
        self.cp_model.add_all_different(all_vars);
    }

    fn find_matching_pairs(
        &self,
        plaintext_filepath: &str,
        observed_filepath: &str,
    ) -> CpModelProto {
        println!("Co-iterating over sorted plaintext and observed files...");
        let mut model = self.cp_model.proto().clone();

        let pt_file = File::open(plaintext_filepath).expect("Could not open plaintext file");
        let obs_file = File::open(observed_filepath).expect("Could not open observed file");

        let pt_reader = BufReader::with_capacity(8 * 1024 * 1024, pt_file);
        let obs_reader = BufReader::with_capacity(8 * 1024 * 1024, obs_file);

        let mut pt_grouper = GroupReader::new(pt_reader);
        let mut obs_grouper = GroupReader::new(obs_reader);

        let mut pt_current = pt_grouper.next_group();
        let mut obs_current = obs_grouper.next_group();

        let mut matched_frequencies = 0;

        // The Merge Join Logic
        while let (Some((f_pt, pt_tuples)), Some((f_obs, obs_tuples))) = (&pt_current, &obs_current)
        {
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
                        if let Some(Some(var)) = self.enc_id_to_intvar.get(encrypted_rec as usize) {
                            vars.push(*var);
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

        println!(
            "Successfully generated constraints for {} unique frequencies.",
            matched_frequencies
        );
        model
    }

    pub fn translate(
        mut self,
        freq_record_match_filepath: &str,
        observed_freq_record_match_filepath: &str,
        encrypted_records: Vec<i64>,
    ) -> (CpModelProto, HashMap<IntVar, (i32, i64)>) {
        self.set_all_vars(encrypted_records);
        let model = self.find_matching_pairs(
            freq_record_match_filepath,
            observed_freq_record_match_filepath,
        );
        (model, self.var_index_map)
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
            table_proto.vars.push(var_index.0);
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
}
