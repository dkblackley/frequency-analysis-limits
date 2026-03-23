use cp_sat::builder::{CpModelBuilder, IntVar};
use cp_sat::proto::constraint_proto::Constraint;
use cp_sat::proto::LinearExpressionProto;
use cp_sat::proto::{
    AllDifferentConstraintProto, ConstraintProto, CpModelProto, CpSolverStatus,
    TableConstraintProto,
};
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::HashMap;

/// Translator: One Formula from All Matching Pairs[cite: 186].
/// Finds matching pairs and translates them into a logical formula constraining
/// value-to-record assignments[cite: 186, 187].
pub struct Translator {
    freq_to_plaintext: HashMap<(i64, u64), Vec<Vec<i64>>>, // the known freq-plaintext analysis.
    cp_model: CpModelBuilder,
    upper: i64,

    // The Cp_sat crate is a little weird, so I have to manually track the variables.
    // This is a map from the Intvar to it's "internal" value and the encrypted ID it's attached to
    var_index_map: HashMap<IntVar, (i32, i64)>,
    // This is a map from an encrypted id to it's IntVar
    enc_id_to_intvar: HashMap<i64, IntVar>,
}

impl Translator {
    pub fn new(freq_to_plaintext: HashMap<(i64, u64), Vec<Vec<i64>>>, largest_val: i64) -> Self {
        Self {
            cp_model: CpModelBuilder::default(),
            freq_to_plaintext,
            upper: largest_val,
            var_index_map: HashMap::new(),
            enc_id_to_intvar: HashMap::new(),
        }
    }

    pub fn set_all_vars(&mut self, encrypted_records: Vec<i64>) {
        let mut count = 0;
        let mut all_vars = Vec::new();

        let pb = ProgressBar::new(encrypted_records.len() as u64);
        pb.set_style(
                ProgressStyle::default_bar()
                    // Added wide_bar, pos (current), len (total), and eta (time remaining)
                    .template(
                        "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})",
                    ).unwrap()
                    .progress_chars("#>-"),
            );

        for encrypted_id in encrypted_records {
            let var = self
                .cp_model
                .new_int_var_with_name([(0, self.upper)], format!("rec_{encrypted_id}"));
            self.enc_id_to_intvar.insert(encrypted_id.clone(), var);
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
        observed_responses: &HashMap<(i64, u64), Vec<Vec<i64>>>,
    ) -> CpModelProto {
        let pb = ProgressBar::new(observed_responses.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                // Added wide_bar, pos (current), len (total), and eta (time remaining)
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})",
                ).unwrap()
                .progress_chars("#>-"),
        );
        {
            // Very silly, but we're just going to side-step everything cp_sat does because it's easier
            let mut model = self.cp_model.proto().clone();

            for (frequency, t_tuples) in observed_responses {
                // Look up the true plaintexts that generate this frequency
                if let Some(allowed_plaintexts) = self.freq_to_plaintext.get(frequency) {
                    // For a 1D mapping (single records):
                    // If freq 42 maps to plaintexts [[2], [31]], then EVERY encrypted tuple
                    // that matched that freq (because we assume we know exactly, there should only be two)
                    // in this list must be constrained to be EITHER 2 or 31 as a true value
                    for encrypted_tuple in t_tuples {
                        let mut vars = Vec::new();

                        for encrypted_rec in encrypted_tuple {
                            if let Some(&var) = self.enc_id_to_intvar.get(encrypted_rec) {
                                vars.push(var);
                            }
                        }

                        // Add the multidimensional constraint to the whole variable grouping
                        self.add_allowed_assignments(&mut model, vars, allowed_plaintexts.clone());
                    }
                } else {
                    panic!(
                        "Warning: Tuple {}, Freq {} not found in precomputed table",
                        frequency.0, frequency.1
                    );
                }
                pb.inc(1);
            }
            pb.finish();
            model
        }
    }

    /// Reconstructs an assignment of values to records.
    ///
    /// # Arguments
    ///
    ///
    /// # Returns
    ///
    pub fn translate(
        mut self,
        freq_record_match: &HashMap<(i64, u64), Vec<Vec<i64>>>,
        encrypted_records: Vec<i64>,
    ) -> (CpModelProto, HashMap<IntVar, (i32, i64)>) {
        self.set_all_vars(encrypted_records);
        let model = self.find_matching_pairs(freq_record_match);
        (model, self.var_index_map)
    }

    /// Re-make the same "AddAllowedAssignments" From python/CPP/wherever. Allowed plaintexts might
    /// be t-tuple of encrypted records, but it's expected that they're all of the same 't' value.
    /// I.e. for 1-tuple we expect a vec of vecs where: The outer vec is some random size
    /// but the inner vec is the same size as Vec<IntVar>.
    fn add_allowed_assignments(
        &self,
        model: &mut CpModelProto,
        vars: Vec<IntVar>,
        allowed_plaintexts: Vec<Vec<i64>>,
    ) {
        let mut table_proto = TableConstraintProto::default();

        // 1. Push all variable indices ONCE
        for var in &vars {
            let var_index = self.var_index_map.get(var).expect("Unknown variable");
            table_proto.vars.push(var_index.0);
        }

        // 2. Flatten all tuples into the values array ONCE.
        // E.g., [[A, B], [C, D]] becomes [A, B, C, D]
        for tuple in allowed_plaintexts {
            for plaintext in tuple {
                table_proto.values.push(plaintext);
            }
        }

        // 3. Push ONE table constraint that contains ALL options (logical OR)
        let mut constraint_proto = ConstraintProto::default();
        constraint_proto.constraint = Some(Constraint::Table(table_proto));

        model.constraints.push(constraint_proto);
    }
}
