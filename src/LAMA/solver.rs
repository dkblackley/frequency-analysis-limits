use cp_sat::builder::{CpModelBuilder, IntVar};
use cp_sat::ffi;
use cp_sat::proto::constraint_proto::Constraint;
use cp_sat::proto::LinearExpressionProto;
use cp_sat::proto::{
    AllDifferentConstraintProto, ConstraintProto, CpModelProto, CpSolverStatus,
    TableConstraintProto,
};
use indicatif::{ProgressBar, ProgressStyle};
use log::error;
use std::collections::HashMap;

/// Solver Reconstruction as Constraint-Satisfaction.
/// Finds an assignment of values to identifiers that satisfies the formula C output by the Translator.
#[derive(Debug)]
pub struct SolverEngine {
    cp_model: CpModelBuilder,
    freq_to_plaintext: HashMap<(i64, u64), Vec<Vec<i64>>>, // (t, frequency) to t-tuples (if t is 1 then vec len 1 vec of unique ids/encodings that match that encoding)
    upper: i64,
    var_index_map: HashMap<IntVar, (i32, i64)>,
}

impl SolverEngine {
    pub fn new(freq_to_plaintext: HashMap<(i64, u64), Vec<Vec<i64>>>, largest_val: i64) -> Self {
        let cp_model = CpModelBuilder::default();
        let var_index_map = HashMap::new();
        Self {
            cp_model,
            freq_to_plaintext,
            upper: largest_val,
            var_index_map,
        }
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

    /// Reconstructs an assignment of values to records.
    ///
    /// # Arguments
    ///
    ///
    /// # Returns
    ///
    pub fn reconstruct(
        &mut self,
        encrypted_records: Vec<i64>, // Vec of all unique records. Remember each 'value' is a single item that represents a d-dim point
        freq_record_match: &HashMap<(i64, u64), Vec<Vec<i64>>>, // mapping of (t, observed frequency) to vec of t-tuples (Is only a single item just now)
    ) -> HashMap<i64, i64> {
        let mut var_map = HashMap::new();
        let mut reconstructed_db = HashMap::new();

        let mut count = 0;
        let mut all_var_indices = Vec::new();
        let mut all_vars = Vec::new();

        let pb = ProgressBar::new(
            (encrypted_records.len() + encrypted_records.len() + freq_record_match.len()) as u64,
        );
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
            var_map.insert(encrypted_id.clone(), var);
            self.var_index_map.insert(var, (count, encrypted_id));
            all_var_indices.push(count);
            all_vars.push(var);
            count += 1;
            pb.inc(1);
        }

        self.cp_model.add_all_different(all_vars);

        // Very silly, but we're just going to side-step everything cp_sat does because it's easier
        let mut model = self.cp_model.proto().clone();

        // // ensure different function is deprecated (hooray) So I have to do this stupid linear
        // constraint. For every pair of variables, strictly enforce: Var_A - Var_B != 0
        // // This guarantees a unique assignment to every record, or it fails.
        // for i in 0..all_var_indices.len() {
        //     for j in (i + 1)..all_var_indices.len() {
        //         let mut lin = cp_sat::proto::LinearConstraintProto::default();
        //         // Select the two variables
        //         lin.vars.extend([all_var_indices[i], all_var_indices[j]]);
        //         // Multiply Var_A by 1, Var_B by -1
        //         lin.coeffs.extend([1, -1]);
        //         // Domain: [-upper, -1] OR [1, upper]. (Excludes 0!)
        //         lin.domain.extend([-self.upper, -1, 1, self.upper]);
        //
        //         let mut constraint = ConstraintProto::default();
        //         constraint.constraint = Some(Constraint::Linear(lin));
        //         model.constraints.push(constraint);
        //     }
        //     pb.inc(1);
        // }

        for (frequency, t_tuples) in freq_record_match {
            // Look up the true plaintexts that generate this frequency
            if let Some(allowed_plaintexts) = self.freq_to_plaintext.get(frequency) {
                // For a 1D mapping (single records):
                // If freq 42 maps to plaintexts [[2], [31]], then EVERY encrypted tuple
                // that matched that freq (because we assume we know exactly, there should only be two)
                // in this list must be constrained to be EITHER 2 or 31 as a true value
                for encrypted_tuple in t_tuples {
                    let mut vars = Vec::new();

                    for encrypted_rec in encrypted_tuple {
                        if let Some(&var) = var_map.get(encrypted_rec) {
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

        let response = ffi::solve(&model);
        let status = response.status();

        // ---------------------------------------------------------
        // 4. EXTRACT RESULTS
        // ---------------------------------------------------------
        if status == CpSolverStatus::Optimal || status == CpSolverStatus::Feasible {
            // Read the final chosen values out of the solver response
            for (intvar, ids) in &self.var_index_map {
                let true_value = intvar.solution_value(&response);
                reconstructed_db.insert(ids.1, true_value);
            }
        } else {
            let numb: i32 = status.into();
            error!("Solver failed to find a consistent reconstruction. Status: {numb}");
        }

        reconstructed_db
    }
    // TODO: more than 1 dimension
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_solver_engine() {
        // PRECOMPUTED UNIVERSE
        let mut freq_to_plaintext = HashMap::new();
        // t=1: Frequency 10 means the plaintext is either 2 or 5
        freq_to_plaintext.insert((1, 10), vec![vec![2], vec![5]]);
        // t=2: Frequency 50 means the joint plaintexts are either [2, 8] or [5, 9]
        freq_to_plaintext.insert((2, 50), vec![vec![2, 8], vec![5, 9]]);

        let mut engine = SolverEngine::new(freq_to_plaintext, 100);

        // ENCRYPTED DATABASE
        let encrypted_records = vec![222, 888, 555]; // Three observed encrypted records

        // OBSERVED FREQUENCIES
        let mut freq_record_match = HashMap::new();
        // Both 999 and 888 independently have a frequency of 10
        freq_record_match.insert((1, 10), vec![vec![222], vec![555]]);
        // When queried together, 999 and 888 have a joint frequency of 50
        freq_record_match.insert((2, 50), vec![vec![222, 888]]);

        // RECONSTRUCT
        let result = engine.reconstruct(encrypted_records, &freq_record_match);

        // Technically we see here the 'parallel freq' issue/mirror DB we always discuss.
        // it is perfectly valid for 222 to be 5 and 555 to be 2... (and vice versa)
        assert_eq!(result.get(&222), Some(&2)); // 999 is forced to 2
        assert_eq!(result.get(&888), Some(&8)); // 888 is forced to 8
        assert_eq!(result.get(&555), Some(&5));
        println!("Database successfully reconstructed: {:?}", result);
    }

    use crate::LAMA::solver::SolverEngine;
    use cp_sat::builder::CpModelBuilder;
    use cp_sat::ffi;
    use cp_sat::proto::constraint_proto::Constraint;
    use cp_sat::proto::{ConstraintProto, CpSolverStatus, TableConstraintProto};
    use std::collections::HashMap;
    // Import the FFI module

    #[test]
    fn test_clone_ffi_and_table_constraints() {
        let mut model = CpModelBuilder::default();

        // 1. Create two variables with a wide domain [0, 100]
        let x = model.new_int_var_with_name([(0, 100)], "x");
        let y = model.new_int_var_with_name([(0, 100)], "y");

        // Since we created them sequentially, x is index 0 and y is index 1.
        let x_index = 0;
        let y_index = 1;

        // 2. THE ESCAPE HATCH: Clone the underlying protobuf so we own a mutable copy.
        // We strictly do this AFTER creating all variables so the clone contains them.
        let mut raw_model = model.proto().clone();

        // 3. Build the manual TableConstraintProto using our shadow indices
        let mut table_proto = TableConstraintProto::default();

        // Bind index 0 (x) and index 1 (y) to the table
        table_proto.vars.push(x_index);
        table_proto.vars.push(y_index);

        // Provide exactly ONE allowed assignment: [x = 42, y = 99]
        // The values must be pushed flatly in the order of the variables.
        table_proto.values.push(42);
        table_proto.values.push(99);

        // 4. Wrap the TableConstraint and inject it directly into our mutable clone
        let mut constraint_proto = ConstraintProto::default();
        constraint_proto.constraint = Some(Constraint::Table(table_proto));
        raw_model.constraints.push(constraint_proto);

        // 5. SOLVE THE RAW MODEL: Bypass the builder and use the crate's FFI function.
        let response = ffi::solve(&raw_model);

        // 6. Verify the solver respected our manual table constraint.
        assert_eq!(response.status(), CpSolverStatus::Optimal);

        // 7. EXTRACT RESULTS
        // Even though we bypassed `model.solve()` and solved `raw_model` instead,
        // `x` and `y` are just opaque index wrappers. They will successfully pull
        // the correct values from the FFI response array!
        let resolved_x = x.solution_value(&response);
        let resolved_y = y.solution_value(&response);

        assert_eq!(resolved_x, 42, "x should be forcefully constrained to 42");
        assert_eq!(resolved_y, 99, "y should be forcefully constrained to 99");

        println!("Success! x = {}, y = {}", resolved_x, resolved_y);
    }
}
