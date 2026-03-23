use cp_sat::builder::{CpModelBuilder, IntVar};
use cp_sat::ffi;
use cp_sat::proto::CpModelProto;
use cp_sat::proto::CpSolverStatus;
use indicatif::{ProgressBar, ProgressStyle};
use log::error;
use std::collections::HashMap;

/// Solver Reconstruction as Constraint-Satisfaction.
/// Finds an assignment of values to identifiers that satisfies the formula C output by the Translator.
#[derive(Debug)]
pub struct Solver {
    var_index_map: HashMap<IntVar, (i32, i64)>,
}

impl Solver {
    pub fn new(var_index_map: HashMap<IntVar, (i32, i64)>) -> Self {
        Self { var_index_map }
    }

    /// Reconstructs an assignment of values to records.
    ///
    /// # Arguments
    ///
    ///
    /// # Returns
    ///
    pub fn solve(&self, model: &mut CpModelProto) -> HashMap<i64, i64> {
        let response = ffi::solve(&model);
        let status = response.status();
        let mut reconstructed_db = HashMap::new();

        if status == CpSolverStatus::Optimal || status == CpSolverStatus::Feasible {
            // Read the final chosen values out of the solver response
            for (intvar, ids) in &self.var_index_map {
                let true_value = intvar.solution_value(&response);
                // ids is the original 'encrypted' db. For convenience, the key is a true encoded
                // val of the record.
                reconstructed_db.insert(ids.1, true_value);
            }
        } else {
            let numb: i32 = status.into();
            error!("Solver failed to find a consistent reconstruction. Status: {numb}");
        }

        reconstructed_db
    }
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

        let mut engine = Solver::new(freq_to_plaintext, 100);

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

    use crate::LAMA::solver::Solver;
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
