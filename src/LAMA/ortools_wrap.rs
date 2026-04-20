use serde::Serialize;
use std::env;
use std::fs::File;
use std::io::Write;
use std::process::Command;

/// A mock for the original cp_sat IntVar
#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
pub struct IntVar(pub usize);

/// Our drop-in replacement for CpModelProto.
/// It holds the global state of the model as it builds up across t-rounds.
#[derive(Clone, Debug, Default)]
pub struct PythonCpModel {
    pub num_vars: usize,
    pub all_different: bool,
    pub table_constraints: Vec<TableConstraint>,
}

#[derive(Clone, Debug)]
pub struct TableConstraint {
    pub vars: Vec<usize>,      // The indices of the variables
    pub values: Vec<Vec<i64>>, // The allowed assignments
}

#[derive(Serialize)]
pub struct ConstraintMeta {
    pub var_ids: Vec<usize>,
    pub start_idx: usize,
    pub length: usize,
    pub tuple_size: usize,
}

impl PythonCpModel {
    pub fn new() -> Self {
        Self::default()
    }

    /// Mirrors the CP-SAT solve_with_parameters logic
    pub fn solve(&self, largest_val: i64, get_one: bool) -> Option<Vec<Vec<i64>>> {
        let mut binary_data: Vec<i64> = Vec::new();
        let mut metadata = Vec::new();

        // Pack the mixed t=1, t=2, t=n constraints into the binary format
        for tc in &self.table_constraints {
            let start_idx = binary_data.len();
            for tuple in &tc.values {
                binary_data.extend_from_slice(tuple);
            }
            metadata.push(ConstraintMeta {
                var_ids: tc.vars.clone(),
                start_idx,
                length: tc.values.len() * tc.vars.len(),
                tuple_size: tc.vars.len(),
            });
        }

        let mut bin_file = File::create("allowed.bin").expect("Failed to create bin file");
        let byte_slice = unsafe {
            std::slice::from_raw_parts(
                binary_data.as_ptr() as *const u8,
                binary_data.len() * std::mem::size_of::<i64>(),
            )
        };
        bin_file
            .write_all(byte_slice)
            .expect("Failed to write binary data");

        let meta_file = File::create("meta.json").expect("Failed to create meta file");
        serde_json::to_writer(meta_file, &metadata).expect("Failed to write metadata");

        let python_exe = env::var("PYTHON_EXEC").unwrap_or_else(|_| "python3".to_string());

        let output = Command::new(python_exe)
            .arg("src/LAMA/solver.py")
            .arg(self.num_vars.to_string())
            .arg(largest_val.to_string())
            .arg(get_one.to_string())
            .output()
            .expect("Failed to execute python solver");

        if !output.status.success() {
            // Print exactly what Python is complaining about
            eprintln!("Python failed with status: {}", output.status);
            eprintln!("Python stderr: {}", String::from_utf8_lossy(&output.stderr));
            eprintln!("Python stdout: {}", String::from_utf8_lossy(&output.stdout));
            return None;
        }

        let sol_str =
            std::fs::read_to_string("solutions.json").unwrap_or_else(|_| "[]".to_string());
        let all_solutions: Vec<Vec<i64>> = serde_json::from_str(&sol_str).unwrap_or_default();

        if all_solutions.is_empty() {
            None
        } else {
            Some(all_solutions)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Brings PythonCpModel and TableConstraint into scope

    #[test]
    fn test_tiny_bruteforce() {
        let mut model = PythonCpModel::new();
        model.num_vars = 3;
        model.all_different = true; // Requires [var_0, var_1, var_2] to be distinct

        // One constraint for t=3 that limits the 3 variables to exactly two valid permutations.
        // It's a tiny search space, so the solver will easily brute-force it instantly.
        model.table_constraints.push(TableConstraint {
            vars: vec![0, 1, 2],
            values: vec![vec![0, 1, 2], vec![2, 1, 0]],
        });

        // Run Python: largest_val = 2, get_one = false
        let result = model
            .solve(2, false)
            .expect("Python solver crashed or failed to run");

        println!("Brute-force solutions: {:?}", result);
        assert_eq!(
            result.len(),
            2,
            "Should find exactly two valid combinations"
        );
        assert!(result.contains(&vec![0, 1, 2]));
        assert!(result.contains(&vec![2, 1, 0]));
    }

    #[test]
    fn test_obvious_t1_and_t2_tuples() {
        let mut model = PythonCpModel::new();
        // 4 variables, domains spanning 0 to 10.
        model.num_vars = 4;
        model.all_different = true; // [5, 6, 9, 2] are all unique values

        // t=1 constraints: The "obvious" single choices
        model.table_constraints.push(TableConstraint {
            vars: vec![0], // var_0 MUST be 5
            values: vec![vec![5]],
        });
        model.table_constraints.push(TableConstraint {
            vars: vec![1], // var_1 MUST be 6
            values: vec![vec![6]],
        });

        // t=2 constraints: The "obvious" links
        model.table_constraints.push(TableConstraint {
            vars: vec![0, 2], // var_0 and var_2
            values: vec![
                vec![5, 9], // Valid because var_0 is 5
                vec![4, 8], // Invalid: var_0 cannot be 4 based on t=1
            ],
        });

        model.table_constraints.push(TableConstraint {
            vars: vec![1, 3], // var_1 and var_3
            values: vec![
                vec![6, 2], // Valid because var_1 is 6
                vec![7, 3], // Invalid: var_1 cannot be 7 based on t=1
            ],
        });

        // Run Python: largest_val = 10, get_one = false
        let result = model
            .solve(10, false)
            .expect("Python solver crashed or failed to run");

        println!("Pruned solutions: {:?}", result);
        assert_eq!(
            result.len(),
            1,
            "Should prune down to exactly one valid combination"
        );
        assert_eq!(result[0], vec![5, 6, 9, 2]);
    }
}
