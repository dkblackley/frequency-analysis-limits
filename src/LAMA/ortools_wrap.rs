use crate::dataloader::unflatten_nd;
use itertools::Itertools;
use log::{debug, info, warn};
use serde::Serialize;
use std::collections::HashMap;
use std::env;
use std::env::var;
use std::fs::File;
use std::io::Write;
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::thread;

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
    pub vars: Vec<usize>,
    pub values: Vec<Vec<i64>>,
    pub costs: Vec<i64>, // NEW FIELD
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

    fn write_model_files(&self) {
        let mut binary_data: Vec<i64> = Vec::new();
        let mut metadata = Vec::new();

        for tc in &self.table_constraints {
            let start_idx = binary_data.len();
            for (idx, tuple) in tc.values.iter().enumerate() {
                binary_data.extend_from_slice(tuple);
                binary_data.push(tc.costs[idx]); // Add cost immediately after the tuple plaintexts
            }
            metadata.push(ConstraintMeta {
                var_ids: tc.vars.clone(),
                start_idx,
                length: tc.values.len() * (tc.vars.len() + 1),
                tuple_size: tc.vars.len() + 1,
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
    }

    pub fn validate(&self, var_index_map: &HashMap<IntVar, (i32, i64)>) {
        let mut found_tracker = HashMap::new();
        let mut found_all = true;
        let mut optimal_cost = 0;

        // Initialize the tracker
        for tc in &self.table_constraints {
            for vars in tc.vars.iter() {
                let orig_val = var_index_map.get(&(IntVar(vars.clone()))).unwrap().1;
                found_tracker.insert(orig_val, false);
            }
        }

        // Validate cohesive rows and calculate true cost
        for tc in &self.table_constraints {
            let mut target_tuple = Vec::with_capacity(tc.vars.len());
            for var_idx in &tc.vars {
                let orig_val = var_index_map.get(&(IntVar(*var_idx))).unwrap().1;
                target_tuple.push(orig_val);
            }

            let mut found_matching_row = false;

            for (row_idx, assignment) in tc.values.iter().enumerate() {
                if assignment == &target_tuple {
                    found_matching_row = true;
                    optimal_cost += tc.costs[row_idx];
                    for val in assignment {
                        found_tracker.insert(*val, true);
                    }
                    break;
                }
            }

            if !found_matching_row {
                warn!(
                    "Table constraint for vars {:?} does not contain the true assignment {:?}",
                    tc.vars, target_tuple
                );
            }
        }

        for (k, v) in &found_tracker {
            if !v {
                found_all = false;
                warn!("There is no valid assignment for {k}!! Solver is VERY LIKELY to crash.");
            }
        }

        if found_all {
            debug!("Every variable has at least 1 valid constraint! (They may still contradict...)")
        }
        debug!("Value for optimal cost was: {}!!", optimal_cost);

        // NEW: Extract and write the true assignment straight to disk
        let mut true_assignment = vec![0; self.num_vars as usize];
        for (_var, (idx, val)) in var_index_map.iter() {
            true_assignment[*idx as usize] = *val;
        }

        let true_sol_file =
            File::create("true_solution.json").expect("Failed to create true solution file");
        serde_json::to_writer(true_sol_file, &true_assignment)
            .expect("Failed to write true solution data");
    }

    /// Mirrors the CP-SAT solve_with_parameters logic
    pub fn solve(&self, largest_val: i64, get_one: bool) -> Option<Vec<Vec<i64>>> {
        let mut binary_data: Vec<i64> = Vec::new();
        let mut metadata = Vec::new();

        for tc in &self.table_constraints {
            let start_idx = binary_data.len();
            for (idx, tuple) in tc.values.iter().enumerate() {
                binary_data.extend_from_slice(tuple);
                binary_data.push(tc.costs[idx]); // Add cost immediately after the tuple plaintexts
            }
            metadata.push(ConstraintMeta {
                var_ids: tc.vars.clone(),
                start_idx,
                length: tc.values.len() * (tc.vars.len() + 1), // Account for cost element
                tuple_size: tc.vars.len() + 1,                 // Account for cost element
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

        drop(bin_file);
        //drop(meta_file);

        let python_exe = env::var("PYTHON_EXEC").unwrap_or_else(|_| "python3".to_string());

        info!("Using python: {:?}", python_exe);

        let mut child = Command::new(python_exe)
            // 1. Force Python to flush output immediately
            .env("PYTHONUNBUFFERED", "1")
            .arg("src/LAMA/solver.py")
            .arg(self.num_vars.to_string())
            .arg(largest_val.to_string())
            //.arg(get_one.to_string())
            .arg(get_one.to_string())
            .arg(true.to_string())
            // 2. Pipe the streams instead of inheriting them
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            // 3. Spawn the process instead of waiting for output()
            .spawn()
            .expect("Failed to execute python solver");

        // Extract the handles so we can read from them
        let stdout = child.stdout.take().expect("Failed to grab stdout");
        let stderr = child.stderr.take().expect("Failed to grab stderr");

        // Spawn a thread to read stdout in real-time
        let stdout_thread = thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                if let Ok(line) = line {
                    debug!("Python stdout: {}", line);
                }
            }
        });

        // Spawn a thread to read stderr in real-time
        let stderr_thread = thread::spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines() {
                if let Ok(line) = line {
                    debug!("Python stderr: {}", line);
                }
            }
        });

        // Wait for the python process to completely finish
        let status = child.wait().expect("Failed to wait on child");

        // Ensure all output has been processed before moving on
        stdout_thread.join().expect("Stdout thread panicked");
        stderr_thread.join().expect("Stderr thread panicked");

        debug!("Python solver finished with: {}", status);

        // if !output.status.success() {
        //     debug!("Python failed with status: {}", output.status);
        //     std::fs::remove_file("allowed.bin").ok();
        //     std::fs::remove_file("meta.json").ok();
        //     std::fs::remove_file("solutions.json").ok();
        //     return None;
        // }

        let sol_str =
            std::fs::read_to_string("solutions.json").unwrap_or_else(|_| "[]".to_string());
        let all_solutions: Vec<Vec<i64>> = serde_json::from_str(&sol_str).unwrap_or_default();

        std::fs::remove_file("allowed.bin").ok();
        std::fs::remove_file("meta.json").ok();
        std::fs::remove_file("solutions.json").ok();

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
            costs: vec![0, 0],
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
            costs: vec![0],
        });
        model.table_constraints.push(TableConstraint {
            vars: vec![1], // var_1 MUST be 6
            values: vec![vec![6]],
            costs: vec![0],
        });

        // t=2 constraints: The "obvious" links
        model.table_constraints.push(TableConstraint {
            vars: vec![0, 2], // var_0 and var_2
            values: vec![
                vec![5, 9], // Valid because var_0 is 5
                vec![4, 8], // Invalid: var_0 cannot be 4 based on t=1
            ],
            costs: vec![0, 0],
        });

        model.table_constraints.push(TableConstraint {
            vars: vec![1, 3], // var_1 and var_3
            values: vec![
                vec![6, 2], // Valid because var_1 is 6
                vec![7, 3], // Invalid: var_1 cannot be 7 based on t=1
            ],
            costs: vec![0, 0],
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
