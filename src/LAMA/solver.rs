use crate::LAMA::ortools_wrap::{IntVar, PythonCpModel};
use indicatif::{ProgressBar, ProgressStyle};
use log::{debug, error, info};
use std::collections::HashMap;

/// A local recreation of the CP-SAT status enum.
/// This prevents your codebase from breaking since you rely on these specific variants.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum CpSolverStatus {
    Unknown = 0,
    ModelInvalid = 1,
    Feasible = 2,
    Infeasible = 3,
    Optimal = 4,
}

impl From<i32> for CpSolverStatus {
    fn from(val: i32) -> Self {
        match val {
            1 => CpSolverStatus::ModelInvalid,
            2 => CpSolverStatus::Feasible,
            3 => CpSolverStatus::Infeasible,
            4 => CpSolverStatus::Optimal,
            _ => CpSolverStatus::Unknown,
        }
    }
}

/// Solver Reconstruction as Constraint-Satisfaction.
/// Finds an assignment of values to identifiers that satisfies the formula C output by the Translator.
#[derive(Debug)]
pub struct Solver {
    var_index_map: HashMap<IntVar, (i32, i64)>,
    pub solution_stat: CpSolverStatus,
    pub num_sols: i32,
}

impl Solver {
    pub fn new(var_index_map: HashMap<IntVar, (i32, i64)>) -> Self {
        Self {
            var_index_map,
            solution_stat: CpSolverStatus::Unknown,
            num_sols: 0,
        }
    }

    /// Reconstructs an assignment of values to records via the external Python solver.
    pub fn solve(
        &mut self,
        model: &mut PythonCpModel,
        largest_enc_val: i64,
        get_one: bool,
    ) -> HashMap<i64, Vec<i64>> {
        info!("Model Validation: Deferred to Python CP-SAT");
        info!(
            "Starting solve with {} variables and {} constraints...",
            model.num_vars,
            model.table_constraints.len()
        );

        let pb = ProgressBar::new_spinner();
        pb.enable_steady_tick(std::time::Duration::from_millis(500));
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.blue} [{elapsed_precise}] Solver thinking (no strict ETA for SAT problems)...")
                .unwrap()
        );

        // Delegate execution and binary formatting to our wrapper
        let response = model.solve(largest_enc_val, get_one);
        pb.finish_with_message(format!("Solver finished in {:?}", pb.elapsed()));

        if response.is_none() {
            let numb: i32 = 3; // Infeasible
            error!("Solver failed to find a consistent reconstruction. Status: {numb} - (3) means infeasible");
            self.solution_stat = CpSolverStatus::Infeasible;
            return HashMap::new();
        }

        let all_solutions = response.unwrap();
        let total_solutions = all_solutions.len();

        info!("Found {} solutions", total_solutions);
        self.num_sols = total_solutions as i32;

        if total_solutions > 0 {
            self.solution_stat = CpSolverStatus::Optimal;
        } else {
            self.solution_stat = CpSolverStatus::Infeasible;
        }

        debug!("Solver status: {:?}", self.solution_stat);

        let mut reconstructed_dbs = HashMap::new();

        // Map the flat Python outputs exactly back to your internal tracking IDs
        for (intvar, ids) in &self.var_index_map {
            let mut found_vals = Vec::new();
            let mut orig_appear = false;

            // Extract the 'primary' solution from the first slot
            let orig_sol = all_solutions[0][ids.0 as usize];

            if get_one {
                orig_appear = true;
                found_vals.push(orig_sol);
            }

            for i in 0..total_solutions {
                let extra_sol = all_solutions[i][ids.0 as usize];
                if orig_sol == extra_sol {
                    orig_appear = true;
                }
                found_vals.push(extra_sol);
            }

            if !orig_appear {
                panic!("Original response was not pushed to all solutions!")
            }

            // ids.1 is the true encoded value of the record
            reconstructed_dbs.insert(ids.1, found_vals);
        }

        reconstructed_dbs
    }
}
