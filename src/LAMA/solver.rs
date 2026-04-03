use cp_sat::builder::{CpModelBuilder, IntVar};
use cp_sat::ffi;
use cp_sat::proto::CpSolverStatus;
use cp_sat::proto::{CpModelProto, CpSolverSolution, SatParameters};
use indicatif::{ProgressBar, ProgressStyle};
use itertools::all;
use log::{error, info};
use std::cmp::max;
use std::collections::HashMap;

/// Solver Reconstruction as Constraint-Satisfaction.
/// Finds an assignment of values to identifiers that satisfies the formula C output by the Translator.
#[derive(Debug)]
pub struct Solver {
    var_index_map: HashMap<IntVar, (i32, i64)>,
    pub solution_stat: CpSolverStatus,
}

impl Solver {
    pub fn new(var_index_map: HashMap<IntVar, (i32, i64)>) -> Self {
        Self {
            var_index_map,
            solution_stat: CpSolverStatus::Unknown,
        }
    }

    /// Reconstructs an assignment of values to records.
    ///
    /// # Arguments
    ///
    ///
    /// # Returns
    ///
    pub fn solve(&mut self, model: &mut CpModelProto, get_one: bool) -> HashMap<i64, Vec<i64>> {
        let mut params = SatParameters::default();

        if get_one {
            params.num_workers = Some(128);
        } else {
            params.enumerate_all_solutions = Some(true);
            params.fill_additional_solutions_in_response = Some(true);
            params.solution_pool_size = Some(self.var_index_map.len() as i32); // Store all solutions found
        }

        //

        // Validate the model structurally before solving
        info!("Model Validation: {}", ffi::validate_cp_model(&model));
        info!(
            "Starting solve with {} variables and {} constraints...",
            model.variables.len(),
            model.constraints.len()
        );

        let pb = ProgressBar::new_spinner();
        pb.enable_steady_tick(std::time::Duration::from_millis(500));
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.blue} [{elapsed_precise}] Solver thinking (no strict ETA for SAT problems)...")
                .unwrap()
        );
        let response = ffi::solve_with_parameters(&model, &params);
        pb.finish_with_message(format!("Solver finished in {:?}", pb.elapsed()));
        let status = response.status();
        let mut reconstructed_dbs = HashMap::new();

        let all_solutions: &Vec<CpSolverSolution> = &response.additional_solutions;
        let total_solutions = all_solutions.len();
        info!("Found {} solutions!", max(total_solutions, 1));

        self.solution_stat = status.into();

        if status == CpSolverStatus::Optimal || status == CpSolverStatus::Feasible {
            // Read the final chosen values out of the solver response
            for (intvar, ids) in &self.var_index_map {
                let mut found_vals = Vec::new();
                let orig_sol = intvar.solution_value(&response);
                let mut orig_appear = false;

                if get_one {
                    orig_appear = true;
                    found_vals.push(orig_sol);
                }

                for i in 0..total_solutions {
                    let extra_sol =
                        all_solutions[i].values[self.var_index_map.get(intvar).unwrap().0 as usize];
                    if orig_sol == extra_sol {
                        orig_appear = true;
                    }
                    found_vals.push(extra_sol);
                }

                if !orig_appear {
                    panic!("Original response was not pushed to all solutions!")
                }
                // ids is the original 'encrypted' db. For convenience, the key is a true encoded
                // val of the record.
                reconstructed_dbs.insert(ids.1, found_vals);
            }
        } else {
            let numb: i32 = status.into();
            error!("Solver failed to find a consistent reconstruction. Status: {numb}");
        }

        reconstructed_dbs
    }
}
