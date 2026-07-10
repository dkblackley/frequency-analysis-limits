use crate::LAMA::mini_solver::{CpModel, IntVar};
use indicatif::{ProgressBar, ProgressStyle};
use log::{debug, error, info};
use rayon::prelude::*;
use std::collections::HashMap;

use pumpkin_solver::conflict_resolvers::resolvers::ResolutionResolver;
use pumpkin_solver::core::results::solution_iterator::IteratedSolution;
use pumpkin_solver::core::results::ProblemSolution;
use pumpkin_solver::core::termination::Indefinite;
// Verified pumpkin-solver v0.4.0 imports
use pumpkin_solver::{all_different, equals, table, Solver as PumpkinSolver};

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

    /// Reconstructs an assignment of values to records natively in Rust
    pub fn solve(
        &mut self,
        model: &mut CpModel,
        largest_enc_val: i64,
        get_one: bool,
    ) -> HashMap<i64, Vec<i64>> {
        info!("Model Validation: Executing via pure Rust Pumpkin Solver (v0.4.0)");
        info!(
            "Starting solve with {} variables and {} constraints...",
            model.num_vars,
            model.table_constraints.len()
        );

        let pb = ProgressBar::new_spinner();
        pb.enable_steady_tick(std::time::Duration::from_millis(500));
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.blue} [{elapsed_precise}] Native Pumpkin Solver thinking...")
                .unwrap()
        );

        let largest_val_i32 = largest_enc_val as i32;

        // Parallel Partitioning Across 64 Cores
        let all_solutions = if get_one {
            // Single thread, break on first result
            Self::solve_partition(model, largest_val_i32, None, true)
        } else {
            // Split the search space entirely by iterating the root variable across threads
            let root_domain: Vec<i32> = (0..=largest_val_i32).collect();

            let sols: Vec<Vec<i64>> = root_domain
                .into_par_iter()
                .flat_map(|fixed_val| {
                    Self::solve_partition(model, largest_val_i32, Some(fixed_val), false)
                })
                .collect();
            sols
        };

        pb.finish_with_message(format!("Solver finished in {:?}", pb.elapsed()));

        let total_solutions = all_solutions.len();
        info!("Found {} solutions", total_solutions);
        self.num_sols = total_solutions as i32;

        if total_solutions > 0 {
            self.solution_stat = CpSolverStatus::Optimal;
        } else {
            let numb: i32 = 3; // Infeasible
            error!("Solver failed to find a consistent reconstruction. Status: {numb} - (3) means infeasible");
            self.solution_stat = CpSolverStatus::Infeasible;
            return HashMap::new();
        }

        debug!("Solver status: {:?}", self.solution_stat);

        let mut reconstructed_dbs = HashMap::new();

        // Map the flat outputs exactly back to your internal tracking IDs
        for (_intvar, ids) in &self.var_index_map {
            let mut found_vals = Vec::new();
            let mut orig_appear = false;

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

    /// Internal function to build and solve a localized Pumpkin instance.
    fn solve_partition(
        model: &CpModel,
        largest_val: i32,
        fixed_val_for_var_0: Option<i32>,
        get_one: bool,
    ) -> Vec<Vec<i64>> {
        let mut solver = PumpkinSolver::default();

        // 1. Initialize variables mapping (0..=largest_val)
        let mut solver_vars = Vec::with_capacity(model.num_vars);
        for _ in 0..model.num_vars {
            solver_vars.push(solver.new_bounded_integer(0, largest_val));
        }

        // 2. Lock root variable for parallel space partitioning
        if let Some(fixed_val) = fixed_val_for_var_0 {
            if model.num_vars > 0 {
                let tag = solver.new_constraint_tag();
                solver.add_constraint(
                    equals(vec![solver_vars[0]], fixed_val, tag)
                ).post();
            }
        }

        // 3. AllDifferent global constraint
        if model.all_different {
            let tag = solver.new_constraint_tag();
            solver.add_constraint(all_different(solver_vars.clone(), tag)).post();
        }

        // 4. Allowed Assignments (Table Constraints)
        for tc in &model.table_constraints {
            let vars: Vec<_> = tc.vars.iter().map(|&idx| solver_vars[idx]).collect();

            // Pumpkin requires i32 for constraints
            let tuples: Vec<Vec<i32>> = tc.values.iter()
                .map(|t| t.iter().map(|&v| v as i32).collect())
                .collect();

            let tag = solver.new_constraint_tag();
            solver.add_constraint(table(vars, tuples, tag)).post();
        }

        // 5. Execute search using Pumpkin's native Iterator
        let mut solutions = Vec::new();
        let mut brancher = solver.default_brancher();
        let mut termination = Indefinite;
        let mut resolver = ResolutionResolver::default();

        let mut solution_iterator = solver.get_solution_iterator(
            &mut brancher,
            &mut termination,
            &mut resolver,
        );

        loop {
            match solution_iterator.next_solution() {
                IteratedSolution::Solution(solution, _, _, _) => {
                    let current_assignment: Vec<i64> = solver_vars.iter()
                        .map(|&v| solution.get_integer_value(v) as i64)
                        .collect();

                    solutions.push(current_assignment);

                    // Stop early if we only need one solution
                    if get_one {
                        break;
                    }
                }
                IteratedSolution::Finished | IteratedSolution::Unknown | IteratedSolution::Unsatisfiable => {
                    break;
                }
            }
        }

        solutions
    }
}