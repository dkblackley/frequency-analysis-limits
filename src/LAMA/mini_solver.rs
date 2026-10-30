use crate::dataloader::unflatten_nd;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use itertools::Itertools;
use log::{debug, info, warn};
use pumpkin_solver::conflict_resolvers::resolvers::ResolutionResolver;
use pumpkin_solver::core::results::solution_iterator::IteratedSolution;
use pumpkin_solver::core::results::ProblemSolution;
use pumpkin_solver::core::termination::Indefinite;
use pumpkin_solver::{all_different, equals, table, Solver as PumpkinSolver};
use rayon::prelude::*;
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
pub struct CpModel {
    pub num_vars: usize,
    pub all_different: bool,
    pub table_constraints: Vec<TableConstraint>,
    run_id: String,
    pub limit: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct TableConstraint {
    pub vars: Vec<usize>,
    pub values: Vec<Vec<i64>>,
    pub costs: Vec<i64>,
}

#[derive(Serialize)]
pub struct ConstraintMeta {
    pub var_ids: Vec<usize>,
    pub start_idx: usize,
    pub length: usize,
    pub tuple_size: usize,
}

impl CpModel {
    pub fn new(limit: Option<usize>) -> Self {
        // Generate a unique ID using the system clock (no extra crates needed)
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
            .to_string();

        Self {
            run_id: id,
            limit,
            ..Default::default()
        }
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
    }

    /// Mirrors the CP-SAT solve_with_parameters logic
    /// Replaces the Python Subprocess with a pure Rust native parallel solver
    pub fn solve(&self, largest_val: i64, get_one: bool) -> Option<Vec<Vec<i64>>> {
        let largest_val_i32 = largest_val as i32;

        let all_solutions = if get_one {
            // Single thread, break on first result
            Self::solve_partition(self, largest_val_i32, None, true)
        } else {
            // Split the search space across 64 cores by partitioning the root variable
            let root_domain: Vec<i32> = (0..=largest_val_i32).collect();

            let pb = ProgressBar::new(root_domain.len() as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {bar:40.cyan/blue} {percent}% {msg} (ETA: {eta_precise})")
                    .unwrap()
            );

            let sols: Vec<Vec<i64>> = root_domain
                .into_par_iter()
                .progress_with(pb) // Attach the progress bar here
                .flat_map(|fixed_val| {
                    Self::solve_partition(self, largest_val_i32, Some(fixed_val), false)
                })
                .collect();
            sols
        };

        if all_solutions.is_empty() {
            None
        } else {
            Some(all_solutions)
        }
    }

    /// Internal function to build and solve a localized Pumpkin instance in memory.
    fn solve_partition(
        model: &CpModel,
        largest_val: i32,
        fixed_val_for_var_0: Option<i32>,
        get_one: bool,
    ) -> Vec<Vec<i64>> {
        let mut solver = PumpkinSolver::default();

        // 1. Initialize variables (0..=largest_val)
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
        if model.all_different && model.num_vars > 0 {
            let tag = solver.new_constraint_tag();
            solver.add_constraint(all_different(solver_vars.clone(), tag)).post();
        }

        // 4. Allowed Assignments (Table Constraints)
        for tc in &model.table_constraints {
            let vars: Vec<_> = tc.vars.iter().map(|&idx| solver_vars[idx]).collect();

            // Cast values to i32 for Pumpkin, discarding the costs array entirely
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
                    // Reconstruct the array of assigned values
                    let current_assignment: Vec<i64> = solver_vars.iter()
                        .map(|&v| solution.get_integer_value(v) as i64)
                        .collect();

                    solutions.push(current_assignment);

                    if get_one || solutions.len() >= model.limit.unwrap_or(usize::MAX) {
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

