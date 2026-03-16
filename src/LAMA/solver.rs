use std::collections::HashMap;
use z3::{
    ast::{Bool, Int}, SatResult,
    Solver,
};

/// Solver Reconstruction as Constraint-Satisfaction.
/// Finds an assignment of values to identifiers that satisfies the formula C output by the Translator.
pub struct SolverEngine {
    z3_solver: Solver,
}

impl SolverEngine {
    pub fn new() -> Self {
        Self {
            z3_solver: Solver::new(),
        }
    }

    /// Reconstructs an assignment of values to records.
    ///
    /// # Arguments
    ///
    /// * `records` - A slice containing unique ids/record identifiers
    /// * `t1_matches` - A map of frequency values to record identifiers - i.e. every record that
    /// appears with frequency ''.
    ///
    /// # Returns
    ///
    /// An `Option` containing a vector of (record, value) pairs if a valid assignment exists.
    pub fn reconstruct(
        &self,
        records: &[u32],
        t1_matches: &HashMap<u32, Vec<i64>>,
    ) -> Option<Vec<(u32, i64)>> {
        let mut variables = HashMap::new();

        // Create Z3 Int variables for each record
        for &rec in records {
            let var_name = format!("rec_{}", rec);
            variables.insert(rec, Int::new_const(var_name.as_str()));
        }

        // Apply domain constraints based on matching pairs (t1_matches)
        for &rec in records {
            if let Some(domain) = t1_matches.get(&rec) {
                let var = variables.get(&rec).unwrap();

                // Construct an OR constraint for all candidate values
                let domain_constraints: Vec<Bool> = domain
                    .iter()
                    .map(|&val| var.eq(&Int::from_i64(val)))
                    .collect();

                let domain_constraints_refs: Vec<&Bool> = domain_constraints.iter().collect();
                if !domain_constraints_refs.is_empty() {
                    let or_constraint = Bool::or(&domain_constraints_refs);
                    self.z3_solver.assert(&or_constraint);
                }
            }
        }

        // Add AllDifferent constraint (enforcing 1-to-1 mapping via pairwise uniqueness)
        let all_vars: Vec<&Int> = records.iter().filter_map(|r| variables.get(r)).collect();
        for i in 0..all_vars.len() {
            for j in (i + 1)..all_vars.len() {
                self.z3_solver.assert(&all_vars[i].eq(all_vars[j]).not());
            }
        }

        // Solve and extract values
        if self.z3_solver.check() == SatResult::Sat {
            let model = self.z3_solver.get_model().unwrap();
            let mut solution = Vec::new();
            for &rec in records {
                if let Some(val) = model.eval(variables.get(&rec).unwrap(), true) {
                    solution.push((rec, val.as_i64().unwrap()));
                }
            }
            Some(solution)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reconstruct_simple() {
        let solver = SolverEngine::new();

        let records = vec![1, 2];
        let mut t1_matches = HashMap::new();

        // Record 1 can be 10 or 20
        t1_matches.insert(1, vec![10, 20]);
        // Record 2 must be 20
        t1_matches.insert(2, vec![20]);

        let solution = solver.reconstruct(&records, &t1_matches).unwrap();

        // Because of the pairwise _eq().not() constraint, Record 1 must resolve to 10
        assert!(solution.contains(&(1, 10)));
        assert!(solution.contains(&(2, 20)));
    }
}
