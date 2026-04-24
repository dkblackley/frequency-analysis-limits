import json
import os
import sys

import numpy as np
from ortools.sat.python import cp_model


class SolutionCollector(cp_model.CpSolverSolutionCallback):
    def __init__(self, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.variables = variables
        self.solutions = []

    def OnSolutionCallback(self):
        self.solutions.append([self.Value(v) for v in self.variables])


def main():
    current_user = os.getenv('USER', os.getenv('USERNAME', ''))
    proj_root = "/home/yelnat/Nextcloud/10TB-STHDD/Sync-Folder-STHDD/programmin/frequency_analysis_limits/" if current_user == 'yelnat' else "/scratch/dblackle/frequency-analysis-limits/"

    num_variables = int(sys.argv[1])
    largest_val = int(sys.argv[2])
    get_one = sys.argv[3].lower() == "true"

    with open(f"{proj_root}meta.json", "r") as f:
        metadata = json.load(f)

    flat_data = np.fromfile(f"{proj_root}allowed.bin", dtype=np.int64)

    model = cp_model.CpModel()
    variables = [model.NewIntVar(0, largest_val, f"var_{i}") for i in range(num_variables)]

    objective_terms = []

    # Apply all accumulated constraints
    for meta in metadata:
        var_ids = meta["var_ids"]
        start = meta["start_idx"]
        length = meta["length"]
        # Tuple size is now var_ids.len() + 1 (the last item is the cost)
        tuple_size = meta["tuple_size"]

        slice_data = flat_data[start: start + length]
        allowed_tuples_with_costs = slice_data.reshape(-1, tuple_size)

        constraint_vars = [variables[i] for i in var_ids]

        # Create a boolean variable for every possible tuple in this constraint block
        tuple_bools = [model.NewBoolVar(f"tuple_{start}_{i}") for i in range(len(allowed_tuples_with_costs))]

        # We MUST pick exactly ONE valid tuple from this block
        model.AddExactlyOne(tuple_bools)

        for i, row in enumerate(allowed_tuples_with_costs):
            t_val = row[:-1]  # The variable assignments
            cost = int(row[-1])  # The cost we passed from Rust

            # Enforce the assignment ONLY IF this boolean is chosen by the solver
            for var, val in zip(constraint_vars, t_val):
                model.Add(var == int(val)).OnlyEnforceIf(tuple_bools[i])

            # Accumulate the cost
            if cost != 0:
                objective_terms.append(tuple_bools[i] * cost)

    model.AddAllDifferent(variables)

    # ---------------------------------------------------------
    # PASS 1: Minimize the total cost to find the optimal score
    # ---------------------------------------------------------
    if objective_terms:
        model.Minimize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 128

    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"Optimal solution found with cost: {solver.ObjectiveValue()}")

        if get_one:
            single_sol = [[solver.Value(v) for v in variables]]
            with open("solutions.json", "w") as f:
                json.dump(single_sol, f)
        else:
            solver.parameters.num_search_workers = 0
            # ---------------------------------------------------------
            # PASS 2: Enumerate all solutions that match the best cost
            # ---------------------------------------------------------
            if objective_terms:
                best_cost = int(solver.ObjectiveValue())
                model.ClearObjective()  # <-- The new, official API method
                model.Add(sum(objective_terms) == best_cost)  # Lock in the best score

            solver.parameters.enumerate_all_solutions = True
            collector = SolutionCollector(variables)
            status = solver.Solve(model, collector)

            with open("solutions.json", "w") as f:
                json.dump(collector.solutions, f)
    else:
        print("Could not find a single solution...")
        sys.exit(1)


if __name__ == "__main__":
    main()
