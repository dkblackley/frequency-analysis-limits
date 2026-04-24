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
        tuple_size = meta["tuple_size"]

        slice_data = flat_data[start: start + length]
        allowed_tuples_with_costs = slice_data.reshape(-1, tuple_size)

        constraint_vars = [variables[i] for i in var_ids]

        # 1. Find the min and max cost in this block to bound our cost variable
        costs = allowed_tuples_with_costs[:, -1]
        min_cost = int(np.min(costs))
        max_cost = int(np.max(costs))

        # 2. Create a single variable to hold the cost for this specific constraint block
        block_cost_var = model.NewIntVar(min_cost, max_cost, f"cost_{start}")

        # 3. Append the cost variable to our list of constrained variables
        block_vars = constraint_vars + [block_cost_var]

        # 4. Convert numpy array to native python list of lists (required by OR-Tools)
        tuples_as_lists = allowed_tuples_with_costs.tolist()

        # 5. Let the C++ backend handle the heavy lifting
        model.AddAllowedAssignments(block_vars, tuples_as_lists)

        # Accumulate the cost variable for the objective
        objective_terms.append(block_cost_var)

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
