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

    if current_user == 'yelnat':
        # Local machine path
        proj_root = "/home/yelnat/Nextcloud/10TB-STHDD/Sync-Folder-STHDD/programmin/frequency_analysis_limits/"
    else:
        # External server path (fallback)
        proj_root = "/scratch/dblackle/frequency-analysis-limits/"
    num_variables = int(sys.argv[1])
    largest_val = int(sys.argv[2])
    get_one = sys.argv[3].lower() == "true"

    with open(f"{proj_root}meta.json", "r") as f:
        metadata = json.load(f)

    flat_data = np.fromfile(f"{proj_root}allowed.bin", dtype=np.int64)

    model = cp_model.CpModel()
    variables = [model.NewIntVar(0, largest_val, f"var_{i}") for i in range(num_variables)]

    # Apply all accumulated constraints (t=1, t=2, etc.)
    for meta in metadata:
        var_ids = meta["var_ids"]
        start = meta["start_idx"]
        length = meta["length"]
        tuple_size = meta["tuple_size"]

        slice_data = flat_data[start: start + length]
        allowed_tuples = slice_data.reshape(-1, tuple_size)

        constraint_vars = [variables[i] for i in var_ids]
        model.AddAllowedAssignments(constraint_vars, allowed_tuples.tolist())

    model.AddAllDifferent(variables)

    solver = cp_model.CpSolver()

    if get_one:
        solver.parameters.num_search_workers = 256
    else:
        solver.parameters.enumerate_all_solutions = True
        solver.parameters.keep_all_feasible_solutions_in_presolve = True

    collector = SolutionCollector(variables)
    status = solver.Solve(model, collector)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) or len(collector.solutions) > 0:
        # If get_one is true, return just the primary solution wrapped in an array
        if get_one and len(collector.solutions) == 0:
            single_sol = [[solver.Value(v) for v in variables]]
            with open("solutions.json", "w") as f:
                json.dump(single_sol, f)
        else:
            with open("solutions.json", "w") as f:
                json.dump(collector.solutions, f)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
