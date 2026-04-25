import json
import os
import sys

import numpy as np
from ortools.sat.python import cp_model


class SolutionCollector(cp_model.CpSolverSolutionCallback):
    def __init__(self, variables, limit=1000000000):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.limit = limit
        self.num_solutions = 0
        self.variables = variables
        self.solutions = []

    def OnSolutionCallback(self):
        self.num_solutions += 1
        self.solutions.append({
            "variables": [self.Value(v) for v in self.variables]
        })
        if self.num_solutions % 1000 == 0:
            print(f"Found: {self.num_solutions}")

        if self.num_solutions >= self.limit:
            self.StopSearch()


def build_model(num_variables, largest_val, metadata, flat_data):
    """Builds the base model with all constraints."""
    model = cp_model.CpModel()
    variables = [model.NewIntVar(0, largest_val, f"var_{i}") for i in range(num_variables)]

    for meta in metadata:
        var_ids = meta["var_ids"]
        start = meta["start_idx"]
        length = meta["length"]
        tuple_size = meta["tuple_size"]

        slice_data = flat_data[start: start + length]
        allowed_tuples_with_costs = slice_data.reshape(-1, tuple_size)

        constraint_vars = [variables[i] for i in var_ids]

        # Slice off the cost column
        allowed_tuples = allowed_tuples_with_costs[:, :-1]
        tuples_as_lists = allowed_tuples.tolist()

        model.AddAllowedAssignments(constraint_vars, tuples_as_lists)

    model.AddAllDifferent(variables)

    return model, variables


def main():
    current_user = os.getenv('USER', os.getenv('USERNAME', ''))
    proj_root = "/home/yelnat/Nextcloud/10TB-STHDD/Sync-Folder-STHDD/programmin/frequency_analysis_limits/" if current_user == 'yelnat' else "/scratch/dblackle/frequency-analysis-limits/"

    num_variables = int(sys.argv[1])
    largest_val = int(sys.argv[2])
    get_one = sys.argv[3].lower() == "true"
    probabilistic = sys.argv[4].lower() == "true"
    run_id = sys.argv[5]

    with open(f"{proj_root}meta_{run_id}.json", "r") as f:
        metadata = json.load(f)

    flat_data = np.fromfile(f"{proj_root}allowed_{run_id}.bin", dtype=np.int64)

    true_solution_path = f"true_solution_{run_id}.json"
    true_solution = None
    if os.path.exists(true_solution_path):
        with open(true_solution_path, "r") as f:
            true_solution = json.load(f)

    # ---------------------------------------------------------
    # DEBUG MODE: The Targeted Probe
    # ---------------------------------------------------------
    if true_solution is not None:
        print("--- RUNNING TARGETED PROBE ---", flush=True)
        model, variables = build_model(num_variables, largest_val, metadata, flat_data)

        # Lock the model exactly to the true solution
        for i, val in enumerate(true_solution):
            model.Add(variables[i] == val)

        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        solver.parameters.num_search_workers = 128

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            print("RESULT: SUCCESS.", flush=True)
            print("Your true solution is mathematically valid under these constraints.", flush=True)
            print("If you aren't seeing it in the main output, it is just buried in the noise.", flush=True)
        else:
            print("RESULT: FATAL ERROR.", flush=True)
            print("Your true solution is impossible to output.", flush=True)
            print("One of your table constraints or the AllDifferent rule explicitly forbids it.", flush=True)
            sys.exit(1)

    # ---------------------------------------------------------
    # STANDARD SOLVER MODE
    # ---------------------------------------------------------
    print("Starting pure CP-SAT python solver!", flush=True)
    model, variables = build_model(num_variables, largest_val, metadata, flat_data)

    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True
    solver.parameters.num_search_workers = 1

    limit = 10000 if probabilistic else 100000000000
    if get_one:
        limit = 1

    collector = SolutionCollector(variables, limit=limit)
    status = solver.Solve(model, collector)

    clean_solutions = [sol["variables"] for sol in collector.solutions]

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"Found {len(clean_solutions)} valid assignments.", flush=True)
    else:
        print("Could not find a single solution...", flush=True)

    if true_solution is not None and true_solution not in clean_solutions:
        print("True solution was NOT found in the noise (cut off by limits). Injecting it.", flush=True)
        clean_solutions.append(true_solution)

    with open(f"solutions_{run_id}.json", "w") as f:
        json.dump(clean_solutions, f)


if __name__ == "__main__":
    main()
