import json
import os
import sys

import numpy as np
from ortools.sat.python import cp_model


class SolutionCollector(cp_model.CpSolverSolutionCallback):
    # Add objective_terms to the init
    def __init__(self, variables, objective_terms, limit=100):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.limit = limit
        self.num_solutions = 0
        self.variables = variables
        self.objective_terms = objective_terms
        self.solutions = []

    def OnSolutionCallback(self):
        self.num_solutions += 1

        # Calculate the actual cost of this specific solution
        current_cost = sum(self.Value(term) for term in self.objective_terms)

        # Store a dictionary so you know the cost associated with the variables
        self.solutions.append({
            "cost": current_cost,
            "variables": [self.Value(v) for v in self.variables]
        })

        if self.num_solutions >= self.limit:
            self.StopSearch()


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
    solver.parameters.num_search_workers = 10
    # solver.parameters.log_search_progress = True
    solver.parameters.linearization_level = 2
    solver.parameters.optimize_with_core = True

    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"Optimal solution found with cost: {solver.ObjectiveValue()}")

        # NEW: Save the exact variable assignments from Pass 1
        pass_1_values = [solver.Value(v) for v in variables]

        if get_one:
            single_sol = [pass_1_values]
            with open("solutions.json", "w") as f:
                json.dump(single_sol, f)
        else:

            # ---------------------------------------------------------
            # PASS 2: Enumerate all solutions that match the best cost
            # ---------------------------------------------------------
            # PASS 2: "Forbid and Loop" Strategy
            if objective_terms:
                best_cost = int(solver.ObjectiveValue())
                model.ClearObjective()

                # model.Minimize(sum(objective_terms))
                target_max_cost = int(best_cost * 1.1)
                model.Add(sum(objective_terms) <= target_max_cost)

            solver.parameters.enumerate_all_solutions = True

            collector = SolutionCollector(variables, objective_terms)
            status = solver.Solve(model, collector)

            # Sort the collected solutions from lowest cost to highest cost
            collector.solutions.sort(key=lambda x: x["cost"])

            clean_solutions = [sol["variables"] for sol in collector.solutions]

            # Keep all 10 workers and aggressive heuristics ON
            # solver.parameters.num_search_workers = 10

            # DO NOT set enumerate_all_solutions = True

            # clean_solutions = []
            # target_number_of_solutions = 25
            #
            # for _ in tqdm(range(target_number_of_solutions)):
            #     status = solver.Solve(model)
            #
            #     if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            #         # 1. Save the solution
            #         current_sol = [solver.Value(v) for v in variables]
            #         clean_solutions.append(current_sol)
            #
            #         # 2. Forbid this exact combination from ever being found again
            #         # This forces the 10 workers to find a NEW optimal solution on the next loop
            #         model.AddForbiddenAssignments(variables, [current_sol])
            #
            #         # print(f"Optimal solution found with cost: {solver.ObjectiveValue()}")
            #
            #         model.ClearHints()
            #         for var, val in zip(variables, current_sol):
            #             model.AddHint(var, val)
            #     else:
            #         print("Exhausted all possible solutions!")
            #         break

            with open("solutions.json", "w") as f:
                json.dump(clean_solutions, f)
    else:
        print("Could not find a single solution...")
        sys.exit(1)


if __name__ == "__main__":
    main()
