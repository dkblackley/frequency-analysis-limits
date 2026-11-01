import os
import json


def print_running_times(base_dir="databases/50x50"):
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist. Please check your path.")
        return

    # Dynamically find all dataset folders (e.g., busstop, cali, drink, etc.)
    datasets = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    datasets.sort()

    methods = ["even_less", "remin", "limits"]

    print(f"{'Dataset':<15} | {'Method':<12} | {'Distribution':<14} | {'Time Taken (s)'}")
    print("-" * 65)

    for db in datasets:
        for method in methods:
            method_dir = os.path.join(base_dir, db, method)
            if not os.path.exists(method_dir):
                continue

            # Iterate through all json files in this method's directory
            for file_name in os.listdir(method_dir):
                if not file_name.endswith(".json"):
                    continue

                # Only look at 'results' files to avoid the massive reconstruction point cloud files
                if "results" not in file_name:
                    continue

                # Identify files that represent perfect knowledge / 100% queries
                is_perfect_knowledge = False
                if method in ["even_less", "remin"] and "prob100.0" in file_name:
                    is_perfect_knowledge = True
                elif method == "limits" and ("e0_d0.9" in file_name or "prob1" in file_name):
                    is_perfect_knowledge = True

                if not is_perfect_knowledge:
                    continue

                # Extract the distribution from the filename
                dist = "unknown"
                for d in ["uniform", "gaussian", "beta"]:
                    if d in file_name:
                        dist = d
                        break

                file_path = os.path.join(method_dir, file_name)

                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        time_taken = data.get("time_taken", "N/A")

                        # Format floats to 4 decimal places for readability
                        if isinstance(time_taken, (float, int)):
                            time_str = f"{time_taken:.4f}"
                        else:
                            time_str = str(time_taken)

                        # Notice some files might have _50 variations in the name, we can print the actual filename prefix if preferred,
                        # but keeping it clean here using the base `db` name.
                        print(f"{db:<15} | {method:<12} | {dist:<14} | {time_str}")

                except Exception as e:
                    print(f"{db:<15} | {method:<12} | {dist:<14} | Error reading file")

        # Print a separator after each dataset block
        print("-" * 65)


if __name__ == "__main__":
    # Ensure this runs relative to the correct directory containing the databases folder
    print_running_times(base_dir="../databases/50x50")