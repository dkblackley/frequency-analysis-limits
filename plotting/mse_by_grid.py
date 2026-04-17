import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from utils import set_sleek_style, format_metric, PALETTE, procrustes_align

logger = logging.getLogger(__name__)


def load_standard_json_for_mse(file_path):
    """Loads Even Less and Remin JSON files and returns the Procrustes aligned MSE."""
    with open(file_path, 'r') as f:
        data = json.load(f)

    true_pts = np.array([pt["true"] for pt in data])
    recon_pts = np.array([pt["reconstructed"] for pt in data])

    # We only care about the MSE value (index 1) for this plot
    _, mse = procrustes_align(true_pts, recon_pts)
    return mse


def load_limits_json_for_mse(file_path):
    """Loads the nested Limits JSON file and returns the Procrustes aligned MSE."""
    with open(file_path, 'r') as f:
        all_data = json.load(f)

    data = all_data[0]
    true_pts = np.array([pt["true"] for pt in data])
    recon_pts = np.array([pt["reconstructed"] for pt in data])

    _, mse = procrustes_align(true_pts, recon_pts)
    return mse


def _render_grid_plot(db, db_data, distributions, output_path):
    """Your provided sleek plotting function."""
    set_sleek_style()
    num_dists = len(distributions)

    fig, axes = plt.subplots(1, num_dists, figsize=(8 * num_dists, 6), layout='constrained')
    if num_dists == 1: axes = [axes]

    formatter = FuncFormatter(format_metric)
    color_map = {"LAMa": PALETTE["blue"], "Even Less": PALETTE["orange"], "Remin": PALETTE["green"]}

    for ax, dist in zip(axes, distributions):
        if dist not in db_data: continue

        ax.set_title(f"{db.capitalize()} {dist.capitalize()} Query Distribution")
        ax.set_xlabel("Grid Size")
        ax.set_ylabel("Mean Squared Error")
        ax.set_yscale('log')

        for method, points in db_data[dist].items():
            if not points: continue
            points = sorted(points, key=lambda x: x[0])
            grids, mses = zip(*points)

            # Clamp MSE to a minimum of 1.0 for the log scale visualization
            mses = [max(1.0, m) for m in mses]

            color = color_map.get(method, PALETTE["black"])
            ax.plot(grids, mses, marker='o', markersize=8, linewidth=4, label=method, color=color)

        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

        # Legend Auto-Dodging: Places the legend cleanly outside the plot area
        ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0.)

    plt.savefig(output_path, format='svg', transparent=True)
    plt.close()


def plot_grid_by_mse(grid_sizes, datasets, methods, distributions):
    """
    Mirrors the outer Rust loop: Iterates over the grid sizes, datasets, and methods,
    dynamically constructs the file paths, calculates the MSE, and dispatches to the plotter.
    """
    base_dir = "../databases"

    for db in datasets:
        # Pre-initialize the structured dictionary your plotting function expects
        db_data = {dist: {"Even Less": [], "Remin": [], "LAMa": []} for dist in distributions}

        for dist in distributions:
            for method in methods:
                for grid_val, grid_str in grid_sizes:
                    path_to_root = f"{base_dir}/{grid_str}/{db}"

                    mse = None
                    try:
                        # Pathfinding logic mirrored from Rust
                        if method == "even_less":
                            path = f"{path_to_root}/even_less/{db}_prob100.0_{dist}_{grid_str}_even_less.json"
                            mse = load_standard_json_for_mse(path)
                            target_key = "Even Less"

                        elif method == "remin":
                            path = f"{path_to_root}/remin/{db}_prob100.0_{dist}_{grid_str}_classic.json"
                            mse = load_standard_json_for_mse(path)
                            target_key = "Remin"

                        elif method == "limits":
                            path = f"{path_to_root}/limits/{db}_{dist}_e0_d0.9_reconstruction.json"
                            mse = load_limits_json_for_mse(path)
                            target_key = "LAMa"

                        if mse is not None:
                            db_data[dist][target_key].append((grid_val, mse))

                    except Exception as e:
                        # Log files that don't exist yet rather than crashing the whole grid
                        logger.debug(f"Skipping {method} for {db} at grid {grid_str}: {e}")

        # Save a combined MSE plot per dataset directly to its 50x50 root folder
        output_path = f"../figures/{db}_mse_vs_grid_size.svg"
        _render_grid_plot(db, db_data, distributions, output_path)