import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from utils import set_sleek_style, PALETTE, procrustes_align

logger = logging.getLogger(__name__)


def load_standard_json(file_path, align=True):
    """Loads Even Less and Remin standard JSON files."""
    with open(file_path, 'r') as f:
        data = json.load(f)

    true_pts = np.array([pt["true"] for pt in data])
    recon_pts = np.array([pt["reconstructed"] for pt in data])

    if align:
        recon_pts, _ = procrustes_align(true_pts, recon_pts)

    return true_pts, recon_pts


def load_limits_json(file_path, align=True):
    """Loads Limits JSON files which are formatted as a list of lists."""
    with open(file_path, 'r') as f:
        all_data = json.load(f)

    # Just grab the first reconstruction exactly like the Rust code: `all_data[0]`
    data = all_data[0]

    true_pts = np.array([pt["true"] for pt in data])
    recon_pts = np.array([pt["reconstructed"] for pt in data])

    if align:
        recon_pts, _ = procrustes_align(true_pts, recon_pts)

    return true_pts, recon_pts


def plot_spatial_reconstruction(db, dist, true_coords, method_coords, output_path):
    set_sleek_style()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), layout='constrained')
    fig.suptitle(f"{db.capitalize()} {dist.capitalize()} Distribution")

    # Mapping internal dictionary keys to their visual representation
    methods = [("LAMa", PALETTE["blue"]), ("Even Less", PALETTE["orange"]), ("Remin", PALETTE["green"])]
    true_pts = np.array(true_coords)

    for ax, (method, color) in zip(axes, methods):
        # 1. Ground Truth (Bottom Layer, zorder=1)
        if true_pts.size > 0:
            ax.scatter(true_pts[:, 0], true_pts[:, 1], c=PALETTE["gray"],
                       s=64, alpha=0.5, label="Ground Truth", zorder=1)

        # 2. Reconstructed Points (Top Layer, zorder=2)
        if method in method_coords:
            recon_pts = np.array(method_coords[method])
            if recon_pts.size > 0:
                ax.scatter(recon_pts[:, 0], recon_pts[:, 1], c=color,
                           s=16, label="Reconstructed", zorder=2)

        ax.set_title(method)
        ax.set_aspect('equal', adjustable='datalim')  # Keeps coordinates physically proportional
        ax.legend(loc='upper right')

    plt.savefig(output_path, format='svg', transparent=True)
    plt.close()


def run_spatial_plots(name, dir_path, dist, grid):
    path_to_root = f"{dir_path}/{name}"

    even_less = f"{path_to_root}/even_less/{name}_prob100.0_{dist}_{grid}x{grid}_even_less.json"
    remin_path = f"{path_to_root}/remin/{name}_prob100.0_{dist}_{grid}x{grid}_classic.json"
    limits = f"{path_to_root}/limits/{name}_{dist}_e0_d0.9_reconstruction.json"

    # Grid overrides
    if grid == 350:
        even_less = f"{path_to_root}/even_less/{name}_prob100.0_{dist}_350x50_even_less.json"
        remin_path = f"{path_to_root}/remin/{name}_prob100.0_{dist}_350x50_classic.json"
    elif grid == 175:
        even_less = f"{path_to_root}/even_less/{name}_prob100.0_{dist}_175x25_even_less.json"
        remin_path = f"{path_to_root}/remin/{name}_prob100.0_{dist}_175x25_classic.json"

    logger.debug(f"About to load data from {even_less}, {remin_path}, {limits}")

    # Load and auto-align the data
    true_el, recon_el = load_standard_json(even_less, align=True)
    true_remin, recon_remin = load_standard_json(remin_path, align=True)
    true_limits, recon_limits = load_limits_json(limits, align=True)

    # Prepare data dictionary mapped to the exact names expected by the plotting function
    data_map = {
        "Even Less": recon_el,
        "Remin": recon_remin,
        "LAMa": recon_limits  # "limits" represents LAMa
    }

    out_path = f"{path_to_root}/{name}_{dist}_spatial_comparison.svg"

    plot_spatial_reconstruction(
        db=name,
        dist=dist,
        true_coords=true_el,  # Using Even Less ground truth as the baseline
        method_coords=data_map,
        output_path=out_path
    )