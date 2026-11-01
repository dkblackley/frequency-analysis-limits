import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter

from data_loader import calculate_mse
from globals_vars import FONT_SIZE, format_db_name, format_y_axis


def plot_reconstruction_histograms(json_path, dataset, dist, output_dir, method_order, colors, labels):
    if not os.path.exists(json_path):
        print(f"[SKIP] Histogram JSON not found: {json_path}")
        return

    with open(json_path, 'r') as f:
        mse_data = json.load(f)

    # Filter to ensure we only plot methods available in the JSON file
    plot_methods = [m for m in method_order if m in mse_data or labels.get(m, m) in mse_data]
    n_cols = len(plot_methods)

    if n_cols == 0:
        return

    fig, axes = plt.subplots(1, n_cols, figsize=(11 * n_cols + 2, 11), sharey=False)
    if n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    fig.set_layout_engine(None)

    for ax, method in zip(axes, plot_methods):
        # Rust outputs labels directly for this struct, so we fetch by the label name
        method_key = labels.get(method, method)

        # Structure is [ [list_of_mses], best_mse ]
        mses = mse_data.get(method_key, [[], 0.0])[0]
        c = colors.get(method, '#000000')

        if not mses:
            continue

        min_mse = 0.0
        local_max_mse = max(mses)

        if local_max_mse <= 0.0:
            local_max_mse = 1.0
        else:
            local_max_mse *= 1.05


        bins = np.linspace(min_mse, local_max_mse, 7)

        ax.hist(mses, bins=bins, color=c, alpha=0.55, edgecolor=c, linewidth=2, rwidth=0.9)

        ax.set_title(f'{method_key}', fontsize=FONT_SIZE + 40)
        if ax == axes[1]:
            ax.set_xlabel('MSE', fontsize=FONT_SIZE + 30)

        if ax == axes[0]:
            ax.set_ylabel('# of Solutions', fontsize=FONT_SIZE + 10, y=0.6)
            ax.set_ylim(0, 8)

        ax.tick_params(axis='x', labelsize=FONT_SIZE - 10)
        ax.tick_params(axis='y', labelsize=FONT_SIZE - 10)

        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))

        ax.yaxis.set_major_formatter(FuncFormatter(format_y_axis))
        ax.xaxis.set_major_formatter(FuncFormatter(format_y_axis))

        ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Massive Rotated Super Title
    fig.text(0.08, 0.65, format_db_name(dataset), va='center', ha='center',
             rotation='vertical', fontsize=FONT_SIZE + 20, fontweight='bold')

    # Centered top title for the current distribution
    fig.text(0.55, 1.0, f'{dist.capitalize()} Dist.', va='center', ha='center',
             fontsize=FONT_SIZE + 40)

    plt.subplots_adjust(left=0.16, bottom=0.4, top=0.75, wspace=0.3, right=1.0)

    filename = f'mse_histogram_{dataset}_{dist}.pdf'
    plt.savefig(os.path.join(output_dir, filename), format='pdf', bbox_inches='tight')
    plt.close(fig)



def plot_lama_flat_distributions(base_dir, dataset, output_dir):
    """Replaces plot_lama_distributions from flat_dist_hist.rs"""
    distributions = ["gaussian", "uniform", "flat"]
    dist_to_mse = {}

    # 1. Load the Data dynamically
    for dist in distributions:

        # path = f"{base_dir}/15x15/{dataset}/limits/{dataset}_{dist}_e0_d0.001_reconstruction.json"
        path = f"{base_dir}/15x15/{dataset}/limits/{dataset}_{dist}_e0_d0.1_reconstruction.json"

        if not os.path.exists(path):
            # Fallback for spitz or others needing d0.1
            path = f"{base_dir}/15x15/{dataset}/limits/{dataset}_{dist}_e0_d0.001_reconstruction.json"
            # path = f"{base_dir}/15x15/{dataset}/limits/{dataset}_{dist}_e0_d0.1_reconstruction.json"


        if not os.path.exists(path):
            print(f"[SKIP] LAMa Flat data not found: {path}")
            continue

        with open(path, 'r') as f:
            all_data = json.load(f)

        mses = []
        for run in all_data:
            # Assuming you still have the calculate_mse() function from your first file
            true_pts = [p["true"] for p in run]
            recon_pts = [p["reconstructed"] for p in run]
            mses.append(calculate_mse(true_pts, recon_pts))

        dist_to_mse[dist] = mses

    n_cols = len(dist_to_mse)
    if n_cols == 0:
        return

    # 2. Setup Canvas
    fig, axes = plt.subplots(1, n_cols, figsize=(11 * n_cols + 2, 11), sharey=False)
    if n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    fig.set_layout_engine(None)
    lama_color = '#0072B2'

    # 3. Plot Histograms
    for ax, dist in zip(axes, distributions):
        mses = dist_to_mse.get(dist, [])
        if not mses:
            continue

        min_mse = 0.0
        local_max_mse = max(mses) if mses else 0.0
        local_max_mse = 1.0 if local_max_mse <= 0.0 else local_max_mse * 1.05

        bins = np.linspace(min_mse, local_max_mse, 7)

        ax.hist(mses, bins=bins, color=lama_color, alpha=0.55, edgecolor=lama_color, linewidth=2, rwidth=0.9)

        ax.set_title(f'{dist.capitalize()} Dist.', fontsize=FONT_SIZE + 25, loc='center', x=0.55)

        if ax == axes[1]:
            ax.set_xlabel('MSE', fontsize=FONT_SIZE + 30)

        if local_max_mse == 1.0:
            ax.set_xticklabels(['0', '2', '4', '6'], fontsize=FONT_SIZE + 10)
            ax.set_ylim(0,2)

        if ax == axes[0]:
            ax.set_ylabel('# of Solutions', fontsize=FONT_SIZE - 1, y=0.4)

        ax.tick_params(axis='x', labelsize=FONT_SIZE - 10)
        ax.tick_params(axis='y', labelsize=FONT_SIZE - 10)

        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))

        ax.yaxis.set_major_formatter(FuncFormatter(format_y_axis))

        if local_max_mse == 1.0:
            # 1. Set the physical anchor points first (this clears the warning)
            ax.set_xticks([0.0, 0.33, 0.66, 1.0])

            # 2. Then set your custom text strings
            ax.set_xticklabels(['0', '2', '4', '6'], fontsize=FONT_SIZE - 10)
        else:
            # Only apply the automatic locator and formatter if we ARE NOT using custom labels
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            ax.xaxis.set_major_formatter(FuncFormatter(format_y_axis))


        ax.grid(axis='y', linestyle='--', alpha=0.7)

    # 4. Massive Rotated Super Title
    fig.text(0.08, 0.55, format_db_name(dataset), va='center', ha='center',
             rotation='vertical', fontsize=FONT_SIZE + 10, fontweight='bold')

    plt.subplots_adjust(left=0.16, bottom=0.4, top=0.75, wspace=0.3, right=1.0)

    filename = f'histogram_{dataset}_lama_flat.pdf'
    plt.savefig(os.path.join(output_dir, filename), format='pdf', bbox_inches='tight')
    plt.close(fig)