import os
import json
import matplotlib.pyplot as plt
from matplotlib.pyplot import xlim

import main
from globals_vars import FONT_SIZE


def plot_vc_bounds(json_filepath, output_dir):
    # Hardcoded loading as requested
    if not os.path.exists(json_filepath):
        print(f"Error: Could not find {json_filepath}")
        return

    with open(json_filepath, 'r') as f:
        data = json.load(f)

    distributions = ["uniform", "gaussian", "beta"]
    n_cols = len(distributions)

    # Matching your exact grid layout sizing
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols + 2, 11), sharey=True)
    if n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    fig.set_layout_engine(None)
    # fig.suptitle(f'VC-dim error bound and true error', fontsize=FONT_SIZE + 20, x=0.4)

    lines_for_legend = []
    labels_for_legend = []

    for ax, dist in zip(axes, distributions):
        # Assumes the JSON is keyed by distribution name: {"uniform": {"true_max_err": [...]}, ...}
        dist_data = data.get(dist, {})
        if not dist_data:
            continue

        max_err = dist_data.get("true_max_err", [])
        avg_err = dist_data.get("true_avg_err", [])
        vc_bnd = dist_data.get("vc_bound", [])

        # Generate X-axis values based on the length of the results array
        x_vals = list(range(1, len(max_err) + 1))

        # Plot Max Error (Solid Blue Square)
        l1, = ax.plot(x_vals, max_err, marker='s', linestyle='-', color='#0072B2',
                      linewidth=4, markersize=12, label='Empirical Error', zorder=3)

        ax.set_yticks([0,  0.25, 0.5, 0.75, 1.0])


        # 2. Explicitly define the text for each tick so it doesn't auto-pad with zeros
        ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1.0'])

        # 2. Add 'pad' to the tick_params to push the labels outward from the axis lines
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE - 25, pad=12)

        # Plot Average Error (Solid Green Circle)
        # l2, = ax.plot(x_vals, avg_err, marker='o', linestyle='-', color='#009E73',
        #               linewidth=4, markersize=12, label='Average Error', zorder=3)

        # Plot VC Bound (Dotted Red Line, no markers to keep it clean)
        l3, = ax.plot(x_vals, vc_bnd, marker='', linestyle=':', color='red',
                      linewidth=6, label='VC Bound', zorder=2)

        # Store legend handles from the first plot to avoid duplicates
        if ax == axes[0]:
            lines_for_legend.extend([l1, l3])
            labels_for_legend.extend(['Empirical Error', 'Theoretical Approximation'])
            ax.set_ylabel('Error', fontsize=FONT_SIZE + 30)

        ax.set_title(f'{dist.capitalize()}', fontsize=FONT_SIZE + 10)
        ax.grid(True, zorder=0)

        ax.set_xlim(0, 100)


    # Single shared legend at the bottom
    fig.legend(lines_for_legend, labels_for_legend,
               loc='lower center', bbox_to_anchor=(0.5, 0.02),
               ncol=3, frameon=False, fontsize=FONT_SIZE - 5,
               handlelength=1.0,     # Shortens the physical length of the lines
               handletextpad=0.4,    # Reduces the space between the line/marker and the text
               columnspacing=1.0,    # Reduces the horizontal space between the three columns
               borderaxespad=0.0)

    plt.ylim(0, 1)

    # (Left, Bottom, Top, Wspace) spacing tweaked to fit the massive labels
    plt.subplots_adjust(left=-0.03, bottom=0.4, top=0.75, wspace=0.3, right=1)

    # Global X-Axis Label (Change 'Iterations' to whatever your X-axis represents)
    fig.text(0.5, 0.24, 'Number of Queries Issued', va='center', ha='center', fontsize=FONT_SIZE + 10)

    os.makedirs(output_dir, exist_ok=True)
    filename = 'vc_bounds_combined.pdf'
    plt.savefig(os.path.join(output_dir, filename), format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to ./{output_dir}/{filename}")


if __name__ == "__main__":
    main.setup_sleek_style()
    # Point this to your generated JSON
    plot_vc_bounds("../vc_bounds_results.json", "plots")