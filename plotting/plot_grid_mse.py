import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sympy.printing.pretty.pretty_symbology import line_width

from globals_vars import format_db_name, FONT_SIZE


def format_y_axis(value, tick_number):
    """Intercepts values near 1.0 to render them as 0 to mimic the Rust log scale visually."""
    if value <= 1.001:
        return "0"
    return f"{value:g}"


def plot_grid_by_mse(grid_data, dataset, distributions, output_dir, method_order, colors, markers, labels):
    db_data = grid_data.get(dataset, {})
    if not db_data:
        return

    n_cols = len(distributions)
    # Scale width based on the number of plots (e.g. 3 distributions -> 32 inches wide)
    fig, axes = plt.subplots(1, n_cols, figsize=(11 * n_cols + 2, 11), sharey=True)
    if n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    fig.set_layout_engine(None)  # Give us full layout control

    lines_for_legend = []
    labels_for_legend = []

    for ax, dist in zip(axes, distributions):
        data = db_data.get(dist, {})

        for method in method_order:
            method_runs = data.get(method, [])
            if not method_runs:
                continue

            method_runs.sort(key=lambda x: x[0])
            x_vals = [r[0] for r in method_runs]
            y_vals = [max(1.0, r[1]) for r in method_runs]

            c = colors.get(method, '#000000')
            m = markers.get(method, 'o')
            l = labels.get(method, method)

            line, = ax.plot(x_vals, y_vals, marker=m, linestyle='-', label=l, color=c, zorder=3)

            # Store handle for the global legend, avoiding duplicates
            if l not in labels_for_legend:
                lines_for_legend.append(line)
                labels_for_legend.append(l)

        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(FuncFormatter(format_y_axis))

        ax.set_xticks([20, 30, 40, 50])
        ax.set_xticklabels(['20', '30', '40', '50'], fontsize=FONT_SIZE - 10)
        ax.set_xlim(19, 51)

        # ax.set_xlabel('Grid Size')
        if ax == axes[0]:
            ax.set_ylabel('MSE (log)', fontsize=FONT_SIZE + 30)
            # ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE - 25, pad=12)

        ax.set_title(f'{dist.capitalize()}', fontsize=FONT_SIZE + 40)
        ax.tick_params(axis='y', labelsize=FONT_SIZE - 10, pad=20)
        ax.grid(True, zorder=0)


    fig.text(0.025, 0.5, format_db_name(dataset), va='center', ha='center',
                 rotation='vertical', fontsize=FONT_SIZE + 20, fontweight='bold')


    if dataset == "spitz":
        fig.legend(lines_for_legend, labels_for_legend,
                   loc='lower center', bbox_to_anchor=(0.5, 0.78),
                   ncol=len(method_order), frameon=False, fontsize=FONT_SIZE + 20,
                   handlelength=1.0,
                   markerscale=1.0,
                   columnspacing=0.5,
                   handletextpad=0.2)


    # (Left, Bottom, Top, Wspace) spacing tweaked to fit the massive labels
    plt.subplots_adjust(left=0.16, bottom=0.3, top=0.75, wspace=0.2, right=1.0)
    fig.text(0.55, 0.12, 'Grid Size', va='center', ha='center', fontsize=FONT_SIZE + 30)

    filename = f'mse_grid_{dataset}.pdf'
    plt.savefig(os.path.join(output_dir, filename), format='pdf', bbox_inches='tight')
    plt.close(fig)