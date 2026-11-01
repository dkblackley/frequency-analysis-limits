import os
import matplotlib.pyplot as plt

from globals_vars import FONT_SIZE


def plot_running_time_lines(time_data, dataset, distributions, output_dir, method_order, colors, markers, labels):
    """
    Expects time_data formatted as: time_data[dataset][distribution][method] = [(10, time10), (20, time20), (30, time30)]
    """
    db_data = time_data.get(dataset, {})
    if not db_data:
        return

    n_cols = len(distributions)
    fig, axes = plt.subplots(1, n_cols, figsize=(11 * n_cols + 2, 11), sharey=True)
    if n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    fig.set_layout_engine(None)

    lines_for_legend = []
    labels_for_legend = []

    for ax, dist in zip(axes, distributions):
        data = db_data.get(dist, {})

        for method in method_order:
            method_runs = data.get(method, [])
            if not method_runs:
                continue

            # Sort by query percent (x-axis)
            method_runs.sort(key=lambda x: x[0])
            x_vals = [r[0] for r in method_runs]

            # Ensure no zeroes exist to prevent log-scale math errors
            y_vals = [max(0.0001, r[1]) for r in method_runs]

            c = colors.get(method, '#000000')
            m = markers.get(method, 'o')
            l = labels.get(method, method)

            line, = ax.plot(x_vals, y_vals, marker=m, linestyle='-', label=l, alpha=0.9, color=c, zorder=3)

            if l not in labels_for_legend:
                lines_for_legend.append(line)
                labels_for_legend.append(l)

        if ax == axes[0]:
            ax.set_ylabel('Time Taken (s)', fontsize=FONT_SIZE + 30)

        # Use log scale due to 1.7s vs 26000s variance
        ax.set_yscale('log')

        # Force strict 10-20-30 axis labels
        ax.set_xticks([10, 20, 30])
        ax.set_xlim(9.5, 30.5)
        ax.set_xticklabels(['10%', '20%', '30%'], fontsize=FONT_SIZE - 10)
        ax.tick_params(axis='y', labelsize=FONT_SIZE - 10)

        ax.set_title(f'{dist.capitalize()}', fontsize=FONT_SIZE + 40)

        # Add horizontal grid lines for easier log scale reading
        ax.grid(True, zorder=0)

    # Replicate the Massive Super Title across the entire width
    fig.text(0.05, 0.55, dataset.capitalize(), va='center', ha='center',
             rotation='vertical', fontsize=FONT_SIZE + 20, fontweight='bold')

    # Single shared legend at the bottom
    fig.legend(lines_for_legend, labels_for_legend,
               loc='lower center', bbox_to_anchor=(0.45, -0.0),
               ncol=len(method_order), frameon=False, fontsize=FONT_SIZE)

    # (Left, Bottom, Top, Wspace, Right) spacing tweaked to fit the massive labels
    plt.subplots_adjust(left=0.17, bottom=0.37, top=0.75, wspace=0.25, right=0.8)

    # Add the single, centered global x-label
    fig.text(0.5, 0.23, 'Query Percent', va='center', ha='center', fontsize=FONT_SIZE + 20)

    filename = f'{dataset}_running_time_lines_combined.pdf'
    plt.savefig(os.path.join(output_dir, filename), format='pdf', bbox_inches='tight')
    plt.close(fig)