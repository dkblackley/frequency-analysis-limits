import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import to_rgba
from globals_vars import FONT_SIZE, format_db_name, format_y_axis


N_YTICKS = 4  # linear axis: 0, ..., largest whisker


def _format_linear_tick(value, _pos):
    """3 significant figures, so the evenly spaced ticks (and the whisker top) read accurately."""
    if value == 0:
        return "0"
    if abs(value) >= 1000:
        return f"{value / 1000:.3g}K"
    return f"{value:.3g}"


def plot_mse_boxplots(json_path, dataset, distributions, output_dir, method_order, colors, labels,
                      queries=(10.0, 20.0, 30.0), legend=None, tag="", log=False):
    """queries: query percents, one evenly spaced group each. legend: None = only for spitz (paper layout).
    log: log-scale MSE axis (the old figure); default linear."""
    if not os.path.exists(json_path):
        print(f"[SKIP] Boxplot JSON not found: {json_path}")
        return

    with open(json_path, 'r') as f:
        db_data = json.load(f)

    n_cols = len(distributions)
    col_w = max(11, 3.2 * len(queries))  # paper layout: 3 queries -> 11
    fig, axes = plt.subplots(1, n_cols, figsize=(col_w * n_cols + 2, 11), sharey=True)
    if n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    fig.set_layout_engine(None)

    y_top = 0.0
    for ax, dist in zip(axes, distributions):
        data = db_data.get(dist, {})

        positions = []
        bp_data = []
        box_colors = []
        present = [m for m in method_order if data.get(m)]
        for q_idx, query in enumerate(queries):
            for m_idx, method in enumerate(present):
                method_runs = data.get(method, [])

                # Retrieve the raw MSEs array mapped to this specific query
                target_mses = next((mses for q, mses in method_runs if q == query), [])

                if target_mses:
                    bp_data.append(target_mses)
                    # Shift the boxes slightly so they cluster around the main query tick mark
                    offset = (m_idx - (len(present) - 1) / 2) * 0.27
                    positions.append(q_idx + offset)
                    box_colors.append(colors.get(method, '#000000'))

        if bp_data:
            if log:
                ax.set_yscale('log')
            ax.yaxis.set_major_formatter(FuncFormatter(format_y_axis))

            bps = ax.boxplot(bp_data, positions=positions, widths=0.22, patch_artist=True,
                             showfliers=False,  # Hides extreme outliers for a cleaner look
                             boxprops=dict(linewidth=1),
                             medianprops=dict(linewidth=1),
                             whiskerprops=dict(linewidth=1),
                             capprops=dict(linewidth=1))

            # Loop through boxes, medians, and colors simultaneously
            for patch, median, color in zip(bps['boxes'], bps['medians'], box_colors):
                # Convert the solid color to RGBA with a low alpha for transparency (e.g., 0.2)
                transparent_color = to_rgba(color, alpha=0.55)

                # Apply the transparent color to the inside of the box
                patch.set_facecolor(transparent_color)

                # Apply the original solid color to the outer box edge
                patch.set_edgecolor(color)

                # Keep the median line solid black
                median.set_color(color)

            if not log:  # top of the highest whisker (the largest 'tail'), shared y-axis
                caps = [y for cap in bps['caps'] for y in cap.get_ydata()]
                if caps:
                    y_top = max(y_top, max(caps))

        ax.set_xticks(range(len(queries)))
        ax.set_xlim(-0.6, len(queries) - 0.4)
        ax.set_xticklabels([f'{q:g}%' for q in queries], fontsize=FONT_SIZE - 10)
        ax.tick_params(axis='y', labelsize=FONT_SIZE - 10)
        pos = [v for mses in bp_data for v in mses if v > 0]
        if pos and log:  # decades spanned by the data (an MSE of 0, an exact reconstruction, sits at the bottom)
            lo, hi = int(np.floor(np.log10(min(pos)))), int(np.ceil(np.log10(max(pos))))
            ax.set_yticks([10.0 ** k for k in range(lo, hi + 1)])
            ax.set_ylim(10.0 ** lo / 2, 10.0 ** hi * 2)
        # ax.set_yticks([1, 100, 100000])

        ax.set_title(f'{dist.capitalize()}', fontsize=FONT_SIZE + 40)
        ax.grid(True, zorder=0, axis='y', which='major', linestyle='--', alpha=1.0)

        if ax == axes[0]:
            ax.set_ylabel('MSE (log)' if log else 'MSE', fontsize=FONT_SIZE + 30)

    if not log and y_top > 0:
        # 0 sits exactly on the bottom of the axes; the top tick is the largest whisker, with a little headroom
        # even step rounded up to a multiple of 10, so every key is a round number and the top key is just
        # above the largest whisker
        step = 10.0 * np.ceil(y_top / (N_YTICKS - 1) / 10.0)
        ticks = step * np.arange(N_YTICKS)
        for ax in axes:
            ax.set_yticks(ticks)
            ax.yaxis.set_major_formatter(FuncFormatter(_format_linear_tick))
        axes[0].set_ylim(0.0, ticks[-1] * 1.06)

    # Massive Rotated Super Title
    k = (11 * n_cols + 2) / fig.get_figwidth()  # keep the paper's margins in inches (k = 1 with 3 queries)
    fig.text((0.025 if log else -0.01) * k, 0.5, format_db_name(dataset), va='center', ha='center',
             rotation='vertical', fontsize=FONT_SIZE + 20, fontweight='bold')

    if legend if legend is not None else dataset == 'spitz':
        legend_elements = [Patch(facecolor=colors[m], label=labels.get(m, m), alpha=0.7)
                           for m in method_order if db_data.get(distributions[0], {}).get(m)]
        fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.78),
                   ncol=len(method_order), frameon=False, fontsize=FONT_SIZE + 20,
                   handlelength=1.0,
                   markerscale=1.0,
                   columnspacing=0.5,
                   handletextpad=0.2)

    plt.subplots_adjust(left=0.185 * k, bottom=0.3, top=0.75, wspace=0.1, right=1.0)
    fig.text((0.185 * k + 1) / 2, 0.12, 'Query Percent', va='center', ha='center', fontsize=FONT_SIZE + 30)

    filename = f'{dataset}{tag}_mse_distribution_boxplot.pdf'
    plt.savefig(os.path.join(output_dir, filename), format='pdf', bbox_inches='tight')
    plt.close(fig)