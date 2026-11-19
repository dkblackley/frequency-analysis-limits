import os
import random
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from globals_vars import FONT_SIZE, format_db_name


def plot_convex_mse(convex_data, dataset, distributions, output_dir, method_order, colors, markers, labels,
                    grid_tag="", xticks=(10, 20, 30), ylim=(-2, 85), yticks=(0, 40, 80), xscale="linear"):
    """xticks: query percents on the x-axis. ylim/yticks: None = matplotlib's automatic range
    (the defaults are the 15x15 paper figure; 50x50 MSEs are on a larger scale). xscale="category" puts
    the query percents at evenly spaced positions, so 1/2/5% stay readable next to 10-30%."""
    db_data = convex_data.get(dataset, {})
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

            method_runs.sort(key=lambda x: x[0])
            x_vals = [r[0] for r in method_runs]
            if xscale == "category":
                x_vals = [list(xticks).index(q) for q in x_vals]
            cent_y = [r[1] for r in method_runs]
            diam_y = [r[2] for r in method_runs]

            c = colors.get(method, '#000000')
            m = markers.get(method, 'o')
            l = labels.get(method, method)

            z_cent = 3
            z_diam = 2
            lw = 8.0  # Matplotlib's default line width

            # Pull "even_less" to the front and make it thinner
            if method == "even_less":
                z_cent = 10
                z_diam = 9
                lw = lw - 2

            # Calculate a darker version of the base color 'c'
            # 0.7 means it will be 30% darker. Adjust this decimal to tune the darkness.
            r, g, b = mcolors.to_rgb(c)
            darken_factor = 0.99
            c_darker = (r * darken_factor, g * darken_factor, b * darken_factor)

            # Solid line (uses original color 'c')
            line1, = ax.plot(x_vals, cent_y, marker=m, linestyle='-', label=l, color=c,
                             zorder=(z_cent-1),  linewidth=lw, alpha=0.9)

            # Dashed line and its marker (uses the new darker color 'c_darker')
            ax.plot(x_vals, diam_y, marker=m, linestyle='--', color=c_darker,
                    zorder=(z_diam*2), linewidth=lw)

            if l not in labels_for_legend:
                lines_for_legend.append(line1)
                labels_for_legend.append(l)

        # ax.set_xlabel('Query Percent')  # Removed to use global label
        if ax == axes[0]:
            ax.set_ylabel('MSE', fontsize=FONT_SIZE + 30)

        # Fixed query-percent axis labels
        if xscale == "category":
            ax.set_xlim(-0.3, len(xticks) - 0.7)
            ax.set_xticks(list(range(len(xticks))))
        else:
            ax.set_xlim(min(xticks) - 1, max(xticks) + 1)
            ax.set_xticks(list(xticks))
        if len(xticks) > 3:  # many ticks: no % sign (the axis title says Query Percent), smaller font
            ax.set_xticklabels([f'{q:g}' for q in xticks], fontsize=FONT_SIZE - 16)
        else:
            ax.set_xticklabels([f'{q:g}%' for q in xticks], fontsize=FONT_SIZE - 10)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if yticks is not None:
            ax.set_yticks(list(yticks))
        ax.tick_params(axis='y', labelsize=FONT_SIZE - 10, pad=40)

        ax.set_title(f'{dist.capitalize()}', fontsize=FONT_SIZE + 40)  # Removed " Dist."
        ax.grid(True, zorder=0)

    # Replicate the Massive Super Title across the entire width
    fig.text(0.025, 0.5, format_db_name(dataset), va='center', ha='center',
                 rotation='vertical', fontsize=FONT_SIZE + 20, fontweight='bold')
    if dataset == "spitz" or dataset == "highway" or dataset == "busstop":
        fig.legend(lines_for_legend, labels_for_legend,
                   loc='lower center', bbox_to_anchor=(0.5, 0.78),
                   ncol=len(method_order), frameon=False, fontsize=FONT_SIZE + 20,
                   handlelength=1.0,
                   markerscale=1.0,
                   columnspacing=0.5,
                   handletextpad=0.2)

    # (Left, Bottom, Top, Wspace) spacing tweaked to fit the massive labels
    # this squeezes the individual plots
    plt.subplots_adjust(left=0.16, bottom=0.3, top=0.75, wspace=0.2, right=1.0)

    # Add the single, centered global x-label
    fig.text(0.55, 0.12, 'Query Percent', va='center', ha='center', fontsize=FONT_SIZE + 30)

    filename = f'{dataset}_convex_mse_combined{grid_tag}.pdf'
    plt.savefig(os.path.join(output_dir, filename), format='pdf', bbox_inches='tight')
    plt.close(fig)