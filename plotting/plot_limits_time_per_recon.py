import os
import matplotlib.pyplot as plt
from globals_vars import FONT_SIZE


def plot_limits_time_per_recon(time_data, dataset, distributions, output_dir, dist_colors, dist_markers):
    db_data = time_data.get(dataset, {})
    if not db_data:
        return

    # Single plot per dataset
    fig, ax = plt.subplots(figsize=(11, 11))
    fig.set_layout_engine(None)

    lines_for_legend = []
    labels_for_legend = []

    # Hardcoded to only evaluate LAMa/limits
    method = "limits"

    for dist in distributions:
        dist_data = db_data.get(dist, {})
        method_runs = dist_data.get(method, [])
        if not method_runs:
            continue

        method_runs.sort(key=lambda x: x[0])
        x_vals = [r[0] for r in method_runs]

        # Calculate time per reconstruction
        y_vals = [(max(0.0001, r[1]) / 10001.0) for r in method_runs]

        c = dist_colors.get(dist, '#000000')
        m = dist_markers.get(dist, 'o')
        l = dist.capitalize()

        line, = ax.plot(x_vals, y_vals, marker=m, linestyle='-', label=l, alpha=0.9, color=c, zorder=3)

        lines_for_legend.append(line)
        labels_for_legend.append(l)

    if not lines_for_legend:
        plt.close(fig)
        return

    ax.set_ylabel('Time per Recon. (s)', fontsize=FONT_SIZE + 20)
    ax.set_xlabel('Query Percent', fontsize=FONT_SIZE + 20)

    # Force strict 10-20-30 axis labels
    ax.set_xticks([10, 20, 30])
    ax.set_xlim(9.5, 30.5)
    ax.set_xticklabels(['10%', '20%', '30%'], fontsize=FONT_SIZE - 10)
    ax.tick_params(axis='y', labelsize=FONT_SIZE - 10)

    ax.set_title(f'{dataset.capitalize()} - LAMa', fontsize=FONT_SIZE + 30)
    ax.grid(True, zorder=0)

    # Legend at the bottom using Distribution mapped handles
    fig.legend(lines_for_legend, labels_for_legend,
               loc='lower center', bbox_to_anchor=(0.5, -0.1),
               ncol=len(distributions), frameon=False, fontsize=FONT_SIZE)

    # Spacing tweaked for a single plot canvas
    plt.subplots_adjust(left=0.25, bottom=0.2, top=0.85, right=0.95)

    filename = f'{dataset}_limits_time_per_recon.pdf'
    plt.savefig(os.path.join(output_dir, filename), format='pdf', bbox_inches='tight')
    plt.close(fig)