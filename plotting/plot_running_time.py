import os
import matplotlib.pyplot as plt

# Assuming FONT_SIZE is imported from your global definitions
from globals_vars import FONT_SIZE


def plot_running_time_bar(time_data, dataset, distributions, output_dir, method_order, colors, labels):
    """
    Plots a grouped bar chart for running times across multiple distributions.
    Expects time_data formatted as: time_data[dataset][distribution][method] = time_in_seconds
    """
    db_data = time_data.get(dataset, {})
    if not db_data:
        return

    n_cols = len(distributions)
    # Maintain the massive canvas size
    fig, axes = plt.subplots(1, n_cols, figsize=(11 * n_cols + 2, 11), sharey=True)
    if n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    fig.set_layout_engine(None)

    bars_for_legend = []
    labels_for_legend = []

    for ax, dist in zip(axes, distributions):
        dist_data = db_data.get(dist, {})

        # Plot a bar for each method
        for i, method in enumerate(method_order):
            time_val = dist_data.get(method, 0.0)
            if time_val == 0.0:
                continue

            c = colors.get(method, '#000000')
            l = labels.get(method, method)

            # Draw the bar (edgecolor and linewidth add a nice crisp border for accessibility)
            bar = ax.bar(i, time_val, color=c, zorder=3, alpha=0.9, edgecolor='black', linewidth=3)

            # Store the first instance of each bar for the global legend
            if l not in labels_for_legend:
                bars_for_legend.append(bar[0])
                labels_for_legend.append(l)

        if ax == axes[0]:
            ax.set_ylabel('Time Taken (s)', fontsize=FONT_SIZE + 30)

        # Log scale is absolutely required here due to massive 1.7s vs 26,000s variance
        ax.set_yscale('log')

        # Remove the x-ticks completely since the global legend handles method identification
        ax.set_xticks([])
        ax.tick_params(axis='y', labelsize=FONT_SIZE - 10)

        ax.set_title(f'{dist.capitalize()}', fontsize=FONT_SIZE + 40)

        # Add horizontal grid lines behind the bars for easier reading of log scales
        ax.grid(True, axis='y', zorder=0, linestyle='--', alpha=0.7)

    # Replicate the Massive Super Title across the entire width (Rotated on the left)
    fig.text(0.05, 0.55, dataset.capitalize(), va='center', ha='center',
             rotation='vertical', fontsize=FONT_SIZE + 20, fontweight='bold')

    # Single shared legend at the bottom
    fig.legend(bars_for_legend, labels_for_legend,
               loc='lower center', bbox_to_anchor=(0.45, 0.0),
               ncol=len(method_order), frameon=False, fontsize=FONT_SIZE)

    # (Left, Bottom, Top, Wspace, Right) spacing tweaked to fit the massive labels
    # Wspace reduced so the distributions sit nicely next to each other
    plt.subplots_adjust(left=0.17, bottom=0.20, top=0.75, wspace=0.1, right=0.95)

    filename = f'{dataset}_running_time_combined.pdf'
    plt.savefig(os.path.join(output_dir, filename), format='pdf', bbox_inches='tight')
    plt.close(fig)