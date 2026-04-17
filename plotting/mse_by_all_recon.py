import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from utils import set_sleek_style, format_metric, PALETTE


def plot_histograms(mse_data, db_name, dist_name, output_path):
    set_sleek_style()

    # Order matches your rust logic
    methods = [("LAMa", PALETTE["blue"]), ("Even Less", PALETTE["orange"]), ("Remin", PALETTE["green"])]

    # constrained layout automatically prevents title/legend overlapping
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), layout='constrained')
    fig.suptitle(f"{db_name.capitalize()} {dist_name.capitalize()} Distribution")

    formatter = FuncFormatter(format_metric)

    for ax, (method, color) in zip(axes, methods):
        if method not in mse_data: continue

        mses = np.array(mse_data[method])
        max_mse = max(1.0, mses.max() * 1.05) if mses.max() <= 0 else mses.max() * 1.05

        # Plot histogram bars
        counts, bins, patches = ax.hist(mses, bins=8, range=(0, max_mse),
                                        color=color, alpha=0.7, edgecolor=color, linewidth=2)

        ax.set_title(method)
        ax.set_xlabel("Mean Squared Error (MSE)")
        if ax is axes[0]: ax.set_ylabel("Number of Solutions")

        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        ax.grid(axis='x')  # Disable vertical grid lines for cleaner look

    plt.savefig(output_path, format='svg', transparent=True)
    plt.close()