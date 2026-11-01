import json
import logging
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from utils import set_sleek_style, format_metric, PALETTE, calculate_mse
from falzon import falzon_mse_distribution

logger = logging.getLogger(__name__)


def plot_histograms(mse_data, db_name, dist_name, output_path, methods=None, share_range=False):
    set_sleek_style()

    # Order matches your rust logic
    if methods is None:
        methods = [("LAMa", PALETTE["blue"]), ("Even Less", PALETTE["orange"]), ("Remin", PALETTE["green"])]

    # Same x-range on every panel, so the panels can be compared bar for bar
    shared_max = None
    if share_range:
        shared_max = max(np.max(mse_data[m]) for m, _ in methods if m in mse_data) * 1.05
        shared_max = shared_max if shared_max > 0 else 1.0

    # constrained layout automatically prevents title/legend overlapping
    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 6), layout='constrained')
    fig.suptitle(f"{db_name.capitalize()} {dist_name.capitalize()} Distribution")

    formatter = FuncFormatter(format_metric)

    for ax, (method, color) in zip(axes, methods):
        if method not in mse_data: continue

        mses = np.array(mse_data[method])
        max_mse = max(1.0, mses.max() * 1.05) if mses.max() <= 0 else mses.max() * 1.05
        if shared_max is not None: max_mse = shared_max

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


def load_limits_mses(file_path):
    """MSE of every LAMa reconstruction, unaligned, exactly like the Rust calculate_mse."""
    with open(file_path, 'r') as f:
        all_data = json.load(f)

    true_pts = np.array([pt["true"] for pt in all_data[0]])
    mses = [calculate_mse(np.array([pt["true"] for pt in recon]),
                          np.array([pt["reconstructed"] for pt in recon])) for recon in all_data]
    return np.array(mses), true_pts


def plot_lama_vs_falzon_histograms(name, domain, dist, output_path, base_dir="../databases"):
    """
    MSE-by-all-reconstructions histogram with only LAMa and Falzon et al. (CCS 2020).
    Falzon et al.'s output E(D) is built directly from the true database (see falzon.py)
    instead of re-running their attack. E(D) does not depend on the query distribution.
    """
    grid = f"{domain[0]}x{domain[1]}"
    path = f"{base_dir}/{grid}/{name}/limits/{name}_{dist}_e0_d0.9_reconstruction.json"
    logger.debug(f"About to load data from {path}")

    limits_mse, true_pts = load_limits_mses(path)
    falzon_mse, r = falzon_mse_distribution(true_pts, domain)

    logger.info(f"{name} {dist}: LAMa returned {len(limits_mse)} solutions, Falzon et al. E(D) has "
                f"4 x 2^{r} = {4 * 2 ** r} (r = {r} reflectable components)")
    if dist == "uniform" and len(limits_mse) != 4 * 2 ** r:
        # Under uniform queries and exact frequencies the two sets should coincide. A larger
        # LAMa set usually means an extra seed database (Theorem 4.4) that falzon.py omits.
        logger.warning(f"{name}: LAMa solution count differs from 4 x 2^r")

    mse_data = {"LAMa": limits_mse, "Falzon et al.": falzon_mse}
    methods = [("LAMa", PALETTE["blue"]), ("Falzon et al.", PALETTE["purple"])]
    plot_histograms(mse_data, name, dist, output_path, methods=methods, share_range=True)
