import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter

from data_loader import calculate_mse
from falzon import falzon_mse_distribution
from globals_vars import FONT_SIZE, format_db_name, format_y_axis


def load_lama_mses(hist_json_path, recon_json_path, label):
    """
    LAMa MSEs, taken from the same histogram JSON plot_reconstruction_histograms uses so the
    LAMa panel is identical. Falls back to the reconstruction JSON the Rust code builds that
    entry from (unaligned calculate_mse of every reconstruction).
    """
    if os.path.exists(hist_json_path):
        with open(hist_json_path, 'r') as f:
            mses = json.load(f).get(label, [[], 0.0])[0]
        if mses:
            return mses

    with open(recon_json_path, 'r') as f:
        all_data = json.load(f)

    return [calculate_mse([p["true"] for p in run], [p["reconstructed"] for p in run])
            for run in all_data if run]


def plot_falzon_histograms(hist_json_path, recon_json_path, grid, dataset, dist, output_dir, colors, labels):
    """
    MSE-by-all-reconstructions histogram with only LAMa and Falzon et al. (CCS 2020), in the
    same style as plot_reconstruction_histograms. Falzon et al.'s output E(D) is built
    directly from the true database (see falzon.py) instead of re-running their attack.
    """
    if not os.path.exists(recon_json_path):
        print(f"[SKIP] LAMa reconstruction JSON not found: {recon_json_path}")
        return

    with open(recon_json_path, 'r') as f:
        true_pts = [p["true"] for p in json.load(f)[0]]

    falzon_mses, r = falzon_mse_distribution(true_pts, grid)
    if 2 ** r > 2 ** 20:
        print(f"[WARN] {dataset} {dist}: Falzon et al. E(D) has 4 x 2^{r} databases, histogram is sampled")

    mse_data = {
        'limits': load_lama_mses(hist_json_path, recon_json_path, labels['limits']),
        'falzon': falzon_mses.tolist(),
    }
    plot_methods = ['limits', 'falzon']
    n_cols = len(plot_methods)

    # Keep the same absolute margins (in inches) as the 3-column histograms
    fig_width = 11 * n_cols + 2
    ref_width = 11 * 3 + 2
    left = 0.16 * ref_width / fig_width
    name_x = 0.08 * ref_width / fig_width

    fig, axes = plt.subplots(1, n_cols, figsize=(fig_width, 11), sharey=False)
    axes = axes.flatten()

    fig.set_layout_engine(None)

    for ax, method in zip(axes, plot_methods):
        method_key = labels.get(method, method)
        mses = mse_data[method]
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
        # No middle column with two panels, so label both x-axes
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
    fig.text(name_x, 0.65, format_db_name(dataset), va='center', ha='center',
             rotation='vertical', fontsize=FONT_SIZE + 20, fontweight='bold')

    # Centered top title for the current distribution
    fig.text((left + 1.0) / 2, 1.0, f'{dist.capitalize()} Dist.', va='center', ha='center',
             fontsize=FONT_SIZE + 40)

    plt.subplots_adjust(left=left, bottom=0.4, top=0.75, wspace=0.3, right=1.0)

    filename = f'falzon_histogram_{dataset}_{dist}.pdf'
    plt.savefig(os.path.join(output_dir, filename), format='pdf', bbox_inches='tight')
    plt.close(fig)
