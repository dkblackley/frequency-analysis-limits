import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter

from data_loader import calculate_mse
from falzon import falzon_mse_distribution
from globals_vars import FONT_SIZE, format_db_name, format_y_axis

# ==========================================
# LAYOUT (all in inches)
# ==========================================
PANEL_WIDTH = 10    # width of each individual histogram
PANEL_HEIGHT = 5  # height of each individual histogram
PAIR_GAP = 1      # gap between the LAMa and Falzon panels of one distribution
DIST_GAP = 1.5       # gap between distributions (e.g. Gaussian Falzon -> Beta LAMa)
FIG_WIDTH = None     # width of the entire row; if set, PANEL_WIDTH is ignored and the panels
                     # are stretched/shrunk so the row is exactly this wide

LEFT_MARGIN = 5.6    # room for the dataset name, y-label and y ticks
BOTTOM_MARGIN = 4.4  # room for the x ticks and MSE label
TOP_MARGIN = 2.75    # room for the method titles
NAME_X = 2.275       # dataset name, from the left edge
NAME_Y = 0.5
LABEL_Y = 0.4
DIST_TITLE_Y = 4.07  # distribution titles, above the top of the panels
HEADER_Y = 6.6       # "LAMa vs Falzon et al.", above the top of the panels


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


def plot_falzon_histograms(paths, grid, dataset, output_dir, colors, labels, show_header=False):
    """
    One row of MSE-by-all-reconstructions histograms: a (LAMa, Falzon et al.) pair per
    distribution, in the same style as plot_reconstruction_histograms. Falzon et al.'s output
    E(D) is built directly from the true database (see falzon.py) instead of re-running
    their attack.

    paths:       {dist: (hist_json_path, recon_json_path)}, in left-to-right order.
    show_header: draw the method/distribution titles and super header (the top row).
    """
    plot_methods = ['limits', 'falzon']

    mse_data = {}
    for dist, (hist_json_path, recon_json_path) in paths.items():
        if not os.path.exists(recon_json_path):
            print(f"[SKIP] LAMa reconstruction JSON not found: {recon_json_path}")
            continue

        with open(recon_json_path, 'r') as f:
            true_pts = [p["true"] for p in json.load(f)[0]]

        falzon_mses, r = falzon_mse_distribution(true_pts, grid)
        if 2 ** r > 2 ** 20:
            print(f"[WARN] {dataset} {dist}: Falzon et al. E(D) has 4 x 2^{r} databases, histogram is sampled")

        mse_data[dist] = {
            'limits': load_lama_mses(hist_json_path, recon_json_path, labels['limits']),
            'falzon': falzon_mses.tolist(),
        }

    if not mse_data:
        return

    def make_bins(mses):
        local_max_mse = max(mses)
        local_max_mse = 1.0 if local_max_mse <= 0.0 else local_max_mse * 1.05
        return np.linspace(0.0, local_max_mse, 7)

    # One y-limit for the whole row, so panels with the same data (e.g. LAMa and Falzon et
    # al. under uniform queries) always get the same y-axis
    y_max = max([8] + [np.histogram(mses, bins=make_bins(mses))[0].max()
                       for methods in mse_data.values() for mses in methods.values() if mses])

    n_dists = len(paths)
    n_methods = len(plot_methods)
    gaps = n_dists * (n_methods - 1) * PAIR_GAP + (n_dists - 1) * DIST_GAP

    if FIG_WIDTH is None:
        panel_width = PANEL_WIDTH
        fig_width = LEFT_MARGIN + n_dists * n_methods * panel_width + gaps
    else:
        fig_width = FIG_WIDTH
        panel_width = (fig_width - LEFT_MARGIN - gaps) / (n_dists * n_methods)
        if panel_width <= 0:
            raise ValueError(f"FIG_WIDTH = {FIG_WIDTH} is too small for the margins and gaps")

    fig_height = BOTTOM_MARGIN + PANEL_HEIGHT + TOP_MARGIN
    panels_top = (BOTTOM_MARGIN + PANEL_HEIGHT) / fig_height

    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.set_layout_engine(None)

    def add_panel(x):
        return fig.add_axes([x / fig_width, BOTTOM_MARGIN / fig_height,
                             panel_width / fig_width, PANEL_HEIGHT / fig_height])

    for d, dist in enumerate(paths):
        pair_x = LEFT_MARGIN + d * (n_methods * panel_width + (n_methods - 1) * PAIR_GAP + DIST_GAP)
        pair_width = n_methods * panel_width + (n_methods - 1) * PAIR_GAP
        axes = [add_panel(pair_x + m * (panel_width + PAIR_GAP)) for m in range(n_methods)]

        if dist not in mse_data:
            for ax in axes:
                ax.set_axis_off()
            continue

        for ax, method in zip(axes, plot_methods):
            mses = mse_data[dist][method]
            c = colors.get(method, '#000000')

            if not mses:
                continue

            ax.hist(mses, bins=make_bins(mses), color=c, alpha=0.55, edgecolor=c, linewidth=2, rwidth=0.9)
            ax.set_ylim(0, y_max)

            if show_header:
                ax.set_title(f'{labels.get(method, method)}', fontsize=FONT_SIZE + 40)

            if d == 0 and ax == axes[0]:
                ax.set_ylabel('# of Solutions', fontsize=FONT_SIZE + 0, y=LABEL_Y)

            ax.tick_params(axis='x', labelsize=FONT_SIZE - 10)
            ax.tick_params(axis='y', labelsize=FONT_SIZE - 10)

            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))

            ax.yaxis.set_major_formatter(FuncFormatter(format_y_axis))
            ax.xaxis.set_major_formatter(FuncFormatter(format_y_axis))

            ax.grid(axis='y', linestyle='--', alpha=0.7)

        # One MSE label per pair, centred under the gap between its two panels
        axes[0].set_xlabel('MSE', fontsize=FONT_SIZE + 30)
        axes[0].xaxis.label.set_x(pair_width / 2 / panel_width)

        if show_header:
            fig.text((pair_x + pair_width / 2) / fig_width, panels_top + DIST_TITLE_Y / fig_height,
                     f'{dist.capitalize()} Dist.', va='center', ha='center', fontsize=FONT_SIZE + 40)


    # Massive Rotated Super Title
    fig.text(NAME_X / fig_width, NAME_Y, format_db_name(dataset), va='center', ha='center',
             rotation='vertical', fontsize=FONT_SIZE + 20, fontweight='bold')

    if show_header:
        fig.text((LEFT_MARGIN + fig_width) / 2 / fig_width, panels_top + HEADER_Y / fig_height,
                 f"{labels['limits']} vs Falzon et al.", va='center', ha='center',
                 fontsize=FONT_SIZE + 60, fontweight='bold')

    filename = f'falzon_histogram_{dataset}.pdf'
    plt.savefig(os.path.join(output_dir, filename), format='pdf', bbox_inches='tight')
    plt.close(fig)
