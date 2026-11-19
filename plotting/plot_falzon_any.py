import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter

from falzon import falzon_mse_distribution
from globals_vars import FONT_SIZE, format_db_name, format_y_axis
from plot_falzon import load_lama_mses

# ==========================================
# LAYOUT (all in inches), separate from plot_falzon.py
# ==========================================
PANEL_WIDTH = 10     # width of each individual histogram
PANEL_HEIGHT = 5     # height of each individual histogram
PANEL_GAP = 1.5      # gap between the LAMa panels (Gaussian -> Beta -> Uniform)
FALZON_GAP = 2.5     # gap between the last LAMa panel and the Falzon (Any Dist.) panel
FIG_WIDTH = None     # width of the entire row; if set, PANEL_WIDTH is ignored and the panels
                     # are stretched/shrunk so the row is exactly this wide

LEFT_MARGIN = 5.6    # room for the dataset name, y-label and y ticks
BOTTOM_MARGIN = 4.4  # room for the x ticks and MSE label
TOP_MARGIN = 2.75    # room for the method titles
NAME_X = 2.275       # dataset name, from the left edge
NAME_Y = 0.5         # dataset name, as a fraction of the figure height
LABEL_Y = 0.4        # "# of Solutions", as a fraction of the panel height
HEADER_LABEL_Y = 0.55  # "# of Solutions" on the header (Spitz) row, which has no long dataset
                       # name below it, so the label is raised to stay inside the saved area
DIST_TITLE_Y = 4.07  # distribution titles, above the top of the panels
HEADER_Y = 6.6       # "LAMa vs Falzon et al.", above the top of the panels
XLABEL_Y = 2.5       # "Mean Squared Error (MSE)", below the bottom of the panels
XLABEL_EVERY_ROW = False  # True: every dataset gets the big x-axis label
                          # False: only the row called with show_xlabel=True (the bottom row)


def plot_falzon_any_histograms(paths, grid, dataset, output_dir, colors, labels, show_header=False,
                               show_xlabel=False):
    """
    One row of MSE-by-all-reconstructions histograms: LAMa under each distribution, then a
    single Falzon et al. (CCS 2020) panel. Falzon et al.'s output E(D) does not depend on the
    query distribution (see falzon.py), so it gets one "Any Dist." column.

    paths:       {dist: (hist_json_path, recon_json_path)}, in left-to-right order.
    show_header: draw the method/distribution titles and super header (the top row).
    show_xlabel: draw the big x-axis label (the bottom row), see XLABEL_EVERY_ROW.
    """
    # (column title, method, MSEs)
    columns = []
    true_pts = None
    for dist, (hist_json_path, recon_json_path) in paths.items():
        if not os.path.exists(recon_json_path):
            print(f"[SKIP] LAMa reconstruction JSON not found: {recon_json_path}")
            columns.append((f'{dist.capitalize()} Dist.', 'limits', []))
            continue

        if true_pts is None:
            with open(recon_json_path, 'r') as f:
                true_pts = [p["true"] for p in json.load(f)[0]]

        columns.append((f'{dist.capitalize()} Dist.', 'limits',
                        load_lama_mses(hist_json_path, recon_json_path, labels['limits'])))

    if true_pts is None:
        return

    falzon_mses, r = falzon_mse_distribution(true_pts, grid)
    if 2 ** r > 2 ** 20:
        print(f"[WARN] {dataset}: Falzon et al. E(D) has 4 x 2^{r} databases, histogram is sampled")
    columns.append(('Any Dist.', 'falzon', falzon_mses.tolist()))

    def make_bins(mses):
        local_max_mse = max(mses)
        local_max_mse = 1.0 if local_max_mse <= 0.0 else local_max_mse * 1.05
        return np.linspace(0.0, local_max_mse, 7)

    # One y-limit for the whole row, so panels with the same data always get the same y-axis
    y_max = max([8] + [np.histogram(mses, bins=make_bins(mses))[0].max() for _, _, mses in columns if mses])

    n_cols = len(columns)
    gaps = (n_cols - 2) * PANEL_GAP + FALZON_GAP

    if FIG_WIDTH is None:
        panel_width = PANEL_WIDTH
        fig_width = LEFT_MARGIN + n_cols * panel_width + gaps
    else:
        fig_width = FIG_WIDTH
        panel_width = (fig_width - LEFT_MARGIN - gaps) / n_cols
        if panel_width <= 0:
            raise ValueError(f"FIG_WIDTH = {FIG_WIDTH} is too small for the margins and gaps")

    fig_height = BOTTOM_MARGIN + PANEL_HEIGHT + TOP_MARGIN
    panels_top = (BOTTOM_MARGIN + PANEL_HEIGHT) / fig_height

    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.set_layout_engine(None)

    x = LEFT_MARGIN
    for i, (title, method, mses) in enumerate(columns):
        if i == n_cols - 1:
            x += FALZON_GAP - PANEL_GAP

        ax = fig.add_axes([x / fig_width, BOTTOM_MARGIN / fig_height,
                           panel_width / fig_width, PANEL_HEIGHT / fig_height])

        if show_header:
            fig.text((x + panel_width / 2) / fig_width, panels_top + DIST_TITLE_Y / fig_height,
                     title, va='center', ha='center', fontsize=FONT_SIZE + 40)

        x += panel_width + PANEL_GAP

        if not mses:
            ax.set_axis_off()
            continue

        c = colors.get(method, '#000000')
        ax.hist(mses, bins=make_bins(mses), color=c, alpha=0.55, edgecolor=c, linewidth=2, rwidth=0.9)
        ax.set_ylim(0, y_max)

        if show_header:
            ax.set_title(f'{labels.get(method, method)}', fontsize=FONT_SIZE + 40)

        if i == 0:
            ax.set_ylabel('# of Solutions', fontsize=FONT_SIZE + 0,
                              y=HEADER_LABEL_Y if show_header else LABEL_Y)

        ax.tick_params(axis='x', labelsize=FONT_SIZE - 10)
        ax.tick_params(axis='y', labelsize=FONT_SIZE - 10)

        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))

        ax.yaxis.set_major_formatter(FuncFormatter(format_y_axis))
        ax.xaxis.set_major_formatter(FuncFormatter(format_y_axis))

        ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Massive Rotated Super Title
    fig.text(NAME_X / fig_width, NAME_Y, format_db_name(dataset), va='center', ha='center',
             rotation='vertical', fontsize=FONT_SIZE + 20, fontweight='bold')

    # One giant x-axis label under the whole row
    if show_xlabel or XLABEL_EVERY_ROW:
        fig.text((LEFT_MARGIN + fig_width) / 2 / fig_width, (BOTTOM_MARGIN - XLABEL_Y) / fig_height,
                 'Mean Squared Error (MSE)', va='center', ha='center', fontsize=FONT_SIZE + 50)

    if show_header:
        fig.text((LEFT_MARGIN + fig_width) / 2 / fig_width, panels_top + HEADER_Y / fig_height,
                 f"{labels['limits']} vs Falzon et al.", va='center', ha='center',
                 fontsize=FONT_SIZE + 60, fontweight='bold')

    filename = f'falzon_any_histogram_{dataset}.pdf'
    plt.savefig(os.path.join(output_dir, filename), format='pdf', bbox_inches='tight')
    plt.close(fig)
