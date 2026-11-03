import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter

from falzon import falzon_mse_distribution
from globals_vars import FONT_SIZE, format_db_name, format_y_axis
from plot_falzon import load_lama_mses

# ==========================================
# LAYOUT (all in inches), separate from plot_falzon.py / plot_falzon_any.py
# ==========================================
PANEL_WIDTH = 10     # width of each individual histogram
PANEL_HEIGHT = 5     # height of each individual histogram
PAIR_GAP = 1.5       # gap between the LAMa and Falzon panels inside one group
GROUP_GAP = 3.0      # gap between the "With Guarantees" and "Without Guarantees" groups
ROW_GAP = 2.75        # vertical gap between dataset rows
FIG_WIDTH = None     # width of the entire figure; if set, PANEL_WIDTH is ignored and the panels
                     # are stretched/shrunk so the figure is exactly this wide

LEFT_MARGIN = 5.6    # room for the dataset name, y-label and y ticks
BOTTOM_MARGIN = 4.4  # room for the x ticks and MSE label
TOP_MARGIN = 2.75    # room for the titles
NAME_X = 2.275       # dataset name, from the left edge
LABEL_Y = 0.4        # "# of Solutions", as a fraction of the panel height
GROUP_TITLE_Y = 4.07 # "With / Without Guarantees", above the top of the top row
HEADER_Y = 6.6       # big bold header, above the top of the top row
XLABEL_Y = 2.5       # "Mean Squared Error (MSE)", below the bottom of the bottom row

# (group title, with guarantees?) left to right
GROUPS = [('With Guarantees', True), ('Without Guarantees', False)]
PLOT_METHODS = ['limits', 'falzon']

# Falzon et al.'s E(D) does not depend on the query distribution, so compute it once per dataset
_falzon_cache = {}


def _falzon_mses(dataset, recon_json_path, grid):
    key = (dataset, tuple(grid))
    if key not in _falzon_cache:
        with open(recon_json_path, 'r') as f:
            true_pts = [p["true"] for p in json.load(f)[0]]
        mses, r = falzon_mse_distribution(true_pts, grid)
        if 2 ** r > 2 ** 20:
            print(f"[WARN] {dataset}: Falzon et al. E(D) has 4 x 2^{r} databases, histogram is sampled")
        _falzon_cache[key] = mses.tolist()
    return _falzon_cache[key]


def panel_mses(dist, method, guarantees, lama_mses, falzon_mses):
    """
    Which MSEs go in which panel.
      LAMa:   always its own reconstructions under `dist`.
      Falzon: with guarantees -> E(D), the same for every distribution (as under uniform).
              without guarantees -> performs like LAMa under a non-uniform distribution;
              under uniform it is E(D) in both groups.
    """
    if method == 'limits':
        return lama_mses
    if guarantees or dist == 'uniform':
        return falzon_mses
    return lama_mses


def plot_falzon_guarantees(paths, grid, dist, output_dir, colors, labels):
    """
    One figure per distribution. Each row is a dataset, with two groups of (LAMa, Falzon et al.)
    MSE-by-all-reconstructions histograms: "With Guarantees" and "Without Guarantees".
    The top row carries the method/group titles and the big header, the bottom row the x label.

    paths: {dataset: (hist_json_path, recon_json_path)} for this distribution, top to bottom.
    grid:  (G0, G1) target grid, see falzon_mse_distribution.
    """
    rows = []  # (dataset, [[mses per method] per group])
    for dataset, (hist_json_path, recon_json_path) in paths.items():
        if not os.path.exists(recon_json_path):
            print(f"[SKIP] LAMa reconstruction JSON not found: {recon_json_path}")
            continue

        lama = load_lama_mses(hist_json_path, recon_json_path, labels['limits'])
        falzon = _falzon_mses(dataset, recon_json_path, grid)
        panels = [[panel_mses(dist, method, guarantees, lama, falzon) for method in PLOT_METHODS]
                  for _, guarantees in GROUPS]
        rows.append((dataset, panels))

    if not rows:
        return

    def make_bins(mses):
        local_max_mse = max(mses)
        local_max_mse = 1.0 if local_max_mse <= 0.0 else local_max_mse * 1.05
        return np.linspace(0.0, local_max_mse, 7)

    n_rows = len(rows)
    n_groups = len(GROUPS)
    n_methods = len(PLOT_METHODS)
    gaps = n_groups * (n_methods - 1) * PAIR_GAP + (n_groups - 1) * GROUP_GAP

    if FIG_WIDTH is None:
        panel_width = PANEL_WIDTH
        fig_width = LEFT_MARGIN + n_groups * n_methods * panel_width + gaps
    else:
        fig_width = FIG_WIDTH
        panel_width = (fig_width - LEFT_MARGIN - gaps) / (n_groups * n_methods)
        if panel_width <= 0:
            raise ValueError(f"FIG_WIDTH = {FIG_WIDTH} is too small for the margins and gaps")

    fig_height = BOTTOM_MARGIN + n_rows * PANEL_HEIGHT + (n_rows - 1) * ROW_GAP + TOP_MARGIN
    panels_top = (fig_height - TOP_MARGIN) / fig_height

    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.set_layout_engine(None)

    group_width = n_methods * panel_width + (n_methods - 1) * PAIR_GAP

    for row, (dataset, panels) in enumerate(rows):
        row_bottom = BOTTOM_MARGIN + (n_rows - 1 - row) * (PANEL_HEIGHT + ROW_GAP)
        is_top = row == 0

        # One y-limit per row, so panels with the same data always get the same y-axis
        y_max = max([8] + [np.histogram(mses, bins=make_bins(mses))[0].max()
                           for group in panels for mses in group if mses])

        for g, (group_title, _) in enumerate(GROUPS):
            group_x = LEFT_MARGIN + g * (group_width + GROUP_GAP)

            if is_top:
                fig.text((group_x + group_width / 2) / fig_width, panels_top + GROUP_TITLE_Y / fig_height,
                         group_title, va='center', ha='center', fontsize=FONT_SIZE + 40)

            for m, method in enumerate(PLOT_METHODS):
                x = group_x + m * (panel_width + PAIR_GAP)
                ax = fig.add_axes([x / fig_width, row_bottom / fig_height,
                                   panel_width / fig_width, PANEL_HEIGHT / fig_height])
                mses = panels[g][m]

                if not mses:
                    ax.set_axis_off()
                    continue

                c = colors.get(method, '#000000')
                ax.hist(mses, bins=make_bins(mses), color=c, alpha=0.55, edgecolor=c, linewidth=2, rwidth=0.9)
                ax.set_ylim(0, y_max)

                if is_top:
                    ax.set_title(f'{labels.get(method, method)}', fontsize=FONT_SIZE + 40)

                if g == 0 and m == 0:
                    ax.set_ylabel('# of Solutions', fontsize=FONT_SIZE - 5, y=LABEL_Y+0.1)

                ax.tick_params(axis='x', labelsize=FONT_SIZE + 10)
                ax.tick_params(axis='y', labelsize=FONT_SIZE + 5)

                ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))

                ax.yaxis.set_major_formatter(FuncFormatter(format_y_axis))
                ax.xaxis.set_major_formatter(FuncFormatter(format_y_axis))

                ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Rotated dataset name for this row
        fig.text(NAME_X / fig_width, (row_bottom + PANEL_HEIGHT / 2) / fig_height, format_db_name(dataset),
                 va='center', ha='center', rotation='vertical', fontsize=FONT_SIZE + 0, fontweight='bold')

    # One giant x-axis label under the bottom row
    fig.text((LEFT_MARGIN + fig_width) / 2 / fig_width, (BOTTOM_MARGIN - XLABEL_Y) / fig_height,
             'Mean Squared Error (MSE)', va='center', ha='center', fontsize=FONT_SIZE + 50)

    # Big bold header over the top row
    fig.text((LEFT_MARGIN + fig_width) / 2 / fig_width, panels_top + HEADER_Y / fig_height,
             f"{labels['limits']} vs Falzon ({dist.capitalize()} Dist.)", va='center', ha='center',
             fontsize=FONT_SIZE + 60, fontweight='bold')

    filename = f'falzon_guarantees_{dist}.pdf'
    plt.savefig(os.path.join(output_dir, filename), format='pdf', bbox_inches='tight')
    plt.close(fig)
