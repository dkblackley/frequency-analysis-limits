import os
import json
import numpy as np
import matplotlib.pyplot as plt
from globals_vars import FONT_SIZE, format_db_name

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from globals_vars import FONT_SIZE, format_db_name


def plot_spatial_reconstruction(json_path, dataset, dist, output_dir, method_order, colors, labels):
    if not os.path.exists(json_path):
        print(f"[SKIP] Spatial JSON not found: {json_path}")
        return

    with open(json_path, 'r') as f:
        export_data = json.load(f)

    true_coords = export_data.get("true_coords", [])
    method_coords = export_data.get("method_coords", {})

    plot_methods = [m for m in method_order if m in method_coords]
    n_cols = len(plot_methods)

    if n_cols == 0:
        return

    SCALE = 0.33

    fig, axes = plt.subplots(1, n_cols,
                             figsize=((11 * n_cols + 2) * SCALE, 11 * SCALE),
                             sharey=True, sharex=True)
    if n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    fig.set_layout_engine(None)


    # 1. Calculate global boundaries for consistent scaling across subplots
    all_pts = list(true_coords)
    for pts in method_coords.values():
        all_pts.extend(pts)

    all_pts = np.array(all_pts)
    if len(all_pts) > 0:
        min_x, max_x = all_pts[:, 0].min(), all_pts[:, 0].max()
        min_y, max_y = all_pts[:, 1].min(), all_pts[:, 1].max()
    else:
        min_x, max_x, min_y, max_y = 0, 1, 0, 1

    edge_margin_x = max((max_x - min_x) * 0.05, 1.0)
    edge_margin_y = max((max_y - min_y) * 0.05, 1.0)

    # 2. Render Subplots
    for ax, method in zip(axes, plot_methods):
        c = colors.get(method, '#000000')
        method_name = labels.get(method, method)
        recon_pts = np.array(method_coords[method])
        true_pts = np.array(true_coords)

        if method_name == "Remin":
            method_name = "Remin+"
            c = colors.get("reminp", "#000000")
        gt_color = '#DCDCDC'  # Light grey for ground truth
        # Plot Ground Truth
        if len(true_pts) > 0:
            ax.scatter(true_pts[:, 0], true_pts[:, 1], c="whitesmoke", s=(20*4)*SCALE, alpha=0.85,
                       label="Ground Truth", zorder=1, edgecolors='gainsboro', linewidths=1, rasterized=True)

        # Plot Reconstructednnnnnnnnnnnhj

        if len(recon_pts) > 0:
            ax.scatter(recon_pts[:, 0], recon_pts[:, 1], c=c, s=20*SCALE, alpha=1.0,
            label="Reconstructed", zorder=2, edgecolors=c, linewidths=1, rasterized=True)

        ax.set_title(method_name, fontsize=(FONT_SIZE + 30)*SCALE, pad=20*SCALE)

        ax.set_xlim(min_x - edge_margin_x, max_x + edge_margin_x)
        ax.set_ylim(min_y - edge_margin_y, max_y + edge_margin_y)

        # ax.tick_params(axis='both', labelsize=FONT_SIZE - 15)
        # ax.grid(True, linestyle='--', alpha=0.5, zorder=0)

        ax.grid(False)
        ax.axis('off')

        legend = ax.legend(
            loc='upper right',
            frameon=True,
            fontsize=(FONT_SIZE - 40)*SCALE,
            handletextpad=-0.5,  # Reduces gap between the point and the text
            borderpad=0.1,  # Reduces padding between the words and the legend border
            labelspacing=0.1  # Reduces vertical spacing between multiple entries
        )

        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        legend.get_frame().set_edgecolor('black')

    # # 3. Super Titles
    # fig.text(0.045, 0.5, format_db_name(dataset), va='center', ha='center',
    #          rotation='vertical', fontsize=FONT_SIZE + 20, fontweight='bold')

    fig.text(0.55, 1.2, f'{format_db_name(dataset)} - {dist.capitalize()} Distribution', va='center', ha='center',
             fontsize=(FONT_SIZE + 40)*SCALE, fontweight='bold')

    plt.subplots_adjust(left=0.0, bottom=0.45, top=1.0, wspace=-0.065, right=1.0)

    filename = f'{dataset}_{dist}_reconstruction.pdf'
    plt.savefig(os.path.join(output_dir, filename), format='pdf', bbox_inches='tight', dpi=400)
    plt.close(fig)


def plot_3d_reconstruction(json_filepath, output_dir):
    """Replaces standalone plot_reconstruction script from Source 9"""
    if not os.path.exists(json_filepath):
        print(f"[SKIP] 3D JSON not found: {json_filepath}")
        return

    with open(json_filepath, 'r') as f:
        data = json.load(f)

    true_points = np.array(data.get("true_points", []))
    centroid_points = np.array(data.get("centroid_points", []))

    if len(true_points) == 0 or len(centroid_points) == 0:
        return

    fig = plt.figure(figsize=(35, 22))
    fig.set_layout_engine(None)

    # Calculate global 3D limits to match camera views
    all_pts = np.vstack((true_points, centroid_points))
    mid_vals = (all_pts.max(axis=0) + all_pts.min(axis=0)) * 0.5
    max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2.0

    axes_config = [
        (1, true_points, 'lightgrey', 'dimgrey', 'Ground Truth'),
        (2, centroid_points, '#0072B2', 'white', 'Centroid Reconstruction')
    ]

    for idx, pts, c, edge_c, title in axes_config:
        ax = fig.add_subplot(1, 2, idx, projection='3d')

        # ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
        #            c=c, s=80, alpha=0.9, edgecolors=edge_c, linewidth=1)

        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=c, s=185, alpha=0.9, edgecolors=edge_c, linewidth=1.25)

        ax.set_title(title, fontsize=FONT_SIZE + 20, fontweight="bold", y=0.95)

        ax.set_xlim(mid_vals[0] - max_range, mid_vals[0] + max_range)
        ax.set_ylim(mid_vals[1] - max_range, mid_vals[1] + max_range)
        ax.set_zlim(mid_vals[2] - max_range, mid_vals[2] + max_range)

        # Remove axes ticks
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        # 2. Force the grid lines to be solid, defined, and grey
        ax.xaxis._axinfo["grid"].update({"color": "lightgrey", "linewidth": 1, "linestyle": "-"})
        ax.yaxis._axinfo["grid"].update({"color": "lightgrey", "linewidth": 1, "linestyle": "-"})
        ax.zaxis._axinfo["grid"].update({"color": "lightgrey", "linewidth": 1, "linestyle": "-"})


        ax.view_init(elev=20, azim=45)

    plt.subplots_adjust(left=0.0, bottom=0.3, top=1.0, wspace=-0.100, right=1.0)

    filename = '3d_reconstruction_plot.pdf'
    plt.savefig(os.path.join(output_dir, filename), format='pdf', bbox_inches='tight', transparent=True)
    plt.close(fig)