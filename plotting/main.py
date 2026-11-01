import os
import matplotlib.pyplot as plt

import plot_grid_mse
from data_loader import load_grid_mse_data, load_convex_mse_data, load_running_time_data, load_running_time_lines_data, \
    load_limits_samples_mse
from plot_boxplots_mse import plot_mse_boxplots
from plot_grid_mse import plot_grid_by_mse
from plot_convex_mse import plot_convex_mse
from plot_histograms import plot_reconstruction_histograms, plot_lama_flat_distributions
from plot_falzon import plot_falzon_histograms
from plot_falzon_any import plot_falzon_any_histograms
from plot_running_time import plot_running_time_bar
from plot_running_time_lines import plot_running_time_lines
from globals_vars import FONT_SIZE
from plot_limits_samples import print_limits_samples_latex

from print_time_per_recon_table import print_time_per_recon_latex
from print_noise_table import load_noise_data, print_noise_latex, save_noise_csv
from plot_limits_time_per_recon import plot_limits_time_per_recon
from spatial_reconstructions import plot_3d_reconstruction, plot_spatial_reconstruction

# ==========================================
# GLOBAL CONSTANTS & AESTHETICS
# ==========================================

COLORS = {
    'limits': '#0072B2',  # Blue (LAMa)
    'even_less': '#E69F00',  # Orange / Yellow
    'remin': '#009E73',  # Green
    'reminp': '#CC79A7',  # Purple (REMIN+)
    'falzon': '#D55E00',  # Vermilion (Falzon et al.)
}

DIST_COLORS = {
    'uniform': '#56B4E9',  # Sky Blue
    'gaussian': '#E69F00',  # Orange
    'beta': '#009E73',  # Bluish Green
}

DIST_MARKERS = {
    'uniform': 'o',
    'gaussian': 's',
    'beta': '^',
}

MARKERS = {
    'limits': 's',  # Square for LAMa
    'even_less': '*',  # Star for Even Less
    'remin': 'o',  # Circle for Remin
    'reminp': '^',  # Triangle for Remin+
    'falzon': 'D',  # Diamond for Falzon et al.
}

LABELS = {
    'limits': 'LAMa',
    'even_less': 'Even Less',
    'remin': 'Remin',
    'reminp': 'Remin+',
    'falzon': 'Falzon',
}

METHOD_ORDER = ['limits', 'even_less', 'remin', 'reminp']


def setup_sleek_style():
    """Applies a modern, flat, and highly readable style to all matplotlib plots."""
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'serif'],
        'axes.titlesize': FONT_SIZE + 5,
        'axes.labelsize': FONT_SIZE,
        'xtick.labelsize': FONT_SIZE - 8,
        'ytick.labelsize': FONT_SIZE - 8,
        'legend.fontsize': FONT_SIZE - 12,
        'axes.linewidth': 2.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.autolayout': False,  # Disabled to allow manual global legend placement
        'grid.color': '#E5E5E5',
        'grid.linestyle': '--',
        'grid.alpha': 0.7,
        'figure.figsize': (17, 11),
        'axes.titlepad': 50,
        'axes.labelpad': 10,
        'lines.linewidth': 8.0,
        'lines.markersize': 30,
    })


if __name__ == "__main__":
    setup_sleek_style()

    BASE_DIR = "../databases"
    OUTPUT_DIR = "plots"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_grid_mse.FONT_SIZE = FONT_SIZE

    datasets = ["busstop", "shopparis", "cali", "drink", "highway", "spitz"]
    # datasets = [ "spitz"]
    methods = ["even_less", "remin", "limits"]
    grid_meth = methods + ["reminp"]
    distributions = ["uniform", "gaussian", "beta"]
    grid_sizes = [(n, f"{n}x{n}") for n in range(20, 51, 10)]

    # # ---------------------------------------------------------
    # # 1. Grid Size vs MSE Plots
    # # ---------------------------------------------------------
    # print("Loading nested data for Grid Size vs MSE...")
    # grid_data = load_grid_mse_data(BASE_DIR, datasets, grid_meth, distributions, grid_sizes)
    #
    # print("Generating Grid vs MSE plots...")
    # for db in datasets:
    #     # Pass the whole list of distributions instead of iterating one by one
    #     plot_grid_by_mse(grid_data, db, distributions, OUTPUT_DIR, METHOD_ORDER, COLORS, MARKERS, LABELS)

    # # ---------------------------------------------------------
    # # 2. Query Percent vs MSE (Convex Hulls)
    # # ---------------------------------------------------------
    # print("Loading nested data for Convex Hull metrics...")
    # convex_data = load_convex_mse_data(BASE_DIR, datasets, methods, distributions)
    #
    # print("Generating Convex Hull Query % vs MSE plots...")
    # for db in datasets:
    #     # Pass the whole list of distributions instead of iterating one by one
    #     plot_convex_mse(convex_data, db, distributions, OUTPUT_DIR, METHOD_ORDER, COLORS, MARKERS, LABELS)
    #
    # print(f"\nDone! All plots saved to ./{OUTPUT_DIR}/")

    # # BOX PLOTS
    # print("Generating Box Plots...")
    # for db in datasets:
    #     box_json_path = f"../figures/box/{db}_mse_distribution_boxplot.json"
    #     plot_mse_boxplots(box_json_path, db, distributions, OUTPUT_DIR, METHOD_ORDER, COLORS, LABELS)

    # # HISTOGRAMS
    # print("Generating Histogram Plots...")
    # for db in datasets:
    #     # for dist in distributions:
    #     #     hist_json_path = f"../figures/histogram_mse_sol_num/mse_histogram_{db}_{dist}.json"
    #     #     plot_reconstruction_histograms(hist_json_path, db, dist, OUTPUT_DIR, METHOD_ORDER, COLORS, LABELS)
    #
    #     hist_json_path = f"../figures/histogram_mse_sol_num/mse_histogram_{db}_uniform.json"
    #     plot_reconstruction_histograms(hist_json_path, db, "uniform", OUTPUT_DIR, METHOD_ORDER, COLORS, LABELS)


    # ---------------------------------------------------------
    # LAMa vs Falzon et al. Histograms (same data as the histograms above)
    # ---------------------------------------------------------
    print("Generating LAMa vs Falzon et al. Histogram Plots...")
    falzon_grid = 50
    falzon_dists = ["gaussian", "beta", "uniform"]  # left to right
    # True: LAMa per distribution + one Falzon "Any Dist." column (plot_falzon_any.py)
    # False: a LAMa/Falzon pair per distribution (plot_falzon.py)
    falzon_any_dist = False
    plot_falzon_row = plot_falzon_any_histograms if falzon_any_dist else plot_falzon_histograms
    for db in datasets:
        paths = {dist: (f"../figures/histogram_mse_sol_num/mse_histogram_{db}_{dist}.json",
                        f"{BASE_DIR}/{falzon_grid}x{falzon_grid}/{db}/limits/{db}_{dist}_e0_d0.9_reconstruction.json")
                 for dist in falzon_dists}
        # Spitz is the top row, so it carries the method/distribution titles, and Paris is the
        # bottom row, so it carries the x-axis label
        plot_falzon_row(paths, (falzon_grid, falzon_grid), db, OUTPUT_DIR, COLORS, LABELS,
                        show_header=(db == "spitz"), show_xlabel=(db == "shopparis"))

    # ---------------------------------------------------------
    # Noisy query distribution (--dist-noise) table
    # ---------------------------------------------------------
    print("Generating noisy-distribution tables...")
    noise = 0.01
    noise_data = load_noise_data(BASE_DIR, datasets, distributions, noise=noise, grid="15x15")
    print_noise_latex(noise_data, datasets, distributions, noise=noise, metric="centroid")
    print_noise_latex(noise_data, datasets, distributions, noise=noise, metric="diameter")
    save_noise_csv(noise_data, OUTPUT_DIR, noise=noise)

    # # ---------------------------------------------------------
    # # LAMa Flat Distribution Histograms
    # # ---------------------------------------------------------
    # print("Generating LAMa Flat Distribution Plots...")
    # for db in datasets:
    #     plot_lama_flat_distributions(BASE_DIR, db, OUTPUT_DIR)

    # print("Generating 2D Spatial Reconstruction Plots...")
    # for db in datasets:
    #     # Assuming you ran the 50x50 grids in Rust, updating the path accordingly
    #     grid_size = 50
    #     json_path = f"../figures/{grid_size}x{grid_size}_recon/{db}_uniform_spatial_data.json"
    #
    #     plot_spatial_reconstruction(
    #         json_path=json_path,
    #         dataset=db,
    #         dist="uniform",
    #         output_dir=OUTPUT_DIR,
    #         method_order=METHOD_ORDER,
    #         colors=COLORS,
    #         labels=LABELS
    #     )

    # # ---------------------------------------------------------
    # # 3D Spatial Reconstruction Plots
    # # ---------------------------------------------------------
    # print("Generating 3D Spatial Reconstruction Plots...")
    # plot_3d_reconstruction("../figures/3d_recon/points_dump.json", OUTPUT_DIR)
