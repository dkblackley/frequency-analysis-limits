import logging
import sys

# Import the plotting functions from your newly created files
# Adjust these imports depending on exactly how you named the functions in those files
from spatial_plot_2d import plot_spatial_reconstruction, run_spatial_plots
from mse_by_grid import plot_grid_by_mse
from mse_by_all_recon import plot_histograms

# Configure Python logging to mirror the Rust log crate behavior
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def do_plotting():
    datasets = ["shopparis", "busstop", "cali", "drink", "highway", "spitz"]
    methods = ["even_less", "remin", "limits"]
    distributions = ["uniform", "gaussian", "beta"]

    # Equivalent to (20..=50).step_by(5)
    databases = [f"../databases/{n}x{n}" for n in range(20, 51, 5)]

    # Equivalent to (20..=50).step_by(10)
    grid_sizes = [(n, f"{n}x{n}") for n in range(20, 51, 10)]

    # 1. First loop: run spatial plots over uniform distribution and different grid sizes
    for grid_val, grid_str in grid_sizes:
        for name in datasets:
            data_dir = f"../databases/{grid_val}x{grid_val}"
            try:
                run_spatial_plots(name, data_dir, "uniform", grid_val)
            except Exception as e:
                logger.warning(f"{name} failed when loaded from {data_dir}:  {e}")

    # 2. Second loop: run spatial plots over different datasets and distributions for 50x50
    for name in datasets:
        for dist in distributions:
            try:
                run_spatial_plots(name, "../databases/50x50", dist, 50)
            except Exception as e:
                # Mimics .expect("SPITZ DIRECT FAILED!")
                logger.error(f"SPITZ DIRECT FAILED! Original error: {e}")


    # 3. Plot grid by MSE
    # In Python, we just call the function. If it fails, it naturally throws an exception (like .unwrap())
    plot_grid_by_mse(grid_sizes, datasets, methods, distributions)

    # 4. JSUT 350x50 spitz stuff
    dir_path = "../databases/350x50"
    name_spitz = "spitz"

    try:
        run_spatial_plots(name_spitz, dir_path, "uniform", 350)
    except Exception as e:
        logger.error(f"SPITZ DIRECT FAILED! Original error: {e}")
        sys.exit(1)

    # 5. Third loop: Histograms
    for dist in distributions:
        # Spitz
        try:
            plot_histograms(name_spitz, (350, 50), dist, f"../figures/{name_spitz}_{dist}_histo.svg")
        except Exception as e:
            logger.warning(f"{name_spitz} failed when loaded from {dir_path}:  {e}")

        # Cali
        try:
            plot_histograms("cali", (50, 50), dist, f"../figures/cali_{dist}_histo.svg")
        except Exception as e:
            logger.warning(f"cali failed when loaded from {dir_path}:  {e}")


if __name__ == "__main__":
    do_plotting()