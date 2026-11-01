import os
import json
import random
import numpy as np
from scipy.spatial import ConvexHull, distance_matrix
from tqdm import tqdm

def procrustes_align(true_pts, recon_pts):
    true_pts = np.array(true_pts)
    recon_pts = np.array(recon_pts)
    n, d = true_pts.shape

    mean_true = np.mean(true_pts, axis=0)
    mean_recon = np.mean(recon_pts, axis=0)

    centered_true = true_pts - mean_true
    centered_recon = recon_pts - mean_recon
    var_recon = np.sum(centered_recon ** 2)

    r_mat = np.eye(d)
    s_factor = 1.0

    if var_recon > 1e-12:
        H = centered_recon.T @ centered_true
        U, S, Vt = np.linalg.svd(H)
        R = U @ Vt
        d_sign = 1.0
        if np.linalg.det(R) < 0:
            d_sign = -1.0
            U[:, -1] *= -1.0
            R = U @ Vt

        trace_sigma = np.sum(S[:-1]) + S[-1] * d_sign
        s_factor = trace_sigma / var_recon
        r_mat = R

    return (centered_recon @ r_mat) * s_factor + mean_true


def calculate_mse(true_pts, recon_pts):
    return np.mean(np.sum((np.array(true_pts) - np.array(recon_pts)) ** 2, axis=1))


def load_standard_method(path):
    with open(path, 'r') as f:
        data = json.load(f)
    if not data:
        return float('nan')
    true_pts = [p["true"] for p in data]
    recon_pts = [p["reconstructed"] for p in data]
    aligned_recon = procrustes_align(true_pts, recon_pts)
    return calculate_mse(true_pts, aligned_recon)


def load_limits_method(path):
    with open(path, 'r') as f:
        data = json.load(f)
    if not data:
        return float('nan')
    mses = []
    for run in data:
        if not run: continue
        true_pts = [p["true"] for p in run]
        recon_pts = [p["reconstructed"] for p in run]
        mses.append(calculate_mse(true_pts, recon_pts))
    return min(mses) if mses else float('inf')


def load_grid_mse_data(base_dir, datasets, methods, distributions, grid_sizes):
    # Dictionary mapping map[dataset][dist][method] -> list of (grid_val, mse)
    nested_data = {db: {dist: {meth: [] for meth in methods} for dist in distributions} for db in datasets}

    for db in datasets:
        for dist in distributions:
            for meth in methods:
                for grid_val, grid_str in grid_sizes:
                    path = ""
                    try:
                        if meth == "even_less":
                            path = f"{base_dir}/{grid_str}/{db}/even_less/{db}_prob100.0_{dist}_{grid_str}_even_less.json"
                            mse = load_standard_method(path)
                        elif meth == "remin":
                            path = f"{base_dir}/{grid_str}/{db}/remin/{db}_prob100.0_{dist}_{grid_str}_classic.json"
                            mse = load_standard_method(path)
                        elif meth == "reminp":
                            path = f"{base_dir}/{grid_str}/{db}/reminp/{db}_prob100.0_{dist}_{grid_str}_classic.json"
                            mse = load_standard_method(path)
                        elif meth == "limits":
                            path = f"{base_dir}/{grid_str}/{db}/limits/{db}_{dist}_e0_d0.9_reconstruction.json"
                            mse = load_limits_method(path)
                        else:
                            continue

                        if np.isfinite(mse):
                            nested_data[db][dist][meth].append((grid_val, mse))
                        else:
                            nested_data[db][dist][meth].append((grid_val, 0.0))
                    except FileNotFoundError:
                        print(f"[SKIP] Missing path for {meth}: {path}")
                    except Exception as e:
                        print(f"[ERROR] Failed plotting for {meth}, {dist}, {db}: {e}")

    return nested_data


# =========================================================
# CONVEX HULL METRICS LOGIC
# =========================================================

def generate_sampled_reconstructions(data, num_samples=10000):
    """Mimics Rust's `generate_sampled_reconstructions` sweep behavior for Remin/Even Less"""
    point_clouds = []
    true_pts = np.array([p['true'] for p in data])
    recon_pts = np.array([p['reconstructed'] for p in data])

    for _ in range(num_samples):
        dx, dy = random.choice([-5.0, -2.0, 0.0, 2.0, 5.0]), random.choice([-5.0, -2.0, 0.0, 2.0, 5.0])
        angle = random.choice([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330])
        sx, sy = random.choice([(1.0, 1.0), (0.9, 0.9), (1.1, 1.1)])

        rad = np.radians(angle)
        cos_t, sin_t = np.cos(rad), np.sin(rad)
        scaled_x, scaled_y = recon_pts[:, 0] * sx, recon_pts[:, 1] * sy

        rot_x = scaled_x * cos_t - scaled_y * sin_t
        rot_y = scaled_x * sin_t + scaled_y * cos_t

        point_clouds.append(np.column_stack((rot_x + dx, rot_y + dy)))
    return point_clouds, true_pts


def compute_dataset_metrics(point_clouds, true_pts):
    num_points = len(true_pts)
    total_cent_mse, total_diam_mse, valid_count = 0.0, 0.0, 0

    for i in range(num_points):
        cloud = np.array([run[i] for run in point_clouds])
        unique_cloud = np.unique(cloud, axis=0)

        if len(unique_cloud) <= 2:
            vertices = unique_cloud
        else:
            try:
                hull = ConvexHull(unique_cloud)
                vertices = unique_cloud[hull.vertices]
            except Exception:
                vertices = unique_cloud

        if len(vertices) == 0:
            continue

        valid_count += 1

        # Centroid Check
        centroid = np.mean(vertices, axis=0)
        total_cent_mse += np.sum((centroid - true_pts[i]) ** 2) / len(true_pts[i])

        # Diameter Midpoint Check
        if len(vertices) > 1:
            dist_mat = distance_matrix(vertices, vertices)
            idx1, idx2 = np.unravel_index(np.argmax(dist_mat), dist_mat.shape)
            midpoint = (vertices[idx1] + vertices[idx2]) / 2.0
        else:
            midpoint = vertices[0]

        total_diam_mse += np.sum((midpoint - true_pts[i]) ** 2) / len(true_pts[i])

    if valid_count == 0:
        return 0.0, 0.0
    return total_cent_mse / valid_count, total_diam_mse / valid_count


from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def load_convex_mse_data(base_dir, datasets, methods, distributions):
    nested_data = {db: {dist: {meth: [] for meth in methods} for dist in distributions} for db in datasets}
    query_percents = [10.0, 20.0, 30.0]
    data_dir = f"{base_dir}/15x15"

    target_cores = max(1, os.cpu_count() - 4)

    # 1. Build a flat list of all parameter combinations
    tasks = []
    for db in datasets:
        for dist in distributions:
            for query in query_percents:
                tasks.append((db, dist, query, base_dir, data_dir, methods))

    # 2. Spin up the pool ONCE and process all tasks concurrently
    with ProcessPoolExecutor(max_workers=target_cores) as executor:
        # Wrap the map function in tqdm.
        # Using list() forces the iterator to evaluate, updating the bar as tasks finish.
        all_results = list(tqdm(
            executor.map(_process_single_configuration, tasks),
            total=len(tasks),
            desc="Processing Configurations"
        ))

    # 3. Unpack the flat results back into your nested dictionary
    for result_group in all_results:
        for db, dist, method, query, c_mse, d_mse in result_group:
            nested_data[db][dist][method].append((query, c_mse, d_mse))

    return nested_data

def _process_single_configuration(args):
    """Worker function to process one specific dataset/distribution/query combination."""
    db, dist, query, base_dir, data_dir, methods = args
    results = []

    # Even Less Setup
    if "even_less" in methods:
        try:
            path = f"{data_dir}/{db}/even_less/{db}_prob{int(query)}.0_{dist}_15x15_even_less.json"
            with open(path, 'r') as f:
                data = json.load(f)
            point_clouds, true_pts = generate_sampled_reconstructions(data)
            c_mse, d_mse = compute_dataset_metrics(point_clouds, true_pts)
            # Append a tuple with the identifiers and the metrics
            results.append((db, dist, "even_less", query, c_mse, d_mse))
        except FileNotFoundError:
            pass

    # Remin Setup
    if "remin" in methods:
        try:
            path = f"{data_dir}/{db}/remin/{db}_prob{int(query)}.0_{dist}_15x15_classic.json"
            with open(path, 'r') as f:
                data = json.load(f)
            point_clouds, true_pts = generate_sampled_reconstructions(data)
            c_mse, d_mse = compute_dataset_metrics(point_clouds, true_pts)
            results.append((db, dist, "remin", query, c_mse, d_mse))
        except FileNotFoundError:
            pass

    # LAMa Limits Setup
    if "limits" in methods:
        try:
            limits_query = query / 100.0
            path = f"{base_dir}/15x15_average/{db}/limits/{db}_{dist}_p{limits_query}_reconstruction.json"
            with open(path, 'r') as f:
                multi_run_data = json.load(f)

            if multi_run_data:
                true_pts = [p['true'] for p in multi_run_data[0]]
                point_clouds = [np.array([p['reconstructed'] for p in run]) for run in multi_run_data]
                c_mse, d_mse = compute_dataset_metrics(point_clouds, true_pts)
                results.append((db, dist, "limits", query, c_mse, d_mse))
        except FileNotFoundError:
            pass

    return results

def load_limits_samples_mse(base_dir, datasets, distributions, target_query=10.0):
    """
    Loads only the LAMa/limits data and evaluates MSE across the LAST N samples.
    Returns: nested_data[dataset][dist] = [(n_samples, centroid_mse, diameter_mse), ...]
    """
    nested_data = {db: {dist: [] for dist in distributions} for db in datasets}
    data_dir = f"{base_dir}/15x15_average"
    sample_steps = [2500, 5000, 7500, 10000]

    for db in datasets:
        for dist in distributions:
            limits_query = target_query / 100.0
            path = f"{data_dir}/{db}/limits/{db}_{dist}_p{limits_query}_reconstruction.json"

            try:
                with open(path, 'r') as f:
                    multi_run_data = json.load(f)

                if not multi_run_data:
                    continue

                for n_samples in sample_steps:
                    # Guard against datasets with fewer than 10,000 runs
                    actual_n = min(n_samples, len(multi_run_data))

                    # SLICE THE LAST `actual_n` RUNS
                    subset_runs = multi_run_data[-actual_n:]

                    true_pts = [p['true'] for p in subset_runs[0]]
                    point_clouds = [np.array([p['reconstructed'] for p in run]) for run in subset_runs]

                    c_mse, d_mse = compute_dataset_metrics(point_clouds, true_pts)
                    nested_data[db][dist].append((n_samples, c_mse, d_mse))

            except FileNotFoundError:
                pass

    return nested_data

def load_running_time_lines_data(base_dir, datasets, methods, distributions):
    """Loads running time data for 10%, 20%, and 30% queries."""
    nested_data = {db: {dist: {meth: [] for meth in methods} for dist in distributions} for db in datasets}
    query_percents = [10.0, 20.0, 30.0]

    for db in datasets:
        for dist in distributions:
            for query in query_percents:
                # Expected string variations for 10%, 20%, 30% across your methods
                query_str_whole = f"prob{int(query)}.0"  # e.g., prob10.0
                query_str_frac = f"p{query / 100}"  # e.g., p0.1
                query_str_frac_alt = f"prob{query / 100}"  # e.g., prob0.1

                # 1. Even Less
                if "even_less" in methods:
                    method_dir = f"{base_dir}/15x15/{db}/even_less"
                    if os.path.exists(method_dir):
                        for fname in os.listdir(method_dir):
                            if "results" in fname and dist in fname and query_str_whole in fname:
                                try:
                                    with open(os.path.join(method_dir, fname), 'r') as f:
                                        time_val = json.load(f).get("time_taken", 0.0)
                                        nested_data[db][dist]["even_less"].append((query, float(time_val)))
                                except Exception:
                                    pass
                                break

                # 2. Remin
                if "remin" in methods:
                    method_dir = f"{base_dir}/15x15/{db}/remin"
                    if os.path.exists(method_dir):
                        for fname in os.listdir(method_dir):
                            if "results" in fname and dist in fname and query_str_whole in fname:
                                try:
                                    with open(os.path.join(method_dir, fname), 'r') as f:
                                        time_val = json.load(f).get("time_taken", 0.0)
                                        nested_data[db][dist]["remin"].append((query, float(time_val)))
                                except Exception:
                                    pass
                                break

                # 3. LAMa Limits
                if "limits" in methods:
                    method_dir = f"{base_dir}/15x15_run_1/{db}/limits"
                    if os.path.exists(method_dir):
                        for fname in os.listdir(method_dir):
                            if "results" in fname and dist in fname and (
                                    query_str_frac in fname or query_str_frac_alt in fname):
                                try:
                                    with open(os.path.join(method_dir, fname), 'r') as f:
                                        time_val = json.load(f).get("time_taken", 0.0)
                                        nested_data[db][dist]["limits"].append((query, float(time_val)))
                                except Exception:
                                    pass
                                break

    return nested_data

def load_running_time_data(base_dir, datasets, methods, distributions):
    """Loads running time data from the 50x50 directory results JSONs."""
    time_data = {db: {dist: {meth: 0.0 for meth in methods} for dist in distributions} for db in datasets}
    dir_50x50 = os.path.join(base_dir, "50x50")

    if not os.path.exists(dir_50x50):
        print(f"[SKIP] Directory {dir_50x50} does not exist.")
        return time_data

    for db in datasets:
        for meth in methods:
            method_dir = os.path.join(dir_50x50, db, meth)
            if not os.path.exists(method_dir):
                continue

            for file_name in os.listdir(method_dir):
                if not file_name.endswith(".json") or "results" not in file_name:
                    continue

                # Identify perfect knowledge / 100% queries
                is_perfect = False
                if meth in ["even_less", "remin"] and "prob100.0" in file_name:
                    is_perfect = True
                elif meth == "limits" and ("e0_d0.9" in file_name or "prob1" in file_name):
                    is_perfect = True

                if not is_perfect:
                    continue

                # Extract distribution
                dist = next((d for d in distributions if d in file_name), None)
                if dist:
                    try:
                        with open(os.path.join(method_dir, file_name), 'r') as f:
                            data = json.load(f)
                            time_data[db][dist][meth] = float(data.get("time_taken", 0.0))
                    except Exception as e:
                        print(f"[ERROR] Reading time data for {db}, {dist}, {meth}: {e}")

    return time_data
