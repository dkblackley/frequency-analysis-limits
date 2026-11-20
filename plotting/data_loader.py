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

def grid_placement_space(embedding, n, num_samples=1000, seed=0, keep_aspect=False):
    """
    A baseline's (Even Less / Remin) reconstruction space on an n x n grid, without using any plaintext:
    num_samples databases drawn uniformly from every placement of its embedding that lies on the grid.

    A placement = (box size w x h, reflection r, rotation theta, shift):
      1. the embedding's bounding box is scaled to a w x h box of grid cells, w, h in 1..n (keep_aspect=True:
         h follows w by the embedding's aspect ratio); w = h = 1 puts every record in one cell,
         w = h = n fills the grid;
      2. optionally reflected (x -> -x), then rotated by theta in {0, 1, ..., 359} degrees about the box
         centre. Reflection + the 360 rotations also give the horizontal, vertical and both diagonal flips
         (e.g. vertical flip = reflection + 180 degrees), so all flips are covered;
      3. shifted by whole cells, then every coordinate rounded to its grid cell.
    A placement with any record outside [1, n]^2 is not used. Draw: (w, h, r, theta) uniformly, accepted with
    probability (#in-grid shifts) / n^2 (rejection sampling: redrawn otherwise), then a uniform in-grid
    shift, i.e. uniform over all (w, h, r, theta, shift) that stay on the grid.
    Returns a list of (len(embedding), 2) float arrays with integer values in 1..n.
    """
    e = np.asarray(embedding, dtype=float)
    lo, hi = e.min(axis=0), e.max(axis=0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    unit = (e - lo) / span - 0.5  # bounding box -> [-0.5, 0.5]^2, centred
    aspect = span[1] / span[0]
    rng = np.random.default_rng(seed)
    clouds = []
    while len(clouds) < num_samples:
        w = int(rng.integers(1, n + 1))
        h = max(1, min(n, int(round(1 + (w - 1) * aspect)))) if keep_aspect else int(rng.integers(1, n + 1))
        refl = rng.integers(0, 2)
        rad = np.radians(rng.integers(0, 360))
        x, y = unit[:, 0] * (w - 1), unit[:, 1] * (h - 1)
        if refl:
            x = -x
        c, s = np.cos(rad), np.sin(rad)
        px, py = x * c - y * s, x * s + y * c
        px, py = px - px.min(), py - py.min()  # rotated shape's lower-left corner at 0
        # in-grid shifts ox, oy (whole cells): 1 + o + extent <= n after rounding
        nx = n - int(np.round(px).max())
        ny = n - int(np.round(py).max())
        if nx <= 0 or ny <= 0 or rng.random() >= nx * ny / (n * n):
            continue  # does not fit / rejected so the draw stays uniform over all placements
        ox, oy = rng.integers(0, nx), rng.integers(0, ny)
        clouds.append(np.column_stack((1 + ox + np.round(px), 1 + oy + np.round(py))))
    return clouds


def generate_sampled_reconstructions(data, num_samples=1000):
    """LEGACY (the paper's Table II): sweeps the RAW embedding coordinates, which are NOT on the grid
    (Remin ~ +-15, Even Less ~ +-1), so its centroid/diameter MSE is just the distance of the truth to the
    origin. Use grid_placement_space instead. Kept only for working_rebuttal.py --original-sweep."""
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


def mass_centroid_mse(point_clouds, true_pts):
    """Per record: mean position over ALL databases (centre of mass, every database weighted equally, unlike
    the hull centroid which only uses the hull's vertices); squared distance to the truth divided by the
    number of dimensions (same convention as compute_dataset_metrics), averaged over records."""
    true_pts = np.asarray(true_pts, dtype=float)
    mean_pos = np.mean(np.stack([np.asarray(c, dtype=float) for c in point_clouds]), axis=0)
    return float(np.mean(np.sum((mean_pos - true_pts) ** 2, axis=1) / true_pts.shape[1]))


# =========================================================
# ORIENTATION ORACLE (default for the box plots, convex-hull figure and rebuttal tables 2b/4)
# Every method is told its orientation, then scored. Why: the 8 symmetries of the square grid (4 rotations x
# reflection) cannot be told apart from the leakage (identical frequencies under uniform; at low query % the
# mirror images also fit within eps), so without this, MSE mostly measures which mirror image a method landed
# in. LAMa: each output database in its best of the 8 grid symmetries (one of 8 choices, uses the truth).
# Baselines: MORE than that, their exact rotation/reflection from Procrustes (any angle, uses the truth), then
# every scale and shift is still sampled (oriented_box_space). Scale and shift never come from the truth.
# =========================================================

def grid_symmetries(a, n):
    """a (..., k, 2) on [1, n]^2 -> (8, ..., k, 2): the 8 symmetries of the grid (4 rotations x reflection)."""
    a = np.asarray(a, dtype=float)
    x, y, m = a[..., 0], a[..., 1], n + 1
    orients = [(x, y), (m - y, x), (m - x, m - y), (y, m - x), (y, x), (m - x, y), (m - y, m - x), (x, m - y)]
    return np.stack([np.stack(o, axis=-1) for o in orients])


def orient_to_truth(true, dbs, n):
    """Each database in whichever of the 8 grid symmetries is closest to the truth."""
    s = grid_symmetries(np.stack(dbs), n)  # (8, #dbs, k, 2)
    best = ((s - np.asarray(true, dtype=float)) ** 2).sum(-1).mean(-1).argmin(0)
    return list(s[best, np.arange(len(dbs))])


def procrustes_rotation_only(true, emb):
    """Embedding rotated/reflected by orthogonal Procrustes against the truth; no scale or shift from it."""
    t = np.asarray(true, dtype=float); e = np.asarray(emb, dtype=float)
    t, e = t - t.mean(0), e - e.mean(0)
    U, _, Vt = np.linalg.svd(e.T @ t)
    return e @ (U @ Vt)


def oriented_box_space(true, embedding, n, num_samples=1000, seed=0):
    """
    A baseline's reconstruction space when told its orientation: the embedding rotated/reflected by
    procrustes_rotation_only, then its bounding box scaled to any w x h cells (w, h = 1..n, from one cell to
    the full grid) at any whole-cell position inside [1, n]^2, rounded to cells. Every such placement is on
    the grid; drawn uniformly over all of them (w with weight n+1-w = its number of positions, h likewise).
    """
    r = procrustes_rotation_only(true, embedding)
    lo, hi = r.min(0), r.max(0)
    u = (r - lo) / np.where(hi - lo > 0, hi - lo, 1.0)
    rng = np.random.default_rng(seed)
    sizes = np.arange(1, n + 1)
    p = (n + 1 - sizes) / (n + 1 - sizes).sum()
    ws, hs = rng.choice(sizes, num_samples, p=p), rng.choice(sizes, num_samples, p=p)
    x0s, y0s = rng.integers(1, n + 2 - ws), rng.integers(1, n + 2 - hs)
    return [np.column_stack((x0 + np.round(u[:, 0] * (w - 1)), y0 + np.round(u[:, 1] * (h - 1))))
            for w, h, x0, y0 in zip(ws, hs, x0s, y0s)]


def baseline_path(base_dir, grid, db, dist, method, query_pct):
    tag = "even_less" if method == "even_less" else "classic"
    return f"{base_dir}/{grid}/{db}/{method}/{db}_prob{float(query_pct)}_{dist}_{grid}_{tag}.json"


def baseline_space(base_dir, grid, db, dist, method, query_pct, num_samples=1000, keep_aspect=False,
                   oriented=True):
    """(true (n,2), reconstruction space of the baseline's embedding), or (None, None) if the file is missing.
    oriented=True (default): oriented_box_space (told its orientation). False: grid_placement_space (every
    rotation/flip/scale/shift, no truth used). Seeded from the config name, so reruns are identical."""
    try:
        with open(baseline_path(base_dir, grid, db, dist, method, query_pct)) as f:
            data = json.load(f)
    except FileNotFoundError:
        return None, None
    if not data:
        return None, None
    seed = int.from_bytes(f"{grid} {db} {dist} {method} {query_pct}".encode(), "little") % (2 ** 32)
    true = np.array([p["true"] for p in data], dtype=float)
    emb, n = [p["reconstructed"] for p in data], int(grid.split("x")[0])
    if oriented:
        space = oriented_box_space(true, emb, n, num_samples=num_samples, seed=seed)
    else:
        space = grid_placement_space(emb, n, num_samples=num_samples, seed=seed, keep_aspect=keep_aspect)
    return true, space


def lama_space(base_dir, grid, db, dist, query_pct, oriented=True):
    """(true, [every database LAMa output]) for a sampled run, or (None, None) if missing.
    oriented=True (default): each database in its best of the 8 grid symmetries (orient_to_truth)."""
    path = f"{base_dir}/{grid}/{db}/limits/{db}_{dist}_p{query_pct / 100.0:g}_reconstruction.json"
    try:
        with open(path) as f:
            runs = json.load(f)
    except FileNotFoundError:
        return None, None
    if not runs:
        return None, None
    true = np.array([p["true"] for p in runs[0]], dtype=float)
    dbs = [np.array([p["reconstructed"] for p in run], dtype=float) for run in runs if run]
    return true, (orient_to_truth(true, dbs, int(grid.split("x")[0])) if oriented else dbs)


def build_box_data(base_dir, db, distributions, methods, query_percents, grid="50x50", num_samples=1000,
                   oriented=False):
    """Box-plot JSON for plot_mse_boxplots: {dist: {method: [[query, [MSE of every database]], ...]}}.
    MSE = calculate_mse to the truth. LAMa: every output database; baselines: num_samples from their space.
    oriented: see the ORIENTATION ORACLE note above (False = raw, the appendix version)."""
    out = {}
    for dist in distributions:
        out[dist] = {m: [] for m in methods}
        for q in query_percents:
            for m in methods:
                if m == "limits":
                    true, space = lama_space(base_dir, grid, db, dist, q, oriented=oriented)
                else:
                    true, space = baseline_space(base_dir, grid, db, dist, m, q, num_samples, oriented=oriented)
                if space is None:
                    print(f"[SKIP] {db} {dist} {m} {q}%: no file")
                    continue
                out[dist][m].append([q, [float(calculate_mse(true, x)) for x in space]])
    return out


from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def load_convex_mse_data(base_dir, datasets, methods, distributions, grid="15x15",
                         limits_grid="15x15_average", query_percents=(10.0, 20.0, 30.0), oriented=True):
    """
    oriented:    see the ORIENTATION ORACLE note above (False = raw, the appendix version).
    grid:        folder (and file-name suffix) of the Even Less / Remin runs, e.g. "15x15" or "50x50".
    limits_grid: folder of the LAMa runs. The paper's 15x15 numbers came from "15x15_average" (not on this
                 machine); the 50x50 runs live in "50x50", the newer 15x15 runs in "15x15".
    """
    nested_data = {db: {dist: {meth: [] for meth in methods} for dist in distributions} for db in datasets}
    data_dir = f"{base_dir}/{grid}"

    target_cores = max(1, os.cpu_count() - 4)

    # 1. Build a flat list of all parameter combinations
    tasks = []
    for db in datasets:
        for dist in distributions:
            for query in query_percents:
                tasks.append((db, dist, query, base_dir, data_dir, methods, grid, limits_grid, oriented))

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
    db, dist, query, base_dir, data_dir, methods, grid, limits_grid, oriented = args
    results = []

    # Even Less / Remin: oriented_box_space (or grid_placement_space if not oriented)
    for meth in ("even_less", "remin"):
        if meth in methods:
            true_pts, point_clouds = baseline_space(base_dir, grid, db, dist, meth, query, oriented=oriented)
            if point_clouds is not None:
                c_mse, d_mse = compute_dataset_metrics(point_clouds, true_pts)
                results.append((db, dist, meth, query, c_mse, d_mse))

    # LAMa: every output database (oriented: each in its best grid symmetry)
    if "limits" in methods:
        true_pts, point_clouds = lama_space(base_dir, limits_grid, db, dist, query, oriented=oriented)
        if point_clouds is not None:
            c_mse, d_mse = compute_dataset_metrics(point_clouds, true_pts)
            results.append((db, dist, "limits", query, c_mse, d_mse))

    return results

def load_limits_samples_mse(base_dir, datasets, distributions, target_query=10.0):
    """
    Loads only the LAMa/limits data and evaluates MSE across the LAST N samples.
    Returns: nested_data[dataset][dist] = [(n_samples, centroid_mse, diameter_mse), ...]
    """
    nested_data = {db: {dist: [] for dist in distributions} for db in datasets}
    data_dir = f"{base_dir}/15x15_average"
    sample_steps = [250, 500, 750, 1000]

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

def load_running_time_lines_data(base_dir, datasets, methods, distributions, grid="50x50", limits_grid="50x50",
                                 query_percents=(10.0, 20.0, 30.0)):
    """Loads running time data per query percent. Even Less / Remin from <grid>/<db>/{even_less,remin},
    LAMa from <limits_grid>/<db>/limits. (The paper's 15x15 version: grid="15x15", limits_grid="15x15_run_1".)"""
    nested_data = {db: {dist: {meth: [] for meth in methods} for dist in distributions} for db in datasets}

    for db in datasets:
        for dist in distributions:
            for query in query_percents:
                # Expected string variations for 10%, 20%, 30% across your methods
                query_str_whole = f"prob{int(query)}.0"  # e.g., prob10.0
                query_str_frac = f"p{query / 100}"  # e.g., p0.1
                query_str_frac_alt = f"prob{query / 100}"  # e.g., prob0.1

                # 1. Even Less
                if "even_less" in methods:
                    method_dir = f"{base_dir}/{grid}/{db}/even_less"
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
                    method_dir = f"{base_dir}/{grid}/{db}/remin"
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
                    method_dir = f"{base_dir}/{limits_grid}/{db}/limits"
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
