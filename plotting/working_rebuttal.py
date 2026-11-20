"""
Dumps the numbers needed for the NDSS rebuttal (Reviewer 356A) to the terminal as markdown tables.
Run from the plotting folder:  python working_rebuttal.py [--items 1,2,3,4,5] [--original-sweep]

Nothing here re-runs an attack; it only reads the existing outputs in ../databases.
Any cell whose source file does not exist is printed as MISSING (never estimated).

MSE conventions (same as the rest of plotting/):
  * "MSE"          = data_loader.calculate_mse: mean over records of the squared Euclidean distance
                     (NOT divided by the number of dimensions), in grid units.
  * "centroid MSE" / "diameter MSE" = data_loader.compute_dataset_metrics (Table II): per record, convex
                     hull of that record's position over all candidate databases, then the squared distance
                     of the hull centroid / diameter midpoint to the truth, divided by the number of
                     dimensions, averaged over records.
  * "mean MSE"     = mean of calculate_mse over every database in a method's reconstruction space
                     (the expected error of picking one of its databases at random).
  * "mass centroid MSE" = data_loader.mass_centroid_mse: per record, its MEAN position over all databases
                     (every database weighted equally, not just the hull vertices), squared distance to the
                     truth divided by the number of dimensions, averaged over records.
  NOTE: mean MSE is NOT divided by dims, the three centroid-type columns are; compare methods within a column.

Reconstruction space in the probabilistic (sampled-query) tables, items 2b and 4:
  * LAMa: every database it output (up to 1000; fewer if that is all the run produced).
  * DEFAULT (--baseline-space oriented): every method is told its orientation (data_loader's ORIENTATION
    ORACLE note): each LAMa database in its best of the 8 grid symmetries; Even Less / Remin rotated/reflected
    by Procrustes (exact angle, more than LAMa gets), then SWEEP_SAMPLES (1000) databases drawn uniformly
    over every w x h box (w, h = 1..N) and whole-cell position (data_loader.oriented_box_space). Scale and
    shift never come from the truth. Without it (the other options) LAMa's databases are scored raw.
  * --baseline-space grid: Even Less / Remin: SWEEP_SAMPLES (1000) databases drawn uniformly from every placement of the
    embedding on the N x N grid (data_loader.grid_placement_space):
    bounding box scaled to any w x h cells (w, h = 1..N), optionally reflected, rotated by any whole degree
    0..359 (reflection + rotation covers the horizontal / vertical / diagonal flips), shifted by whole cells,
    rounded to cells; placements with any record off the grid are not used (rejection sampling).
    --baseline-space boxes (older): only the 8 axis-aligned orientations, any of the 8 axis-aligned orientations (4 rotations x
    reflection) of its bounding box, squeezed into any w x h box (w, h = 1..N) at any position inside
    [1, N]^2, rounded to grid cells: 8 * (N(N+1)/2)^2 placements (13,005,000 at 50x50). Drawn without
    enumerating them: w with weight N+1-w, h likewise, then a uniform position and orientation.
    --baseline-space sweep instead draws like data_loader.generate_sampled_reconstructions (shift dx, dy in
    {-5,-2,0,2,5}, rotation in 30-degree steps, scale in {0.9, 1, 1.1}: 900 configurations) on the grid
    placement (bounding box -> [1, N]^2), about the grid centre. (generate_sampled_reconstructions itself
    sweeps the RAW embedding coordinates, which are not on the grid: Remin ~ +-15, Even Less ~ +-1.)
    Both are seeded per config, so reruns are identical.
"""
import argparse
import itertools
import json
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from data_loader import (calculate_mse, compute_dataset_metrics, generate_sampled_reconstructions,
                         grid_placement_space, mass_centroid_mse, oriented_box_space, orient_to_truth,
                         procrustes_align)
from falzon import falzon_mse_distribution
from globals_vars import format_db_name

BASE_DIR = "../databases"
HIST_DIR = "../figures/histogram_mse_sol_num"
DATASETS = os.environ.get("REBUTTAL_DATASETS", "busstop,shopparis,cali,drink,highway,spitz").split(",")
DISTS = os.environ.get("REBUTTAL_DISTS", "uniform,gaussian,beta").split(",")
BASELINES = ["even_less", "remin"]
LABEL = {"limits": "LAMa", "even_less": "Even Less", "remin": "Remin", "reminp": "Remin+"}
MISSING = "MISSING"


# =========================================================
# Small helpers
# =========================================================

def rust_float(v):
    """Rust's Display for f64 in file names: 1.0 -> '1', 0.1 -> '0.1'."""
    return f"{v:g}"


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def fmt(v, nd=2):
    if v is None:
        return MISSING
    if isinstance(v, str):
        return v
    if isinstance(v, (int, np.integer)):
        return str(v)
    return f"{v:.{nd}f}"


def md_table(header, rows, range_cols=None):
    """Prints a markdown table. range_cols: indices of numeric columns to summarise in a min-max row."""
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")
    for r in rows:
        print("| " + " | ".join(fmt(c) for c in r) + " |")
    if range_cols and rows:
        rng =["**min–max**"] + [""] * (len(header) - 1)
        for c in range_cols:
            vals = [r[c] for r in rows if isinstance(r[c], (int, float, np.integer, np.floating))]
            if vals:
                lo, hi = min(vals), max(vals)
                rng[c] = f"{fmt(lo)}–{fmt(hi)}"
            else:
                rng[c] = MISSING
        print("| " + " | ".join(rng) + " |")
    print()


def parallel_rows(fn, tasks, default_workers=6):
    """fn(*task) for every task on several CPUs (each config is independent); results in task order."""
    workers = max(1, min(len(tasks), (os.cpu_count() or 2) - 1,
                         int(os.environ.get("REBUTTAL_WORKERS", default_workers))))
    if workers == 1:
        return [fn(*t) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, *zip(*tasks)))


def h(title):
    print("\n" + "#" * 3 + " " + title + "\n", flush=True)


def runs_to_arrays(runs):
    """LAMa reconstruction JSON (list of runs) -> (true (n,d), [recon (n,d), ...])."""
    true = np.array([p["true"] for p in runs[0]], dtype=float)
    recons = [np.array([p["reconstructed"] for p in run], dtype=float) for run in runs if run]
    return true, recons


def lama_paths(grid, db, dist, query=None):
    """query=None -> ideal case (all queries, exact frequencies), else fraction e.g. 0.1."""
    d = f"{BASE_DIR}/{grid}/{db}/limits"
    if query is None:
        name = f"{db}_{dist}_e0_d0.9"
        return f"{d}/{name}_reconstruction.json", f"{d}/results_{name}.json"
    name = f"{db}_{dist}_p{rust_float(query)}"
    return f"{d}/{name}_reconstruction.json", f"{d}/results_{name}.json"


def baseline_path(grid, db, dist, method, query_pct):
    tag = "even_less" if method == "even_less" else "classic"
    return f"{BASE_DIR}/{grid}/{db}/{method}/{db}_prob{float(query_pct)}_{dist}_{grid}_{tag}.json"


def lama_true_in_output(true, recons):
    return any(np.array_equal(true, r) for r in recons)


# =========================================================
# Item 2 alignment: plaintext-free, 8 axis-aligned orientations
# =========================================================

def bbox_align_candidates(embedding, n):
    """
    Scale and translate the embedding so its bounding box fills the known domain [1, n]^2 (per axis),
    round to the grid, then return the 8 axis-aligned orientations (4 rotations x reflection) of the
    grid square. Uses no plaintext.
    """
    e = np.asarray(embedding, dtype=float)
    lo, hi = e.min(axis=0), e.max(axis=0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    g = np.round(1 + (e - lo) / span * (n - 1))
    x, y = g[:, 0], g[:, 1]
    m = n + 1  # reflections about the centre of [1, n]
    orients = [
        (x, y), (m - y, x), (m - x, m - y), (y, m - x),          # rotations 0/90/180/270
        (y, x), (m - x, y), (m - y, m - x), (x, m - y),          # the same, reflected
    ]
    return [np.column_stack(o) for o in orients]


SWEEP_SAMPLES = 1000
SWEEP_SHIFTS = [-5.0, -2.0, 0.0, 2.0, 5.0]
SWEEP_ANGLES = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
SWEEP_SCALES = [1.0, 0.9, 1.1]


def sweep_reconstruction_space(embedding, n, seed, num_samples=SWEEP_SAMPLES):
    """
    A baseline's reconstruction space: num_samples databases, each the plaintext-free grid placement of the
    embedding (bounding box -> [1, n]^2, as in bbox_align_candidates, not rounded) scaled and rotated about
    the grid centre and shifted, with the transformation drawn like generate_sampled_reconstructions.
    """
    e = np.asarray(embedding, dtype=float)
    lo, hi = e.min(axis=0), e.max(axis=0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    placed = 1 + (e - lo) / span * (n - 1)
    centre = (n + 1) / 2.0
    rng = random.Random(seed)
    clouds = []
    for _ in range(num_samples):
        dx, dy = rng.choice(SWEEP_SHIFTS), rng.choice(SWEEP_SHIFTS)
        rad = np.radians(rng.choice(SWEEP_ANGLES))
        s = rng.choice(SWEEP_SCALES)
        x, y = (placed[:, 0] - centre) * s, (placed[:, 1] - centre) * s
        c, sn = np.cos(rad), np.sin(rad)
        clouds.append(np.column_stack((x * c - y * sn + centre + dx, x * sn + y * c + centre + dy)))
    return clouds


def box_reconstruction_space(embedding, n, seed, num_samples=SWEEP_SAMPLES):
    """
    num_samples databases drawn uniformly from every placement of the embedding on the n x n grid: one of the
    8 axis-aligned orientations of its bounding box, scaled into a w x h box (w, h in 1..n) at any position
    inside [1, n]^2, rounded to grid cells. A uniform draw over all 8 * (n(n+1)/2)^2 placements = w with
    weight n+1-w (its number of positions), h likewise, then a uniform position and orientation.
    """
    e = np.asarray(embedding, dtype=float)
    lo, hi = e.min(axis=0), e.max(axis=0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    u, v = ((e - lo) / span).T  # unit square
    orients = [(u, v), (1 - v, u), (1 - u, 1 - v), (v, 1 - u), (v, u), (1 - u, v), (1 - v, 1 - u), (u, 1 - v)]
    rng = np.random.default_rng(seed)
    sizes = np.arange(1, n + 1)
    p = (n + 1 - sizes) / (n + 1 - sizes).sum()
    ws, hs = rng.choice(sizes, num_samples, p=p), rng.choice(sizes, num_samples, p=p)
    x0s = rng.integers(1, n + 2 - ws)  # upper bound exclusive: x0 in 1..n+1-w
    y0s = rng.integers(1, n + 2 - hs)
    os_ = rng.integers(0, 8, num_samples)
    clouds = []
    for w, h, x0, y0, o in zip(ws, hs, x0s, y0s, os_):
        ou, ov = orients[o]
        clouds.append(np.column_stack((x0 + np.round(ou * (w - 1)), y0 + np.round(ov * (h - 1)))))
    return clouds


BASELINE_SPACE = os.environ.get("REBUTTAL_BASELINE_SPACE", "oriented")  # set from --baseline-space


def config_seed(*parts):
    """Stable seed from the config name (Python's hash() of a str changes between runs)."""
    return int.from_bytes(" ".join(map(str, parts)).encode(), "little") % (2 ** 32)


def load_baseline_space(grid, db, dist, method, query_pct, n):
    """(true, reconstruction space) of a baseline run, or (None, None) if the file is missing."""
    data = load_json(baseline_path(grid, db, dist, method, query_pct))
    if not data:
        return None, None
    true = np.array([p["true"] for p in data], dtype=float)
    emb, seed = [p["reconstructed"] for p in data], config_seed(grid, db, dist, method, query_pct)
    if BASELINE_SPACE == "oriented":
        return true, oriented_box_space(true, emb, n, seed=seed)
    make = {"grid": grid_placement_space, "boxes": box_reconstruction_space,
            "sweep": sweep_reconstruction_space}[BASELINE_SPACE]
    return true, make(emb, n, seed=seed)


def lama_scored(true, recons, n):
    """LAMa's databases as scored in items 2b/4: with --baseline-space oriented (default) every method is told
    its orientation, so each LAMa database is put in its best of the 8 grid symmetries; else raw."""
    return orient_to_truth(true, recons, n) if BASELINE_SPACE == "oriented" else recons


def space_metrics(true, space):
    """[mean MSE over the space, hull centroid MSE, hull diameter MSE, mass centroid MSE] (module docstring)."""
    c, d = compute_dataset_metrics(space, true)
    return [float(np.mean([calculate_mse(true, x) for x in space])), c, d, mass_centroid_mse(space, true)]


def load_baseline_candidates(grid, db, dist, method, query_pct, n):
    data = load_json(baseline_path(grid, db, dist, method, query_pct))
    if not data:
        return None, None, None
    true = np.array([p["true"] for p in data], dtype=float)
    cands = bbox_align_candidates([p["reconstructed"] for p in data], n)
    return true, cands, data


def grid_n(grid):
    return int(grid.split("x")[0])


def mse_stats(true, cands):
    mses = [calculate_mse(true, c) for c in cands]
    return len(mses), min(mses), float(np.mean(mses)), max(mses), mses


# =========================================================
# Item 1
# =========================================================

def item1():
    h("Item 1a — LAMa vs Falzon et al. (figures falzon_guarantees_*): ideal case, 50x50 grid")
    print("Grid: 50x50 (TwoDMap target grid (50,50), coordinates 1..50 in the saved JSON). LAMa: all queries,"
          " exact frequencies (perfect setting, eps ~ 1e-6).")
    print("Falzon et al.: NOT re-run. falzon.py constructs their output E(D) directly from the true DB "
          "(Thm 4.4: reflect any subset of the r reflectable components, then the 4 global flips) -> 4*2^r DBs.")
    print("Falzon et al. do not output a single best guess: E(D) is an unordered set and contains the true DB by "
          "construction. In the 'Without Guarantees' panels the figure re-uses LAMa's histogram for "
          "Gaussian/Beta (plot_falzon_guarantees.panel_mses), i.e. nothing was computed for Falzon there.\n")
    rows = []
    for db in ["spitz", "busstop", "cali", "drink", "highway", "shopparis"]:
        falzon_n = falzon_r = None
        for dist in DISTS:
            rpath, _ = lama_paths("50x50", db, dist)
            runs = load_json(rpath)
            if not runs:
                rows.append([format_db_name(db), dist, None, None, None, None, None, None])
                continue
            true, recons = runs_to_arrays(runs)
            hist = load_json(f"{HIST_DIR}/mse_histogram_{db}_{dist}.json")
            hist_n = len(hist.get("LAMa", [[]])[0]) if hist else None
            if falzon_n is None:
                _, r = falzon_mse_distribution(true, (50, 50), max_exact=1)
                falzon_n, falzon_r = 4 * 2 ** r, r
            rows.append([format_db_name(db), dist, len(recons), hist_n if hist_n is not None else "no hist JSON",
                         "yes" if lama_true_in_output(true, recons) else "NO", falzon_r, falzon_n,
                         "n/a (set; D ∈ E(D))"])
    md_table(["Dataset", "Dist", "# DBs LAMa (recon JSON)", "# DBs LAMa (fig. hist JSON)", "true DB in LAMa out",
              "r (reflectable comps)", "# DBs Falzon (4·2^r)", "Falzon single best guess = true?"], rows,
             range_cols=[2, 6])

    h("Item 1b — TV-distance experiment (tab:lama_noise_sweep)")
    print("- Grid: 15x15 (databases/15x15, records on 1..15 per axis; sweep run with --query-percent 1.0).")
    print("- Frequencies: EXACT. With query_percent == 1.0 no queries are sampled: the observed frequency of a")
    print("  record tuple is the exact enclosing probability under the perturbed distribution "
          "(LAMA/mod.rs, `else` branch of get_observed_prob).")
    print("- Perturbation (QueryDistribution::add_noise): p' = (1-λ)p + λq, q = i.i.d. Uniform(0,1) weight per query,")
    print("  normalised, drawn from StdRng seed 1337 in sorted-query order; λ = noise / TV(p,q) so TV(p,p') = noise")
    print("  exactly (λ capped at 1). Attacker uses the unperturbed p; eps = max over MBQs of |P'(enclose) - P(enclose)|.")
    print("- Trials per cell: 1 (single run per dataset/dist/noise; q is seeded so a rerun gives the same p').")
    rows = []
    for db in DATASETS:
        for dist in DISTS:
            row = [format_db_name(db), dist]
            for nz in (0.005, 0.01, 0.02, 0.03, 0.04, 0.05):
                res = load_json(f"{BASE_DIR}/15x15/{db}/limits/results_{db}_{dist}_p1_noise{rust_float(nz)}.json")
                row.append(f"eps={res['eps']:.4f}" if res else None)
            rows.append(row)
    print()
    md_table(["Dataset", "Dist", "TV 0.005", "TV 0.01", "TV 0.02", "TV 0.03", "TV 0.04", "TV 0.05"], rows)

    h("Item 1c — Query distribution definitions (src/LAMA/query.rs)")
    print("Let Q be the list of all M = (n(n+1)/2)^2 axis-aligned range queries of the grid (index 0..n-1 per axis),")
    print("in the order returned by Selector::get_dom_pairs. Query i gets weight w_i; P(query i) = w_i / Σ_j w_j.")
    print("- Uniform:  w_i = 1.")
    print("- Gaussian: w_i = N(i; μ = M/2, σ = M/5) pdf (\"params taken from REMIN\").")
    print("- Beta:     w_i = Beta(α=2, β=1) pdf at x = (i + 0.5)/M, i.e. w_i = 2x (linearly increasing).")
    print("Probability the attacker uses for a record tuple = Σ of P(query) over all queries enclosing the tuple's")
    print("minimum bounding box (prefix-sum precompute, QueryDistribution::precompute_cumulative_results).")
    print("CAVEAT: get_dom_pairs builds Q with rayon par_bridge, so the ORDER of Q (and hence which query gets which")
    print("Gaussian/Beta weight) is not deterministic across runs. Within one run attacker and sampler share the same")
    print("order, so results are self-consistent, but the Gaussian/Beta distributions are not a fixed function of the")
    print("query's geometry. Uniform is unaffected.\n")

    h("Item 1d — Largest level (tuple size t) matched before stopping")
    rows = []
    for grid, name in [("16x16x14", "nh"), ("5x5", "grid"), ("6x6", "grid"), ("6x6x6", "grid"),
                       ("6x6x6x6", "grid"), ("6x6x6x6x6", "grid")]:
        for dist in DISTS:
            for suffix in ("e0_d0.001", "e0_d0.9"):
                res = load_json(f"{BASE_DIR}/{grid}/{name}/limits/results_{name}_{dist}_{suffix}.json")
                if not res:
                    continue
                meta = res.get("translator_meta")
                if meta:
                    t = max(int(k) for k in meta["recons_per_t"])
                    level = f"t={t} (recons at t: {meta['recons_per_t'][str(t)]})"
                else:
                    level = "not recorded (result predates translator_meta)"
                runs = load_json(f"{BASE_DIR}/{grid}/{name}/limits/{name}_{dist}_{suffix}_reconstruction.json")
                rows.append([grid, name, dist, suffix, level, len(runs) if runs else None, res["total_db_size"]])
                break
    md_table(["Grid", "Data", "Dist", "Run", "Largest level", "# DBs output", "# records"], rows)
    print("Note: 'number_of_reconstructions' in results_*.json is the number of RECORDS (len of the record->values")
    print("map), not the number of databases; '# DBs output' above is counted from the reconstruction JSON.\n")


# =========================================================
# Items 2 & 3
# =========================================================

def table2_row(grid, n, q, db, dist, original_sweep):
    row = [format_db_name(db), dist]
    runs = load_json(lama_paths(grid, db, dist, q / 100)[0])
    if runs:
        true, recons = runs_to_arrays(runs)
        row += [len(recons)] + space_metrics(true, lama_scored(true, recons, n))
    else:
        row += [None] * 5
    for m in BASELINES:
        true, space = load_baseline_space(grid, db, dist, m, q, n)
        if space is None:
            row += [None] * (6 if original_sweep else 4)
            continue
        row += space_metrics(true, space)
        if original_sweep:
            data = load_json(baseline_path(grid, db, dist, m, q))
            random.seed(0)
            clouds, t = generate_sampled_reconstructions(data)
            oc, od = compute_dataset_metrics(clouds, t)
            row += [oc, od]
    return row


def item4_row(grid, n, q, db, dist):
    return [f"{q:g}%", format_db_name(db), dist] + realistic_row(grid, n, db, dist, q)


def item2_and_3(original_sweep=False):
    grid, n = "50x50", 50
    h("Item 2a — Plaintext-free alignment (bbox -> [1,N]^2, round, 8 orientations): ideal case, 50x50")
    print("LAMa: every database it outputs, raw (no alignment). Baselines: prob100.0 runs, 8 candidates each.\n")
    rows, exp_rows = [], []
    for db in DATASETS:
        for dist in DISTS:
            row, erow = [format_db_name(db), dist], [format_db_name(db), dist]
            runs = load_json(lama_paths(grid, db, dist)[0])
            if runs:
                true, recons = runs_to_arrays(runs)
                k, lo, mean, hi, _ = mse_stats(true, recons)
                row += [k, lo, mean, hi]
                erow += [mean]
            else:
                row += [None] * 4
                erow += [None]
            for m in BASELINES + ["reminp"]:
                true, cands, _ = load_baseline_candidates(grid, db, dist, m, 100.0, n)
                if cands is None:
                    if m != "reminp":
                        row += [None] * 4
                    erow += [None]
                    continue
                k, lo, mean, hi, _ = mse_stats(true, cands)
                k_unique = len({c.tobytes() for c in cands})
                if m != "reminp":
                    row += [f"8 ({k_unique} unique)", lo, mean, hi]
                erow += [mean]
            rows.append(row)
            exp_rows.append(erow)
    hdr = ["Dataset", "Dist"]
    for m in ["limits"] + BASELINES:
        hdr += [f"{LABEL[m]} #", f"{LABEL[m]} best", f"{LABEL[m]} mean", f"{LABEL[m]} worst"]
    md_table(hdr, rows, range_cols=[2, 3, 4, 5, 7, 8, 9, 11, 12, 13])

    h("Item 3 — Expected error: mean MSE over all returned DBs (uniform pick), ideal case, 50x50")
    print("Baselines use the item-2 alignment (mean over the 8 candidates). Remin+ included for reference.\n")
    md_table(["Dataset", "Dist", "LAMa", "Even Less", "Remin", "Remin+"], exp_rows, range_cols=[2, 3, 4, 5])

    grid, n, q = "50x50", 50, 30.0
    h("Item 2b — Table II eps with item-2 alignment: 50x50, 30% queries")
    print("Mean MSE over the reconstruction space, and centroid / diameter MSE of the per-record convex hull "
          "over it. LAMa: every database it output (*_p0.3_reconstruction.json). Even Less / Remin: "
          f"{SWEEP_SAMPLES} databases from their reconstruction space ('{BASELINE_SPACE}', module docstring).")
    if original_sweep:
        print("'orig' columns: the paper's Table II procedure (random rotation/scale/translation sweep, 10000 "
              "samples, random.seed(0)), recomputed here for reference.")
    print()
    rows = parallel_rows(table2_row, [(grid, n, q, db, dist, original_sweep) for db in DATASETS for dist in DISTS])
    hdr = ["Dataset", "Dist", "LAMa #", "LAMa mean MSE", "LAMa cent.", "LAMa diam.", "LAMa mass cent."]
    for m in BASELINES:
        hdr += [f"{LABEL[m]} mean MSE", f"{LABEL[m]} cent.", f"{LABEL[m]} diam.", f"{LABEL[m]} mass cent."]
        if original_sweep:
            hdr += [f"{LABEL[m]} cent. (orig)", f"{LABEL[m]} diam. (orig)"]
    md_table(hdr, rows, range_cols=list(range(3, len(hdr))))
    print_wins(rows, hdr, key_cols=2)


# =========================================================
# Item 4
# =========================================================

def realistic_row(grid, n, db, dist, q_pct):
    """LAMa [#DBs, true in out, mean MSE, cent., diam., eps, time, consistent, true pairs outside] +
    per baseline [mean MSE, cent., diam.] over its swept reconstruction space."""
    row = []
    runs = load_json(lama_paths(grid, db, dist, q_pct / 100)[0])
    res = load_json(lama_paths(grid, db, dist, q_pct / 100)[1])
    if runs:
        true, recons = runs_to_arrays(runs)
        lns = (res or {}).get("lns")  # only in results from the pair-LNS code
        row += [len(recons), "yes" if lama_true_in_output(true, recons) else "NO"] + space_metrics(true, lama_scored(true, recons, n)) + [
                f"{res['eps']:.4f}" if res else None, res["time_taken"] if res else None,
                ("yes" if lns["output_is_consistent"] else "NO (best found)") if lns else "n/a (old code)",
                lns["true_db_pairs_outside"] if lns else "n/a"]
    else:
        row += [None] * 10
    for m in BASELINES:
        true, space = load_baseline_space(grid, db, dist, m, q_pct, n)
        row += space_metrics(true, space) if space is not None else [None] * 4
    return row


METRICS = ["mean MSE", "cent.", "diam.", "mass cent."]


def print_wins(rows, hdr, key_cols):
    """Per metric: in how many configs LAMa is strictly lower than BOTH baselines, plus the configs where it
    is not (lower = better). Configs missing a method are not counted."""
    print("**Does LAMa win?** (lower is better; LAMa must beat both Even Less and Remin)\n")
    out = []
    for metric in METRICS:
        li = hdr.index(f"LAMa {metric}")
        bi = [hdr.index(f"{LABEL[m]} {metric}") for m in BASELINES]
        wins, total, losses = 0, 0, []
        for r in rows:
            vals = [r[li]] + [r[i] for i in bi]
            if any(not isinstance(v, (float, int, np.floating)) for v in vals):
                continue
            total += 1
            if vals[0] < min(vals[1:]):
                wins += 1
            else:
                losses.append(" ".join(str(c) for c in r[:key_cols]) +
                              f" (LAMa {vals[0]:.1f} vs " + ", ".join(f"{v:.1f}" for v in vals[1:]) + ")")
        out.append([metric, f"{wins}/{total}", "; ".join(losses) if losses else "none"])
    md_table(["Metric", "LAMa wins", "Configs where LAMa does not win"], out)


def item4():
    hdr_tail = ["LAMa #DBs", "true DB in out", "LAMa mean MSE", "LAMa cent.", "LAMa diam.", "LAMa mass cent.", "eps", "LAMa time (s)",
                "all pairs within eps", "true DB pairs outside eps"]
    for m in BASELINES:
        hdr_tail += [f"{LABEL[m]} mean MSE", f"{LABEL[m]} cent.", f"{LABEL[m]} diam.", f"{LABEL[m]} mass cent."]
    numeric = {"LAMa #DBs", "LAMa time (s)", "true DB pairs outside eps"}
    for grid, qs in [("50x50", [1.0, 2.0, 5.0, 10.0, 20.0, 30.0])]:
        n = grid_n(grid)
        h(f"Item 4 — Realistic case (sampled queries), {grid}")
        print(f"Reconstruction spaces: LAMa = every database it output; Even Less / Remin = {SWEEP_SAMPLES} "
              f"databases from '{BASELINE_SPACE}' (module docstring). MISSING = run never done / not on disk.\n")
        rows = parallel_rows(item4_row, [(grid, n, q, db, dist) for q in qs for db in DATASETS for dist in DISTS])
        hdr = ["Query %", "Dataset", "Dist"] + hdr_tail
        md_table(hdr, rows, range_cols=[i for i, c in enumerate(hdr)
                                        if c in numeric or c.endswith(("mean MSE", "cent.", "diam."))])
        print_wins(rows, hdr, key_cols=3)


def item5():
    h("Item 5 — Non-independent (correlated) queries")
    print("MISSING: no correlated-query runs exist and the code has no option for it. "
          "Selector::sample_percent_responses only draws i.i.d. from the WeightedIndex sampler. "
          "Needs a new sampler (e.g. Metropolis–Hastings walk over neighbouring queries with stay/step prob rho) "
          "and new runs for LAMa, Even Less and Remin.\n")


# =========================================================
# Round 2 (second request from the response writer): items r1-r5, 2D only, 50x50.
# One protocol for every method ("domainfit"): baselines = the item-2a alignment (bbox -> [1,N]^2, round,
# 8 orientations), no Procrustes, no random boxes; LAMa = its output databases as they are.
# MSE = calculate_mse (mean squared Euclidean distance, NOT divided by dims) everywhere in r1-r3.
# =========================================================

R2_GRID, R2_N = "50x50", 50
R2_QS = [1.0, 2.0, 5.0, 10.0, 20.0, 30.0]
R2_TVS = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
# Same 8 maps (and order) as bbox_align_candidates / data_loader.grid_symmetries; m = n + 1.
_SYMS = [lambda x, y, m: (x, y), lambda x, y, m: (m - y, x), lambda x, y, m: (m - x, m - y),
         lambda x, y, m: (y, m - x), lambda x, y, m: (y, x), lambda x, y, m: (m - x, y),
         lambda x, y, m: (m - y, m - x), lambda x, y, m: (x, m - y)]


def sym(a, n, s):
    """Grid symmetry s (0..7) of a (..., 2) array of points on [1, n]^2."""
    return np.stack(_SYMS[s](a[..., 0], a[..., 1], n + 1), axis=-1)


def sym_mse_matrix(true, stack, n):
    """(8, K): calculate_mse of every database in stack (K, R, 2) under every grid symmetry."""
    return np.stack([((sym(stack, n, s) - true) ** 2).sum(-1).mean(-1) for s in range(8)])


def align_to_first(stack, n):
    """Each database in whichever of the 8 symmetries is closest (MSE) to database 0. No plaintext used."""
    ref, out = stack[0], np.empty_like(stack)
    best = np.stack([((sym(stack, n, s) - ref) ** 2).sum(-1).mean(-1) for s in range(8)]).argmin(0)
    for s in range(8):
        if (best == s).any():
            out[best == s] = sym(stack[best == s], n, s)
    return out


def _circle2(a, b):
    c = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    return c, math.hypot(a[0] - c[0], a[1] - c[1])


def _circle3(a, b, c):
    d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    if abs(d) < 1e-12:  # collinear: the widest pair
        return max((_circle2(a, b), _circle2(a, c), _circle2(b, c)), key=lambda t: t[1])
    sa, sb, sc = a[0] ** 2 + a[1] ** 2, b[0] ** 2 + b[1] ** 2, c[0] ** 2 + c[1] ** 2
    ux = (sa * (b[1] - c[1]) + sb * (c[1] - a[1]) + sc * (a[1] - b[1])) / d
    uy = (sa * (c[0] - b[0]) + sb * (a[0] - c[0]) + sc * (b[0] - a[0])) / d
    return (ux, uy), math.hypot(a[0] - ux, a[1] - uy)


def min_enclosing_circle(points, rng):
    """Smallest enclosing ball (2D) of a set of points, Welzl's randomised incremental algorithm. (centre, r)"""
    p = [tuple(map(float, q)) for q in np.unique(np.asarray(points), axis=0)]
    rng.shuffle(p)
    inside = lambda c, r, q: math.hypot(q[0] - c[0], q[1] - c[1]) <= r + 1e-9
    c, r = p[0], 0.0
    for i in range(1, len(p)):
        if not inside(c, r, p[i]):
            c, r = p[i], 0.0
            for j in range(i):
                if not inside(c, r, p[j]):
                    c, r = _circle2(p[i], p[j])
                    for k in range(j):
                        if not inside(c, r, p[k]):
                            c, r = _circle3(p[i], p[j], p[k])
    return c, r


def single_db_scores(true, db, n):
    """(up to symmetry, expected) of one database: min and mean of calculate_mse over its 8 images."""
    m = sym_mse_matrix(true, db[None], n)[:, 0]
    return float(m.min()), float(m.mean())


def method_metrics(true, stack, n, seed):
    """r1 + r2 numbers for one method's output stack (K, R, 2)."""
    m = sym_mse_matrix(true, stack, n)
    out = {"k": len(stack), "sym": float(m.min(0).mean()), "exp": float(m[0].mean())}
    al = align_to_first(stack, n)
    mass = al.mean(0)
    rng = random.Random(seed)
    balls = [min_enclosing_circle(al[:, i], rng) for i in range(al.shape[1])]
    seb = np.array([b[0] for b in balls])
    radii = np.array([b[1] for b in balls])
    out["mass_sym"], out["mass_exp"] = single_db_scores(true, mass, n)
    out["seb_sym"], out["seb_exp"] = single_db_scores(true, seb, n)
    out["max_r"], out["mean_r"] = float(radii.max()), float(radii.mean())
    return out


def procrustes_mse(true, emb):
    """MSE after Procrustes (rotation, scale, shift) allowing a REFLECTION too: the better of the embedding and
    its mirror image. data_loader.procrustes_align forces det(R) = +1 (rotations only), so a mirrored output
    collapses to scale ~0 and scores the truth's variance (~362 at 50x50)."""
    e = np.asarray(emb, dtype=float)
    return float(min(calculate_mse(true, procrustes_align(true, e)),
                     calculate_mse(true, procrustes_align(true, e * [1.0, -1.0]))))


def r2_config(q, db, dist):
    """Everything items r1-r4 need for one config; the LAMa JSON is loaded once. q = 100 -> ideal case."""
    grid, n = R2_GRID, R2_N
    seed = config_seed(grid, db, dist, q)
    out = {"q": q, "db": db, "dist": dist}
    rpath, respath = lama_paths(grid, db, dist, None if q == 100 else q / 100)
    res = load_json(respath)
    if res:
        out["res"] = {k: res.get(k) for k in ("eps", "lns", "merged_from", "time_taken", "time_taken_per_run")}
    runs = load_json(rpath)
    if runs:
        true = np.array([p["true"] for p in runs[0]], dtype=float)
        stack = np.array([[p["reconstructed"] for p in run] for run in runs if run], dtype=float)
        del runs
        mm = method_metrics(true, stack, n, seed)
        procs = [procrustes_mse(true, d) for d in stack]
        mm["proc_best"], mm["proc_first"] = float(min(procs)), float(procs[0])
        mm["true_in_out"] = bool((stack == true).all(axis=(1, 2)).any())
        out["limits"] = mm
    for m in BASELINES:
        data = load_json(baseline_path(grid, db, dist, m, q))
        if not data:
            continue
        true = np.array([p["true"] for p in data], dtype=float)
        emb = [p["reconstructed"] for p in data]
        mm = method_metrics(true, np.stack(bbox_align_candidates(emb, n)), n, seed)
        mm["proc"] = procrustes_mse(true, emb)
        out[m] = mm
    return out


def g(cfg, method, key):
    return cfg.get(method, {}).get(key)


def consistent_flag(cfg):
    lns = (cfg.get("res") or {}).get("lns")
    if not cfg.get("limits"):
        return None
    if not lns:
        return "n/a (no lns)"
    return "yes" if lns["output_is_consistent"] else "NO"


def print_wins_cols(rows, hdr, key_cols, metrics):
    """metrics: [(name, LAMa column, [baseline columns])]. LAMa wins = strictly lower than every baseline."""
    out = []
    for name, lc, bcs in metrics:
        li, bi = hdr.index(lc), [hdr.index(c) for c in bcs]
        wins, total, losses = 0, 0, []
        for r in rows:
            vals = [r[li]] + [r[i] for i in bi]
            if any(not isinstance(v, (float, int, np.floating)) for v in vals):
                continue
            total += 1
            if vals[0] < min(vals[1:]):
                wins += 1
            else:
                losses.append(" ".join(str(c) for c in r[:key_cols]) +
                              f" (LAMa {vals[0]:.1f} vs " + ", ".join(f"{v:.1f}" for v in vals[1:]) + ")")
        out.append([name, f"{wins}/{total}", "; ".join(losses) if losses else "none"])
    md_table(["Metric", "LAMa wins", "Configs where LAMa does not win"], out)


R1_WINS = [("up to symmetry", "LAMa sym", ["Even Less sym", "Remin sym"]),
           ("expected", "LAMa exp", ["Even Less exp", "Remin exp"])]


def r1_rows(cfgs):
    rows = []
    for c in cfgs:
        if c["q"] == 100:
            continue
        rows.append([f"{c['q']:g}%", format_db_name(c["db"]), c["dist"], g(c, "limits", "k"), consistent_flag(c),
                     g(c, "limits", "sym"), g(c, "limits", "exp")] +
                    [g(c, m, k) for m in BASELINES for k in ("sym", "exp")])
    hdr = ["Query %", "Dataset", "Dist", "LAMa #DBs", "all pairs within eps", "LAMa sym", "LAMa exp"] + \
          [f"{LABEL[m]} {k}" for m in BASELINES for k in ("sym", "exp")]
    return hdr, rows


def item_r1(cfgs):
    h("Item r1 — Sampled queries, one protocol ('domainfit'), 50x50, 1/2/5/10/20/30%")
    print("LAMa: every output database as written (no alignment). Even Less / Remin: their single embedding, "
          "bbox scaled to [1,50]^2, rounded, 8 orientations (= item 2a); no Procrustes, no random boxes, no "
          "plaintext.")
    print("- sym = up to symmetry: per output database min MSE over the 8 grid symmetries, then the mean over the "
          "output (baseline: the best of its 8 candidates, since they are each other's symmetries).")
    print("- exp = expected: mean raw MSE over the whole output (baseline: mean of its 8 candidates).")
    print("- MSE = mean squared Euclidean distance in grid cells (not /dims). 'all pairs within eps' = NO: the run "
          "found no fully consistent database and output its best ones instead (see r4).\n")
    hdr, rows = r1_rows(cfgs)
    md_table(hdr, rows, range_cols=[3] + list(range(5, len(hdr))))
    print("**Does LAMa win?** (lower is better; LAMa must beat both Even Less and Remin)\n")
    print_wins_cols(rows, hdr, 3, R1_WINS)


def item_r2(cfgs):
    h("Item r2 — Single database without plaintext (per-record mass centroid / smallest-enclosing-ball centre)")
    print("Each method's output databases are aligned to its FIRST one (each database in its best of the 8 grid "
          "symmetries w.r.t. database 0, MSE; no plaintext). Per record: mass centroid (mean position over all "
          "aligned databases) and the centre / radius of the smallest enclosing ball (Welzl) of its candidate "
          "positions. Each resulting single database is scored up to symmetry (min MSE over its 8 images) and "
          "expected (mean MSE over its 8 images). MSE not /dims. max r / mean r = max / mean over records of the "
          "ball radius, in grid cells (max r = the new Table II epsilon). Hull-diameter midpoint dropped.")
    print("Baselines: the same on their 8 domainfit candidates. These are exact symmetries of each other, so "
          "aligning them to the first makes all 8 identical: radius 0 and the single database = candidate 0 "
          "(scores equal r1's best-of-8 / mean-of-8). Without alignment every centre is the grid centre.\n")
    rows = []
    for c in cfgs:
        if c["q"] == 100:
            continue
        row = [f"{c['q']:g}%", format_db_name(c["db"]), c["dist"], g(c, "limits", "k")]
        row += [g(c, "limits", k) for k in ("mass_sym", "mass_exp", "seb_sym", "seb_exp", "max_r", "mean_r")]
        for m in BASELINES:
            row += [g(c, m, k) for k in ("mass_sym", "mass_exp", "seb_sym", "seb_exp", "max_r")]
        rows.append(row)
    hdr = ["Query %", "Dataset", "Dist", "LAMa #DBs", "LAMa mass sym", "LAMa mass exp", "LAMa ball sym",
           "LAMa ball exp", "LAMa max r", "LAMa mean r"]
    for m in BASELINES:
        hdr += [f"{LABEL[m]} {k}" for k in ("mass sym", "mass exp", "ball sym", "ball exp", "max r")]
    md_table(hdr, rows, range_cols=list(range(3, len(hdr))))
    print("**Does LAMa win?**\n")
    print_wins_cols(rows, hdr, 3, [
        (f"{a} {b}", f"LAMa {a} {b}", [f"{LABEL[m]} {a} {b}" for m in BASELINES])
        for a in ("mass", "ball") for b in ("sym", "exp")])


def item_r3(cfgs):
    h("Item r3 — Baselines' own protocol: Procrustes (rotation, reflection, scale, shift) to the truth")
    print("data_loader.procrustes_align (uses the truth), best of the embedding and its mirror image (that "
          "function alone allows rotations only), then MSE (not /dims). Even Less / Remin: their single "
          "embedding. LAMa best = min over its output databases (uses the truth to pick); LAMa first = database 0, "
          "the lowest attacker-side error (picked without the truth). 'ideal' = all queries, exact frequencies "
          "(LAMa e0_d0.9 run, baselines prob100).\n")
    rows = []
    for c in cfgs:
        rows.append(["ideal" if c["q"] == 100 else f"{c['q']:g}%", format_db_name(c["db"]), c["dist"],
                     g(c, "limits", "proc_best"), g(c, "limits", "proc_first")] +
                    [g(c, m, "proc") for m in BASELINES])
    hdr = ["Query %", "Dataset", "Dist", "LAMa best", "LAMa first"] + [LABEL[m] for m in BASELINES]
    md_table(hdr, rows, range_cols=[3, 4, 5, 6])
    print_wins_cols(rows, hdr, 3, [("Procrustes, LAMa best DB", "LAMa best", [LABEL[m] for m in BASELINES]),
                                   ("Procrustes, LAMa first DB", "LAMa first", [LABEL[m] for m in BASELINES])])


def item_r4(cfgs):
    h("Item r4 — Runs that did not find a fully consistent database ('all pairs within eps' = NO)")
    print("Time limit is NOT stored in the results; run_me.sh gives the LNS (1 - 0.15) * limit - 0.25 h, so for a "
          "run that hit its deadline the limit is inferred as (LNS seconds / 3600 + 0.25) / 0.85 (job limits are "
          "1/2/4/8/16 ... h). For a merged config (2D, since 2026-11-18) the lns fields are those of the run that "
          "was copied (no run was consistent); 'runs' lists every run's (#DBs, time h).\n")
    rows = []
    for c in cfgs:
        if c["q"] == 100 or consistent_flag(c) != "NO":
            continue
        res = c["res"]
        lns = res["lns"]
        lim = (lns["lns_seconds"] / 3600 + 0.25) / 0.85 if lns.get("hit_deadline") else None
        mf = res.get("merged_from") or {}
        runs = ", ".join(f"{k}: {v['returned']} DBs {v['time_taken'] / 3600:.1f} h" for k, v in sorted(mf.items()))
        rows.append([f"{c['q']:g}%", format_db_name(c["db"]), c["dist"], lns["returned"],
                     lns["best_pairs_outside"], lns["true_db_pairs_outside"],
                     "yes" if lns.get("hit_deadline") else "no", lns["lns_seconds"] / 3600,
                     f"≈{lim:.1f} h (inferred)" if lim else "n/a (no deadline hit)",
                     lns.get("pipeline_attempt"), runs or "single run"])
    md_table(["Query %", "Dataset", "Dist", "#DBs output", "best DB pairs outside eps", "true DB pairs outside eps",
              "hit deadline", "LNS time (h)", "time limit", "pipeline attempt", "runs (merged)"], rows,
             range_cols=[3, 4, 7])
    print(f"{len(rows)} configs without a fully consistent output.\n")
    hdr, all_rows = r1_rows(cfgs)
    kept = [r for r in all_rows if r[4] != "NO"]
    print(f"**r1 win counts with those configs excluded** ({len(kept)} of {len(all_rows)} configs left)\n")
    print_wins_cols(kept, hdr, 3, R1_WINS)


def item_r5():
    h("Item r5a — TV sweep at 50x50 (as tab:lama_noise_sweep): all queries, exact frequencies, true "
      "distribution TV away from the attacker's")
    print("Perfect pipeline (--query-percent 1.0 --dist-noise <tv> --lazy-pairs, run_me.sh NOISE mode); noise q "
          "seeded (StdRng 1337), 1 run per cell. Cell: #DBs / true DB in output / eps. #DBs = 0 or no "
          "reconstruction -> infeasible.\n")
    rows = []
    for db in DATASETS:
        for dist in DISTS:
            row = [format_db_name(db), dist]
            for tv in R2_TVS:
                name = f"{db}_{dist}_p1_noise{rust_float(tv)}"
                d = f"{BASE_DIR}/{R2_GRID}/{db}/limits"
                res, runs = load_json(f"{d}/results_{name}.json"), load_json(f"{d}/{name}_reconstruction.json")
                if res is None:
                    row.append(None)
                    continue
                k = len(runs) if runs else 0
                tin = "yes" if runs and lama_true_in_output(*runs_to_arrays(runs)) else "NO"
                row.append(f"{k} DBs / true {tin} / eps={res['eps']:.2e}" if k else
                           f"infeasible (0 DBs) / eps={res['eps']:.2e}")
            rows.append(row)
    md_table(["Dataset", "Dist"] + [f"TV {tv:g}" for tv in R2_TVS], rows)

    h("Item r5b — Misspecified query distribution (Reviewer C)")
    print("MISSING: no runs, and the code cannot do it yet. lama_attack builds ONE QueryDistribution from --dist "
          "and uses it both to sample the queries and as the attacker's model (only --dist-noise perturbs the "
          "sampling side, towards random weights). Needed: a separate attacker distribution, e.g. "
          "--attacker-dist / --attacker-sigma-scale (gaussian sigma = 1.1 * M/5 and 1.25 * M/5 while sampling "
          "with M/5; uniform attacker while sampling Beta(2,1)), then runs for the six datasets. Per case the "
          "table would give: feasible, true DB in output, #DBs, eps.\n")


def round2(items):
    cfgs = parallel_rows(r2_config, [(q, db, dist) for q in R2_QS + [100.0] for db in DATASETS for dist in DISTS],
                         default_workers=3)  # 3: json.load of a 1000-DB file takes a few GB
    if "r1" in items:
        item_r1(cfgs)
    if "r2" in items:
        item_r2(cfgs)
    if "r3" in items:
        item_r3(cfgs)
    if "r4" in items:
        item_r4(cfgs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="1,2,4,5,r1,r2,r3,r4,r5",
                    help="comma list of items (2 also prints item 3); r1-r5 = second request (50x50, domainfit)")
    ap.add_argument("--original-sweep", action="store_true",
                    help="also recompute the paper's random-sweep Table II eps for the baselines (slow)")
    ap.add_argument("--datasets", default=",".join(DATASETS), help="comma list (items 1b, 2, 4)")
    ap.add_argument("--dists", default=",".join(DISTS), help="comma list (items 1b, 2, 4)")
    ap.add_argument("--baseline-space", choices=["oriented", "grid", "boxes", "sweep"], default="oriented",
                    help="Even Less / Remin reconstruction space in items 2b and 4 (see the module docstring)")
    args = ap.parse_args()
    BASELINE_SPACE = args.baseline_space
    DATASETS, DISTS = args.datasets.split(","), args.dists.split(",")
    os.environ["REBUTTAL_DATASETS"], os.environ["REBUTTAL_DISTS"] = args.datasets, args.dists
    os.environ["REBUTTAL_BASELINE_SPACE"] = BASELINE_SPACE  # worker processes re-import this module
    items = set(args.items.split(","))
    t0 = time.time()
    if "1" in items:
        item1()
    if "2" in items or "3" in items:
        item2_and_3(args.original_sweep)
    if "4" in items:
        item4()
    if "5" in items:
        item5()
    if items & {"r1", "r2", "r3", "r4"}:
        round2(items)
    if "r5" in items:
        item_r5()
    print(f"\n_Generated in {time.time() - t0:.0f}s from {os.path.abspath(BASE_DIR)}_")
