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
"""
import argparse
import itertools
import json
import os
import random
import time

import numpy as np

from data_loader import calculate_mse, compute_dataset_metrics, generate_sampled_reconstructions
from falzon import falzon_mse_distribution
from globals_vars import format_db_name

BASE_DIR = "../databases"
HIST_DIR = "../figures/histogram_mse_sol_num"
DATASETS = ["busstop", "shopparis", "cali", "drink", "highway", "spitz"]
DISTS = ["uniform", "gaussian", "beta"]
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
    if range_cols:
        rng = ["**min–max**"] + [""] * (len(header) - 1)
        for c in range_cols:
            vals = [r[c] for r in rows if isinstance(r[c], (int, float, np.integer, np.floating))]
            if vals:
                lo, hi = min(vals), max(vals)
                rng[c] = f"{fmt(lo)}–{fmt(hi)}"
            else:
                rng[c] = MISSING
        print("| " + " | ".join(rng) + " |")
    print()


def h(title):
    print("\n" + "#" * 3 + " " + title + "\n")


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

    grid, n, q = "15x15", 15, 30.0
    h("Item 2b — Table II eps with item-2 alignment: 15x15, 30% queries")
    print("Centroid / diameter MSE of the per-record convex hull over the candidate set. LAMa: its sampled "
          "reconstructions (file *_p0.3_reconstruction.json). Baselines: the 8 bbox-aligned candidates.")
    if original_sweep:
        print("'orig' columns: the paper's Table II procedure (random rotation/scale/translation sweep, 10000 "
              "samples, random.seed(0)), recomputed here for reference.")
    print()
    rows = []
    for db in DATASETS:
        for dist in DISTS:
            row = [format_db_name(db), dist]
            runs = load_json(lama_paths(grid, db, dist, q / 100)[0])
            if runs:
                true, recons = runs_to_arrays(runs)
                c, d = compute_dataset_metrics(recons, true)
                row += [len(recons), c, d]
            else:
                row += [None] * 3
            for m in BASELINES:
                true, cands, data = load_baseline_candidates(grid, db, dist, m, q, n)
                if cands is None:
                    row += [None] * (4 if original_sweep else 2)
                    continue
                c, d = compute_dataset_metrics(cands, true)
                row += [c, d]
                if original_sweep:
                    random.seed(0)
                    clouds, t = generate_sampled_reconstructions(data)
                    oc, od = compute_dataset_metrics(clouds, t)
                    row += [oc, od]
            rows.append(row)
    hdr = ["Dataset", "Dist", "LAMa #", "LAMa cent.", "LAMa diam."]
    for m in BASELINES:
        hdr += [f"{LABEL[m]} cent.", f"{LABEL[m]} diam."]
        if original_sweep:
            hdr += [f"{LABEL[m]} cent. (orig)", f"{LABEL[m]} diam. (orig)"]
    md_table(hdr, rows, range_cols=list(range(3, len(hdr))))


# =========================================================
# Item 4
# =========================================================

def realistic_row(grid, n, db, dist, q_pct):
    """[LAMa #, true in out, median MSE, centroid MSE, eps, time] + per baseline [median MSE, centroid MSE]."""
    row = []
    runs = load_json(lama_paths(grid, db, dist, q_pct / 100)[0])
    res = load_json(lama_paths(grid, db, dist, q_pct / 100)[1])
    if runs:
        true, recons = runs_to_arrays(runs)
        mses = [calculate_mse(true, r) for r in recons]
        c, _ = compute_dataset_metrics(recons, true)
        lns = (res or {}).get("lns")  # only in results from the pair-LNS code
        row += [len(recons), "yes" if lama_true_in_output(true, recons) else "NO", float(np.median(mses)), c,
                res["eps"] if res else None, res["time_taken"] if res else None,
                ("yes" if lns["output_is_consistent"] else "NO (best found)") if lns else "n/a (old code)",
                lns["true_db_pairs_outside"] if lns else "n/a"]
    else:
        row += [None] * 8
    for m in BASELINES:
        true, cands, _ = load_baseline_candidates(grid, db, dist, m, q_pct, n)
        if cands is None:
            row += [None, None]
            continue
        mses = [calculate_mse(true, cnd) for cnd in cands]
        c, _ = compute_dataset_metrics(cands, true)
        row += [float(np.median(mses)), c]
    return row


def item4():
    hdr_tail = ["LAMa #DBs", "true DB in out", "LAMa median MSE", "LAMa cent.", "eps", "LAMa time (s)",
                "all pairs within eps", "true DB pairs outside eps",
                "EL median MSE", "EL cent.", "Remin median MSE", "Remin cent."]
    for grid, qs in [("15x15", [1.0, 2.0, 5.0, 10.0, 20.0, 30.0]), ("30x30", [10.0, 20.0, 30.0]),
                     ("50x50", [1.0, 2.0, 5.0, 10.0, 20.0, 30.0])]:
        n = grid_n(grid)
        h(f"Item 4 — Realistic case (sampled queries), {grid}")
        print("Baselines use the item-2 alignment (8 candidates). MISSING = run never done / not on disk.\n")
        rows = []
        for q in qs:
            for db in DATASETS:
                for dist in DISTS:
                    rows.append([f"{q:g}%", format_db_name(db), dist] + realistic_row(grid, n, db, dist, q))
        md_table(["Query %", "Dataset", "Dist"] + hdr_tail, rows, range_cols=[3, 5, 6, 7, 8, 12, 13, 14, 15])


def item5():
    h("Item 5 — Non-independent (correlated) queries")
    print("MISSING: no correlated-query runs exist and the code has no option for it. "
          "Selector::sample_percent_responses only draws i.i.d. from the WeightedIndex sampler. "
          "Needs a new sampler (e.g. Metropolis–Hastings walk over neighbouring queries with stay/step prob rho) "
          "and new runs for LAMa, Even Less and Remin.\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="1,2,4,5", help="comma list of items (2 also prints item 3)")
    ap.add_argument("--original-sweep", action="store_true",
                    help="also recompute the paper's random-sweep Table II eps for the baselines (slow)")
    args = ap.parse_args()
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
    print(f"\n_Generated in {time.time() - t0:.0f}s from {os.path.abspath(BASE_DIR)}_")
