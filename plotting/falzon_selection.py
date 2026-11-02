"""
Simulates the "Automatically finding D in E(D)" step of Falzon et al. (CCS 2020, Sec. 7) on top
of the E(D) built in falzon.py, without running their pipeline (no train/test split, no search
for a training set).

Their attack picks one database out of E(D) with auxiliary knowledge:
    * MSE attack:  a training histogram H_train over the domain; pick the D_i minimising
                   (1/|D|) * sum_x (H_train(x) - H_i(x))^2
    * mean attack: only the training mean mu_train; pick the D_i minimising |mu_train - mu_i|
Their training data is similar DATA (e.g. other hospitals), not queries. We simulate it as the
true data histogram moved `train_noise` away in total variation (same mixing as --dist-noise).

We also give Falzon et al. the knowledge LAMa has (the query distribution) and do the same two
selections on the query side, with the per-record frequencies (how often each record is returned):
    * qhist: pick the D_i minimising (1/n) * sum_r (o(r) - e_i(r))^2, where o(r) is the observed
             frequency of record r and e_i(r) the probability (under the known query distribution)
             that a query returns D_i's point for r
    * qmean: pick the D_i minimising |mean_r o(r) - mean_r e_i(r)|
o(r) is either exact (every query seen, optionally from a TV-noised query distribution) or
estimated from a sample of queries (e.g. 30%).

Ties are broken uniformly at random, so we report the EXPECTED MSE of the pick over the tied
databases, and the probability that the pick is exactly the true database (their "accuracy").

Query weights follow query.rs: weight by the query's index in the query list, with
    gaussian: Normal(mu = M / 2, sigma = M / 5).pdf(i)
    beta:     Beta(2, 1).pdf((i + 0.5) / M)
    uniform:  1
over the M queries [lo, hi] (lo <= hi) in lexicographic order of (lo, hi), which is the order
Selector::get_dom_pairs' iterators produce before par_bridge reorders them across threads.
"""
import csv
import json
import os

import numpy as np
from scipy.stats import beta as beta_dist, norm

from data_loader import calculate_mse
from falzon import global_flips, reflect, reflectable_components
from globals_vars import format_db_name

MAX_EXACT_COMPONENTS = 12  # enumerate all 4 * 2^r databases up to this r, otherwise sample
NUM_SAMPLED_MASKS = 4096


# ------------------------------------------------------------------
# Query distribution (query.rs) and enclosing probabilities
# ------------------------------------------------------------------
def query_weight_array(dist, n0, n1):
    """W[lx, ly, hx, hy] = probability of query [lo, hi] on a 0..n0-1 x 0..n1-1 domain (0 if lo > hi)."""
    lx, ly, hx, hy = np.indices((n0, n1, n0, n1), dtype=np.int32)
    valid = (lx <= hx) & (ly <= hy)
    m = int(valid.sum())
    i = np.arange(m, dtype=float)  # C-order of the valid entries = lexicographic (lo, hi)
    if dist == "gaussian":
        w = norm.pdf(i, loc=m / 2.0, scale=m / 5.0)
    elif dist == "beta":
        w = beta_dist.pdf((i + 0.5) / m, 2.0, 1.0)
    elif dist == "uniform":
        w = np.ones(m)
    else:
        raise ValueError(f"Unknown distribution {dist}")
    weights = np.zeros((n0, n1, n0, n1))
    weights[valid] = w / w.sum()
    return weights


def add_tv_noise(probs, noise, seed=1337):
    """(1 - lambda) * p + lambda * q with q seeded random over the support of p, so TV(p, p') = noise."""
    if noise <= 0.0:
        return probs
    support = probs > 0
    rng = np.random.default_rng(seed)
    q = np.zeros_like(probs)
    q[support] = rng.random(int(support.sum()))
    q /= q.sum()
    tv = 0.5 * np.abs(probs - q).sum()
    lam = min(1.0, noise / tv)
    return (1.0 - lam) * probs + lam * q


def enclosing_table(weights):
    """S[a, b, c, d] = total weight of queries with lo <= (a, b) and hi >= (c, d), so the probability
    that a query returns point (x, y) is S[x, y, x, y] (LAMa's cumulative probability of an MBQ)."""
    s = np.cumsum(np.cumsum(weights, axis=0), axis=1)
    s = np.flip(np.cumsum(np.cumsum(np.flip(s, axis=(2, 3)), axis=2), axis=3), axis=(2, 3))
    return s


def point_probs(table, grid_pts):
    g = np.asarray(grid_pts, dtype=int)
    return table[g[..., 0], g[..., 1], g[..., 0], g[..., 1]]


def observed_record_freqs(true_grid, dist, n0, n1, mode="exact", query_noise=0.0, query_percent=0.3, seed=0):
    """o(r): how often each true record is returned. mode "exact" = every query seen (in the limit),
    "sampled" = from query_percent * M queries sampled from the (possibly noisy) true distribution."""
    weights = add_tv_noise(query_weight_array(dist, n0, n1), query_noise)
    if mode == "exact":
        return point_probs(enclosing_table(weights), true_grid)
    flat = weights.ravel()
    m = int((flat > 0).sum())
    k = int(np.ceil(m * query_percent))
    rng = np.random.default_rng(seed)
    counts = np.bincount(rng.choice(flat.size, size=k, p=flat), minlength=flat.size).reshape(weights.shape)
    return point_probs(enclosing_table(counts / k), true_grid)


# ------------------------------------------------------------------
# E(D) as explicit candidates
# ------------------------------------------------------------------
def _to_grid(true_pts, grid):
    """Saved coordinates -> grid values 0..G and back (same mapping as falzon_mse_distribution)."""
    true_pts = np.asarray(true_pts, dtype=float)
    grid = np.asarray(grid, dtype=float)
    t_min = true_pts.min(axis=0)
    t_span = true_pts.max(axis=0) - t_min
    unit = np.where(t_span > 0, t_span / grid, 1.0)
    g = np.round((true_pts - t_min) / unit)
    return g, (lambda p: p * unit + t_min)


def candidate_databases(true_grid, n0, n1, seed=0):
    """
    Every database in E(D) (grid values 0..N-1, shape (num_candidates, n, 2)) and the index of the
    true database. With more than MAX_EXACT_COMPONENTS reflectable components, a uniform sample of
    NUM_SAMPLED_MASKS component subsets (each with the 4 flips) is used instead (sampled = True).
    """
    pts = true_grid + 1  # Falzon et al. use 1..N
    refl = reflect(pts, n0, n1)
    comps = reflectable_components(pts, n0, n1)
    r = len(comps)
    sampled = r > MAX_EXACT_COMPONENTS
    if sampled:
        rng = np.random.default_rng(seed)
        masks = rng.integers(0, 2, size=(NUM_SAMPLED_MASKS, r)).astype(bool)
    else:
        masks = ((np.arange(2 ** r)[:, None] >> np.arange(r)) & 1).astype(bool)

    in_comp = np.full((len(masks), len(pts)), False)
    for c, comp in enumerate(comps):
        in_comp[:, comp] = masks[:, [c]]
    seeds = np.where(in_comp[..., None], refl[None], pts[None])  # (masks, n, 2)

    cands = np.concatenate([flip(seeds) for flip in global_flips(n0, n1)]) - 1
    cands = np.round(cands).astype(int)
    true_idx = np.flatnonzero(np.all(cands == true_grid.astype(int), axis=(1, 2)))
    return cands, (true_idx[0] if len(true_idx) else None), r, sampled


# ------------------------------------------------------------------
# Selection
# ------------------------------------------------------------------
def _pick(scores, mses, true_idx):
    """Expected MSE and accuracy of a uniformly random pick among the minimisers of scores."""
    best = np.isclose(scores, scores.min(), rtol=1e-9, atol=1e-15)
    acc = float(best[true_idx]) / best.sum() if true_idx is not None else 0.0
    return float(mses[best].mean()), acc


def data_histogram(grid_pts, n0, n1):
    g = np.asarray(grid_pts, dtype=int)
    h = np.zeros((n0, n1))
    np.add.at(h, (g[:, 0], g[:, 1]), 1.0)
    return h / len(g)


def falzon_selection(true_pts, grid, dist, train_noise=0.05, obs_modes=None, train_hists=None, seed=0):
    """
    Expected MSE (and accuracy) of each selection rule for one database.
    obs_modes: {label: dict(mode=..., query_noise=..., query_percent=...)} for the query-side rules.
    train_hists: {label: histogram over the (n0, n1) grid} extra training data for the data-side rules,
                 e.g. the other datasets' histograms ("loo"), giving rules data_hist_<label>/data_mean_<label>.
    Returns {rule_label: (expected_mse, accuracy)}, plus "random" (a uniform pick from E(D)) and
    "r" / "sampled" for the size of E(D).
    """
    obs_modes = obs_modes or {"exact": dict(mode="exact")}
    true_pts = np.asarray(true_pts, dtype=float)
    true_grid, to_saved = _to_grid(true_pts, grid)
    n0, n1 = int(grid[0]) + 1, int(grid[1]) + 1

    cands, true_idx, r, sampled = candidate_databases(true_grid, n0, n1, seed)
    mses = np.array([calculate_mse(true_pts, to_saved(c)) for c in cands])
    out = {"r": r, "sampled": sampled, "random": (float(mses.mean()), 1.0 / len(cands))}

    # Data side (Falzon et al. Sec. 7): training data = true histogram, train_noise away in TV
    cells = np.stack(np.indices((n0, n1)), axis=-1).reshape(-1, 2)
    trains = {"": add_tv_noise(data_histogram(true_grid, n0, n1), train_noise, seed=7)}
    trains.update({f"_{label}": h / h.sum() for label, h in (train_hists or {}).items()})
    hist_scores_all = np.zeros((len(trains), len(cands)))
    for i, c in enumerate(cands):  # one candidate histogram at a time (all of them at once can be GBs)
        h_c = data_histogram(c, n0, n1)
        for t, h_train in enumerate(trains.values()):
            hist_scores_all[t, i] = np.mean((h_train - h_c) ** 2)
    for (suffix, h_train), hist_scores in zip(trains.items(), hist_scores_all):
        mu_train = (h_train.reshape(-1, 1) * cells).sum(axis=0)
        mean_scores = np.linalg.norm(cands.mean(axis=1) - mu_train, axis=1)
        out[f"data_hist{suffix}"] = _pick(hist_scores, mses, true_idx)
        out[f"data_mean{suffix}"] = _pick(mean_scores, mses, true_idx)

    # Query side: the attacker knows the noiseless query distribution
    expected = point_probs(enclosing_table(query_weight_array(dist, n0, n1)), cands)  # (cands, n)
    for label, kwargs in obs_modes.items():
        obs = observed_record_freqs(true_grid, dist, n0, n1, seed=seed, **kwargs)
        out[f"query_hist_{label}"] = _pick(np.mean((expected - obs) ** 2, axis=1), mses, true_idx)
        out[f"query_mean_{label}"] = _pick(np.abs(expected.mean(axis=1) - obs.mean()), mses, true_idx)
    return out


# ------------------------------------------------------------------
# Experiments over the datasets + table
# ------------------------------------------------------------------
# Query-side observations: every query seen, or 30% of them sampled. Add e.g.
# "noise0.02": dict(mode="exact", query_noise=0.02) for a TV-noised true query distribution.
DEFAULT_OBS_MODES = {
    "exact": dict(mode="exact"),
    "sampled0.3": dict(mode="sampled", query_percent=0.3),
}


def _obs_header(label, kwargs):
    if kwargs.get("mode") == "sampled":
        pct = f"{kwargs.get('query_percent', 0.3) * 100:g}\\%"
    else:
        pct = "100\\%"
    noise = kwargs.get("query_noise", 0.0)
    return pct + (f", TV {noise:g}" if noise > 0 else "")


def run_falzon_selection(base_dir, datasets, distributions, grid=50, train_noise=0.05, obs_modes=None,
                         lama_suffix="e0_d0.9"):
    """
    results[db][dist] = falzon_selection(...) output plus "lama": (avg MSE, min MSE, #recons), using
    the true database and LAMa reconstructions of {base_dir}/{grid}x{grid}/{db}/limits/.
    """
    obs_modes = obs_modes or DEFAULT_OBS_MODES
    n = grid + 1

    def load_runs(db, dist):
        path = f"{base_dir}/{grid}x{grid}/{db}/limits/{db}_{dist}_{lama_suffix}_reconstruction.json"
        if not os.path.exists(path):
            return None
        with open(path, 'r') as f:
            return [run for run in json.load(f) if run]

    # Leave-one-out training data (closer to Falzon et al.'s setting: the aggregate of OTHER databases)
    true_hists = {}
    for db in datasets:
        runs = next((r for r in (load_runs(db, d) for d in distributions) if r), None)
        if runs:
            true_hists[db] = data_histogram(_to_grid([p["true"] for p in runs[0]], (grid, grid))[0], n, n)

    results = {}
    for db in datasets:
        others = [h for other, h in true_hists.items() if other != db]
        train_hists = {"loo": sum(others)} if others else {}
        for dist in distributions:
            runs = load_runs(db, dist)
            if runs is None:
                print(f"[SKIP] {db} {dist}: no LAMa reconstruction JSON")
                continue
            true_pts = [p["true"] for p in runs[0]]
            res = falzon_selection(true_pts, (grid, grid), dist, train_noise, obs_modes, train_hists)
            lama = [calculate_mse([p["true"] for p in run], [p["reconstructed"] for p in run]) for run in runs]
            res["lama"] = (float(np.mean(lama)), float(np.min(lama)), len(lama))
            if res["sampled"]:
                print(f"[WARN] {db} {dist}: E(D) has 4 x 2^{res['r']} databases, selection uses a sample")
            results.setdefault(db, {})[dist] = res
    return results


def _fmt(val):
    return "--" if val is None else f"{val:.1f}"


def print_falzon_selection_latex(results, datasets, distributions, obs_modes=None):
    """One row per dataset/dist: LAMa avg MSE, Falzon random pick, data-side and query-side picks."""
    obs_modes = obs_modes or DEFAULT_OBS_MODES
    cols = [("random", "Rand."), ("data_mean", "Mean"), ("data_hist", "Hist."),
            ("data_mean_loo", "Mean LOO"), ("data_hist_loo", "Hist. LOO")]
    for label, kwargs in obs_modes.items():
        h = _obs_header(label, kwargs)
        cols += [(f"query_mean_{label}", f"Q-Mean {h}"), (f"query_hist_{label}", f"Q-Hist. {h}")]

    print("\n" + "=" * 50)
    print("LATEX TABLE: FALZON ET AL. WITH A SELECTION STEP (EXPECTED MSE OF THE PICK)")
    print("=" * 50 + "\n")
    print("\\begin{table*}[htpb]")
    print("\\centering")
    print("\\footnotesize")
    print("\\setlength{\\tabcolsep}{3pt}")
    print("\\begin{tabular}{ll" + "r" * (1 + len(cols)) + "}")
    print("\\toprule")
    header = " & ".join(f"\\textbf{{{name}}}" for _, name in cols)
    print(f"\\textbf{{Dataset}} & \\textbf{{Dist.}} & \\textbf{{LAMa}} & {header} \\\\")
    print("\\midrule")
    for db in datasets:
        db_res = results.get(db, {})
        if not db_res:
            continue
        print(f"\\multirow{{{len(distributions)}}}{{*}}{{{format_db_name(db)}}}")
        for dist in distributions:
            res = db_res.get(dist)
            if res is None:
                print(f" & {dist.capitalize()} & " + " & ".join(["--"] * (1 + len(cols))) + " \\\\")
                continue
            cells = [_fmt(res["lama"][0])] + [_fmt(res[key][0]) for key, _ in cols]
            print(f" & {dist.capitalize()} & {' & '.join(cells)} \\\\")
        print("\\midrule")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{MSE of LAMa (average over its reconstructions) and the expected MSE of the single "
          "database Falzon et al.'s attack picks from E(D): uniformly at random (Rand.), by matching the mean "
          "or histogram of training data (Mean/Hist., their Sec.~7; training data is the true data moved "
          "a TV distance away, or the other datasets for LOO), or by matching the mean or per-record "
          "frequencies under the known query distribution (Q-Mean/Q-Hist.).}")
    print("\\label{tab:falzon_selection}")
    print("\\end{table*}\n")


def save_falzon_selection_csv(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = f"{output_dir}/falzon_selection.csv"
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "dist", "rule", "expected_mse", "accuracy", "r", "sampled"])
        for db, db_res in results.items():
            for dist, res in db_res.items():
                writer.writerow([db, dist, "lama_avg", res["lama"][0], None, res["r"], res["sampled"]])
                writer.writerow([db, dist, "lama_min", res["lama"][1], None, res["r"], res["sampled"]])
                for rule, val in res.items():
                    if isinstance(val, tuple) and rule != "lama":
                        writer.writerow([db, dist, rule, val[0], val[1], res["r"], res["sampled"]])
    print(f"Saved {path}")
