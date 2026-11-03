"""
LAMa vs Falzon et al. as "Guaranteed output" vs "Best guess", one column per query distribution:

    top row    (Guaranteed output): what each attack can PROVE contains the true database. LAMa's output
               is every database consistent with the frequencies (1 under Gaussian/Beta), Falzon et al.'s is
               E(D) (4 * 2^r databases, whatever the distribution). Bars are the average MSE over that
               output, the count of databases is written above each bar.
    bottom row (Best guess): each attack commits to one database, picked with Falzon et al.'s mean attack
               (Sec. 7): the database whose mean is closest to a training mean = true mean + N(0, sigma^2)
               grid cells per axis. Bars are the expected MSE of that pick.

One bar pair (LAMa, Falzon) per dataset. The y axis is symlog because LAMa is often exactly 0.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from data_loader import calculate_mse
from falzon import falzon_mse_distribution
from falzon_selection import _to_grid, candidate_databases
from globals_vars import FONT_SIZE, format_db_name


def _expected_mean_match(cand_means, mses, mu_true, sigma, trials, rng):
    """Expected MSE of the candidate whose mean is closest to mu_true + N(0, sigma^2) (ties at random)."""
    if len(mses) == 1:
        return float(mses[0])
    picks = []
    for _ in range(trials if sigma > 0 else 1):
        mu = mu_true + rng.normal(0.0, sigma, size=2) if sigma > 0 else mu_true
        scores = np.linalg.norm(cand_means - mu, axis=1)
        best = np.isclose(scores, scores.min(), rtol=1e-9, atol=1e-12)
        picks.append(mses[best].mean())
    return float(np.mean(picks))


def load_guarantee_data(base_dir, datasets, distributions, grid=50, sigma=2.0, trials=1000,
                        lama_suffix="e0_d0.9", seed=0):
    """
    data[dist][db] = dict(lama_n, lama_mse, lama_guess, falzon_r, falzon_mse, falzon_guess), from
    {base_dir}/{grid}x{grid}/{db}/limits/{db}_{dist}_{lama_suffix}_reconstruction.json.
    Falzon et al.'s output does not depend on the distribution, only the LAMa numbers do.
    """
    data = {dist: {} for dist in distributions}
    falzon_cache = {}
    for db in datasets:
        for dist in distributions:
            path = f"{base_dir}/{grid}x{grid}/{db}/limits/{db}_{dist}_{lama_suffix}_reconstruction.json"
            if not os.path.exists(path):
                print(f"[SKIP] {path} not found")
                continue
            with open(path, 'r') as f:
                runs = [run for run in json.load(f) if run]
            true_pts = np.array([p["true"] for p in runs[0]], dtype=float)
            true_grid, _ = _to_grid(true_pts, (grid, grid))
            mu_true = true_grid.mean(axis=0)
            rng = np.random.default_rng(seed)

            # LAMa: every reconstruction it outputs, in grid units for the mean matching
            t_min = true_pts.min(axis=0)
            unit = np.where(np.ptp(true_pts, axis=0) > 0, np.ptp(true_pts, axis=0) / grid, 1.0)
            recons = [np.array([p["reconstructed"] for p in run], dtype=float) for run in runs]
            lama_mses = np.array([calculate_mse(true_pts, rec) for rec in recons])
            lama_means = np.array([((rec - t_min) / unit).mean(axis=0) for rec in recons])

            # Falzon et al.: E(D) (same for every distribution, so computed once per dataset)
            if db not in falzon_cache:
                all_mses, r = falzon_mse_distribution(true_pts, (grid, grid))
                n = grid + 1
                cands, _, _, _ = candidate_databases(true_grid, n, n, seed)
                _, to_saved = _to_grid(true_pts, (grid, grid))
                cand_mses = np.array([calculate_mse(true_pts, to_saved(c)) for c in cands])
                falzon_cache[db] = dict(
                    falzon_r=r,
                    falzon_mse=float(np.mean(all_mses)),
                    falzon_guess=_expected_mean_match(cands.mean(axis=1), cand_mses, mu_true, sigma, trials, rng),
                )

            data[dist][db] = dict(
                lama_n=len(runs),
                lama_mse=float(lama_mses.mean()),
                lama_guess=_expected_mean_match(lama_means, lama_mses, mu_true, sigma, trials, rng),
                **falzon_cache[db],
            )
    return data


def _count_label(n=None, r=None):
    if r is not None:
        return f"{4 * 2 ** r:,}" if r <= 10 else f"$4{{\\cdot}}2^{{{r}}}$"
    return f"{n:,}"


def plot_guarantee_vs_best_guess(data, datasets, distributions, output_dir, colors, labels, sigma=2.0,
                                 font_size=FONT_SIZE, filename="falzon_guarantee_vs_guess.pdf"):
    """2 x len(distributions) grid of grouped bars (see the module docstring)."""
    n_cols = len(distributions)
    fig, axes = plt.subplots(2, n_cols, figsize=(14 * n_cols, 22), sharey='row', sharex=True, squeeze=False)
    fig.subplots_adjust(hspace=0.12, wspace=0.08)
    x = np.arange(len(datasets))
    width = 0.38
    rows = [("Guaranteed output\nMSE", "lama_mse", "falzon_mse"),
            (f"Best guess ($\\sigma={sigma:g}$)\nMSE", "lama_guess", "falzon_guess")]

    y_max = max([1.0] + [v[k] for d in data.values() for v in d.values()
                         for k in ("lama_mse", "falzon_mse", "lama_guess", "falzon_guess")])

    for col, dist in enumerate(distributions):
        for row, (row_title, lama_key, falzon_key) in enumerate(rows):
            ax = axes[row][col]
            dist_data = data.get(dist, {})
            for offset, method, key in [(-width / 2, 'limits', lama_key), (width / 2, 'falzon', falzon_key)]:
                vals = [dist_data[db][key] if db in dist_data else np.nan for db in datasets]
                bars = ax.bar(x + offset, vals, width, color=colors.get(method), alpha=0.85,
                              edgecolor=colors.get(method), linewidth=2,
                              label=labels.get(method, method) if (row == 0 and col == 0) else None)
                if row == 0:  # how many databases the guaranteed output holds
                    for bar, db in zip(bars, datasets):
                        if db not in dist_data:
                            continue
                        d = dist_data[db]
                        text = _count_label(n=d["lama_n"]) if method == 'limits' else _count_label(r=d["falzon_r"])
                        ax.text(bar.get_x() + bar.get_width() / 2, max(bar.get_height(), 0) * 1.2 + 0.1, text,
                                ha='center', va='bottom', fontsize=font_size * 0.4)

            ax.set_yscale('symlog', linthresh=1.0)
            ax.set_ylim(0, y_max * 40)
            ax.set_xticks(x)
            if row == len(rows) - 1:
                ax.set_xticklabels([format_db_name(db) for db in datasets], rotation=45, ha='right',
                                   fontsize=font_size * 0.55)
            ax.tick_params(axis='y', labelsize=font_size * 0.55)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if col == 0:
                ax.set_ylabel(row_title, fontsize=font_size * 0.6)
            if row == 0:
                ax.set_title(f"{dist.capitalize()} Dist.", fontsize=font_size * 0.7, pad=20)

    fig.legend(loc='lower center', ncol=2, fontsize=font_size * 0.6, frameon=False, bbox_to_anchor=(0.5, 0.94))
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, filename)
    plt.savefig(out, format=os.path.splitext(filename)[1][1:], bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


def print_guarantee_latex(data, datasets, distributions, sigma=2.0):
    """Table fallback: per distribution, #DBs and avg MSE of the guaranteed output, and the best-guess MSE."""
    print("\\begin{table*}[htpb]")
    print("\\centering")
    print("\\footnotesize")
    print("\\setlength{\\tabcolsep}{3pt}")
    print("\\begin{tabular}{l" + "rrrrrr" * len(distributions) + "}")
    print("\\toprule")
    print(" & " + " & ".join(f"\\multicolumn{{6}}{{c}}{{\\textbf{{{d.capitalize()}}}}}" for d in distributions) + " \\\\")
    print(" ".join(f"\\cmidrule(lr){{{2 + 6 * i}-{7 + 6 * i}}}" for i in range(len(distributions))))
    print(" & " + " & ".join(["\\multicolumn{4}{c}{Guaranteed} & \\multicolumn{2}{c}{Best guess}"] * len(distributions)) + " \\\\")
    sub = "\\#L & \\#F & MSE L & MSE F & L & F"
    print("\\textbf{Dataset} & " + " & ".join([sub] * len(distributions)) + " \\\\")
    print("\\midrule")
    for db in datasets:
        cells = []
        for dist in distributions:
            d = data.get(dist, {}).get(db)
            if d is None:
                cells += ["--"] * 6
                continue
            cells += [_count_label(n=d["lama_n"]), _count_label(r=d["falzon_r"]),
                      f"{d['lama_mse']:.1f}", f"{d['falzon_mse']:.1f}", f"{d['lama_guess']:.1f}", f"{d['falzon_guess']:.1f}"]
        print(f"{format_db_name(db)} & {' & '.join(cells)} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print(f"\\caption{{LAMa (L) vs Falzon et al. (F). Guaranteed: number of databases each attack outputs (all "
          f"guaranteed to include the true one) and their average MSE. Best guess: expected MSE of the single "
          f"database picked by mean matching with a training mean $\\sigma={sigma:g}$ grid cells off.}}")
    print("\\label{tab:falzon_guarantee}")
    print("\\end{table*}")
