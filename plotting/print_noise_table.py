import csv
import json
import os

import numpy as np

from data_loader import compute_dataset_metrics, calculate_mse
from globals_vars import format_db_name


def _rust_float(val):
    """Formats a float the way Rust's Display does in the result file names (1.0 -> '1', 0.1 -> '0.1')."""
    return f"{val:g}"


def _load_limits_metrics(path):
    """Centroid and diameter-midpoint MSE (convex hull of every reconstruction) of one LAMa output."""
    try:
        with open(path, 'r') as f:
            multi_run_data = json.load(f)
    except FileNotFoundError:
        return None

    if not multi_run_data:
        return None

    true_pts = [p['true'] for p in multi_run_data[0]]
    point_clouds = [np.array([p['reconstructed'] for p in run]) for run in multi_run_data]
    c_mse, d_mse = compute_dataset_metrics(point_clouds, true_pts)
    return c_mse, d_mse, len(multi_run_data)


def _load_time(path):
    try:
        with open(path, 'r') as f:
            return json.load(f).get("time_taken")
    except FileNotFoundError:
        return None


def load_noise_data(base_dir, datasets, distributions, noise=0.01, query_percents=(10.0, 20.0, 30.0, 100.0),
                    grid="15x15", clean_grid=None):
    """
    Loads the noisy-distribution LAMa runs (--dist-noise) and, where it exists, the same run without noise.
    Returns: nested_data[dataset][dist] = [(query, noisy_metrics, clean_metrics, noisy_time), ...]
    where *_metrics = (centroid_mse, diameter_mse, num_reconstructions) or None if the file is missing.
    """
    clean_grid = clean_grid or grid
    noise_str = _rust_float(noise)
    nested_data = {db: {dist: [] for dist in distributions} for db in datasets}

    for db in datasets:
        for dist in distributions:
            for query in query_percents:
                q = _rust_float(query / 100.0)
                noisy_name = f"{db}_{dist}_p{q}_noise{noise_str}"
                clean_name = f"{db}_{dist}_p{q}"

                noisy = _load_limits_metrics(f"{base_dir}/{grid}/{db}/limits/{noisy_name}_reconstruction.json")
                clean = _load_limits_metrics(f"{base_dir}/{clean_grid}/{db}/limits/{clean_name}_reconstruction.json")
                noisy_time = _load_time(f"{base_dir}/{grid}/{db}/limits/results_{noisy_name}.json")

                if noisy is None and clean is None:
                    continue
                nested_data[db][dist].append((query, noisy, clean, noisy_time))

    return nested_data


def _fmt(val):
    return "--" if val is None else f"{val:.2f}"


def print_noise_latex(noise_data, datasets, distributions, noise=0.01, metric="centroid"):
    """
    Prints a LaTeX table of LAMa's MSE under a noisy query distribution, with the noiseless MSE in brackets.
    metric: "centroid" or "diameter" (the two placement strategies of the output quality section).
    """
    m_idx = 0 if metric == "centroid" else 1
    query_percents = sorted({r[0] for db in noise_data.values() for runs in db.values() for r in runs})
    if not query_percents:
        print("No noisy-distribution results found.")
        return

    print("\n" + "=" * 50)
    print(f"LATEX TABLE: LAMA {metric.upper()} MSE WITH A NOISY QUERY DISTRIBUTION")
    print("=" * 50 + "\n")

    print("\\begin{table}[htpb]")
    print("\\centering")
    print("\\begin{tabular}{ll" + "r" * len(query_percents) + "}")
    print("\\toprule")
    header = " & ".join(f"\\textbf{{{int(q)}\\%}}" for q in query_percents)
    print(f"\\textbf{{Dataset}} & \\textbf{{Dist.}} & {header} \\\\")
    print("\\midrule")

    for db in datasets:
        db_data = noise_data.get(db, {})
        if not any(db_data.values()):
            continue

        print(f"\\multirow{{{len(distributions)}}}{{*}}{{{format_db_name(db)}}}")
        for dist in distributions:
            runs = {r[0]: r for r in db_data.get(dist, [])}
            cells = []
            for q in query_percents:
                _, noisy, clean, _ = runs.get(q, (q, None, None, None))
                noisy_val = noisy[m_idx] if noisy else None
                clean_val = clean[m_idx] if clean else None
                cells.append(f"{_fmt(noisy_val)} ({_fmt(clean_val)})")
            print(f" & {dist.capitalize()} & {' & '.join(cells)} \\\\")

        print("\\midrule")

    print("\\bottomrule")
    print("\\end{tabular}")
    print(f"\\caption{{LAMa {metric} MSE when the true query distribution is {noise} away (total variation) "
          f"from the one the attacker knows, at varying query percentages. "
          f"The MSE with the exact distribution is in brackets.}}")
    print(f"\\label{{tab:lama_noise_{metric}}}")
    print("\\end{table}\n")


def save_noise_csv(noise_data, output_dir, noise=0.01):
    """Writes every number behind the tables to one CSV (one row per dataset/dist/query)."""
    os.makedirs(output_dir, exist_ok=True)
    path = f"{output_dir}/noise_{_rust_float(noise)}_table.csv"
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "dist", "query_percent", "noisy_centroid_mse", "noisy_diameter_mse",
                         "noisy_num_recons", "noisy_time_s", "clean_centroid_mse", "clean_diameter_mse",
                         "clean_num_recons"])
        for db, db_data in noise_data.items():
            for dist, runs in db_data.items():
                for query, noisy, clean, noisy_time in runs:
                    writer.writerow([db, dist, query,
                                     *(noisy if noisy else (None, None, None)), noisy_time,
                                     *(clean if clean else (None, None, None))])
    print(f"Saved {path}")


def _load_eps(path):
    try:
        with open(path, 'r') as f:
            return json.load(f).get("eps")
    except FileNotFoundError:
        return None


def _load_sweep_metrics(path):
    """(avg_mse, centroid_mse, diameter_mse, num_reconstructions) of one LAMa output, or None if missing.
    avg_mse is the MSE of each reconstruction against the truth, averaged over all reconstructions."""
    try:
        with open(path, 'r') as f:
            multi_run_data = json.load(f)
    except FileNotFoundError:
        return None

    runs = [run for run in multi_run_data if run]
    if not runs:
        return None

    true_pts = [p['true'] for p in runs[0]]
    point_clouds = [np.array([p['reconstructed'] for p in run]) for run in runs]
    avg_mse = float(np.mean([calculate_mse([p['true'] for p in run], [p['reconstructed'] for p in run])
                             for run in runs]))
    c_mse, d_mse = compute_dataset_metrics(point_clouds, true_pts)
    return avg_mse, c_mse, d_mse, len(runs)


def load_noise_sweep_data(base_dir, datasets, distributions, noises=(0.01, 0.02, 0.03), grid="15x15"):
    """
    Loads the 100%-of-queries noise sweep (run_LAMA_noise_sweep.slurm): every query seen, but the true
    distribution is `noise` away (TV) from --dist, and eps is the exact largest error (no VC bound).
    Returns: nested_data[dataset][dist] = {"clean": metrics, noise: (metrics, eps, time), ...}
    where metrics = (avg_mse, centroid_mse, diameter_mse, num_reconstructions) or None if the file is missing.
    "clean" is the noiseless 100% run on the same grid, if it exists.
    """
    nested_data = {db: {dist: {} for dist in distributions} for db in datasets}

    for db in datasets:
        for dist in distributions:
            limits_dir = f"{base_dir}/{grid}/{db}/limits"
            nested_data[db][dist]["clean"] = _load_sweep_metrics(f"{limits_dir}/{db}_{dist}_p1_reconstruction.json")
            for noise in noises:
                name = f"{db}_{dist}_p1_noise{_rust_float(noise)}"
                nested_data[db][dist][noise] = (
                    _load_sweep_metrics(f"{limits_dir}/{name}_reconstruction.json"),
                    _load_eps(f"{limits_dir}/results_{name}.json"),
                    _load_time(f"{limits_dir}/results_{name}.json"),
                )

    return nested_data


def _fmt1(val):
    return "--" if val is None else f"{val:.1f}"


def print_noise_sweep_latex(sweep_data, datasets, distributions, noises=(0.01, 0.02, 0.03)):
    """
    Prints one compact LaTeX table of the noise sweep (100% of queries). One column group per noise level
    (TV distance), each with 4 sub-columns: average MSE over all reconstructions, centroid and
    diameter-midpoint hull MSE, and the number of reconstructions found. Everything to 1 decimal place.
    """
    print("\n" + "=" * 50)
    print("LATEX TABLE: LAMA MSE VS DISTRIBUTION NOISE (100% QUERIES)")
    print("=" * 50 + "\n")

    sub_cols = ["Avg.", "Cent.", "Diam.", "\\#Rec."]
    print("\\begin{table*}[htpb]")
    print("\\centering")
    print("\\footnotesize")
    print("\\setlength{\\tabcolsep}{3pt}")
    print("\\begin{tabular}{ll" + "rrrr" * len(noises) + "}")
    print("\\toprule")
    groups = " & ".join(f"\\multicolumn{{4}}{{c}}{{\\textbf{{{noise * 100:g}\\%}}}}" for noise in noises)
    print(f" & & {groups} \\\\")
    rules = " ".join(f"\\cmidrule(lr){{{3 + 4 * i}-{6 + 4 * i}}}" for i in range(len(noises)))
    print(rules)
    print(f"\\textbf{{Dataset}} & \\textbf{{Dist.}} & {' & '.join(sub_cols * len(noises))} \\\\")
    print("\\midrule")

    for db in datasets:
        db_data = sweep_data.get(db, {})
        if not any(runs.get(n, (None,))[0] for runs in db_data.values() for n in noises):
            continue

        print(f"\\multirow{{{len(distributions)}}}{{*}}{{{format_db_name(db)}}}")
        for dist in distributions:
            runs = db_data.get(dist, {})
            cells = []
            for n in noises:
                metrics = runs.get(n, (None, None, None))[0]
                if metrics:
                    avg, cent, diam, num = metrics
                    cells += [_fmt1(avg), _fmt1(cent), _fmt1(diam), str(num)]
                else:
                    cells += ["--"] * 4
            print(f" & {dist.capitalize()} & {' & '.join(cells)} \\\\")

        print("\\midrule")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{LAMa with every query observed, when the true query distribution is the given total "
          "variation distance away from the one the attacker knows (the attacker uses the exact largest error "
          "in any enclosing probability as $\\epsilon$). Avg.: MSE averaged over all reconstructions; "
          "Cent./Diam.: centroid and diameter-midpoint convex hull MSE; \\#Rec.: reconstructions found.}")
    print("\\label{tab:lama_noise_sweep}")
    print("\\end{table*}\n")


def save_noise_sweep_csv(sweep_data, output_dir, noises=(0.01, 0.02, 0.03)):
    """Writes every number behind the sweep table to one CSV (one row per dataset/dist/noise; noise 0 = clean)."""
    os.makedirs(output_dir, exist_ok=True)
    path = f"{output_dir}/noise_sweep_table.csv"
    empty = (None, None, None, None)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "dist", "noise", "eps", "avg_mse", "centroid_mse", "diameter_mse",
                         "num_recons", "time_s"])
        for db, db_data in sweep_data.items():
            for dist, runs in db_data.items():
                clean = runs.get("clean")
                writer.writerow([db, dist, 0.0, None, *(clean if clean else empty), None])
                for n in noises:
                    metrics, eps, t = runs.get(n, (None, None, None))
                    writer.writerow([db, dist, n, eps, *(metrics if metrics else empty), t])
    print(f"Saved {path}")
