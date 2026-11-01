def format_latex_time(val):
    """Formats values >= 0.01 to 2 decimals, and smaller values to scientific notation."""
    if val == 0.0:
        return "0.00"
    if val >= 0.01:
        return f"{val:.2f}"
    else:
        sci_str = f"{val:.2e}"
        base, exp = sci_str.split('e')
        return f"${base} \\times 10^{{{int(exp)}}}$"


def print_time_per_recon_latex(time_data, datasets, distributions):
    """Prints a LaTeX table of Time Per Reconstruction ONLY for LAMa."""

    print("\n" + "=" * 50)
    print("LATEX TABLE: LAMA TIME PER RECONSTRUCTION")
    print("=" * 50 + "\n")

    print("\\begin{table}[htpb]")
    print("\\centering")
    print("\\begin{tabular}{llrrr}")
    print("\\toprule")
    print("\\textbf{Dataset} & \\textbf{Dist.} & \\textbf{10\\%} & \\textbf{20\\%} & \\textbf{30\\%} \\\\")
    print("\\midrule")

    for db in datasets:
        db_data = time_data.get(db, {})
        if not db_data:
            continue

        print(f"\\multirow{{3}}{{*}}{{{db.capitalize()}}}")
        for dist in distributions:
            # "limits" is the internal name for LAMa
            method_runs = db_data.get(dist, {}).get("limits", [])

            method_runs.sort(key=lambda x: x[0])
            times = {r[0]: (r[1] / 10001.0) for r in method_runs}

            t10 = format_latex_time(times.get(10.0, 0.0))
            t20 = format_latex_time(times.get(20.0, 0.0))
            t30 = format_latex_time(times.get(30.0, 0.0))

            print(f" & {dist.capitalize()} & {t10} & {t20} & {t30} \\\\")

        print("\\midrule")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{LAMa time per reconstruction (seconds) at varying query percentages.}")
    print("\\label{tab:lama_time_per_recon}")
    print("\\end{table}\n")