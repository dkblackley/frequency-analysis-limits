def format_latex_mse(val):
    """Formats values >= 0.01 to 2 decimals, and smaller values to scientific notation."""
    if val == 0.0:
        return "0.00"
    if val >= 0.01:
        return f"{val:.2f}"
    else:
        sci_str = f"{val:.2e}"
        base, exp = sci_str.split('e')
        # Wrap in $ $ for LaTeX math mode, and remove leading zeros from the exponent
        return f"${base} \\times 10^{{{int(exp)}}}$"


def print_limits_samples_latex(sample_data, datasets, distributions):
    """Prints a LaTeX table of Centroid MSE Convergence for LAMa."""

    print("\n" + "=" * 50)
    print("LATEX TABLE: LAMA CONVERGENCE (MSE)")
    print("=" * 50 + "\n")

    print("\\begin{table}[htpb]")
    print("\\centering")
    print("\\begin{tabular}{llrrrr}")
    print("\\toprule")
    print(
        "\\multirow{2}{*}{\\textbf{Dataset}} & \\multirow{2}{*}{\\textbf{Dist.}} & \\multicolumn{4}{c}{\\textbf{Number of Samples}} \\\\")
    print("\\cmidrule(lr){3-6}")
    print("& & \\textbf{2500} & \\textbf{5000} & \\textbf{7500} & \\textbf{10000} \\\\")
    print("\\midrule")

    for db in datasets:
        db_data = sample_data.get(db, {})
        if not db_data:
            continue

        print(f"\\multirow{{3}}{{*}}{{{db.capitalize()}}}")
        for dist in distributions:
            runs = db_data.get(dist, [])

            # Map the sample size to its Centroid MSE (index 1 in the tuple)
            mses = {r[0]: r[1] for r in runs}

            v2500 = format_latex_mse(mses.get(2500, 0.0))
            v5000 = format_latex_mse(mses.get(5000, 0.0))
            v7500 = format_latex_mse(mses.get(7500, 0.0))
            v10000 = format_latex_mse(mses.get(10000, 0.0))

            print(f" & {dist.capitalize()} & {v2500} & {v5000} & {v7500} & {v10000} \\\\")

        print("\\midrule")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{LAMa Centroid MSE convergence over varying sample sizes (10\\% queries).}")
    print("\\label{tab:lama_convergence}")
    print("\\end{table}\n")