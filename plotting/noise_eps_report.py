"""
eps / #databases per TV for the perfect-setting noise sweep, from the results_*.json files ONLY (no
reconstruction JSON needed). Run from plotting/:

    python noise_eps_report.py                 # 50x50, all datasets / dists
    python noise_eps_report.py --grid 15x15 --datasets drink,spitz --dists gaussian

How the noise works (src/LAMA/query.rs add_noise): true distribution p' = (1 - lam) p + lam q, q = a fixed
random weight per query (StdRng seed 1337, handed out in sorted-query order), lam = TV / TV(p, q). The
attacker uses p, and eps = max over MBQs |P'(enclose) - P(enclose)| = lam * max|P_q - P_p|. With the query
list sorted (results field query_order = "sorted") p and q are the same in every run, so eps / TV is one
constant per (dataset, dist) and the output databases are nested: #DBs can only grow with TV.
Flags: "eps/TV varies" or "#DBs drops" = runs that used different distributions (old, unsorted runs).
"""
import argparse
import json
import os

TVS = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="50x50")
    ap.add_argument("--datasets", default="busstop,shopparis,cali,drink,highway,spitz")
    ap.add_argument("--dists", default="uniform,gaussian,beta")
    ap.add_argument("--base", default="../databases")
    a = ap.parse_args()

    print(f"| Dataset | Dist | " + " | ".join(f"TV {tv:g}" for tv in TVS) + " | eps/TV | flags |")
    print("|" + "---|" * (len(TVS) + 4))
    for db in a.datasets.split(","):
        for dist in a.dists.split(","):
            cells, ratios, counts, flags = [], [], [], set()
            for tv in TVS:
                r = load(f"{a.base}/{a.grid}/{db}/limits/results_{db}_{dist}_p1_noise{tv:g}.json")
                if r is None:
                    cells.append("—")
                    continue
                n = r.get("number_of_databases")
                tin = r.get("true_db_in_output")
                if r.get("query_order") != "sorted":
                    flags.add("old (unsorted) run")
                ratios.append(r["eps"] / tv)
                counts.append(n)
                cells.append(f"{n if n is not None else '?'} DBs, eps {r['eps']:.2e}"
                             + ("" if tin is None else f", true {'yes' if tin else 'NO'}"))
            if ratios and max(ratios) - min(ratios) > 1e-6 * max(ratios):
                flags.add("eps/TV varies")
            known = [c for c in counts if c is not None]
            if any(b < a_ for a_, b in zip(known, known[1:])):
                flags.add("#DBs drops")
            ratio = f"{ratios[0]:.4f}" if ratios and "eps/TV varies" not in flags else \
                (f"{min(ratios):.4f}–{max(ratios):.4f}" if ratios else "—")
            print(f"| {db} | {dist} | " + " | ".join(cells) + f" | {ratio} | {', '.join(sorted(flags)) or 'ok'} |")
    print(f"\n_From {os.path.abspath(a.base)}/{a.grid}_")


if __name__ == "__main__":
    main()
