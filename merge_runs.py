#!/usr/bin/env python3
"""Merge independent sampled-setting LAMa runs of one config into the main grid directory.

    python3 merge_runs.py --grid 50x50 --name drink --dist beta --prob 0.03 [--runs 3] [--outputs 1000]

Sources: databases/<grid>_run<k>/<name>/limits/{results_,}<name>_<dist>_p<q>{.json,_reconstruction.json}
for k = 1..runs, plus <grid>_run0: at the first merge, the best earlier SINGLE pair-LNS run of this config
(the one in databases/<grid>/<name>/limits/, or one in its archive_short_runs/ or archive_other_runs/;
best by the attacker-side rank below) is copied to run0, so it stays in the pool for every later merge. A previous MERGE there is
replaced (it is rebuilt from the same runs).

Merge: only sources whose output is fully consistent (every pair within that run's epsilon). Each run's
databases are already sorted by its own total error; they are interleaved round-robin (db 0 of every run,
then db 1, ...), duplicates dropped (a database = the set of (true point, reconstructed point) pairs, so
record order does not matter), and cut at --outputs. Every database is written with records sorted by
true point (one fixed order for all of them). If no source is consistent, the source with the best
attacker-side rank (fewest pairs outside epsilon, then lowest error; never the truth) is copied as is.

NOTE: each run draws its own query sample (and epsilon), so a merged database is consistent with the
sample of the run it came from, not necessarily with the others. "merged_from" records which run gave
how many databases.

Whatever was in databases/<grid>/<name>/limits/ before is OVERWRITTEN (no archive copy; since 2026-11-20).
The run results in <grid>_run<k>/ are never touched. Up to date (merge newer than every run result) -> does nothing, unless --force.
"""
import argparse
import glob
import json
import os
import shutil
import sys


def rank(res):
    """Attacker-side quality (never uses the truth), larger is better. Matches sampled_rank in mod.rs."""
    lns = res.get("lns")
    if not lns:
        return (False, False, 0, float("-inf"), float("-inf"))
    cons = bool(lns.get("output_is_consistent"))
    return (True, cons, lns.get("returned", 0) if cons else 0,
            -lns.get("best_pairs_outside", 0), -lns.get("best_total_error", 0))


def canonical(db):
    """Database -> (hashable key, records sorted by true point)."""
    recs = sorted(db, key=lambda r: tuple(r["true"]))
    key = tuple((tuple(r["true"]), tuple(r["reconstructed"])) for r in recs)
    return key, recs


def mse_info(dbs):
    """Info only (uses the truth): centroid MSE, mean per-database MSE, MSE of database 0."""
    if not dbs or not dbs[0]:
        return None, None, None
    sq = lambda a, b: sum((x - y) ** 2 for x, y in zip(a, b))
    db_mse = lambda db: sum(sq(r["true"], r["reconstructed"]) for r in db) / len(db)
    n, dims = len(dbs[0]), len(dbs[0][0]["true"])
    cent = 0.0
    for i in range(n):  # records are in the same (sorted) order in every merged database
        mean = [sum(db[i]["reconstructed"][d] for db in dbs) / len(dbs) for d in range(dims)]
        cent += sq(dbs[0][i]["true"], mean)
    return cent / n, sum(db_mse(db) for db in dbs) / len(dbs), db_mse(dbs[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="databases")
    ap.add_argument("--grid", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--dist", required=True)
    ap.add_argument("--prob", required=True, help="as in the file names (Rust f64 Display: 0.01, 0.1, 0.3)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--outputs", type=int, default=1000)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    stem = f"{a.name}_{a.dist}_p{a.prob}"
    out_dir = os.path.join(a.root, a.grid, a.name, "limits")
    out_res = os.path.join(out_dir, f"results_{stem}.json")
    out_rec = os.path.join(out_dir, f"{stem}_reconstruction.json")

    sources = []  # (label, results path, reconstruction path)
    for k in range(1, a.runs + 1):
        d = os.path.join(a.root, f"{a.grid}_run{k}", a.name, "limits")
        sources.append((f"run{k}", os.path.join(d, f"results_{stem}.json"), os.path.join(d, f"{stem}_reconstruction.json")))
    missing = [lab for lab, r, c in sources if not (os.path.exists(r) and os.path.exists(c))]
    if missing:
        print(f"{stem}: not all runs finished (missing {', '.join(missing)}); not merging")
        return 2

    existing = json.load(open(out_res)) if os.path.exists(out_res) else None
    if existing is not None and "merged_from" in existing:
        newest_run = max(os.path.getmtime(r) for _, r, _ in sources)
        if os.path.getmtime(out_res) >= newest_run and not a.force:
            print(f"{stem}: merge is up to date")
            return 0
    d0 = os.path.join(a.root, f"{a.grid}_run0", a.name, "limits")
    if not os.path.exists(os.path.join(d0, f"results_{stem}.json")):
        # first merge: the best earlier SINGLE pair-LNS run (the one in limits/ or one archived by
        # run_me.sh / lama_attack; attacker-side rank) joins the pool for good, as <grid>_run0
        cands = []
        if existing is not None and "merged_from" not in existing:
            cands.append((out_res, out_rec))
        for sub in ("archive_short_runs", "archive_other_runs"):
            for d in sorted(glob.glob(os.path.join(out_dir, sub, "*"))):
                cands.append((os.path.join(d, f"results_{stem}.json"), os.path.join(d, f"{stem}_reconstruction.json")))
        best = None
        for r, c in cands:
            if not (os.path.exists(r) and os.path.exists(c)):
                continue
            res = json.load(open(r))
            if not res.get("lns") or "merged_from" in res:
                continue
            if best is None or rank(res) > best[0]:
                best = (rank(res), r, c)
        if best is not None:
            os.makedirs(d0, exist_ok=True)
            shutil.copy2(best[1], os.path.join(d0, f"results_{stem}.json"))
            shutil.copy2(best[2], os.path.join(d0, f"{stem}_reconstruction.json"))
            print(f"{stem}: run0 = {best[1]}")
    if os.path.exists(os.path.join(d0, f"results_{stem}.json")):
        sources.insert(0, ("run0", os.path.join(d0, f"results_{stem}.json"), os.path.join(d0, f"{stem}_reconstruction.json")))

    loaded = []
    for lab, r, c in sources:
        res = json.load(open(r))
        loaded.append((lab, res, r, c))

    consistent = [(lab, res, r, c) for lab, res, r, c in loaded if (res.get("lns") or {}).get("output_is_consistent")]
    per_source = {lab: {"rank_fields": list(rank(res))[1:], "returned": (res.get("lns") or {}).get("returned"),
                        "output_is_consistent": (res.get("lns") or {}).get("output_is_consistent"),
                        "time_taken": res.get("time_taken"), "eps": res.get("eps"), "used": 0}
                  for lab, res, _, _ in loaded}

    if consistent:
        streams = []
        for lab, res, r, c in consistent:
            streams.append((lab, json.load(open(c))))
        merged, seen = [], set()
        i = 0
        while len(merged) < a.outputs and any(i < len(dbs) for _, dbs in streams):
            for lab, dbs in streams:
                if i < len(dbs) and len(merged) < a.outputs:
                    key, recs = canonical(dbs[i])
                    if key not in seen:
                        seen.add(key)
                        merged.append(recs)
                        per_source[lab]["used"] += 1
            i += 1
        base = max(consistent, key=lambda s: rank(s[1]))[1]
        out = json.loads(json.dumps(base))
        out["lns"]["returned"] = len(merged)
        out["lns"]["output_is_consistent"] = True
        out["lns"]["restarts_consistent"] = sum(s[1]["lns"].get("restarts_consistent", 0) for s in consistent)
        out["lns"]["true_db_in_output"] = any(s[1]["lns"].get("true_db_in_output") for s in consistent)
    else:
        lab, base, r, c = max(loaded, key=lambda s: rank(s[1]))
        merged = [canonical(db)[1] for db in json.load(open(c))]
        per_source[lab]["used"] = len(merged)
        out = json.loads(json.dumps(base))

    out["number_of_databases"] = len(merged)
    cent, mean_db, first = mse_info(merged)
    out["lns"]["centroid_mse"], out["lns"]["mean_db_mse"], out["lns"]["first_db_mse"] = cent, mean_db, first
    out["merged_from"] = per_source
    out["time_taken_per_run"] = {lab: s["time_taken"] for lab, s in per_source.items()}

    # overwrite whatever is there (written to .tmp first, then renamed over it)
    os.makedirs(out_dir, exist_ok=True)
    with open(out_rec + ".tmp", "w") as fh:
        json.dump(merged, fh)
    os.replace(out_rec + ".tmp", out_rec)
    with open(out_res + ".tmp", "w") as fh:
        json.dump(out, fh, indent=2)
    os.replace(out_res + ".tmp", out_res)
    used = ", ".join(f"{lab} {s['used']}" for lab, s in per_source.items())
    print(f"{stem}: merged {len(merged)} databases ({'consistent' if consistent else 'NONE consistent: best source copied'}; {used})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
