"""Writes ../figures/3d_recon/points_dump.json for spatial_reconstructions.plot_3d_reconstruction.

From a LAMa reconstruction JSON (a list of databases): every database is put in whichever grid symmetry
is closest to the truth (same orientation oracle as the 2D figures). Then, per record, the convex hull of
its positions across all databases gives a hull centroid and a diameter midpoint. What is written as
"centroid_points" (the name plot_3d_reconstruction reads) depends on --points:
    best        hull centroid or diameter midpoint, whichever has the lower MSE (default)
    centroid    hull centroid
    diameter    diameter midpoint
    lowest-cost database 0 of the reconstruction JSON = lowest total error (attacker-side cost)

    python dump_3d_points.py                                   # nh gaussian 30%
    python dump_3d_points.py --points diameter
    python dump_3d_points.py --points lowest-cost
    python dump_3d_points.py --recon ../databases/16x16x14/nh/limits/nh_beta_p0.3_reconstruction.json
"""
import argparse
import itertools
import json
import os

import numpy as np
from scipy.spatial import ConvexHull, distance_matrix


def symmetries(dbs, dims):
    """dbs (#dbs, k, d) on [1, n_i] -> (#syms, #dbs, k, d): each axis reflected or not, and equal-length
    axes permuted (for 16x16x14: x/y swap x 8 reflections = 16)."""
    d = len(dims)
    perms = [p for p in itertools.permutations(range(d)) if all(dims[p[i]] == dims[i] for i in range(d))]
    out = []
    for p in perms:
        a = dbs[..., list(p)]
        for flips in itertools.product((False, True), repeat=d):
            b = a.copy()
            for i, f in enumerate(flips):
                if f:
                    b[..., i] = dims[i] + 1 - b[..., i]
            out.append(b)
    return np.stack(out)


def hull_points(cloud):
    """Hull centroid and diameter midpoint of one record's positions over all databases."""
    u = np.unique(cloud, axis=0)
    verts = u
    if len(u) > cloud.shape[1]:
        try:
            verts = u[ConvexHull(u).vertices]
        except Exception:  # coplanar / collinear positions: use every position
            pass
    centroid = verts.mean(0)
    if len(verts) > 1:
        i, j = np.unravel_index(np.argmax(distance_matrix(verts, verts)), (len(verts),) * 2)
        mid = (verts[i] + verts[j]) / 2.0
    else:
        mid = verts[0]
    return centroid, mid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", default="../databases/16x16x14/nh/limits/nh_gaussian_p0.3_reconstruction.json")
    ap.add_argument("--dims", default="16,16,14")
    ap.add_argument("--out", default="../figures/3d_recon/points_dump.json")
    ap.add_argument("--points", choices=["best", "centroid", "diameter", "lowest-cost"], default="best")
    ap.add_argument("--no-orient", action="store_true", help="use the databases as LAMa output them")
    args = ap.parse_args()
    dims = [int(x) for x in args.dims.split(",")]

    with open(args.recon) as f:
        data = json.load(f)
    true = np.array([r["true"] for r in data[0]], dtype=float)
    dbs = np.array([[r["reconstructed"] for r in db] for db in data], dtype=float)

    if not args.no_orient:
        s = symmetries(dbs, dims)
        best = ((s - true) ** 2).sum(-1).mean(-1).argmin(0)
        dbs = s[best, np.arange(len(dbs))]

    cents, mids = zip(*(hull_points(dbs[:, i]) for i in range(len(true))))
    cents, mids = np.array(cents), np.array(mids)
    d = true.shape[1]
    candidates = {"centroid": cents, "diameter": mids, "lowest-cost": dbs[0]}
    mse = {k: float(((v - true) ** 2).sum(1).mean() / d) for k, v in candidates.items()}
    choice = args.points if args.points != "best" else min(("centroid", "diameter"), key=mse.get)
    pts = candidates[choice]
    print(f"{len(dbs)} databases, {len(true)} records; MSE (/dims): centroid {mse['centroid']:.3f}, "
          f"diameter midpoint {mse['diameter']:.3f}, lowest-cost DB {mse['lowest-cost']:.3f} -> using {choice}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"true_points": true.tolist(), "centroid_points": pts.tolist(),
                   "method": choice, "mse": mse, "source": args.recon,
                   "num_databases": len(dbs), "oriented": not args.no_orient}, f)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
