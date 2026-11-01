"""
Simulate LAMa (perfect knowledge, all queries observed) against Falzon et al. without
re-running either attack.

Falzon et al. get the response multiset RM(D) and output E(D), which does not depend on
the query distribution (falzon.py). LAMa keeps a database only if every record set has
the same probability of being returned under the known distribution P. So, restricted to
E(D), LAMa's output is the subset of E(D) whose probabilities under P match D's.

We check singletons and pairs (T_2). Probabilities are P(query contains the bounding box
of the set), i.e. QueryDistribution::cumulative_prob_lookup of its MBQ. T_2 survivors are
an upper bound on what LAMa's T_4 keeps inside E(D).

Query weights mirror src/LAMA/query.rs: queries (c, d), c <= d, listed in lexicographic
order (c0, c1, d0, d1) as get_dom_pairs would without par_bridge; Gaussian weight is
Normal(len/2, len/5).pdf(i) and Beta weight Beta(2, 1).pdf((i + 0.5) / len) for index i.
"""
import argparse
import numpy as np
from scipy.stats import norm, beta as beta_dist
from falzon import enumerate_equivalent_databases, reflectable_components


def query_weights(n0, n1, dist):
    """W[c0, c1, d0, d1] on 0-indexed grid of n0 x n1 values (zero where c > d)."""
    c0, c1, d0, d1 = np.meshgrid(np.arange(n0), np.arange(n1), np.arange(n0), np.arange(n1), indexing='ij')
    valid = (c0 <= d0) & (c1 <= d1)
    idx = np.cumsum(valid.ravel()).reshape(valid.shape) - 1    # lexicographic index of valid queries
    total = valid.sum()
    if dist == "uniform":
        w = np.ones(valid.shape)
    elif dist == "gaussian":
        w = norm(total / 2, total / 5).pdf(idx)
    elif dist == "beta":
        w = beta_dist(2, 1).pdf((idx + 0.5) / total)
    else:
        raise ValueError(dist)
    return np.where(valid, w, 0.0) / np.where(valid, w, 0.0).sum()


def enclosing_prob(W):
    """S[l0, l1, h0, h1] = P(query contains box [l, h]) = sum_{c <= l, d >= h} W[c, d]."""
    S = W.cumsum(0).cumsum(1)
    S = S[:, :, ::-1, ::-1].cumsum(2).cumsum(3)[:, :, ::-1, ::-1]
    return S


def t2_signature(pts, S):
    """Probabilities of every singleton and pair, indexed by record."""
    p = np.asarray(pts, dtype=int)
    lo = np.minimum(p[:, None, :], p[None, :, :])
    hi = np.maximum(p[:, None, :], p[None, :, :])
    return S[lo[..., 0], lo[..., 1], hi[..., 0], hi[..., 1]]


def simulate(grid_pts, n0, n1, dists=("uniform", "gaussian", "beta"), max_r=14):
    """grid_pts: 0-indexed integer points on [0, n0) x [0, n1). Returns {dist: (|E(D)|, |LAMa|, lama_mses, falzon_mses)}."""
    pts1 = np.asarray(grid_pts) + 1
    r = len(reflectable_components(pts1, n0, n1))
    if r > max_r:
        raise ValueError(f"r = {r} too large to enumerate")
    cands = [np.round(c).astype(int) - 1 for c in enumerate_equivalent_databases(pts1, n0, n1)]
    true = np.asarray(grid_pts)
    mses = np.array([np.mean(np.sum((c - true) ** 2, axis=1)) for c in cands])

    out = {}
    for dist in dists:
        S = enclosing_prob(query_weights(n0, n1, dist))
        sig = t2_signature(true, S)
        keep = np.array([np.allclose(t2_signature(c, S), sig, rtol=1e-9, atol=1e-15) for c in cands])
        out[dist] = (len(cands), int(keep.sum()), mses[keep], mses)
    return r, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="LAMa reconstruction JSON; its true points are used")
    ap.add_argument("--grid", type=int, nargs=2, default=(50, 50), help="grid passed to TwoDMap::new")
    ap.add_argument("--pad", type=int, default=0, help="empty cells added on every side")
    args = ap.parse_args()

    if args.json:
        import json
        with open(args.json) as f:
            t = np.array([p["true"] for p in json.load(f)[0]])
        span = t.max(0) - t.min(0)
        g = np.round((t - t.min(0)) / np.where(span > 0, span / np.array(args.grid), 1)).astype(int)
        pts = g + args.pad
        n0, n1 = args.grid[0] + 1 + 2 * args.pad, args.grid[1] + 1 + 2 * args.pad
    else:
        # Demo: a diagonal "route" in a padded 30 x 30 domain, like a single Spitz day
        pts = [(i + 1, i) for i in range(4, 24, 3)] + [(i, i + 2) for i in range(6, 22, 5)]
        n0 = n1 = 30
    r, res = simulate(pts, n0, n1)
    print(f"r = {r}")
    for d, (ne, nl, lm, fm) in res.items():
        print(f"{d:8s}  Falzon |E(D)| = {ne:5d} (mean MSE {fm.mean():.2f}, max {fm.max():.2f})   "
              f"LAMa = {nl:5d} (mean MSE {lm.mean():.2f}, max {lm.max():.2f})")
