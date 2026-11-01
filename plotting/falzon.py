"""
Direct construction of the output of Falzon et al., "Full Database Reconstruction in
Two Dimensions" (CCS 2020), without re-running their attack.

Given the full response multiset RM(D), their attack returns E(D): every database with
the same response multiset as D. By their Theorem 4.4 (and their remark that real data
only ever needed a single seed database), E(D) is obtained from D by

    1. reflecting any subset of the reflectable components of D (Lemma 4.2), then
    2. reflecting the result horizontally and/or vertically,

giving 4 * 2^r databases, where r is the number of reflectable components.

Notation follows the paper: domain D = [N0] x [N1] with values 1..N, reflection
    sigma(w) = (w1 * (N0 + 1) / (N1 + 1), w0 * (N1 + 1) / (N0 + 1))     (Eq. 4)
and p <= q means p is dominated by q in both coordinates.

Since RM(D) counts every query once, E(D) does not depend on the query distribution.
"""
import numpy as np


def reflect(points, n0, n1):
    """sigma(w) from Eq. (4) of Falzon et al. Points are 1-indexed, shape (n, 2)."""
    points = np.asarray(points, dtype=float)
    return np.stack([points[:, 1] * (n0 + 1) / (n1 + 1),
                     points[:, 0] * (n1 + 1) / (n0 + 1)], axis=1)


def is_reflectable(points, n0, n1, tol=1e-9):
    """Lemma 3.1: sigma(w) is an integral point of [N0] x [N1]."""
    refl = reflect(points, n0, n1)
    integral = np.all(np.abs(refl - np.round(refl)) < tol, axis=1)
    in_domain = (refl[:, 0] >= 1 - tol) & (refl[:, 0] <= n0 + tol) & \
                (refl[:, 1] >= 1 - tol) & (refl[:, 1] <= n1 + tol)
    return integral & in_domain


def find_components(points, n0, n1):
    """
    Partition D into components (Sec. 4). For p in C and q not in C we need either
        q <= p and q <= sigma(p),   or   p <= q and sigma(p) <= q.
    Whenever that fails for an ordered pair (p, q), p and q must share a component, so
    the components are the connected components of the "fails" graph (Lemma 4.1).
    Returns a list of index arrays.
    """
    pts = np.asarray(points, dtype=float)
    refl = reflect(pts, n0, n1)
    n = len(pts)

    # le[i, j] = point i <= point j (dominance)
    def dom(a, b):
        return (a[:, None, 0] <= b[None, :, 0]) & (a[:, None, 1] <= b[None, :, 1])

    q_below = dom(pts, pts).T & dom(pts, refl).T     # [p, q]: q <= p and q <= sigma(p)
    q_above = dom(pts, pts) & dom(refl, pts)          # [p, q]: p <= q and sigma(p) <= q
    fails = ~(q_below | q_above)
    fails = fails | fails.T
    np.fill_diagonal(fails, False)

    # Union-find over the fails graph
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in zip(*np.nonzero(np.triu(fails))):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [np.array(g) for g in groups.values()]


def reflectable_components(points, n0, n1):
    """
    Components whose points are all reflectable. A component whose reflection leaves
    it unchanged (e.g. a single point on the main diagonal, which the paper defines as
    non-reflectable) is dropped, since reflecting it does not give a new database.
    """
    pts = np.asarray(points, dtype=float)
    ok = is_reflectable(pts, n0, n1)
    refl = reflect(pts, n0, n1)
    comps = []
    for comp in find_components(pts, n0, n1):
        if ok[comp].all() and not np.allclose(refl[comp], pts[comp]):
            comps.append(comp)
    return comps


def global_flips(n0, n1):
    """Identity, horizontal, vertical and both, on 1-indexed points."""
    def make(fx, fy):
        def f(p):
            out = np.array(p, dtype=float, copy=True)
            if fx: out[..., 0] = n0 + 1 - out[..., 0]
            if fy: out[..., 1] = n1 + 1 - out[..., 1]
            return out
        return f
    return [make(fx, fy) for fx in (False, True) for fy in (False, True)]


def enumerate_equivalent_databases(points, n0, n1):
    """
    Explicitly list E(D) (as per-record coordinate arrays, 1-indexed). Only intended for
    small r, e.g. tests. Use falzon_mse_distribution for plotting.
    """
    pts = np.asarray(points, dtype=float)
    refl = reflect(pts, n0, n1)
    comps = reflectable_components(pts, n0, n1)
    out = []
    for mask in range(2 ** len(comps)):
        db = pts.copy()
        for c, comp in enumerate(comps):
            if mask >> c & 1:
                db[comp] = refl[comp]
        for flip in global_flips(n0, n1):
            out.append(flip(db))
    return out


def falzon_mse_distribution(true_pts, grid, max_exact=2 ** 20, num_samples=1_000_000, seed=0):
    """
    MSE (same definition as utils.calculate_mse / Rust calculate_mse) of every database in
    Falzon et al.'s output E(D), measured against the true database.

    true_pts: (n, 2) true coordinates as saved in the LAMa JSON ("true" field).
    grid:     (G0, G1) target grid passed to TwoDMap::new, e.g. (50, 50) or (350, 50).
              The grid domain is 0..G inclusive, i.e. N = G + 1 values per axis.

    TwoDMap stores grid value g and saves (g + offset) / scale, which is an affine map
    whose minimum is g = 0 and maximum is g = G, so we recover g exactly from the
    saved coordinates and map the reflected databases back into the same units.

    Returns (mses, r): mses has 4 * 2^r entries when 2^r <= max_exact, otherwise
    4 * num_samples entries drawn uniformly from E(D).
    """
    true_pts = np.asarray(true_pts, dtype=float)
    grid = np.asarray(grid, dtype=float)
    t_min = true_pts.min(axis=0)
    t_span = true_pts.max(axis=0) - t_min
    unit = np.where(t_span > 0, t_span / grid, 1.0)    # saved units per grid cell (1 / scale)

    g = np.round((true_pts - t_min) / unit)
    n0, n1 = int(grid[0]) + 1, int(grid[1]) + 1
    pts = g + 1                                         # Falzon et al. use [N] = 1..N

    def to_saved(p):
        return (p - 1) * unit + t_min

    refl = reflect(pts, n0, n1)
    comps = reflectable_components(pts, n0, n1)
    in_comp = np.zeros(len(pts), dtype=bool)
    for comp in comps:
        in_comp[comp] = True
    n = len(pts)
    r = len(comps)

    rng = np.random.default_rng(seed)
    all_mses = []
    for flip in global_flips(n0, n1):
        sq_orig = np.sum((to_saved(flip(pts)) - true_pts) ** 2, axis=1)
        sq_refl = np.sum((to_saved(flip(refl)) - true_pts) ** 2, axis=1)

        fixed = sq_orig[~in_comp].sum()
        cost = np.array([[sq_orig[c].sum(), sq_refl[c].sum()] for c in comps]).reshape(r, 2)

        if 2 ** r <= max_exact:
            # SSE of every subset, built up one component at a time
            sse = np.array([fixed])
            for a, b in cost:
                sse = np.concatenate([sse + a, sse + b])
        else:
            bits = rng.integers(0, 2, size=(num_samples, r)).astype(bool)
            sse = fixed + np.where(bits, cost[:, 1], cost[:, 0]).sum(axis=1)
        all_mses.append(sse / n)

    return np.concatenate(all_mses), r
