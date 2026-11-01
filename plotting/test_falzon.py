"""
Sanity checks for falzon.py. Run with: cd plotting && python test_falzon.py

Brute force on tiny domains: every database we generate must have the same response
multiset RM(D) as D. We also report how often the true E(D) is bigger than ours,
which happens when E(D) needs more than one seed database (Theorem 4.4 of Falzon et
al.). Tiny grids have many density coincidences, so this is far more common here
than on real data.
"""
import itertools
import random
from collections import Counter
import numpy as np
from falzon import enumerate_equivalent_databases, reflectable_components


def response_multiset(db, n0, n1):
    rm = Counter()
    for a0 in range(1, n0 + 1):
        for b0 in range(a0, n0 + 1):
            for a1 in range(1, n1 + 1):
                for b1 in range(a1, n1 + 1):
                    rm[frozenset(i for i, (x, y) in enumerate(db) if a0 <= x <= b0 and a1 <= y <= b1)] += 1
    return frozenset(rm.items())


def check(n0, n1, n, trials, rng):
    cells = [(x, y) for x in range(1, n0 + 1) for y in range(1, n1 + 1)]
    by_rm = {}
    for db in itertools.permutations(cells, n):
        by_rm.setdefault(response_multiset(db, n0, n1), []).append(db)

    extra_seeds = 0
    for _ in range(trials):
        db = rng.sample(cells, n)
        truth = set(by_rm[response_multiset(db, n0, n1)])
        ours = {tuple(map(tuple, np.round(d).astype(int).tolist()))
                for d in enumerate_equivalent_databases(db, n0, n1)}
        assert ours <= truth, f"non-equivalent database generated for {db}"
        extra_seeds += ours != truth
    print(f"[{n0}]x[{n1}], {n} points: all generated databases equivalent; "
          f"E(D) needed extra seeds in {extra_seeds}/{trials}")


if __name__ == "__main__":
    rng = random.Random(1)
    check(4, 4, 3, 60, rng)
    check(3, 5, 3, 60, rng)
    check(5, 3, 4, 40, rng)

    # Theorem 4.3: points (i + 1, i) on [R + 1]^2 give R reflectable components
    R = 6
    assert len(reflectable_components([(i + 1, i) for i in range(1, R + 1)], R + 1, R + 1)) == R
    print("Theorem 4.3 example: OK")
