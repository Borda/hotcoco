"""gen_assign_fixtures.py — freeze scipy's exact LSAP output as a Rust test oracle.

primitives::assign::lsap is a semantic port of scipy's rectangular_lsap. Optimality
is checked in-Rust by brute force, but matching scipy's *specific* co-optimal
assignment (its tie-breaking) needs scipy itself. This emits a diverse set of
(cost matrix, maximize, scipy row_ind/col_ind) — weighted toward small-integer
matrices where co-optima exist and tie-breaking actually matters — for the Rust
test at crates/hotcoco/src/primitives/testdata/lsap_scipy.json.

    uv run python scripts/gen_assign_fixtures.py
"""

import json
import random
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

OUT = Path(__file__).parent.parent / "crates/hotcoco/src/primitives/testdata/lsap_scipy.json"

rng = random.Random(0xA551617)


def case(nr, nc, maximize, hi, kind):
    if kind == "int":
        cost = [[rng.randint(0, hi) for _ in range(nc)] for _ in range(nr)]
    else:
        cost = [[round(rng.uniform(0.0, hi), 4) for _ in range(nc)] for _ in range(nr)]
    m = np.array(cost, dtype=float)
    r, c = linear_sum_assignment(m, maximize=maximize)
    return {
        "nr": nr,
        "nc": nc,
        "maximize": maximize,
        "cost": [float(x) for row in cost for x in row],  # row-major flat
        "row_ind": [int(x) for x in r],
        "col_ind": [int(x) for x in c],
    }


def main():
    cases = []
    # Tie-heavy: small-integer matrices (many co-optima) across shapes/sizes.
    for _ in range(400):
        nr = rng.randint(1, 6)
        nc = rng.randint(1, 6)
        hi = rng.choice([1, 2, 3])  # tiny value range => lots of ties
        cases.append(case(nr, nc, rng.random() < 0.5, hi, "int"))
    # Continuous (unique optima) — guards the common path.
    for _ in range(150):
        nr = rng.randint(1, 7)
        nc = rng.randint(1, 7)
        cases.append(case(nr, nc, rng.random() < 0.5, 10.0, "float"))
    # A few all-equal matrices — maximal degeneracy stresses tie order directly.
    for n in (2, 3, 4, 5):
        cases.append({"nr": n, "nc": n, "maximize": False, "cost": [1.0] * (n * n), **_scipy_all_equal(n)})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cases, separators=(",", ":")) + "\n")
    print(f"wrote {len(cases)} cases to {OUT} ({OUT.stat().st_size} bytes)")


def _scipy_all_equal(n):
    r, c = linear_sum_assignment(np.ones((n, n)))
    return {"row_ind": [int(x) for x in r], "col_ind": [int(x) for x in c]}


if __name__ == "__main__":
    main()
