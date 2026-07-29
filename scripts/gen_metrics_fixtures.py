"""gen_metrics_fixtures.py — freeze sklearn and netcal outputs as Rust oracles.

`metrics::confusion` and `metrics::calibration` are the two functional-layer
modules with a well-known external reference that nothing compared against. Their
in-crate tests are hand-derived and good, but hand-derived numbers verify the
author's arithmetic, not agreement with the field: an ECE that binned slightly
differently from everyone else would reproduce every fixture and still report a
number nobody could compare to a paper.

Frozen rather than compared live, following `lsap_scipy.json`: the Rust tests then
run in CI with no Python, no sklearn, and no netcal installed.

Two mappings matter.

*Confusion.* hotcoco takes `Option` labels and reserves index `num_classes` for
background, because a detection can match nothing and a ground truth can go
unpredicted — cases `sklearn.metrics.confusion_matrix` has no way to express. So
the comparison is scoped to records where **both** labels are present, which is
exactly sklearn's domain, and the background row and column are checked separately
by the in-crate marginal property test.

*Calibration.* netcal's `ECE(bins)` uses equal-width bins over [0, 1] and weights
each bin by occupancy, which is what `calibration_error` computes over
`calibration_curve`'s output. Scores are drawn with deliberately lopsided
occupancy — a uniform draw fills every bin roughly equally and makes the weighting
unobservable.

    uv run --with scikit-learn --with netcal python scripts/gen_metrics_fixtures.py
"""

import json
import random
from pathlib import Path

import numpy as np
from netcal.metrics import ECE
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).parent.parent
CONFUSION_OUT = ROOT / "crates/hotcoco/src/metrics/testdata/confusion_sklearn.json"
CALIBRATION_OUT = ROOT / "crates/hotcoco/src/metrics/testdata/calibration_netcal.json"

rng = random.Random(0xC0FFEE)
nprng = np.random.default_rng(0xC0FFEE)


def gen_confusion_cases(n_cases=200):
    cases = []
    for _ in range(n_cases):
        num_classes = rng.randint(2, 8)
        n = rng.randint(1, 60)

        # Skewed class frequencies: a uniform draw makes every row sum alike, and
        # a row/column mix-up is then much harder to see.
        heavy = rng.randrange(num_classes)

        def draw():
            return heavy if rng.random() < 0.5 else rng.randrange(num_classes)

        gt = [draw() for _ in range(n)]
        dt = [draw() for _ in range(n)]

        labels = list(range(num_classes))
        m = confusion_matrix(gt, dt, labels=labels).tolist()
        cases.append({"num_classes": num_classes, "gt": gt, "dt": dt, "matrix": m})
    return cases


def gen_calibration_cases(n_cases=200):
    cases = []
    for _ in range(n_cases):
        n_bins = rng.choice([5, 10, 15, 20])
        n = rng.randint(20, 300)

        style = rng.choice(["lopsided_low", "lopsided_high", "uniform", "bimodal"])
        if style == "lopsided_low":
            scores = np.clip(nprng.beta(1.2, 8.0, n), 0.0, 1.0)
        elif style == "lopsided_high":
            scores = np.clip(nprng.beta(8.0, 1.2, n), 0.0, 1.0)
        elif style == "bimodal":
            half = n // 2
            scores = np.clip(np.concatenate([nprng.beta(1.5, 9, half), nprng.beta(9, 1.5, n - half)]), 0.0, 1.0)
        else:
            scores = nprng.uniform(0.0, 1.0, n)

        # Correctness correlated with score, so the model is imperfectly calibrated
        # rather than trivially so — ECE 0 would not distinguish anything.
        skew = nprng.uniform(-0.35, 0.35)
        p = np.clip(scores + skew, 0.0, 1.0)
        matched = (nprng.uniform(0.0, 1.0, n) < p).astype(int)

        ece = float(ECE(int(n_bins)).measure(scores, matched))
        cases.append(
            {
                "n_bins": int(n_bins),
                "style": style,
                "scores": [float(s) for s in scores],
                "matched": [bool(m) for m in matched],
                "ece": ece,
            }
        )
    return cases


def main():
    CONFUSION_OUT.parent.mkdir(parents=True, exist_ok=True)

    conf = gen_confusion_cases()
    CONFUSION_OUT.write_text(json.dumps(conf, indent=1) + "\n")
    print(f"wrote {len(conf)} confusion cases -> {CONFUSION_OUT.relative_to(ROOT)}")

    cal = gen_calibration_cases()
    CALIBRATION_OUT.write_text(json.dumps(cal, indent=1) + "\n")
    nonzero = sum(1 for c in cal if c["ece"] > 1e-9)
    print(f"wrote {len(cal)} calibration cases -> {CALIBRATION_OUT.relative_to(ROOT)}")
    print(f"  {nonzero} with non-zero ECE; max {max(c['ece'] for c in cal):.4f}")


if __name__ == "__main__":
    main()
