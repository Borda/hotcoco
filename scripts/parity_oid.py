"""parity_oid.py — hotcoco Open Images AP vs the TensorFlow reference.

Compares hotcoco's `oid_style=True` evaluation against frozen output from the TF
Object Detection API's `OpenImagesDetectionEvaluator(group_of_weight=1.0)` — the
Open Images Challenge detection metric.

Fixtures are checked in (`fixtures/oid_tf_expected.json`), so this needs no network
and no TensorFlow. Regenerate them with `scripts/gen_oid_fixtures.py`.

    uv run python scripts/parity_oid.py [--tol 1e-9] [-v]

Both mAP and per-class AP are compared. Per-class matters on its own: two
categories whose APs are swapped leave the mean identical, which is the same
argument `test_adversarial.py` makes for diffing per-detection decisions rather
than only metrics.

What this does and does not cover
---------------------------------
Covered: group-of absorption (one TP per group-of box, surplus detections ignored,
undetected group-of box is a miss), IoA as the containment measure, and VOC 2010
all-points AP.

Not covered: the non-exhaustive image-level-label rule, the Challenge's third
mechanism, which hotcoco does not implement — the oracle is generated with the
evaluator that omits it, so this comparison says nothing about it. Also not covered:
hierarchy expansion, which hotcoco applies before evaluation rather than inside it.

A mismatch is not automatically a hotcoco bug. It can mean the reference moved;
check `reference` in the fixture file before assuming.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gen_oid_fixtures import MIN_DISCRIMINATING
from helpers import suppress_output
from hotcoco import COCO, COCOeval

FIXTURES = Path(__file__).parent / "fixtures" / "oid_tf_expected.json"


def hotcoco_aps(case) -> tuple[float, dict[str, float]]:
    """Return (mAP, {category_id: AP}) for one case."""
    gt = COCO(
        {
            "images": [{"id": 1, "width": 800, "height": 800, "file_name": "a.jpg"}],
            "annotations": case["annotations"],
            "categories": case["categories"],
        }
    )
    dt = gt.load_res(case["detections"])
    ev = COCOeval(gt, dt, "bbox", oid_style=True)
    # Every OID run prints its `Provenance::Extension` deviation to fd 2. That is
    # correct — and is exactly the gap this script narrows — but one copy per case
    # would bury the result seventy times over.
    with suppress_output():
        ev.run()

    # `per_class` is keyed by category *name*; the fixture is keyed by id, since
    # that is what the TF result keys carry. Map through the case's own category
    # list rather than assuming the `c{i}` convention a second time.
    by_name = ev.results(per_class=True).get("per_class", {})
    per_class = {str(c["id"]): by_name.get(c["name"]) for c in case["categories"]}
    return float(ev.stats[0]), per_class


def per_class_mismatches(expected: dict, got: dict, tol: float) -> list[str]:
    """Per-category disagreements, as human-readable strings.

    `None` on the reference side means TF returned NaN — a category with no ground
    truth, which it excludes from the mean. hotcoco reports its `-1.0` "not
    computed" sentinel for the same case, so the two agree by *both* declining to
    score it; anything else is a real disagreement about whether a category is
    evaluable.
    """
    out = []
    for cid, exp in expected.items():
        g = got.get(cid)
        if exp is None:
            if g is not None and g >= 0.0:
                out.append(f"c{cid}: reference has no AP, hotcoco reports {g:.6f}")
        elif g is None or g < 0.0:
            out.append(f"c{cid}: reference {exp:.6f}, hotcoco reports nothing")
        elif abs(g - exp) > tol:
            out.append(f"c{cid}: ref={exp:.6f} got={g:.6f} diff={abs(g - exp):.3e}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tol", type=float, default=1e-9, help="max absolute AP difference")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every case")
    args = ap.parse_args()

    if not FIXTURES.exists():
        print(f"missing {FIXTURES} — run scripts/gen_oid_fixtures.py")
        return 1

    data = json.loads(FIXTURES.read_text())
    cases = data["cases"]

    print("=" * 68)
    print(f"  hotcoco (oid_style) vs {data['reference']['evaluator']}")
    print(f"  {len(cases)} cases, tolerance {args.tol:g}, mAP + per-class")
    print("=" * 68)

    failures: list[tuple[str, str]] = []
    worst, worst_name = 0.0, "-"

    for case in cases:
        expected = case["expected"]
        try:
            got_map, got_per_class = hotcoco_aps(case)
        except Exception as exc:  # noqa: BLE001
            failures.append((case["name"], f"{type(exc).__name__}: {exc}"))
            continue

        diff = abs(got_map - expected["mAP"])
        if diff > worst:
            worst, worst_name = diff, case["name"]

        why = []
        if diff > args.tol:
            why.append(f"mAP ref={expected['mAP']:.6f} got={got_map:.6f} diff={diff:.3e}")
        why += per_class_mismatches(expected["per_class"], got_per_class, args.tol)
        if why:
            failures.append((case["name"], "; ".join(why)))
        elif args.verbose:
            print(f"  {case['name']:36s} mAP={got_map:.6f}  OK")

    # A corpus that is all 0.0/1.0 agrees with a stub implementation. The floor is
    # `gen_oid_fixtures.MIN_DISCRIMINATING`, imported rather than restated:
    # asserting it here as well means a regenerated corpus cannot quietly weaken
    # the consumer, and there is one number to change if the bar moves.
    discriminating = sum(1 for c in cases if 0.0 < c["expected"]["mAP"] < 1.0)
    print(f"\n  {discriminating}/{len(cases)} cases score strictly between 0 and 1")
    if discriminating < len(cases) * MIN_DISCRIMINATING:
        print("  FAIL: fixture set is mostly saturated and discriminates little.")
        return 1

    if failures:
        print(f"\n  {len(failures)} of {len(cases)} cases FAILED")
        for name, why in failures[:20]:
            print(f"    {name:36s} {why}")
        if len(failures) > 20:
            print(f"    ... and {len(failures) - 20} more")
        return 1

    print(f"  worst mAP difference {worst:.3e} on '{worst_name}'")
    print("\n  ALL OPEN IMAGES PARITY TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
