"""Parity check: hotcoco vs pycocotools on COCO val2017.

Uses third-party published fake result files from ppwwyyxx/cocoapi.
See docs/getting-started/installation.md for download instructions.

Tolerances are 1e-12 across the board — tight enough that only floating-point
noise passes.

They used to be 1e-4 (bbox) and 2e-4 (segm), sized around real divergences that
have since been removed: the threshold grids now match `numpy.linspace`
bit-for-bit, and polygon rasterization reproduces the reference's per-architecture
rounding (FMA on arm64, two roundings on x86-64).
Measured worst case across all 34 metrics is now 3.7e-14, so a 1e-4 gate would
have accepted a number wrong in the fourth decimal place. A gate has to be sized
to what the code actually does, or it stops being a gate.

Usage:
    uv run python scripts/parity.py
    just parity
"""

import json
import sys
from pathlib import Path

from helpers import VAL2017, suppress_output
from hotcoco import COCO, COCOeval
from pycocotools.coco import COCO as PyCOCO
from pycocotools.cocoeval import COCOeval as PyCOCOeval

# Files come from `helpers.VAL2017`; the tolerance is this script's own, and is
# the same for every iou_type — see the module docstring for how it was sized.
TOL = 1e-12
TOL_LABEL = f"<= {TOL:.0e}"


def run_pycocotools(gt_file, dt_file, iou_type):
    with suppress_output(stderr=False):
        gt = PyCOCO(str(gt_file))
        dt = gt.loadRes(str(dt_file))
        ev = PyCOCOeval(gt, dt, iou_type)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return ev.stats.tolist()


def run_hotcoco(gt_file, dt_file, iou_type):
    with suppress_output(stderr=False):
        gt = COCO(str(gt_file))
        dt = gt.load_res(str(dt_file))
        ev = COCOeval(gt, dt, iou_type)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return ev.stats, ev.metric_keys()


# A pinned baseline alongside the live comparison.
#
# The live run catches hotcoco drifting away from pycocotools. It cannot catch
# both of them drifting *together*: a pycocotools upgrade that changed a metric
# would move the reference and the comparison would stay green. The baseline was
# produced by pycocotools itself (see scripts/gen_val2017_baseline.py), so a
# mismatch means the reference moved and someone has to decide whether that is
# intended.
# Loaded unconditionally, and a missing or unreadable file is fatal. It used to
# default to `{}` and every per-type lookup was guarded by `is not None`, so
# deleting the fixture — or renaming one key inside it — left the script printing
# ALL METRICS PASS having checked nothing against the pin. A baseline that
# disappears silently is worse than no baseline: it reports the check it isn't
# running.
BASELINE_PATH = Path(__file__).parent / "fixtures" / "val2017_expected.json"
if not BASELINE_PATH.exists():
    sys.exit(f"missing baseline {BASELINE_PATH} — regenerate with scripts/gen_val2017_baseline.py")
try:
    BASELINE = json.loads(BASELINE_PATH.read_text())
    BASELINE_METRICS = BASELINE["metrics"]
    BASELINE_REF = BASELINE["reference"]
except (json.JSONDecodeError, KeyError) as exc:
    sys.exit(f"unreadable baseline {BASELINE_PATH}: {exc} — regenerate with scripts/gen_val2017_baseline.py")

all_pass = True

for iou_type, files in VAL2017.items():
    print(f"\n{'=' * 68}")
    print(f"  {iou_type}  ({TOL_LABEL})")
    print(f"{'=' * 68}")
    print(f"  {'Metric':<8} {'pycocotools':>14} {'hotcoco':>14} {'diff':>12}  status")
    print(f"  {'-' * 58}")

    py = run_pycocotools(files["gt"], files["dt"], iou_type)
    rs, metric_names = run_hotcoco(files["gt"], files["dt"], iou_type)

    type_pass = True
    for i, name in enumerate(metric_names):
        diff = abs(py[i] - rs[i])
        ok = diff <= TOL
        if not ok:
            type_pass = False
            all_pass = False
        status = "PASS" if ok else "FAIL"
        print(f"  {name:<8} {py[i]:>14.8f} {rs[i]:>14.8f} {diff:>12.2e}  {status}")

    # Compare against the pinned reference values as well. An absent key is a
    # failure, not a skip — the alternative is a green run over an empty check.
    expected = BASELINE_METRICS.get(iou_type)
    if expected is None:
        print(f"\n  BASELINE: no pinned values for '{iou_type}' — nothing was checked against the pin. FAIL")
        print("    Regenerate with scripts/gen_val2017_baseline.py.")
        type_pass = False
        all_pass = False
    elif len(expected) != len(py):
        print(f"\n  BASELINE: expected {len(expected)} metrics, reference produced {len(py)} — FAIL")
        type_pass = False
        all_pass = False
    else:
        drift = [(metric_names[i], expected[i], py[i]) for i in range(len(expected)) if abs(expected[i] - py[i]) > TOL]
        if drift:
            print(f"\n  BASELINE DRIFT — pinned values were produced by {BASELINE_REF}:")
            for nm, want, got in drift:
                print(f"    {nm:<8} pinned={want:.8f}  reference now={got:.8f}  diff={abs(want - got):.2e}")
            print("    The reference implementation moved. Confirm intended, then regenerate.")
            type_pass = False
            all_pass = False
        else:
            print(f"  {'(baseline)':<8} {len(expected)} pinned reference values match")

    result_label = "ALL PASS" if type_pass else "SOME METRICS FAILED"
    print(f"\n  {result_label}")

print(f"\n{'=' * 68}")
if all_pass:
    print("ALL METRICS PASS")
else:
    print("PARITY FAILURES DETECTED")
    sys.exit(1)
