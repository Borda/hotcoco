#!/usr/bin/env python3
"""Parity check: hotcoco tide_errors() vs tidecv reference.

Usage:
    uv run python scripts/parity_tide.py

Requires: uv pip install tidecv

Tolerances (ΔAP in [0,1] scale):
  Cls, Loc, Both, Dupe, Bkg: ±0.005
  Miss: ±0.10 — a documented deviation, not a parity target. See below.

Why Miss is not held to +/-0.005
--------------------------------
A previous version blamed "COCO per-category eval vs tidecv global top-100 per
image". That explanation is FALSE and was never measured: the largest image in
bbox_val2017_results.json holds 76 detections, so neither cap drops any of the
43,715. Recorded here because this is where someone comes to relax the tolerance.

The real cause is crowd handling, and it is deliberate. hotcoco follows COCO -- an
iscrowd region takes part in matching and a detection landing on one is ignored.
tidecv removes crowd regions from the GT and ignores a detection afterwards only if
it is unmatched AND same-class AND clears IoA 0.5, so it books as false positives
some detections COCO-style evaluation ignores.

Measured on val2017: 4,305 detections sit over a same-class crowd region; tidecv
reports 1,042 more FPs across the five types, 617 more Loc, and 573 fewer Miss.
Miss is the residue after Loc/Cls claim their GT, so it inherits all of it.

Note what this tolerance does NOT cover: the FP counts differ by ~10% on Loc and
Bkg while both pass at +/-0.005, because a low-ranked error barely moves the AP
integral. ΔAP parity is not count parity.

Full write-up, including why matching tidecv would be worse: the "Coming from
tidecv" section of docs/guide/evaluation.md.

tidecv API notes (v1.0.1):
  tide.run_thresholds[key] is a list of TIDERun objects (one per threshold).
  fix_main_errors() returns ΔAP in [0, 100] scale; divide by 100 for [0,1].
"""

import sys
from pathlib import Path

_DATA = Path(__file__).resolve().parents[1] / "data"
GT_PATH = str(_DATA / "annotations/instances_val2017.json")
DT_PATH = str(_DATA / "bbox_val2017_results.json")

# --- hotcoco ---
from hotcoco import COCO, COCOeval  # noqa: E402

gt = COCO(GT_PATH)
dt = gt.load_res(DT_PATH)
ev = COCOeval(gt, dt, "bbox")
ev.evaluate()
hc = ev.tide_errors(pos_thr=0.5, bg_thr=0.1)

print("=== hotcoco tide_errors ===")
print(f"ap_base: {hc['ap_base']:.4f}")
print("delta_ap:")
for k in ["Cls", "Loc", "Both", "Dupe", "Bkg", "Miss", "FP", "FN"]:
    print(f"  {k:4s}: {hc['delta_ap'][k]:.4f}")
print("counts:")
for k in ["Cls", "Loc", "Both", "Dupe", "Bkg", "Miss"]:
    print(f"  {k:4s}: {hc['counts'][k]}")

# --- tidecv ---
try:
    from tidecv import TIDE, datasets
    from tidecv.errors.main_errors import BackgroundError, BoxError, ClassError, DuplicateError, MissedError, OtherError

    tide = TIDE()
    tide.evaluate_range(datasets.COCO(GT_PATH), datasets.COCOResult(DT_PATH), mode=TIDE.BOX)
    tide.summarize()

    # Get the run at pos_thresh=0.5 (first in the list)
    thr_key = list(tide.run_thresholds.keys())[0]
    r50 = tide.run_thresholds[thr_key][0]
    assert abs(r50.pos_thresh - 0.5) < 1e-6, f"Expected pos_thresh=0.5, got {r50.pos_thresh}"

    # ΔAP from fix_main_errors() is in [0, 100] scale; convert to [0, 1]
    main_errors = r50.fix_main_errors()

    err_map = [
        (ClassError, "Cls"),
        (BoxError, "Loc"),
        (OtherError, "Both"),
        (DuplicateError, "Dupe"),
        (BackgroundError, "Bkg"),
        (MissedError, "Miss"),
    ]

    print("\n=== tidecv (pos_thr=0.5) ===")
    tc_delta = {}
    for err_type, name in err_map:
        val = main_errors.get(err_type, 0.0) / 100.0  # [0,100] → [0,1]
        cnt = len(r50.error_dict.get(err_type, []))
        tc_delta[name] = val
        print(f"  {name:4s}: delta_ap={val:.4f}  count={cnt}")

    tc_counts = {name: len(r50.error_dict.get(err_type, [])) for err_type, name in err_map}

    print("\n=== Comparison (hotcoco vs tidecv) ===")
    # Miss is a documented deviation from crowd handling, not a parity target --
    # see the module docstring for the measurement that killed the previous
    # (false) explanation. The five FP types are held to real parity.
    tol = 0.005
    tol_miss = 0.10  # bounded, not pinned: crowd-handling deviation (see docstring)
    all_ok = True

    for name in ["Cls", "Loc", "Both", "Dupe", "Bkg", "Miss"]:
        hc_v = hc["delta_ap"][name]
        tc_v = tc_delta.get(name, float("nan"))
        hc_cnt = hc["counts"][name]
        tc_cnt = tc_counts.get(name, 0)
        diff = abs(hc_v - tc_v)
        cur_tol = tol_miss if name == "Miss" else tol
        status = "OK" if diff <= cur_tol else "FAIL"
        if status == "FAIL":
            all_ok = False
        note = " (crowd-handling deviation)" if name == "Miss" else ""
        print(
            f"  {name:4s}: hc={hc_v:.4f} (n={hc_cnt:5d})  "
            f"tc={tc_v:.4f} (n={tc_cnt:5d})  diff={diff:.4f}  {status}{note}"
        )

    if not all_ok:
        print("\nSome values exceed tolerance.")
        sys.exit(1)

    print("\nAll ΔAP values within tolerance (±0.005 for Cls/Loc/Both/Dupe/Bkg; ±0.10 for Miss).")

except ImportError:
    # Exit non-zero: without the reference this script has printed hotcoco's own
    # numbers and compared them to nothing. Exiting 0 made it look like a passing
    # parity check, which is how the TIDE gate went unverified for a whole
    # refactor of the TIDE classifier.
    print("\ntidecv not installed — cannot compare against the reference.")
    print("Install with: uv pip install tidecv")
    sys.exit(1)
except Exception as e:
    import traceback

    traceback.print_exc()
    print(f"\ntidecv comparison failed: {e}")
    sys.exit(1)
