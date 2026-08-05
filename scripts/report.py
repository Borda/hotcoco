"""Generate an evaluation report PDF for COCO val2017."""

import argparse

import hotcoco
from helpers import VAL2017
from hotcoco.plot import report

# `kpt` is the spelling `just report type=kpt` passes; the evaluator's own name
# for it is `keypoints`, and passing `kpt` through was a hard error.
_ALIASES = {"kpt": "keypoints"}

parser = argparse.ArgumentParser()
parser.add_argument("--type", default="bbox", choices=["bbox", "segm", "kpt", "keypoints"])
parser.add_argument("--out", default="report.pdf")
args = parser.parse_args()

iou_type = _ALIASES.get(args.type, args.type)
# Absolute paths from helpers.DATA_DIR: relative strings only worked from the
# repo root, so `just report` broke anywhere else.
gt_path = str(VAL2017[iou_type]["gt"])
dt_path = str(VAL2017[iou_type]["dt"])

gt = hotcoco.COCO(gt_path)
dt = gt.load_res(dt_path)
ev = hotcoco.COCOeval(gt, dt, iou_type)
ev.run()
report(ev, save_path=args.out, gt_path=gt_path, dt_path=dt_path)
print(f"Saved {args.out}")
