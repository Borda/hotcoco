"""gen_val2017_baseline.py — pin pycocotools' val2017 metrics as a standing reference.

`parity.py` compares hotcoco against pycocotools live, which catches hotcoco
drifting away from the reference. It cannot catch both drifting *together*: a
pycocotools upgrade that changed a metric would move the reference, and the
comparison would stay green while every published number quietly shifted.

So the expected values are also written down. They come from **pycocotools**, not
from hotcoco — a recording of our own output would pin whatever the code happened
to do the day it was written, bug included, and would agree with itself forever.

A baseline mismatch is not automatically a bug. It means the reference moved, and
someone has to decide whether that was intended before regenerating.

    uv run python scripts/gen_val2017_baseline.py
"""

from __future__ import annotations

import importlib.metadata as md
import json
import sys
from pathlib import Path

from helpers import VAL2017, suppress_output
from pycocotools.coco import COCO as PyCOCO
from pycocotools.cocoeval import COCOeval as PyCOCOeval

OUT = Path(__file__).parent / "fixtures" / "val2017_expected.json"

COMMENT = (
    "Expected COCO val2017 metrics, produced by PYCOCOTOOLS (not hotcoco) on the "
    "published ppwwyyxx/cocoapi fake-result files. A second, standing check "
    "alongside the live comparison in parity.py: the live run catches hotcoco "
    "drifting from the reference, this catches BOTH of them drifting together - "
    "a pycocotools upgrade that silently changes a metric would otherwise keep "
    "the live comparison green. Regenerate with scripts/gen_val2017_baseline.py."
)


def main() -> int:
    out = {
        "_comment": COMMENT,
        "reference": {"pycocotools": md.version("pycocotools"), "numpy": md.version("numpy")},
        "metrics": {},
    }

    missing = []
    for name, files in VAL2017.items():
        gt, dt = files["gt"], files["dt"]
        if not gt.exists() or not dt.exists():
            missing.append(name)
            continue
        with suppress_output(stderr=False):
            coco_gt = PyCOCO(str(gt))
            coco_dt = coco_gt.loadRes(str(dt))
            ev = PyCOCOeval(coco_gt, coco_dt, name)
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        out["metrics"][name] = [float(v) for v in ev.stats]
        print(f"  {name:<10} {len(ev.stats)} metrics")

    if missing:
        print(f"\nMissing data for: {', '.join(missing)}. Run `just download-coco` first.")
        print("Refusing to write a partial baseline — it would silently stop checking them.")
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {OUT} (pycocotools {out['reference']['pycocotools']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
