"""Benchmark: pycocotools vs faster-coco-eval vs hotcoco on COCO val2017.

Generates deterministic synthetic detections from GT annotations (seed=42).
Only the val2017 annotation files are needed — no result files required.
See docs/getting-started/installation.md for download instructions.

Wall clock time for all three evaluation types. Optional --scale flag
multiplies detections to simulate higher load.

Usage:
    uv run python scripts/bench.py
    uv run python scripts/bench.py --scale 10
    uv run python scripts/bench.py --types bbox segm
    just bench
"""

import argparse
import json
import os
import random
import tempfile
import time

import pycocotools.mask as mask_utils
from faster_coco_eval import COCO as FcCOCO
from faster_coco_eval import COCOeval_faster
from helpers import VAL2017, suppress_output
from hotcoco import COCO, COCOeval
from pycocotools.coco import COCO as PyCOCO
from pycocotools.cocoeval import COCOeval as PyCOCOeval

# Detections are synthesized from the GT, so only the annotation files in
# `helpers.VAL2017` are used — the result files are not read here.

# ~1 detection per GT annotation in val2017 instances (~36,781 annotations).
BASE_DETS = 36_781


def generate_detections(gt_path, iou_type, n_dets, seed=42):
    """Generate deterministic synthetic detections for timing benchmarks.

    AP scores are meaningless, but detection count and format are representative.
    Fixed seed ensures identical output across runs.
    """
    rng = random.Random(seed)
    with open(gt_path) as f:
        gt = json.load(f)

    images = {img["id"]: img for img in gt["images"]}
    cat_ids = [c["id"] for c in gt["categories"]]
    img_ids = list(images.keys())

    dets = []
    for _ in range(n_dets):
        img_id = rng.choice(img_ids)
        img = images[img_id]
        w, h = img["width"], img["height"]
        cat_id = rng.choice(cat_ids)
        score = rng.uniform(0.01, 0.99)

        bw = rng.uniform(10, max(11, w * 0.5))
        bh = rng.uniform(10, max(11, h * 0.5))
        x = rng.uniform(0, max(0, w - bw))
        y = rng.uniform(0, max(0, h - bh))

        det = {"image_id": img_id, "category_id": cat_id, "score": score}

        if iou_type == "bbox":
            det["bbox"] = [x, y, bw, bh]
        elif iou_type == "segm":
            poly = [[x, y, x + bw, y, x + bw, y + bh, x, y + bh]]
            rle = mask_utils.frPyObjects(poly, int(h), int(w))[0]
            rle["counts"] = rle["counts"].decode("utf-8") if isinstance(rle["counts"], bytes) else rle["counts"]
            det["segmentation"] = rle
        elif iou_type == "keypoints":
            kpts = []
            for _ in range(17):
                kx = rng.uniform(x, x + bw)
                ky = rng.uniform(y, y + bh)
                kpts.extend([kx, ky, 2])
            det["keypoints"] = kpts
            det["bbox"] = [x, y, bw, bh]
            det["category_id"] = cat_ids[0]  # person

        dets.append(det)

    return dets


def bench_pycocotools(gt_file, dt_file, iou_type):
    """Returns (load_seconds, eval_seconds); load = ctor + loadRes."""
    with suppress_output(stderr=False):
        t0 = time.perf_counter()
        gt = PyCOCO(str(gt_file))
        dt = gt.loadRes(str(dt_file))
        t_load = time.perf_counter() - t0
        t0 = time.perf_counter()
        ev = PyCOCOeval(gt, dt, iou_type)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        t_eval = time.perf_counter() - t0
    return t_load, t_eval


def bench_faster_coco_eval(gt_file, dt_file, iou_type):
    """Returns (load_seconds, eval_seconds); load = ctor + loadRes."""
    with suppress_output(stderr=False):
        t0 = time.perf_counter()
        gt = FcCOCO(str(gt_file))
        dt = gt.loadRes(str(dt_file))
        t_load = time.perf_counter() - t0
        t0 = time.perf_counter()
        ev = COCOeval_faster(gt, dt, iou_type)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        t_eval = time.perf_counter() - t0
    return t_load, t_eval


def bench_hotcoco(gt_file, dt_file, iou_type):
    """Returns (load_seconds, eval_seconds); load = ctor + loadRes."""
    with suppress_output(stderr=False):
        t0 = time.perf_counter()
        gt = COCO(str(gt_file))
        dt = gt.load_res(str(dt_file))
        t_load = time.perf_counter() - t0
        t0 = time.perf_counter()
        ev = COCOeval(gt, dt, iou_type)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        t_eval = time.perf_counter() - t0
    return t_load, t_eval


def strip_segmentation(gt_path, out_path):
    """Write a copy of a GT file with every `segmentation` field removed.

    The official instances files carry a polygon on every annotation — about
    two-thirds of the file bytes — which bbox evaluation never reads. This
    variant represents datasets that never had masks (custom bbox datasets,
    YOLO conversions, Objects365).
    """
    with open(gt_path) as f:
        gt = json.load(f)
    for ann in gt["annotations"]:
        ann.pop("segmentation", None)
    with open(out_path, "w") as f:
        json.dump(gt, f)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--scale",
        type=int,
        default=1,
        metavar="N",
        help=f"Multiply baseline detection count ({BASE_DETS:,}) by N (default: 1)",
    )
    p.add_argument(
        "--types",
        nargs="+",
        choices=["bbox", "segm", "keypoints"],
        default=["bbox", "segm", "keypoints"],
        help="Eval types to run (default: all three)",
    )
    p.add_argument(
        "--phases",
        action="store_true",
        help="Split each result into load (ctor + loadRes) and eval "
        "(evaluate + accumulate + summarize), and add a bbox-only-GT variant "
        "row (segmentation fields stripped) when bbox is among the types",
    )
    return p.parse_args()


def main():
    args = parse_args()
    n_dets = BASE_DETS * args.scale

    scale_note = f" ({args.scale}× detections)" if args.scale > 1 else ""
    print(f"\nCOCO val2017{scale_note} — {n_dets:,} synthetic detections (seed=42)")
    print("=" * 68)
    if args.phases:
        print(f"  {'Eval Type':<22} {'Phase':<6} {'pycocotools':>12} {'faster-coco-eval':>17} {'hotcoco':>9}")
    else:
        print(f"  {'Eval Type':<12} {'pycocotools':>13} {'faster-coco-eval':>17} {'hotcoco':>9}")
    print("=" * 68)

    with tempfile.TemporaryDirectory() as tmpdir:
        # (row label, GT file, iou_type). The label and the iou_type coincide for
        # the three real types and diverge only for the stripped-GT variant below.
        rows = [(iou_type, files["gt"], iou_type) for iou_type, files in VAL2017.items() if iou_type in args.types]
        if args.phases and "bbox" in args.types:
            # Same detections, GT without its (unread) polygon masks.
            stripped = os.path.join(tmpdir, "gt_bbox_only.json")
            strip_segmentation(VAL2017["bbox"]["gt"], stripped)
            rows.append(("bbox (bbox-only GT)", stripped, "bbox"))

        for name, gt_path, iou_type in rows:
            # Detections are generated from the *official* GT so the
            # bbox-only-GT variant times the same workload on a lighter file.
            src_gt = VAL2017["bbox"]["gt"] if name.startswith("bbox") else gt_path
            dets = generate_detections(src_gt, iou_type, n_dets)
            dt_path = os.path.join(tmpdir, "dt.json")
            with open(dt_path, "w") as f:
                json.dump(dets, f)

            py = bench_pycocotools(gt_path, dt_path, iou_type)
            fc = bench_faster_coco_eval(gt_path, dt_path, iou_type)
            hc = bench_hotcoco(gt_path, dt_path, iou_type)

            if args.phases:
                for phase, i in (("load", 0), ("eval", 1)):
                    label = name if phase == "load" else ""
                    print(
                        f"  {label:<22} {phase:<6} {py[i]:>11.2f}s "
                        f"{fc[i]:>10.2f}s ({py[i] / fc[i]:.1f}×) "
                        f"{hc[i]:>6.2f}s ({py[i] / hc[i]:.1f}×)"
                    )
            else:
                py_t, fc_t, hc_t = sum(py), sum(fc), sum(hc)
                print(
                    f"  {name:<12} {py_t:>11.2f}s  {fc_t:>6.2f}s ({py_t / fc_t:.1f}×)  "
                    f"{hc_t:>6.2f}s ({py_t / hc_t:.1f}×)"
                )

    print("=" * 68)
    print("  Speedups are relative to pycocotools.")


if __name__ == "__main__":
    main()
