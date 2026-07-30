"""Fast parity regression tests for CI.

Hand-crafted edge-case tests (bbox/segm/keypoints) and OID evaluation tests.
All tests run in under 30 seconds and are safe to run on every commit.

To hunt for new parity bugs, use the hypothesis fuzzer instead:
    just fuzz

Usage:
    uv run pytest scripts/test_parity.py -v -x --tb=short
    just test
"""

import contextlib
import json
import os
import sys
import tempfile
import warnings

import pytest
from helpers import COCO_KEYPOINT_NAMES, COCO_SKELETON, suppress_stdout
from hotcoco import COCO, COCOeval, Hierarchy
from hotcoco import COCO as RsCOCO
from hotcoco import COCOeval as RsCOCOeval
from pycocotools.coco import COCO as PyCOCO
from pycocotools.cocoeval import COCOeval as PyCOCOeval

TOLERANCE = 1e-10

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _written(gt_dataset, dt_results):
    """Write a GT/DT pair to temp files and yield their paths."""
    gt_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    dt_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    try:
        json.dump(gt_dataset, gt_file)
        gt_file.close()
        json.dump(dt_results, dt_file)
        dt_file.close()
        with suppress_stdout():
            yield gt_file.name, dt_file.name
    finally:
        os.unlink(gt_file.name)
        os.unlink(dt_file.name)


def run_both(gt_dataset, dt_results, iou_type):
    """Run evaluation through both pycocotools and hotcoco, return stats."""
    with _written(gt_dataset, dt_results) as (gt_path, dt_path):
        py_gt = PyCOCO(gt_path)
        py_dt = py_gt.loadRes(dt_path)
        py_ev = PyCOCOeval(py_gt, py_dt, iou_type)
        py_ev.evaluate()
        py_ev.accumulate()
        py_ev.summarize()
        py_stats = py_ev.stats.tolist()

        rs_gt = RsCOCO(gt_path)
        rs_dt = rs_gt.load_res(dt_path)
        rs_ev = RsCOCOeval(rs_gt, rs_dt, iou_type)
        rs_ev.evaluate()
        rs_ev.accumulate()
        rs_ev.summarize()
        rs_stats = rs_ev.stats

    return py_stats, rs_stats


def _metric_names_for(iou_type):
    """Get canonical metric names from the Rust evaluator for a given iou_type."""
    empty = RsCOCOeval(COCO(), COCO(), iou_type)
    return empty.metric_keys()


def assert_metrics_match(py_stats, rs_stats, iou_type, min_numeric=0):
    """Assert all metrics match within tolerance.

    Returns the number of metrics compared numerically — i.e. excluding those
    where both sides are the -1.0 "not computed for this configuration" sentinel.
    Agreeing that a metric is undefined is a real assertion (a hotcoco value of
    0.5 against pycocotools' -1.0 fails here), but it exercises no arithmetic, so
    a test whose every metric is a sentinel has verified very little. Pass
    ``min_numeric`` to require that a test actually reached the numeric paths.
    """
    metric_names = _metric_names_for(iou_type)
    expected_len = len(metric_names)
    assert len(py_stats) == expected_len, f"pycocotools returned {len(py_stats)}, expected {expected_len}"
    assert len(rs_stats) == expected_len, f"hotcoco returned {len(rs_stats)}, expected {expected_len}"

    mismatches = []
    numeric = 0
    for i in range(expected_len):
        py_val, rs_val = py_stats[i], rs_stats[i]
        if py_val == -1.0 and rs_val == -1.0:
            continue
        numeric += 1
        diff = abs(py_val - rs_val)
        if diff > TOLERANCE:
            mismatches.append(f"  [{i}] {metric_names[i]}: py={py_val:.15f} rs={rs_val:.15f} diff={diff:.2e}")

    if mismatches:
        raise AssertionError(f"\n{iou_type} metric mismatch (tol={TOLERANCE}):\n" + "\n".join(mismatches))

    assert numeric >= min_numeric, (
        f"{iou_type}: only {numeric} of {expected_len} metrics were compared numerically "
        f"(the rest were -1.0 on both sides); this test asked for at least {min_numeric}"
    )
    return numeric


def _make_minimal_gt(iou_type, images=None, categories=None, annotations=None):
    """Build a minimal GT dataset dict."""
    if images is None:
        images = [{"id": 1, "width": 640, "height": 480, "file_name": "test.jpg"}]
    if categories is None:
        if iou_type == "keypoints":
            categories = [
                {
                    "id": 1,
                    "name": "person",
                    "supercategory": "person",
                    "keypoints": COCO_KEYPOINT_NAMES,
                    "skeleton": COCO_SKELETON,
                }
            ]
        else:
            categories = [{"id": 1, "name": "cat", "supercategory": "none"}]
    if annotations is None:
        annotations = []
    return {"images": images, "annotations": annotations, "categories": categories}


def _make_bbox_ann(ann_id, img_id=1, cat_id=1, bbox=None, iscrowd=0, **extra):
    """Shortcut to create a bbox GT annotation."""
    if bbox is None:
        bbox = [10.0, 10.0, 100.0, 100.0]
    ann = {
        "id": ann_id,
        "image_id": img_id,
        "category_id": cat_id,
        "bbox": bbox,
        "area": bbox[2] * bbox[3],
        "iscrowd": iscrowd,
    }
    ann.update(extra)
    return ann


def _make_bbox_det(img_id=1, cat_id=1, bbox=None, score=0.9):
    """Shortcut to create a bbox detection."""
    if bbox is None:
        bbox = [10.0, 10.0, 100.0, 100.0]
    return {"image_id": img_id, "category_id": cat_id, "bbox": bbox, "score": score}


# ---------------------------------------------------------------------------
# OID helpers
# ---------------------------------------------------------------------------


def _make_coco(dataset: dict) -> COCO:
    """Write dataset to a temp file and load as COCO."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(dataset, f)
        path = f.name
    try:
        return COCO(path)
    finally:
        os.unlink(path)


def _img(id: int = 1) -> dict:
    return {"id": id, "file_name": f"img{id}.jpg", "height": 640, "width": 640}


def _ann(
    id: int,
    image_id: int,
    category_id: int,
    bbox: list,
    area: float,
    score: float | None = None,
    is_group_of: bool | None = None,
) -> dict:
    ann = {"id": id, "image_id": image_id, "category_id": category_id, "bbox": bbox, "area": area, "iscrowd": 0}
    if score is not None:
        ann["score"] = score
    if is_group_of is not None:
        ann["is_group_of"] = is_group_of
    return ann


def _cat(id: int, name: str, supercategory: str | None = None) -> dict:
    cat = {"id": id, "name": name}
    if supercategory is not None:
        cat["supercategory"] = supercategory
    return cat


# ---------------------------------------------------------------------------
# Edge-case tests: bbox / segm / keypoints
# ---------------------------------------------------------------------------


def test_empty_gt():
    """No GT annotations, some detections → all metrics -1.0.

    Every metric here is the sentinel on both sides, so this asserts agreement
    about undefinedness and nothing numeric. That is the whole point of the case,
    but the expectation has to be pinned on *hotcoco* — the original checked
    ``py_stats``, which tests pycocotools against itself.
    """
    for iou_type in ["bbox", "segm"]:
        gt = _make_minimal_gt(iou_type)
        dts = [_make_bbox_det(score=0.5)]
        py_stats, rs_stats = run_both(gt, dts, iou_type)
        assert_metrics_match(py_stats, rs_stats, iou_type)
        assert all(s == -1.0 for s in rs_stats[:6]), f"hotcoco: expected -1.0 AP metrics, got {rs_stats[:6]}"
        assert all(s == -1.0 for s in py_stats[:6]), f"pycocotools: expected -1.0 AP metrics, got {py_stats[:6]}"


def test_all_crowd():
    """Every GT is iscrowd=1."""
    anns = [
        _make_bbox_ann(1, bbox=[10, 10, 100, 100], iscrowd=1),
        _make_bbox_ann(2, bbox=[200, 200, 50, 50], iscrowd=1),
    ]
    gt = _make_minimal_gt("bbox", annotations=anns)
    dts = [_make_bbox_det(bbox=[10, 10, 100, 100], score=0.9), _make_bbox_det(bbox=[200, 200, 50, 50], score=0.5)]
    py_stats, rs_stats = run_both(gt, dts, "bbox")
    assert_metrics_match(py_stats, rs_stats, "bbox")
    # Crowd GTs are ignored *and* rematchable: the detections they absorb are
    # neither TPs nor FPs, and no non-crowd GT remains to measure recall against.
    # Every metric is therefore undefined rather than zero — pin that on hotcoco,
    # since agreement alone would also hold if both sides were wrong the same way.
    assert all(s == -1.0 for s in rs_stats), f"hotcoco: all-crowd GT should give all -1.0, got {rs_stats}"


def test_identical_boxes():
    """All GTs and DTs have the exact same bbox."""
    bbox = [50.0, 50.0, 200.0, 150.0]
    anns = [_make_bbox_ann(i + 1, bbox=bbox) for i in range(3)]
    gt = _make_minimal_gt("bbox", annotations=anns)
    dts = [_make_bbox_det(bbox=bbox, score=round(0.3 + i * 0.3, 1)) for i in range(3)]
    py_stats, rs_stats = run_both(gt, dts, "bbox")
    assert_metrics_match(py_stats, rs_stats, "bbox")


def test_area_at_boundaries():
    """Annotations with area exactly at 1024 and 9216 (small/medium/large boundaries)."""
    anns = [
        _make_bbox_ann(1, bbox=[10.0, 10.0, 32.0, 32.0]),  # area = 1024
        _make_bbox_ann(2, bbox=[100.0, 100.0, 96.0, 96.0]),  # area = 9216
    ]
    gt = _make_minimal_gt("bbox", annotations=anns)
    dts = [
        _make_bbox_det(bbox=[10.0, 10.0, 32.0, 32.0], score=0.8),
        _make_bbox_det(bbox=[100.0, 100.0, 96.0, 96.0], score=0.7),
    ]
    py_stats, rs_stats = run_both(gt, dts, "bbox")
    assert_metrics_match(py_stats, rs_stats, "bbox")


def test_single_image_single_cat():
    """Minimal dataset: one image, one category, one GT, one DT."""
    anns = [_make_bbox_ann(1, bbox=[100.0, 100.0, 50.0, 50.0])]
    gt = _make_minimal_gt("bbox", annotations=anns)
    dts = [_make_bbox_det(bbox=[100.0, 100.0, 50.0, 50.0], score=1.0)]
    py_stats, rs_stats = run_both(gt, dts, "bbox")
    assert_metrics_match(py_stats, rs_stats, "bbox")


def test_zero_area_boxes():
    """Boxes with zero width or height."""
    anns = [
        _make_bbox_ann(1, bbox=[10.0, 10.0, 0.0, 50.0]),  # zero width
        _make_bbox_ann(2, bbox=[50.0, 50.0, 50.0, 0.0]),  # zero height
        _make_bbox_ann(3, bbox=[100.0, 100.0, 80.0, 80.0]),  # normal
    ]
    gt = _make_minimal_gt("bbox", annotations=anns)
    dts = [
        _make_bbox_det(bbox=[10.0, 10.0, 0.0, 50.0], score=0.9),
        _make_bbox_det(bbox=[50.0, 50.0, 50.0, 0.0], score=0.8),
        _make_bbox_det(bbox=[100.0, 100.0, 80.0, 80.0], score=0.7),
    ]
    py_stats, rs_stats = run_both(gt, dts, "bbox")
    assert_metrics_match(py_stats, rs_stats, "bbox")


def test_many_detections_few_gt():
    """Many detections for a single GT — tests maxDet handling."""
    anns = [_make_bbox_ann(1, bbox=[100.0, 100.0, 50.0, 50.0])]
    gt = _make_minimal_gt("bbox", annotations=anns)
    dts = [_make_bbox_det(bbox=[100.0 + i * 2, 100.0, 50.0, 50.0], score=round(1.0 - i * 0.005, 4)) for i in range(200)]
    py_stats, rs_stats = run_both(gt, dts, "bbox")
    assert_metrics_match(py_stats, rs_stats, "bbox")


def test_kpt_no_visible():
    """Keypoint GT with num_keypoints=0 should be ignored."""
    kpts_zero = [0, 0, 0] * 17
    kpts_some = []
    for i in range(17):
        kpts_some.extend([float(100 + i * 10), float(100 + i * 5), 2])

    anns = [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [50.0, 50.0, 200.0, 200.0],
            "area": 40000.0,
            "iscrowd": 0,
            "keypoints": kpts_zero,
            "num_keypoints": 0,
        },
        {
            "id": 2,
            "image_id": 1,
            "category_id": 1,
            "bbox": [50.0, 50.0, 200.0, 200.0],
            "area": 40000.0,
            "iscrowd": 0,
            "keypoints": kpts_some,
            "num_keypoints": 17,
        },
    ]
    gt = _make_minimal_gt("keypoints", annotations=anns)
    dts = [{"image_id": 1, "category_id": 1, "bbox": [50.0, 50.0, 200.0, 200.0], "score": 0.9, "keypoints": kpts_some}]
    py_stats, rs_stats = run_both(gt, dts, "keypoints")
    assert_metrics_match(py_stats, rs_stats, "keypoints")


def test_segm_polygon_rasterization():
    """Segmentation with polygon GTs and bbox DTs — tests the full segm pipeline."""
    anns = [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [10.0, 10.0, 100.0, 100.0],
            "area": 10000.0,
            "iscrowd": 0,
            "segmentation": [[10, 10, 110, 10, 110, 110, 10, 110]],
        },
        {
            "id": 2,
            "image_id": 1,
            "category_id": 1,
            "bbox": [200.0, 200.0, 50.0, 50.0],
            "area": 2500.0,
            "iscrowd": 0,
            "segmentation": [[200, 200, 250, 200, 250, 250, 200, 250]],
        },
    ]
    gt = _make_minimal_gt("segm", annotations=anns)
    dts = [
        _make_bbox_det(bbox=[10.0, 10.0, 100.0, 100.0], score=0.95),
        _make_bbox_det(bbox=[200.0, 200.0, 50.0, 50.0], score=0.8),
    ]
    py_stats, rs_stats = run_both(gt, dts, "segm")
    assert_metrics_match(py_stats, rs_stats, "segm")


# ---------------------------------------------------------------------------
# OID evaluation tests
# ---------------------------------------------------------------------------


def test_basic_hierarchy():
    """Dog detection on Poodle GT: correct at Dog level, wrong at Poodle."""
    hierarchy = Hierarchy.from_parent_map({1: 2, 2: 3})

    gt = _make_coco(
        {
            "images": [_img()],
            "annotations": [_ann(1, 1, 1, [10, 10, 100, 100], 10000)],  # Poodle
            "categories": [_cat(1, "poodle", "dog"), _cat(2, "dog", "animal"), _cat(3, "animal")],
        }
    )
    dt = _make_coco(
        {
            "images": [_img()],
            "annotations": [_ann(1, 1, 2, [10, 10, 100, 100], 10000, score=0.9)],  # Dog
            "categories": [_cat(1, "poodle", "dog"), _cat(2, "dog", "animal"), _cat(3, "animal")],
        }
    )

    ev = COCOeval(gt, dt, "bbox", oid_style=True, hierarchy=hierarchy)
    ev.run()
    results = ev.results(per_class=True)

    per_class = results["per_class"]
    assert "dog" in per_class, f"Expected 'dog' in per_class, got {list(per_class.keys())}"
    assert abs(per_class["dog"] - 1.0) < 1e-6, f"Dog AP should be 1.0, got {per_class['dog']}"


def test_group_of_scores_one_tp_and_absorbs_the_rest():
    """A group-of box is worth one TP; surplus detections inside it are ignored.

    Open Images protocol: "If at least one detection is inside group-of box a
    single True Positive is scored. ... Multiple correct detections inside the
    same group-of box is still count as a single True Positive."

    The old version asserted "all should be TPs" and used detections at IoU 0.16
    against the group box, so the group-of path never executed -- it passed on VOC
    interpolation. Both halves are fixed here.
    """
    gt = _make_coco(
        {
            "images": [_img()],
            "annotations": [
                _ann(1, 1, 1, [300, 300, 100, 100], 10000),  # Normal GT
                _ann(2, 1, 1, [0, 0, 200, 200], 40000, is_group_of=True),  # Group-of
            ],
            "categories": [_cat(1, "person")],
        }
    )
    dt = _make_coco(
        {
            "images": [_img()],
            "annotations": [
                _ann(1, 1, 1, [300, 300, 100, 100], 10000, score=0.9),  # ordinary TP
                _ann(2, 1, 1, [10, 10, 80, 80], 6400, score=0.8),  # scores the group box
                _ann(3, 1, 1, [50, 50, 80, 80], 6400, score=0.7),  # absorbed, ignored
            ],
            "categories": [_cat(1, "person")],
        }
    )

    ev = COCOeval(gt, dt, "bbox", oid_style=True)
    ev.run()
    assert ev.stats[0] > 0.99, f"both GTs found, surplus ignored -> AP ~1.0, got {ev.stats[0]:.4f}"

    e = ev.eval_imgs[0]
    ignored = [i for i, ig in zip(e["dtIds"], e["dtIgnore"][0]) if ig]
    assert ignored == [3], f"only the surplus detection should be ignored, got {ignored}"
    assert sum(e["gtInDenominator"]) == 2, "ordinary GT + group-of box = 2 in the denominator"


def test_group_of_matches_on_ioa_not_iou():
    """A detection smaller than the group-of box must still be absorbed.

    Group-of matching uses IoA (intersection / detection area), so a detection
    wholly inside the box scores 1.0 however small it is. Under plain IoU this
    80x80 detection scores 0.16 against the 200x200 box and leaks out as a false
    positive -- the defect this guards.

    The inside-the-box detection deliberately outranks the ordinary true positive
    so a regression lands its false positive *before* full recall. Reverse the
    scores and AP reads 1.0 either way, proving nothing.
    """
    gt = _make_coco(
        {
            "images": [_img()],
            "annotations": [
                _ann(1, 1, 1, [300, 300, 100, 100], 10000),
                _ann(2, 1, 1, [0, 0, 200, 200], 40000, is_group_of=True),
            ],
            "categories": [_cat(1, "person")],
        }
    )
    dt = _make_coco(
        {
            "images": [_img()],
            "annotations": [
                _ann(1, 1, 1, [10, 10, 80, 80], 6400, score=0.9),
                _ann(2, 1, 1, [300, 300, 100, 100], 10000, score=0.8),
            ],
            "categories": [_cat(1, "person")],
        }
    )

    ev = COCOeval(gt, dt, "bbox", oid_style=True)
    ev.run()
    # Matched on IoU instead, AP collapses to ~0.25.
    assert ev.stats[0] > 0.99, f"detection inside a group-of box must be absorbed, got {ev.stats[0]:.4f}"


def test_undetected_group_of_is_a_miss():
    """An undetected group-of box is a single false negative.

    Protocol: "Otherwise, the group-of box is counted as a single False Negative."
    This previously asserted the opposite, which is the Open Images *V2* metric
    (TF `group_of_weight=0.0`), not the Challenge metric hotcoco targets.
    """
    gt = _make_coco(
        {
            "images": [_img()],
            "annotations": [
                _ann(1, 1, 1, [0, 0, 100, 100], 10000),  # Normal GT
                _ann(2, 1, 1, [400, 400, 100, 100], 10000, is_group_of=True),  # Group-of (far away)
            ],
            "categories": [_cat(1, "person")],
        }
    )
    dt = _make_coco(
        {
            "images": [_img()],
            "annotations": [
                _ann(1, 1, 1, [0, 0, 100, 100], 10000, score=0.9)  # Matches normal GT only
            ],
            "categories": [_cat(1, "person")],
        }
    )

    ev = COCOeval(gt, dt, "bbox", oid_style=True)
    ev.run()
    assert abs(ev.stats[0] - 0.5) < 0.02, f"one of two GTs found -> AP ~0.5, got {ev.stats[0]:.4f}"


def test_pre_expanded_idempotent():
    """Pre-expanded GTs should produce same results as unexpanded."""
    hierarchy = Hierarchy.from_parent_map({1: 2})

    gt_unexpanded = _make_coco(
        {
            "images": [_img()],
            "annotations": [_ann(1, 1, 1, [10, 10, 100, 100], 10000)],
            "categories": [_cat(1, "dog", "animal"), _cat(2, "animal")],
        }
    )
    gt_expanded = _make_coco(
        {
            "images": [_img()],
            "annotations": [
                _ann(1, 1, 1, [10, 10, 100, 100], 10000),
                _ann(2, 1, 2, [10, 10, 100, 100], 10000),  # Expanded animal
            ],
            "categories": [_cat(1, "dog", "animal"), _cat(2, "animal")],
        }
    )
    dt = _make_coco(
        {
            "images": [_img()],
            "annotations": [_ann(1, 1, 1, [10, 10, 100, 100], 10000, score=0.9)],
            "categories": [_cat(1, "dog", "animal"), _cat(2, "animal")],
        }
    )

    ev1 = COCOeval(gt_unexpanded, dt, "bbox", oid_style=True, hierarchy=hierarchy)
    ev1.run()

    dt2 = _make_coco(
        {
            "images": [_img()],
            "annotations": [_ann(1, 1, 1, [10, 10, 100, 100], 10000, score=0.9)],
            "categories": [_cat(1, "dog", "animal"), _cat(2, "animal")],
        }
    )
    ev2 = COCOeval(gt_expanded, dt2, "bbox", oid_style=True, hierarchy=hierarchy)
    ev2.run()

    assert abs(ev1.stats[0] - ev2.stats[0]) < 1e-10, (
        f"Pre-expanded should match unexpanded: {ev1.stats[0]:.6f} vs {ev2.stats[0]:.6f}"
    )


def test_dt_expansion():
    """expand_dt=True: Dog prediction gets credit at Animal level."""
    hierarchy = Hierarchy.from_parent_map({1: 2})

    gt = _make_coco(
        {
            "images": [_img()],
            "annotations": [_ann(1, 1, 2, [10, 10, 100, 100], 10000)],  # Animal GT
            "categories": [_cat(1, "dog", "animal"), _cat(2, "animal")],
        }
    )
    dt = _make_coco(
        {
            "images": [_img()],
            "annotations": [_ann(1, 1, 1, [10, 10, 100, 100], 10000, score=0.9)],  # Dog
            "categories": [_cat(1, "dog", "animal"), _cat(2, "animal")],
        }
    )

    ev1 = COCOeval(gt, dt, "bbox", oid_style=True, hierarchy=hierarchy)
    ev1.run()
    ap_no_expand = ev1.stats[0]

    gt2 = _make_coco(
        {
            "images": [_img()],
            "annotations": [_ann(1, 1, 2, [10, 10, 100, 100], 10000)],
            "categories": [_cat(1, "dog", "animal"), _cat(2, "animal")],
        }
    )
    dt2 = _make_coco(
        {
            "images": [_img()],
            "annotations": [_ann(1, 1, 1, [10, 10, 100, 100], 10000, score=0.9)],
            "categories": [_cat(1, "dog", "animal"), _cat(2, "animal")],
        }
    )

    ev2 = COCOeval(gt2, dt2, "bbox", oid_style=True, hierarchy=hierarchy)
    p = ev2.params
    p.expand_dt = True
    ev2.params = p
    ev2.run()
    ap_expand = ev2.stats[0]

    assert ap_expand > ap_no_expand, f"DT expansion should improve AP: {ap_expand:.4f} vs {ap_no_expand:.4f}"


def test_virtual_nodes():
    """Supercategory not in categories list: should still work via virtual node."""
    gt = _make_coco(
        {
            "images": [_img()],
            "annotations": [_ann(1, 1, 1, [10, 10, 100, 100], 10000)],
            "categories": [_cat(1, "chair", "furniture")],  # "furniture" → virtual node
        }
    )
    dt = _make_coco(
        {
            "images": [_img()],
            "annotations": [_ann(1, 1, 1, [10, 10, 100, 100], 10000, score=0.9)],
            "categories": [_cat(1, "chair", "furniture")],
        }
    )

    ev = COCOeval(gt, dt, "bbox", oid_style=True)
    ev.run()
    # chair matches (AP=1.0), virtual_furniture has GT but no DT (AP=0.0) → mean=0.5
    assert abs(ev.stats[0] - 0.5) < 1e-6, f"AP should be 0.5 (chair + virtual), got {ev.stats[0]:.4f}"
    results = ev.results(per_class=True)
    assert results["per_class"]["chair"] > 0.99, (
        f"Chair per-class AP should be 1.0, got {results['per_class']['chair']:.4f}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# EvalReport
# ---------------------------------------------------------------------------


def _eval_for_report(iou_type="bbox"):
    gt = _make_minimal_gt(
        iou_type, annotations=[_make_bbox_ann(1, bbox=[10, 10, 50, 50]), _make_bbox_ann(2, bbox=[100, 100, 40, 40])]
    )
    dt = [_make_bbox_det(bbox=[10, 10, 50, 50], score=0.9), _make_bbox_det(bbox=[100, 100, 40, 40], score=0.8)]
    coco_gt = _make_coco(gt)
    coco_dt = coco_gt.load_res(dt)
    ev = RsCOCOeval(coco_gt, coco_dt, iou_type)
    with suppress_stdout():
        ev.run()
    return ev


def test_report_dict_shape():
    report = _eval_for_report().report()

    assert set(report) == {"task", "provenance", "metrics", "per_class", "per_group", "curves", "params"}
    assert report["task"] == "detection"
    assert report["metrics"]["AP"] == pytest.approx(1.0)
    # Nested per-class, not flattened "AP/name".
    assert all(isinstance(v, dict) for v in report["per_class"].values())


def test_report_provenance_distinguishes_extension_from_parity():
    """OBB has no reference implementation, so it must not read as leaderboard-comparable."""
    assert _eval_for_report("bbox").report()["provenance"] == "parity_verified"
    assert _eval_for_report("obb").report()["provenance"] == "extension"


def test_provenance_survives_into_the_render_layer():
    """The marker has to reach whatever draws the numbers, not stop at the dict.

    A PDF or dashboard is the artifact that gets circulated to someone who never
    ran the eval, so an unmarked one is the failure `Provenance` exists to
    prevent. `report()` computed it correctly for a whole release while every
    Python renderer ignored it.
    """
    from hotcoco.plot.data import PlotData

    verified = PlotData.from_coco_eval(_eval_for_report())
    assert verified.provenance == "parity_verified"
    assert verified.is_benchmark_standard
    assert verified.deviations == []

    # The sharp case. Still `iou_type="bbox"` in plain COCO mode, so a renderer
    # deriving the marker from eval_mode or geometry — the obvious shortcut —
    # would call this leaderboard-comparable.
    ev = _eval_for_report()
    ev.params.recThrs = [i / 10 for i in range(11)]
    with suppress_stdout(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ev.run()

    extension = PlotData.from_coco_eval(ev)
    assert extension.eval_mode == "coco" and extension.iou_type == "bbox"
    assert extension.provenance == "extension"
    assert not extension.is_benchmark_standard
    assert any("rec_thrs" in d for d in extension.deviations), extension.deviations


def test_provenance_accessor_sees_unevaluated_param_changes():
    """`ev.params.x = ...` mutates a Python-side copy that only syncs on entry.

    Reading the evaluator directly reported `parity_verified` for a run that was
    about to be an extension — comparability answered from stale state, which is
    worse than not answering. Checked here because the accessor's whole point is
    working *before* a long evaluation.
    """
    ev = _eval_for_report()
    assert ev.provenance() == "parity_verified"  # control

    off = _eval_for_report()
    off.params.iouThrs = [0.5]
    assert off.provenance() == "extension"
    assert any("iou_thrs" in d for d in off.reference_deviations())


def test_report_curves_are_plottable():
    report = _eval_for_report().report()
    curves = report["curves"]

    rec_thrs = curves["rec_thrs"]
    assert len(rec_thrs) == 101
    # One curve per IoU threshold, each sharing the rec_thrs x-axis.
    pr_curves = {k: v for k, v in curves.items() if k.startswith("pr@")}
    assert len(pr_curves) == 10
    for name, curve in pr_curves.items():
        assert len(curve) == len(rec_thrs), name


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-x", "--tb=short"]))


# ---------------------------------------------------------------------------
# Drop-in behaviours that are not about metric values
# ---------------------------------------------------------------------------


def test_params_in_place_mutation_takes_effect():
    """`ev.params.imgIds = [...]` must configure the run, as it does in pycocotools.

    This is the canonical idiom — it appears in pycocotools' own demo — and it
    used to be a silent no-op here: the `params` getter cloned into a fresh object
    each access, so the assignment mutated a temporary and evaluation proceeded
    over the whole dataset.

    Asserted through a *result*, not just the attribute, because reading the
    attribute back would also pass if params were a persistent object the
    evaluator never consulted.
    """
    anns = [_make_bbox_ann(1, img_id=1, bbox=[10, 10, 50, 50]), _make_bbox_ann(2, img_id=2, bbox=[10, 10, 50, 50])]
    images = [
        {"id": 1, "width": 640, "height": 480, "file_name": "a.jpg"},
        {"id": 2, "width": 640, "height": 480, "file_name": "b.jpg"},
    ]
    gt = _make_minimal_gt("bbox", images=images, annotations=anns)
    dts = [
        _make_bbox_det(img_id=1, bbox=[10, 10, 50, 50], score=0.9),
        _make_bbox_det(img_id=2, bbox=[500, 400, 20, 20], score=0.9),
    ]

    with _written(gt, dts) as (gt_path, dt_path):
        coco_gt = COCO(gt_path)
        coco_dt = coco_gt.load_res(dt_path)

        both = COCOeval(coco_gt, coco_dt, "bbox")
        both.evaluate()
        both.accumulate()
        both.summarize()

        only_good = COCOeval(coco_gt, coco_dt, "bbox")
        only_good.params.imgIds = [1]
        assert list(only_good.params.imgIds) == [1], "params did not retain the assignment"
        only_good.evaluate()
        only_good.accumulate()
        only_good.summarize()

    # Image 1 is a perfect detection, image 2 is a total miss. Restricting to
    # image 1 must therefore score strictly higher.
    assert only_good.stats[0] > both.stats[0], (
        f"restricting to imgIds=[1] changed nothing: {only_good.stats[0]} vs {both.stats[0]} "
        "- params mutation is not reaching the evaluator"
    )


def test_non_reference_params_warn_and_downgrade_provenance():
    """Off-reference configuration is visible from Python, both ways.

    `summarize()` writes its warnings with `eprintln!`, straight to file
    descriptor 2 — which bypasses `sys.stderr`, so they are invisible in a
    notebook, invisible to `capsys`, and uncatchable by `warnings.catch_warnings`.
    They are re-raised as real Python warnings for that reason.
    """
    gt = _make_minimal_gt("bbox", annotations=[_make_bbox_ann(1, bbox=[10, 10, 50, 50])])
    dts = [_make_bbox_det(bbox=[10, 10, 50, 50], score=0.9)]

    with _written(gt, dts) as (gt_path, dt_path):
        coco_gt = COCO(gt_path)
        coco_dt = coco_gt.load_res(dt_path)

        ref = COCOeval(coco_gt, coco_dt, "bbox")
        ref.evaluate()
        ref.accumulate()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ref.summarize()
        assert not caught, f"default params should not warn, got {[str(w.message) for w in caught]}"
        assert ref.results()["provenance"] == "parity_verified"

        off = COCOeval(coco_gt, coco_dt, "bbox")
        off.params.recThrs = [i / 10 for i in range(11)]
        off.evaluate()
        off.accumulate()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            off.summarize()

    messages = [str(w.message) for w in caught]
    assert any("rec_thrs" in m for m in messages), f"expected a rec_thrs warning catchable from Python, got {messages}"
    # Provenance has to survive into the archived artifact, not just the report.
    assert off.results()["provenance"] == "extension"
