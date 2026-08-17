# LVIS & Open Images

COCO's protocol is not the only one hotcoco speaks. LVIS needs federated evaluation over its 1,200-category long tail, and Open Images has its own single-threshold AP with category hierarchies and group-of boxes. Both are built in — this page covers when each applies and how to run them.

## LVIS evaluation

[LVIS](https://www.lvisdataset.org/) ([Gupta et al., ECCV 2019](https://arxiv.org/abs/1908.03195)) is a large-vocabulary instance segmentation dataset with ~1,200 categories. It uses **federated annotation** — each image is only exhaustively labeled for a subset of categories. Running standard COCO eval on LVIS over-penalizes detectors by treating every unannotated category as a missed detection. hotcoco handles this correctly out of the box.

### Drop-in replacement for lvis-api

If your pipeline uses lvis-api (Detectron2, MMDetection, or any code that does `from lvis import LVISEval`), call `init_as_lvis()` once at startup:

```python
from hotcoco import init_as_lvis
init_as_lvis()

# Existing lvis-api code works unchanged
from lvis import LVIS, LVISEval, LVISResults

lvis_gt = LVIS("lvis_v1_val.json")
lvis_dt = LVISResults(lvis_gt, "detections.json")
ev = LVISEval(lvis_gt, lvis_dt, "bbox")
ev.run()
ev.print_results()
results = ev.get_results()
```

### Direct usage

If you're not using lvis-api, use `LVISeval` or pass `lvis_style=True` to `COCOeval`:

```python
from hotcoco import COCO, LVISeval

lvis_gt = COCO("lvis_v1_val.json")
lvis_dt = lvis_gt.load_res("detections.json")

ev = LVISeval(lvis_gt, lvis_dt, "segm")  # lvis_style=True is set automatically
ev.run()
results = ev.get_results()
# {"AP": 0.42, "APr": 0.38, "APc": 0.44, "APf": 0.45, "AR@300": ..., ...}
```

Or equivalently:

```python
from hotcoco import COCO, COCOeval

ev = COCOeval(lvis_gt, lvis_dt, "segm", lvis_style=True)
ev.evaluate()
ev.accumulate()
ev.summarize()
results = ev.get_results()
```

### The 13 LVIS metrics

| Metric | Description |
|--------|-------------|
| AP | mAP @ IoU[0.5:0.05:0.95] |
| AP50 | mAP @ IoU=0.5 |
| AP75 | mAP @ IoU=0.75 |
| APs | AP for small objects (area < 32²) |
| APm | AP for medium objects (32² ≤ area < 96²) |
| APl | AP for large objects (area ≥ 96²) |
| APr | AP for rare categories (1–10 training images) |
| APc | AP for common categories (11–100 training images) |
| APf | AP for frequent categories (100+ training images) |
| AR@300 | Mean recall @ max 300 detections per image |
| ARs@300 | AR for small objects |
| ARm@300 | AR for medium objects |
| ARl@300 | AR for large objects |

The frequency split (rare / common / frequent) is determined by the `frequency` field on each category in the LVIS annotation file (`"r"`, `"c"`, `"f"`). These correspond to the number of training images in which the category appears, as defined in the LVIS paper.

`get_results()` returns all 13 metrics as a dict for programmatic access.

## Open Images evaluation

[Open Images](https://storage.googleapis.com/openimages/web/index.html) uses a different evaluation protocol from COCO: a single AP@IoU=0.5, a category hierarchy for annotation expansion, and `is_group_of` annotations in place of `iscrowd`. hotcoco supports all three.

### Quick start

```python
from hotcoco import COCO, COCOeval

coco_gt = COCO("oid_annotations.json")
coco_dt = coco_gt.load_res("detections.json")

ev = COCOeval(coco_gt, coco_dt, "bbox", oid_style=True)
ev.run()

result = ev.get_results()
# {"AP": 0.573}
```

Starting from the CSV files Open Images actually ships, rather than from COCO
JSON, load them directly:

```python
coco_gt = COCO.from_oid(
    "challenge-2019-validation-detection-bbox.csv",
    class_descriptions="class-descriptions-boxable.csv",
)
coco_dt = coco_gt.load_res_oid("predictions.csv")
```

`IsGroupOf` carries through to the group-of matching described
[below](#group-of-annotations). The readers, their optional arguments, and what
happens without image dimensions are documented under
[format conversion](datasets.md#open-images).

`oid_style=True` sets:

- IoU threshold = 0.5 (single threshold, no sweep)
- Area range = "all" only
- Max detections = 100

### Category hierarchy

Open Images categories form a hierarchy — a "Dog" detection also counts as an "Animal" detection if Animal is an ancestor of Dog. Pass a `Hierarchy` to expand GT annotations automatically at evaluation time.

```python
from hotcoco import COCO, COCOeval, Hierarchy

# From the OID hierarchy JSON (bbox_labels_600_hierarchy.json)
label_to_id = {cat["name"]: cat["id"] for cat in coco_gt.dataset["categories"]}
h = Hierarchy.from_file("bbox_labels_600_hierarchy.json", label_to_id=label_to_id)

ev = COCOeval(coco_gt, coco_dt, "bbox", oid_style=True, hierarchy=h)
ev.run()
```

If you don't have a hierarchy JSON, omit `hierarchy=` entirely. `oid_style=True` then
derives one from the `supercategory` field of each category:

```python
# Parent→child relationships come from Category.supercategory
ev = COCOeval(coco_gt, coco_dt, "bbox", oid_style=True)
ev.run()
```

Categories without a `supercategory` produce a flat hierarchy, which makes expansion a
no-op — matching still uses OID semantics (group-of handling, a single IoU threshold).

Or build a hierarchy manually from a parent map:

```python
h = Hierarchy.from_parent_map({
    3: 1,   # cat 3's parent is cat 1
    4: 1,   # cat 4's parent is cat 1
    5: 2,   # cat 5's parent is cat 2
})
```

### Detection expansion

By default only GT annotations are expanded up the hierarchy. To also expand detections (so a "Dog" detection also counts as an "Animal" detection):

```python
ev = COCOeval(coco_gt, coco_dt, "bbox", oid_style=True, hierarchy=h)
ev.params.expand_dt = True
ev.run()
```

### Group-of annotations

OID uses `is_group_of: true` on annotations covering a *cluster* of objects — five or more instances of the same class, occluding each other, where no individual box can be drawn. A cluster is one thing you either found or didn't:

- **A group-of box is worth exactly one ground truth.** The best-scoring detection inside it is a true positive. Every other detection inside it is ignored — neither true positive nor false positive. Detecting the pile twice earns nothing extra.
- **Missing it costs one false negative.** An undetected group-of box counts once against recall.
- **"Inside" is IoA, not IoU** — intersection divided by the *detection's* area. A detection wholly inside the box qualifies however small it is, which is the point: individual objects are much smaller than the cluster that contains them.

This is the [Open Images Challenge protocol](https://storage.googleapis.com/openimages/web/evaluation.html), equivalently TensorFlow's `group_of_weight = 1.0`, and it is what FiftyOne implements. It is checked against the TensorFlow Object Detection API on every commit — see [verification](#verification) below.

Open Images AP also uses **VOC 2010 all-points integration** — the exact area under the precision-recall curve — rather than COCO's 101-point recall grid. The protocol specifies it and both reference implementations do it, so an OID number here is not directly comparable to a COCO number computed on the same data.

!!! note "Not the same as the Open Images V2 metric"

    The older V2 detection metric ignored group-of boxes entirely — they contributed to neither the numerator nor the denominator (`group_of_weight = 0.0`). Both are real published protocols. hotcoco implements the Challenge metric, so **numbers here will not match a V2-era leaderboard**.

Mechanically this is COCO's `iscrowd` with the scoring changed: same intersection-over-area measure, same "many detections may fall inside one region", but where a crowd region is dropped from the denominator, a group-of box is counted once and can be found.

Your annotations need `"is_group_of": true` in the JSON for this to take effect. Standard annotations without this field default to `false`.

### The OID metric

`summarize()` reports a single metric:

| Metric | IoU | Area | MaxDets |
|--------|-----|------|---------|
| **AP** | 0.50 | all | 100 |

`get_results()` returns `{"AP": <float>}`.

### Verification

Open Images evaluation is compared against the [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) — the reference the official protocol page points to — over 70 cases covering group-of absorption, IoA containment at and around the 0.5 boundary, undetected group-of boxes, overlapping group-of boxes, and randomized multi-class scenes. Both mAP and per-class AP are compared, and **every case agrees to within one ulp** (worst difference 1.11e-16). It runs in CI on every commit.

Two things that comparison does **not** cover, and why `provenance` still reports `"extension"`:

- **Non-exhaustive image-level labels.** The challenge ignores detections of a class not verified on an image, and counts detections of a negatively-labeled class as false positives. hotcoco does not implement this — it needs per-image label data that COCO-format JSON cannot carry. A real challenge submission would score differently.
- **Hierarchy expansion** is applied to annotations before evaluation rather than inside it, so it sits outside the compared surface.

See [Hierarchy](../api/hierarchy.md) in the API reference for full construction and query methods.
