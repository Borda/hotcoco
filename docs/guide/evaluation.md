# Evaluation

hotcoco supports four evaluation types: bounding box, segmentation, keypoints, and oriented bounding box (OBB). All four follow the same workflow.

This page covers the core pipeline. [LVIS and Open Images](lvis-open-images.md) have their own protocols, and the analysis tools that explain *why* a model scores what it does — confusion matrices, TIDE, calibration, per-image diagnostics — live in [Model Diagnostics](diagnostics.md).

## The three-step pipeline

Every COCO evaluation follows the same pattern:

=== "Python"

    ```python
    from hotcoco import COCO, COCOeval

    coco_gt = COCO("annotations.json")
    coco_dt = coco_gt.load_res("detections.json")

    ev = COCOeval(coco_gt, coco_dt, iou_type)
    ev.evaluate()    # Per-image matching
    ev.accumulate()  # Aggregate into precision/recall curves
    ev.summarize()   # Print and compute the 12 summary metrics
    ```

=== "Rust"

    ```rust
    use hotcoco::{COCO, COCOeval};
    use hotcoco::params::IouType;
    use std::path::Path;

    let coco_gt = COCO::new(Path::new("annotations.json"))?;
    let coco_dt = coco_gt.load_res(Path::new("detections.json"))?;

    let mut ev = COCOeval::new(coco_gt, coco_dt, iou_type);
    ev.evaluate();    // Per-image matching
    ev.accumulate();  // Aggregate into precision/recall curves
    ev.summarize();   // Print and compute the 12 summary metrics
    ```

The only thing that changes between eval types is the `iou_type` parameter and the format of your detections.

## Bounding box evaluation

Set `iou_type` to `"bbox"` (Python) or `IouType::Bbox` (Rust).

Detection format — each result needs `image_id`, `category_id`, `bbox` as `[x, y, width, height]`, and `score`:

```json
[
  {"image_id": 42, "category_id": 1, "bbox": [10.0, 20.0, 30.0, 40.0], "score": 0.95},
  ...
]
```

IoU is computed as the intersection-over-union of the two bounding boxes.

## Segmentation evaluation

Set `iou_type` to `"segm"` (Python) or `IouType::Segm` (Rust).

Detection format — each result needs `image_id`, `category_id`, `segmentation` as a compressed RLE dict, and `score`:

```json
[
  {
    "image_id": 42,
    "category_id": 1,
    "segmentation": {"counts": "abc123...", "size": [480, 640]},
    "score": 0.95
  },
  ...
]
```

**What is RLE?** Run-Length Encoding stores a binary mask as the lengths of alternating runs of 0s and 1s — `size` is `[height, width]` and `counts` is a compact string (`str` in JSON files, `bytes` from `mask.encode`; `load_res` accepts either). For encoding masks from numpy arrays, see the [Mask Operations guide](masks.md).

IoU is computed on the binary masks after RLE decoding.

!!! tip
    If your results only have bounding boxes, use bbox evaluation instead. `load_res` generates polygon segmentations from bboxes, but these are axis-aligned rectangles — not instance masks.

## Keypoint evaluation

Set `iou_type` to `"keypoints"` (Python) or `IouType::Keypoints` (Rust).

Detection format — each result needs `image_id`, `category_id`, `keypoints` as a flat list of `[x1, y1, v1, x2, y2, v2, ...]`, and `score`:

```json
[
  {
    "image_id": 42,
    "category_id": 1,
    "keypoints": [x1, y1, v1, x2, y2, v2, ...],
    "score": 0.95
  },
  ...
]
```

Each keypoint has an `(x, y)` position and a visibility flag `v` (0 = not labeled, 1 = labeled but not visible, 2 = labeled and visible).

Similarity is measured using Object Keypoint Similarity ([OKS](https://cocodataset.org/#keypoints-eval)) instead of IoU. OKS uses per-keypoint sigma values that account for annotation noise — keypoints with higher variance (like hips) are weighted less strictly than precise ones (like eyes).

**Differences from bbox/segm:**

- 10 metrics instead of 12 (no small area range — keypoints are only meaningful on medium and large objects)
- Default max detections is `[20]` instead of `[1, 10, 100]`
- Ground truth annotations with `num_keypoints == 0` are automatically ignored

## Oriented bounding box (OBB) evaluation

Set `iou_type` to `"obb"` (Python) or `IouType::Obb` (Rust).

OBB evaluation is for rotated/oriented detection tasks — aerial imagery, document analysis, and scene text. Each annotation uses a 5-parameter representation: center coordinates, dimensions, and rotation angle.

Detection format — each result needs `image_id`, `category_id`, `obb` as `[cx, cy, w, h, angle]`, and `score`:

```json
[
  {"image_id": 42, "category_id": 1, "obb": [150.0, 200.0, 80.0, 40.0, 0.785], "score": 0.95},
  ...
]
```

The `obb` field is `[cx, cy, w, h, angle]` where:

- `cx, cy` — center of the rotated rectangle
- `w, h` — width and height of the rectangle
- `angle` — rotation angle in **radians**, counter-clockwise positive

IoU is computed via polygon intersection of the two rotated rectangles (Sutherland-Hodgman clipping). This is exact — no approximation or rasterization.

**Differences from bbox:**

- Uses rotated IoU instead of axis-aligned IoU
- Same 12 metrics as bbox/segm
- `load_res` automatically computes `area` (w × h) and an axis-aligned `bbox` (for area-range filtering) from the OBB
- No pycocotools equivalent exists — hotcoco defines the evaluation protocol

Conversion to and from DOTA, the standard aerial-detection label format, is built in — `COCO.from_dota()` / `to_dota()` in Python, `coco convert --from dota` on the CLI, and the `hotcoco::convert` functions in Rust. See [format conversion](datasets.md#dota).

## The 12 COCO metrics

`summarize()` computes and prints these metrics (10 for keypoints). The evaluation protocol is defined in the [COCO detection evaluation](https://cocodataset.org/#detection-eval) specification ([Lin et al., ECCV 2014](https://arxiv.org/abs/1405.0312)):

| Index | Metric | IoU | Area | MaxDets |
|-------|--------|-----|------|---------|
| 0 | **AP** | 0.50:0.95 | all | 100 |
| 1 | AP | 0.50 | all | 100 |
| 2 | AP | 0.75 | all | 100 |
| 3 | AP | 0.50:0.95 | small | 100 |
| 4 | AP | 0.50:0.95 | medium | 100 |
| 5 | AP | 0.50:0.95 | large | 100 |
| 6 | AR | 0.50:0.95 | all | 1 |
| 7 | AR | 0.50:0.95 | all | 10 |
| 8 | AR | 0.50:0.95 | all | 100 |
| 9 | AR | 0.50:0.95 | small | 100 |
| 10 | AR | 0.50:0.95 | medium | 100 |
| 11 | AR | 0.50:0.95 | large | 100 |

Reading the table: a detection counts only when its IoU with a ground truth clears the threshold — the headline **AP** averages over thresholds 0.50–0.95, while AP50 and AP75 fix a single one. **AR** is the recall achieved with at most 1, 10, or 100 detections per image. The small/medium/large rows restrict to objects under 32², between 32² and 96², and over 96² pixels respectively — models often perform very differently across sizes. For a full treatment of the underlying concepts, see the [COCO evaluation spec](https://cocodataset.org/#detection-eval).

## Customizing parameters

Modify `ev.params` before calling `evaluate()`:

=== "Python"

    ```python
    ev = COCOeval(coco_gt, coco_dt, "bbox")

    # Evaluate a subset of categories
    ev.params.cat_ids = [1, 3]

    # Evaluate a subset of images
    ev.params.img_ids = [42, 139]

    # Custom IoU thresholds
    ev.params.iou_thrs = [0.5, 0.75, 0.9]

    # Custom max detections
    ev.params.max_dets = [1, 10, 100]

    # Category-agnostic evaluation (pool all categories)
    ev.params.use_cats = False

    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    ```

=== "Rust"

    ```rust
    let mut ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);

    ev.params.cat_ids = vec![1, 3];
    ev.params.img_ids = vec![42, 139];
    ev.params.iou_thrs = vec![0.5, 0.75, 0.9];
    ev.params.max_dets = vec![1, 10, 100];
    ev.params.use_cats = false;

    ev.evaluate();
    ev.accumulate();
    ev.summarize();
    ```

!!! note
    Changing `iou_thrs`, `max_dets`, or `area_rng` from their defaults affects what `summarize()` can display. The 12-metric output format is fixed — for example, AP50 looks for IoU=0.50 in your thresholds and shows `-1.000` if it's not there. A `UserWarning` is emitted when your parameters don't match the expected defaults. Filtering by `img_ids`, `cat_ids`, or setting `use_cats` is safe and won't trigger warnings.

See [Params](../api/params.md) for the full list of configurable parameters.

## Sliced evaluation

`slice_by()` re-computes all summary metrics for named subsets of images — without re-running IoU computation. This is useful for comparing model performance across data splits (e.g., indoor vs outdoor, day vs night, small images vs large images).

```python
ev = COCOeval(coco_gt, coco_dt, "bbox")
ev.evaluate()

# Define slices as {name: [img_ids]}
slices = {
    "indoor": [42, 139, 203, ...],
    "outdoor": [78, 412, 901, ...],
}

results = ev.slice_by(slices)
```

Each slice in `results` contains the full set of summary metrics plus a delta vs the overall baseline:

```python
for name, sr in results.items():
    if name == "_overall":
        continue
    print(f"{name}: AP={sr['AP']:.3f} (Δ{sr['delta']['AP']:+.3f})")
```

You can also pass a callable instead of a dict — it receives each image dict and returns a slice name (or `None` to skip):

```python
results = ev.slice_by(lambda img: "large" if img["width"] > 1000 else "small")
```

### From the CLI

Pass `--slices <path>` to `coco eval` with a JSON file mapping slice names to image ID lists:

```bash
coco eval --gt annotations.json --dt detections.json --slices slices.json
```

## Where to next

- [LVIS & Open Images](lvis-open-images.md) — federated AP, category hierarchies, and group-of matching
- [Model Diagnostics](diagnostics.md) — confusion matrix, TIDE error analysis, calibration, F-scores, model comparison, and per-image failure mining
- [Working with Results](results.md) — the evaluation report, provenance, per-category AP, JSON export, and experiment-tracker logging
