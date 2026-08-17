# COCOeval

Run COCO evaluation to compute AP/AR metrics.

=== "Python"

    ```python
    from hotcoco import COCO, COCOeval

    coco_gt = COCO("instances_val2017.json")
    coco_dt = coco_gt.load_res("detections.json")

    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    ```

=== "Rust"

    ```rust
    use hotcoco::{COCO, COCOeval};
    use hotcoco::params::IouType;
    use std::path::Path;

    let coco_gt = COCO::new(Path::new("instances_val2017.json"))?;
    let coco_dt = coco_gt.load_res(Path::new("detections.json"))?;

    let mut ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    ev.evaluate();
    ev.accumulate();
    ev.summarize();
    ```

---

## Constructor

=== "Python"

    ```python
    COCOeval(
        coco_gt: COCO,
        coco_dt: COCO,
        iou_type: str,
        *,
        lvis_style: bool = False,
        oid_style: bool = False,
        hierarchy: Hierarchy | None = None,
    )
    ```

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `coco_gt` | `COCO` | — | Ground truth COCO object |
    | `coco_dt` | `COCO` | — | Detections COCO object (from `load_res`) |
    | `iou_type` | `str` | — | `"bbox"`, `"segm"`, `"keypoints"`, or `"obb"` |
    | `lvis_style` | `bool` | `False` | Enable LVIS federated evaluation mode |
    | `oid_style` | `bool` | `False` | Enable Open Images evaluation mode (IoU=0.5, group-of matching) |
    | `hierarchy` | <code>Hierarchy &#124; None</code> | `None` | Category hierarchy for GT expansion in OID mode |

    !!! note "pycocotools keyword spellings"
        pycocotools spells the constructor keywords `cocoGt`, `cocoDt`, and
        `iouType`, and consumers pass them that way (torchmetrics' pycocotools
        backend calls `COCOeval(gt, dt, iouType=...)`). Both spellings are
        accepted; mixing the two spellings of one argument is an error.

=== "Rust"

    ```rust
    // Standard COCO
    COCOeval::new(coco_gt: COCO, coco_dt: COCO, iou_type: IouType) -> Self

    // LVIS federated
    COCOeval::new_lvis(coco_gt: COCO, coco_dt: COCO, iou_type: IouType) -> Self

    // Open Images
    COCOeval::new_oid(coco_gt: COCO, coco_dt: COCO, hierarchy: Option<Hierarchy>) -> Self
    ```

    | Parameter | Type | Description |
    |-----------|------|-------------|
    | `coco_gt` | `COCO` | Ground truth COCO object |
    | `coco_dt` | `COCO` | Detections COCO object (from `load_res`) |
    | `iou_type` | `IouType` | `IouType::Bbox`, `IouType::Segm`, `IouType::Keypoints`, or `IouType::Obb` |
    | `hierarchy` | `Option<Hierarchy>` | Category hierarchy for GT expansion; `None` to skip expansion |

`"obb"` evaluates oriented boxes with a rotated IoU kernel — see [OBB evaluation](../guide/evaluation.md#oriented-bounding-box-obb-evaluation).

The same classes are also reachable under `hotcoco.detection` (`from hotcoco.detection import COCOeval`), an explicit namespace for the detection metric family. Both spellings return the same objects.

---

## Properties

### `params`

=== "Python"

    ```python
    params: Params
    ```

    Evaluation parameters. Modify before calling `evaluate()`.

    ```python
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.params.cat_ids = [1, 2, 3]
    ev.params.max_dets = [1, 10, 100]
    ```

=== "Rust"

    ```rust
    pub params: Params
    ```

    ```rust
    let mut ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
    ev.params.cat_ids = vec![1, 2, 3];
    ev.params.max_dets = vec![1, 10, 100];
    ```

See [Params](params.md) for all configurable fields.

---

### `stats`

=== "Python"

    ```python
    stats: np.ndarray
    ```

    The 12 summary metrics (10 for keypoints) as a `float64` numpy array, populated after `summarize()`. An empty list before `summarize()` is called — both states match pycocotools.

    ```python
    ev.summarize()
    print(f"AP: {ev.stats[0]:.3f}")
    print(f"AP50: {ev.stats[1]:.3f}")
    ```

=== "Rust"

    ```rust
    fn stats(&self) -> Option<&[f64]>
    ```

    ```rust
    ev.summarize();
    if let Some(stats) = ev.stats() {
        println!("AP: {:.3}", stats[0]);
        println!("AP50: {:.3}", stats[1]);
    }
    ```

---

### `eval_imgs`

Per-image evaluation results, populated after `evaluate()`. See [Working with Results](../guide/results.md) for details.

=== "Python"

    ```python
    eval_imgs: list[dict | None]
    ```

=== "Rust"

    ```rust
    fn eval_imgs(&self) -> &[Option<EvalImg>]
    ```

---

### `eval`

Accumulated precision/recall arrays, populated after `accumulate()`. See [Working with Results](../guide/results.md) for details.

=== "Python"

    ```python
    eval: dict | None
    ```

    Contains `"precision"`, `"recall"`, and `"scores"` arrays.

=== "Rust"

    ```rust
    fn accumulated(&self) -> Option<&AccumulatedEval>
    ```

    Access elements with `precision_idx(t, r, k, a, m)` and `recall_idx(t, k, a, m)`.

---

## Methods

### `evaluate`

```python
evaluate() -> None
```

Run per-image evaluation. Matches detections to ground truth annotations using greedy matching sorted by confidence. Must be called before `accumulate()`.

Populates `eval_imgs`.

---

### `accumulate`

```python
accumulate() -> None
```

Accumulate per-image results into precision/recall curves using interpolated precision at 101 recall thresholds.

Populates `eval`.

---

### `summarize`

```python
summarize() -> None
```

Compute and print the standard COCO metrics. Populates `stats`.

!!! warning "Non-default parameters"
    `summarize()` uses a fixed display format that assumes default `iou_thrs`, `max_dets`, and `area_rng_lbl`. If you've changed any of these, a `UserWarning` is emitted (catchable with `warnings.catch_warnings`, visible in Jupyter) and some metrics may show `-1.000` (e.g. AP50 when `iou_thrs` doesn't include 0.50). The `stats` array always has 12 entries (10 for keypoints) regardless of your parameters. `-1.000` always means "not computed for this configuration" — an unknown area label or max-dets value degrades to `-1.0` rather than silently substituting the `"all"` slice.

Prints 12 lines for bbox/segm (10 for keypoints):

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.382
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.584
 ...
```

---

### `run`

```python
run() -> None
```

Run the full pipeline in one call: `evaluate()` → `accumulate()` → `summarize()`. Primarily used with LVIS pipelines (Detectron2, MMDetection) that expect a single `run()` call.

---

### `metric_keys`

```python
metric_keys() -> list[str]
```

Return metric names in canonical display order for the current evaluation mode. This is the authoritative ordering — the same list that drives `summarize()` and `get_results()`.

```python
ev = COCOeval(gt, dt, "bbox")
ev.metric_keys()
# ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl']
```

Does not require `evaluate()` or `run()` — only depends on the evaluation mode and IoU type.

---

### `metric_defs`

```python
metric_defs() -> list[dict]
```

Return the metric catalog as structured data, one dict per metric in `metric_keys()`
order: `name` (str), `ap` (bool — AP vs AR), `iou_thr` (float or `None` for the full
0.50:0.05:0.95 sweep), `area` (str), `max_det` (int), and `freq_group` (`"rare"` /
`"common"` / `"frequent"` or `None`; LVIS only).

```python
ev.metric_defs()[1]
# {'name': 'AP50', 'ap': True, 'iou_thr': 0.5, 'area': 'all', 'max_det': 100, 'freq_group': None}
```

This exists so renderers read a metric's axes instead of parsing them back out of its
name — `"AR10"` is ambiguous between a detection cap of 10 and an IoU of 0.10, and only
the catalog knows which. hotcoco's own PDF report and dashboard consume it. Like
`metric_keys()`, works before `run()`.

---

### `get_results`

```python
get_results(prefix: str | None = None, per_class: bool = False) -> dict[str, float]
```

Return the summary metrics as a dict. Must be called after `summarize()` (or `run()`). Returns an empty dict if `summarize()` has not been called.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prefix` | <code>str &#124; None</code> | `None` | If given, each key is prefixed as `"{prefix}/{metric}"`. |
| `per_class` | `bool` | `False` | If `True`, include per-category AP values keyed as `"AP/{cat_name}"` (or `"{prefix}/AP/{cat_name}"` with a prefix). |

Standard bbox/segm keys: `AP`, `AP50`, `AP75`, `APs`, `APm`, `APl`, `AR1`, `AR10`, `AR100`, `ARs`, `ARm`, `ARl`.

Keypoint keys: `AP`, `AP50`, `AP75`, `APm`, `APl`, `AR`, `AR50`, `AR75`, `ARm`, `ARl`.

LVIS keys: `AP`, `AP50`, `AP75`, `APs`, `APm`, `APl`, `APr`, `APc`, `APf`, `AR@300`, `ARs@300`, `ARm@300`, `ARl@300`.

```python
ev.run()

# Basic usage (unchanged)
results = ev.get_results()
print(f"AP: {results['AP']:.3f}, AP50: {results['AP50']:.3f}")

# Prefixed keys — ready for any logger
results = ev.get_results(prefix="val/bbox")
# {"val/bbox/AP": 0.578, "val/bbox/AP50": 0.861, ...}

# With per-class AP
results = ev.get_results(prefix="val/bbox", per_class=True)
# {"val/bbox/AP": 0.578, ..., "val/bbox/AP/person": 0.82, ...}
```

---

### `print_results`

```python
print_results() -> None
```

Print a formatted results table to stdout. For LVIS, matches the lvis-api `print_results()` style. Must be called after `summarize()` (or `run()`).

---

### `summary_lines`

```python
summary_lines() -> list[str]
```

The same lines `summarize()` prints, returned instead of written to stdout — one string per metric, already formatted. Use it to route the summary into a logger, a report, or a test assertion. Must be called after `summarize()` (or `run()`).

---

### `virtual_cat_names`

```python
virtual_cat_names: list[str]   # property
```

Category names added by Open Images hierarchy expansion — ancestor categories that exist in the hierarchy but not in the dataset's own taxonomy. Empty when not in OID mode, when no hierarchy expansion occurred, or before `evaluate()`. Use it to distinguish expanded ancestor categories from the model's native classes:

```python
ev = COCOeval(gt, dt, "bbox", oid_style=True, hierarchy=h)
ev.evaluate()
ev.virtual_cat_names   # e.g. ['Carnivore', 'Mammal'] — no parentheses; it's a property
```

---

### `slice_by`

```python
slice_by(slices: dict[str, list[int]] | Callable[[dict], str]) -> dict[str, Any]
```

Re-accumulate metrics for named subsets of images without recomputing IoU, and return one metrics dict per slice. Pass either an explicit `{name: [image_ids]}` mapping or a function that takes an image dict and returns a slice name.

```python
ev.run()
by_light = ev.slice_by({"day": day_ids, "night": night_ids})
by_light["night"]["AP"]
```

Requires `evaluate()` to have run. Matching is done once and reused for every slice, so slicing a dozen ways costs barely more than slicing one way. See [sliced evaluation](../guide/evaluation.md#sliced-evaluation).

---

### `report`

```python
report() -> dict
```

Return a full evaluation report. Must be called after `summarize()` (or `run()`). Raises `RuntimeError` otherwise.

This is the shape every hotcoco metric family reports in, so code that renders a detection report will render a panoptic or tracking one unchanged.

**Returns** a dict with:

| Key | Type | Description |
|-----|------|-------------|
| `"task"` | `str` | `"detection"`. |
| `"provenance"` | `str` | `"parity_verified"` or `"extension"` — see below. |
| `"metrics"` | `dict[str, float]` | Summary metrics keyed by name. |
| `"per_class"` | `dict[str, dict[str, float]]` | `{class_name: {metric: value}}`. |
| `"per_group"` | `dict[str, dict[str, float]]` | LVIS frequency buckets in LVIS mode; empty otherwise. |
| `"curves"` | `dict[str, list[float]]` | One precision-recall curve per IoU threshold, plus `"rec_thrs"`. |
| `"params"` | `dict` | The evaluation parameters used. |

```python
ev.run()
report = ev.report()

report["metrics"]["AP"]              # 0.377
report["per_class"]["person"]["AP"]  # 0.521
```

#### Checking provenance

`provenance` records whether these numbers may be compared against a published leaderboard:

| Value | Meaning |
|-------|---------|
| `"parity_verified"` | Checked against the reference implementation. bbox, segm, and keypoints match pycocotools. |
| `"extension"` | A real metric or configuration that is not leaderboard-comparable: oriented boxes (no reference protocol exists), Open Images (checked against the TensorFlow reference for group-of handling and AP, but missing the challenge's image-level-label rule), or any run with non-default `iou_thrs`, `rec_thrs`, `max_dets`, area ranges, `use_cats`, or `kpt_oks_sigmas`. Fine for comparing your own models; not a leaderboard number. |

```python
if report["provenance"] != "parity_verified":
    print(f"note: {report['provenance']} — not benchmark-standard")
```

For the same marker without building a report — and without evaluating at all — see
[`provenance`](#provenance) and [`reference_deviations`](#reference_deviations).

#### Plotting the curves

`curves` holds the aggregate precision-recall curve for each IoU threshold, averaged over
categories at `area="all"` and the largest `max_dets` — the slice a chart actually draws.
All PR curves share the `"rec_thrs"` x-axis.

```python
import matplotlib.pyplot as plt

curves = report["curves"]
for iou in ("pr@0.50", "pr@0.75", "pr@0.95"):
    plt.plot(curves["rec_thrs"], curves[iou], label=iou)
plt.xlabel("recall")
plt.ylabel("precision")
plt.legend()
```

For the full per-category arrays use [`eval["precision"]`](#eval) instead — on COCO that
is roughly a million floats, which is why the report carries only the aggregate.

---

### `provenance`

```python
provenance() -> str
```

Return `"parity_verified"` or `"extension"` — the same value as `report()["provenance"]`
and `results()["provenance"]`, but read from the configuration alone, so **this one works
before `run()`**. Check it ahead of a long evaluation rather than discovering afterwards
that the numbers cannot be published.

```python
ev = hotcoco.COCOeval(gt, dt, "bbox")
ev.params.iouThrs = [0.5]
ev.provenance()   # 'extension' — already, before evaluating
```

Never infer comparability from `iou_type` or the eval mode instead. Parity is a property
of the whole configuration, so the run above is an extension despite being ordinary COCO
bbox evaluation.

---

### `is_benchmark_standard`

```python
is_benchmark_standard() -> bool
```

`True` exactly when `provenance()` is `"parity_verified"` — the predicate itself, so
renderers don't re-derive it with a string compare. Default-deny: a provenance variant
added in a future release reads as *needs a caveat* until a renderer is taught what it
means. Works before `run()`.

```python
if not ev.is_benchmark_standard():
    print("caveat:", ev.reference_deviations())
```

---

### `reference_deviations`

```python
reference_deviations() -> list[str]
```

Return one human-readable sentence per way this run departs from the reference
configuration, empty exactly when `provenance()` is `"parity_verified"`. Also works
before `run()`.

```python
for reason in ev.reference_deviations():
    print(reason)
# iou_thrs differ from default (0.50:0.05:0.95). AP50/AP75 lines may show -1.000.
```

This is the same predicate behind the warnings `summarize()` prints, so a report cannot
claim parity while the warnings disagree. hotcoco's own renderers — the PDF report, the
browse dashboard, and `coco eval --json` — read these rather than recomputing them.

---

### `results`

```python
results(per_class: bool = False) -> dict
```

Return evaluation results as a serializable dict. Must be called after `summarize()` (or `run()`). Raises `RuntimeError` if `summarize()` has not been called.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `per_class` | `bool` | `False` | If `True`, include per-category AP values under the `"per_class"` key. |

**Returns** a dict with:

| Key | Type | Description |
|-----|------|-------------|
| `"hotcoco_version"` | `str` | hotcoco version that produced these results. |
| `"provenance"` | `str` | `"parity_verified"` or `"extension"` — same value as `report()["provenance"]`. |
| `"params"` | `dict` | Evaluation parameters: `iou_type`, `eval_mode`, `iou_thresholds`, `recall_thresholds`, `area_ranges`, `max_dets`, `use_cats`, `kpt_oks_sigmas`, and `reference_deviations` — enough for a saved run to explain its own provenance. |
| `"metrics"` | `dict[str, float]` | Summary metrics keyed by name (same keys as `get_results()`). |
| `"per_class"` | `dict[str, float]` \| absent | Per-category AP values keyed by category name. Only present if `per_class=True`. |

```python
ev.run()
r = ev.results()
print(r["metrics"]["AP"])

# With per-category breakdown
r = ev.results(per_class=True)
print(r["per_class"]["person"])
```

---

### `save_results`

```python
save_results(path: str, per_class: bool = False) -> None
```

Save evaluation results to a JSON file. Must be called after `summarize()` (or `run()`). Raises `RuntimeError` if `summarize()` has not been called, or `IOError` if the file cannot be written.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | — | Output file path. |
| `per_class` | `bool` | `False` | If `True`, include per-category AP values. |

```python
ev = COCOeval(coco_gt, coco_dt, "bbox")
ev.run()
ev.save_results("results.json")

# With per-category AP
ev.save_results("results_per_class.json", per_class=True)
```

The JSON structure matches the dict returned by `results()`.

---

### `confusion_matrix`

```python
confusion_matrix(
    iou_thr: float = 0.5,
    max_det: int | None = None,
    min_score: float | None = None,
) -> dict
```

Compute a per-category confusion matrix. Unlike `evaluate()`, this method compares **all** detections in an image against **all** ground truth boxes regardless of category, enabling cross-category confusion analysis.

This method is **standalone** — no `evaluate()` call is needed first.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `iou_thr` | `float` | `0.5` | IoU threshold for a DT↔GT match |
| `max_det` | `int \| None` | last `params.max_dets` value | Max detections per image by score |
| `min_score` | `float \| None` | `None` | Discard detections below this confidence before `max_det` truncation |

**Returns** a dict with:

| Key | Type | Description |
|-----|------|-------------|
| `"matrix"` | `np.ndarray[uint64]` shape `(K+1, K+1)` | Raw confusion counts. Rows = GT category, cols = predicted. Index `K` is background. Unsigned — cast before subtracting counts (e.g. `matrix.astype(np.int64)`) to avoid underflow. |
| `"normalized"` | `np.ndarray[float64]` shape `(K+1, K+1)` | Row-normalized version (rows sum to 1.0; zero rows stay zero). |
| `"cat_ids"` | `list[int]` | Category IDs for rows/cols `0..K-1`. |
| `"cat_names"` | `list[str]` | Category names for rows/cols `0..K-1`, in the same order as `cat_ids`. |
| `"num_cats"` | `int` | Number of categories `K`. |
| `"iou_thr"` | `float` | IoU threshold used. |

**Matrix layout** (rows = GT, cols = predicted):

- `matrix[i][j]` where `i ≠ K, j ≠ K` — GT category `i` matched to predicted category `j`. On-diagonal = TP; off-diagonal = class confusion.
- `matrix[i][K]` — GT category `i` unmatched (false negative).
- `matrix[K][j]` — Predicted category `j` unmatched (false positive).

```python
ev = COCOeval(coco_gt, coco_dt, "bbox")
cm = ev.confusion_matrix(iou_thr=0.5, max_det=100)

matrix = cm["matrix"]
cat_ids = cm["cat_ids"]

# True positives per category
tp = matrix.diagonal()[:-1]

# False negatives per category
fn = matrix[:-1, -1]

# False positives per category
fp = matrix[-1, :-1]

# Normalized view
print(cm["normalized"])
```

See [Confusion Matrix](../guide/diagnostics.md#confusion-matrix) in the evaluation guide for a full walkthrough.

---

### `tide_errors`

```python
tide_errors(
    pos_thr: float = 0.5,
    bg_thr: float = 0.1,
) -> dict
```

Decompose detection errors into six TIDE error types ([Bolya et al., ECCV 2020](https://arxiv.org/abs/2008.08115)) and compute ΔAP — the AP gain from eliminating each error type.

Requires `evaluate()` to have been called first.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pos_thr` | `float` | `0.5` | IoU threshold for TP/FP classification |
| `bg_thr` | `float` | `0.1` | Background IoU threshold for Loc/Both/Bkg discrimination |

**Returns** a dict with:

| Key | Type | Description |
|-----|------|-------------|
| `"delta_ap"` | `dict[str, float]` | ΔAP for each error type. Keys: `"Cls"`, `"Loc"`, `"Both"`, `"Dupe"`, `"Bkg"`, `"Miss"`, plus tidecv's special oracles `"FP"` (suppress every false positive) and `"FN"` (drop every missed GT from the denominator; a superset of `"Miss"`). |
| `"counts"` | `dict[str, int]` | Count of each error type. Keys: `"Cls"`, `"Loc"`, `"Both"`, `"Dupe"`, `"Bkg"`, `"Miss"`. |
| `"ap_base"` | `float` | Baseline mean AP at `pos_thr`. |
| `"pos_thr"` | `float` | IoU threshold used. |
| `"bg_thr"` | `float` | Background threshold used. |

```python
ev = COCOeval(coco_gt, coco_dt, "bbox")
ev.evaluate()
result = ev.tide_errors(pos_thr=0.5, bg_thr=0.1)

print(f"ap_base: {result['ap_base']:.3f}")
for k, v in sorted(result["delta_ap"].items(), key=lambda x: -x[1]):
    if k not in ("FP", "FN"):
        print(f"  {k}: ΔAP={v:.4f}  n={result['counts'].get(k, '—')}")
```

See [TIDE Error Analysis](../guide/diagnostics.md#tide-error-analysis) in the evaluation guide for a detailed walkthrough.

---

### `calibration`

```python
calibration(
    n_bins: int = 10,
    iou_threshold: float = 0.5,
) -> dict
```

Compute confidence calibration metrics — how well confidence scores predict actual detection accuracy.

Requires `evaluate()` to have been called first. Bins all non-ignored detections by confidence score and compares the mean confidence in each bin to the fraction of true positives.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_bins` | `int` | `10` | Number of equal-width confidence bins in [0, 1]. |
| `iou_threshold` | `float` | `0.5` | IoU threshold for TP/FP classification. Must match one of `params.iouThrs`. |

**Returns** a dict with:

| Key | Type | Description |
|-----|------|-------------|
| `"ece"` | `float` | Expected Calibration Error — weighted mean of per-bin \|accuracy - confidence\|. |
| `"mce"` | `float` | Maximum Calibration Error — worst per-bin gap. |
| `"bins"` | `list[dict]` | Per-bin breakdown. Each dict has `bin_lower`, `bin_upper`, `avg_confidence`, `avg_accuracy`, `count`. |
| `"per_category"` | `dict[str, float]` | Per-category ECE, keyed by category name. |
| `"iou_threshold"` | `float` | IoU threshold used. |
| `"n_bins"` | `int` | Number of bins. |
| `"num_detections"` | `int` | Total non-ignored detections analyzed. |

```python
ev = COCOeval(coco_gt, coco_dt, "bbox")
ev.evaluate()

cal = ev.calibration(n_bins=10, iou_threshold=0.5)
print(f"ECE: {cal['ece']:.4f}, MCE: {cal['mce']:.4f}")

# Per-category breakdown
for name, ece in sorted(cal["per_category"].items(), key=lambda x: -x[1])[:5]:
    print(f"  {name}: ECE={ece:.4f}")
```

See [Confidence Calibration](../guide/diagnostics.md#confidence-calibration) in the evaluation guide for a full walkthrough.

---

### `f_scores`

```python
f_scores(beta: float = 1.0) -> dict[str, float]
```

Compute F-beta scores after `accumulate()` (or `run()`).

For each (IoU threshold, category), finds the confidence operating point that maximizes F-beta, then averages across categories — analogous to how mAP averages precision. Returns three metrics mirroring AP/AP50/AP75.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `beta` | `float` | `1.0` | Trade-off weight. `beta=1` → F1 (equal weight). `beta<1` → weights precision. `beta>1` → weights recall. |

**Returns** a dict with three keys:

| Key | Description |
|-----|-------------|
| `"F1"` | Mean max-F1 across IoU 0.50:0.05:0.95, all categories |
| `"F1_50"` | Max-F1 at IoU=0.50 |
| `"F1_75"` | Max-F1 at IoU=0.75 |

Key names reflect `beta`, formatted with no trailing zeros: `"F0.5"`, `"F0.5_50"`, `"F0.5_75"` for `beta=0.5`; `"F2"`, `"F2_50"`, `"F2_75"` for `beta=2.0`.

Returns an empty dict if `accumulate()` has not been called.

```python
ev = COCOeval(coco_gt, coco_dt, "bbox")
ev.run()

# F1 (default)
scores = ev.f_scores()
print(f"F1: {scores['F1']:.3f}, F1@50: {scores['F1_50']:.3f}")

# Precision-weighted
print(ev.f_scores(beta=0.5))   # {"F0.5": ..., "F0.5_50": ..., "F0.5_75": ...}

# Recall-weighted
print(ev.f_scores(beta=2.0))   # {"F2": ..., "F2_50": ..., "F2_75": ...}
```

---

### `image_diagnostics`

```python
image_diagnostics(
    iou_thr: float = 0.5,
    score_thr: float = 0.5,
) -> dict
```

Per-image diagnostics: annotation TP/FP/FN index, per-image F1 and AP scores, error profiles, and label error candidates.

Requires `evaluate()` to have been called first.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `iou_thr` | `float` | `0.5` | IoU threshold for TP/FP classification (snapped to nearest in `params.iouThrs`). |
| `score_thr` | `float` | `0.5` | Minimum detection confidence to consider for label error detection. |

**Returns** a dict with:

| Key | Type | Description |
|-----|------|-------------|
| `"dt_status"` | `dict[int, str]` | Detection annotation ID → `"tp"` or `"fp"`. |
| `"gt_status"` | `dict[int, str]` | GT annotation ID → `"matched"` or `"fn"`. |
| `"dt_match"` | `dict[int, int]` | TP detection → matched GT annotation ID. |
| `"gt_match"` | `dict[int, int]` | Matched GT → the detection that matched it. |
| `"img_summary"` | `dict[int, dict]` | Per-image summary (see below). |
| `"label_errors"` | `list[dict]` | Suspected label errors, sorted by detection score descending (see below). |
| `"iou_thr"` | `float` | Actual IoU threshold used (snapped). |
| `"score_thr"` | `float` | Score threshold used for label error detection. |

Each **image summary** dict contains:

| Key | Type | Description |
|-----|------|-------------|
| `"tp"` | `int` | True positive count. |
| `"fp"` | `int` | False positive count. |
| `"fn"` | `int` | False negative count. |
| `"f1"` | `float` | F1 score: `2*tp / (2*tp + fp + fn)`. 1.0 for empty images. |
| `"ap"` | `float` | AP at the selected IoU threshold (101-point interpolation). |
| `"error_profile"` | `str` | One of `"perfect"`, `"fp_heavy"`, `"fn_heavy"`, `"mixed"`. |

Each **label error** dict contains:

| Key | Type | Description |
|-----|------|-------------|
| `"image_id"` | `int` | Image containing the suspected error. |
| `"dt_id"` | `int` | Detection annotation ID. |
| `"dt_score"` | `float` | Detection confidence. |
| `"dt_category"` | `str` | Detection category name. |
| `"dt_category_id"` | `int` | Detection category ID. |
| `"gt_id"` | `int \| None` | Overlapping GT annotation ID (`None` for missing_annotation). |
| `"gt_category"` | `str \| None` | GT category name. |
| `"gt_category_id"` | `int \| None` | GT category ID. |
| `"iou"` | `float` | Bbox IoU between detection and GT (0.0 for missing_annotation). |
| `"type"` | `str` | `"wrong_label"` or `"missing_annotation"`. |

```python
ev = COCOeval(coco_gt, coco_dt, "bbox")
ev.evaluate()

diag = ev.image_diagnostics(iou_thr=0.5, score_thr=0.5)

# Worst images by F1
worst = sorted(diag["img_summary"].items(), key=lambda x: x[1]["f1"])[:5]

# Label errors
for le in diag["label_errors"]:
    print(f"{le['type']}: {le['dt_category']}→{le.get('gt_category', 'N/A')}")
```

See [Per-image diagnostics](../guide/diagnostics.md#per-image-diagnostics-label-error-detection) in the evaluation guide.

---

## LVIS evaluation

LVIS uses federated AP: each category is scored only over the images where it was
exhaustively annotated. These three names are drop-in replacements for the
`lvis-api` package, so Detectron2 and MMDetection pipelines run unchanged. See the
[LVIS guide](../guide/lvis-open-images.md#lvis-evaluation) for the protocol.

### `LVISeval`

```python
hotcoco.LVISeval(gt: COCO, dt: COCO, iou_type: str = "segm") -> COCOeval
```

Returns a `COCOeval` configured for federated evaluation — equivalent to
`COCOeval(gt, dt, iou_type, lvis_style=True)`. Supports `run()`,
`print_results()`, and `get_results()`, which is what those pipelines call.
Note the default `iou_type` is `"segm"`, matching `lvis-api`.

`LVISEval` (capital E) is an alias: that is the spelling `lvis-api` exports and
the one `from lvis import LVISEval` expects.

```python
from hotcoco import LVIS, LVISeval, LVISResults

lvis_gt = LVIS("lvis_v1_val.json")
lvis_dt = LVISResults(lvis_gt, "predictions.json", max_dets=300)

ev = LVISeval(lvis_gt, lvis_dt, "segm")
ev.run()
ev.print_results()
```

Evaluation reports 13 metrics — the 12 COCO metrics with `AR@300` in place of the
detection-count variants, plus `APr` / `APc` / `APf` for rare, common, and
frequent categories. See [the 13 LVIS metrics](../guide/lvis-open-images.md#the-13-lvis-metrics).

### `LVIS`

An alias for `COCO`, provided because `lvis-api` names its dataset class `LVIS`.
Loading is identical.

### `LVISResults`

```python
hotcoco.LVISResults(lvis_gt: COCO, results, max_dets: int = 300) -> COCO
```

Returns a `COCO` detections object, equivalent to `lvis_gt.load_res(results)`.
`max_dets` is accepted for API compatibility but not applied here — the 300-detection
cap is a `Params` setting that `LVISeval` already configures.

---

## Module-level functions

### `compare`

```python
hotcoco.compare(
    eval_a: COCOeval,
    eval_b: COCOeval,
    n_bootstrap: int = 0,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict
```

Pairwise model comparison. Both evaluators must have had `evaluate()` called and use the same `eval_mode`, `iou_type`, and evaluation grid — mismatched `iou_thrs`, `rec_thrs`, `max_dets`, or area ranges raise `ValueError` rather than summarizing one run under the other's catalog. Accumulation and summarization are performed internally on the shared image set.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eval_a` | `COCOeval` | — | Baseline model evaluation. |
| `eval_b` | `COCOeval` | — | Improved model evaluation. |
| `n_bootstrap` | `int` | `0` | Bootstrap samples for CIs (0 = disabled). |
| `seed` | `int` | `42` | Random seed for reproducibility. |
| `confidence` | `float` | `0.95` | Confidence level (e.g. 0.95 for 95% CI). |

**Returns** a dict with:

| Key | Type | Description |
|-----|------|-------------|
| `metric_keys` | `list[str]` | Metric names in canonical display order. |
| `metrics_a` | `dict[str, float]` | Summary metrics for model A. |
| `metrics_b` | `dict[str, float]` | Summary metrics for model B. |
| `deltas` | `dict[str, float]` | Per-metric delta (B − A). |
| `ci` | `dict` or `None` | Bootstrap CIs per metric (`lower`, `upper`, `confidence`, `prob_positive`, `std_err`). `None` if `n_bootstrap=0`. |
| `per_category` | `list[dict]` | Per-category AP comparison, sorted by delta ascending. Each entry has `cat_id`, `cat_name`, `ap_a`, `ap_b`, `delta`. |
| `n_bootstrap` | `int` | Number of bootstrap samples used. |
| `num_images` | `int` | Number of shared images. |

```python
import hotcoco

gt = hotcoco.COCO("annotations.json")
ev_a = hotcoco.COCOeval(gt, gt.load_res("baseline.json"), "bbox")
ev_a.evaluate()
ev_b = hotcoco.COCOeval(gt, gt.load_res("improved.json"), "bbox")
ev_b.evaluate()

# Without bootstrap
result = hotcoco.compare(ev_a, ev_b)
print(result["deltas"]["AP"])  # e.g. +0.033

# With bootstrap CIs
result = hotcoco.compare(ev_a, ev_b, n_bootstrap=1000)
ci = result["ci"]["AP"]
print(f"[{ci['lower']:+.3f}, {ci['upper']:+.3f}]")  # e.g. [+0.01, +0.05]
```
