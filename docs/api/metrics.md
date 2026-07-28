# metrics

Metric functions over flat arrays — no evaluator required.

=== "Python"

    ```python
    from hotcoco import metrics
    ```

=== "Rust"

    ```rust
    use hotcoco::metrics;
    ```

Every function here is pure: arrays in, numbers out. This is the same shape
`sklearn.metrics` and `torchmetrics.functional` use, and it means you can score
predictions that never went through a COCO JSON file.

```python
from hotcoco import metrics

scores  = [0.95, 0.88, 0.71, 0.40]
matched = [True, True, False, True]

ap = metrics.average_precision(scores, matched, num_gt=5)
ece, mce = metrics.calibration_error(scores, matched)
```

!!! tip "`COCOeval` still does the whole pipeline"
    `COCOeval` is unchanged and is the right tool when you have COCO-format
    ground truth and detections. Its analysis methods — `calibration()`,
    `confusion_matrix()` — call straight into these functions. Reach for
    `metrics` when you have arrays rather than a dataset, or when you are
    scoring something that is not detection.

## `metrics` vs `primitives`

The two split by what a function produces:

| Module | Produces | Contains |
|---|---|---|
| [`primitives`](primitives.md) | matches and similarities | `lsap`, `bbox_iou`, `mask_iou` |
| `metrics` | numbers from matches | `average_precision`, `calibration_error`, `confusion_matrix` |

`primitives.lsap` decides *which prediction pairs with which ground truth*.
`metrics.average_precision` turns that decision into a number.

!!! warning "Provisional through 1.x"
    These APIs are public so drivers and users can share them, but they are not
    frozen until 1.4. Expect additive change rather than reshaping. `COCOeval`
    and the `pycocotools` drop-in surface are the permanent, frozen part of the
    API and are unaffected.

---

## Functions

### `average_precision`

Average precision from per-prediction scores and match flags.

=== "Python"

    ```python
    average_precision(
        scores: Sequence[float],
        matched: Sequence[bool],
        num_gt: int,
        ignored: Sequence[bool] | None = None,
        rec_thrs: Sequence[float] | None = None,
    ) -> float
    ```

    | Parameter | Type | Description |
    |-----------|------|-------------|
    | `scores` | `Sequence[float]` | Confidence per prediction, in any order |
    | `matched` | `Sequence[bool]` | Whether each prediction is correct |
    | `num_gt` | `int` | Total ground truths — the recall denominator |
    | `ignored` | `Sequence[bool] \| None` | Predictions counting as neither TP nor FP |
    | `rec_thrs` | `Sequence[float] \| None` | Recall grid; defaults to COCO's 101 points |

=== "Rust"

    ```rust
    metrics::counts::average_precision(
        scores: &[f64],
        matched: &[bool],
        ignored: Option<&[bool]>,
        num_gt: usize,
        rec_thrs: &[f64],
    ) -> f64
    ```

Sorts by score descending, classifies each prediction as TP or FP, and
interpolates precision onto the recall grid using PASCAL VOC interpolation —
the same computation that produces COCO's AP.

```python
>>> round(metrics.average_precision([0.9, 0.8, 0.3], [True, False, True], num_gt=2), 3)
0.835
```

!!! warning "`num_gt` must count ground truths you never predicted"
    It is the recall denominator. Passing only the matched count silently
    overstates recall, and therefore AP.

Returns `0.0` when there are no predictions or no ground truth. If your metric
wants a different answer for the empty case — per-image diagnostics call an
empty image perfect — branch before calling.

---

### `precision_recall_curve`

Precision interpolated onto a recall grid, from cumulative TP/FP counts.

=== "Python"

    ```python
    precision_recall_curve(
        tp_cum: Sequence[float],
        fp_cum: Sequence[float],
        num_gt: int,
        rec_thrs: Sequence[float] | None = None,
    ) -> tuple[float, list[tuple[int, float, int]]]
    ```

=== "Rust"

    ```rust
    metrics::counts::precision_recall_curve(
        tp_cum: &[f64], fp_cum: &[f64], num_gt: usize, rec_thrs: &[f64],
    ) -> (f64, Vec<(usize, f64, usize)>)
    ```

Lower level than `average_precision` — use it when you already hold cumulative
counts, or want the curve rather than the scalar. `tp_cum` and `fp_cum` must
already be prefix-summed over predictions sorted by descending score.

Returns `(final_recall, points)`, where each point is
`(threshold_index, precision, rank)`. Recall thresholds the predictions never
reach are **omitted** rather than reported as zero, so `points` can be shorter
than `rec_thrs`.

---

### `calibration_curve`

Reliability bins: predicted confidence against observed accuracy.

=== "Python"

    ```python
    calibration_curve(
        scores: Sequence[float], matched: Sequence[bool], n_bins: int = 10
    ) -> list[dict]
    ```

=== "Rust"

    ```rust
    metrics::calibration::calibration_curve(
        scores: &[f64], matched: &[bool], n_bins: usize,
    ) -> Vec<CalibrationBin>
    ```

The data behind a reliability diagram. Each dict has `bin_lower`, `bin_upper`,
`avg_confidence`, `avg_accuracy`, and `count`. Empty bins are included with
`count = 0`, so the list always has `n_bins` entries and plots without gaps.

A perfectly calibrated model has `avg_confidence == avg_accuracy` in every bin —
that diagonal is what the diagram compares against.

---

### `calibration_error`

Expected and Maximum Calibration Error.

=== "Python"

    ```python
    calibration_error(
        scores: Sequence[float], matched: Sequence[bool], n_bins: int = 10
    ) -> tuple[float, float]
    ```

=== "Rust"

    ```rust
    // Rust splits binning from scoring, so bins can be reused.
    let bins = metrics::calibration::calibration_curve(&scores, &matched, n_bins);
    let (ece, mce) = metrics::calibration::calibration_error(&bins);
    ```

Returns `(ece, mce)`:

- **ECE** — the occupancy-weighted mean gap between confidence and accuracy.
  The headline number.
- **MCE** — the worst single bin's gap, unweighted. Catches a badly calibrated
  region that ECE averages away.

```python
>>> # Always claims 0.9 confidence, right half the time.
>>> ece, mce = metrics.calibration_error([0.9] * 100, [True] * 50 + [False] * 50)
>>> round(ece, 3)
0.4
```

Both are `0.0` for empty input.

---

### `confusion_matrix`

Confusion counts over matched ground-truth/prediction pairs.

=== "Python"

    ```python
    confusion_matrix(
        gt: Sequence[int | None], dt: Sequence[int | None], num_classes: int
    ) -> numpy.ndarray
    ```

=== "Rust"

    ```rust
    metrics::confusion::confusion_matrix(
        gt_labels: &[Option<usize>], dt_labels: &[Option<usize>], num_classes: usize,
    ) -> Vec<u64>
    ```

`sklearn.metrics.confusion_matrix` assumes every sample has both a true and a
predicted label. Detection and tracking don't: a prediction can match nothing,
and a ground truth can go unpredicted. So this takes **optional** labels and
reserves index `num_classes` for background.

| `gt[i]` | `dt[i]` | Meaning | Lands at |
|---|---|---|---|
| `g` | `d` | matched pair (correct when `g == d`) | `[g][d]` |
| `g` | `None` | ground truth with no prediction | `[g][num_classes]` |
| `None` | `d` | prediction matching no ground truth | `[num_classes][d]` |
| `None` | `None` | nothing happened | ignored |

```python
>>> m = metrics.confusion_matrix([0, 1, None], [0, None, 1], num_classes=2)
>>> m[0, 0], m[1, 2], m[2, 1]   # correct, missed, spurious
(1, 1, 1)
```

The entries are one per **match record**, not one per prediction — producing
those records is the caller's job, and it is the only family-specific step.
Counts are integers, so accumulating per-image and summing gives the same answer
as one whole-dataset call. That is what lets you parallelize and reduce.

Class indices outside `range(num_classes)` are dropped rather than raising, so a
stray label can't take down an evaluation run.

---

## Not yet exposed to Python

**Bootstrap confidence intervals** (`metrics::bootstrap::bootstrap_ci` in Rust)
take the statistic as a closure, and calling back into Python from the parallel
resampling loop would mean re-acquiring the GIL per sample — which would make it
slower than doing the whole thing in Python. `compare()` uses it internally and
returns the intervals, which covers the case people actually ask for.

**Greedy matching** (`primitives::greedy::greedy_match`) carries pycocotools'
crowd and ignore semantics in its signature. Exposing that faithfully needs a
Python-facing shape designed on purpose rather than transliterated; it lands in
a 1.x minor. `COCOeval.evaluate()` uses it today.
