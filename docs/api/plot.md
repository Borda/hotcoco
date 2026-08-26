# plot

```python
from hotcoco.plot import (
    report,
    pr_curve, pr_curve_iou_sweep, pr_curve_by_category, pr_curve_top_n,
    confusion_matrix, top_confusions, per_category_ap, tide_errors,
    reliability_diagram, comparison_bar, category_deltas,
    style, SERIES_COLORS, CHROME, SEQUENTIAL,
)
```

Requires `pip install hotcoco[plot]` (matplotlib >= 3.5).

All functions share these common parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `theme` | `str` | Visual theme: `"cyanotype"` (default) or `"cyanotype-dark"`. |
| `paper_mode` | `bool` | Set both figure and axes background to white. Useful for LaTeX / PowerPoint. Default `False`. |
| `ax` | <code>Axes &#124; None</code> | Draw on an existing axes. If `None`, creates a new figure. |
| `save_path` | <code>str &#124; Path &#124; None</code> | Save figure to this path (150 DPI). |

All functions return `(Figure, Axes)`.

---

## `pr_curve_iou_sweep`

```python
pr_curve_iou_sweep(
    coco_eval, *,
    iou_thrs=None, area_rng="all", max_det=None,
    theme="cyanotype", paper_mode=False, ax=None, save_path=None,
)
```

Plot one precision-recall curve per IoU threshold, with precision averaged across all categories. The primary line (lowest IoU) is annotated at its F1 peak, and filled under only when the plot has at most two curves — a fill runs to zero, so under the topmost curve of a wider sweep it would tint every other curve's area too. Past four curves the legend moves outside the axes rather than sitting on the data.

| Parameter | Type | Description |
|-----------|------|-------------|
| `coco_eval` | `COCOeval` | Must have `run()` called first. |
| `iou_thrs` | <code>list[float] &#124; None</code> | IoU thresholds to include. Default: all thresholds in params. Matched against the run's grid within a tolerance, since the default grid stores 0.90 as `0.8999999999999999`. Raises `ValueError` for a threshold the run was not accumulated at. |
| `area_rng` | `str` | Area range: `"all"`, `"small"`, `"medium"`, `"large"`. Default `"all"`. |
| `max_det` | <code>int &#124; None</code> | Max detections. Default: last entry in `params.max_dets`. |

---

## `pr_curve_by_category`

```python
pr_curve_by_category(
    coco_eval, cat_id, *,
    iou_thr=0.5, area_rng="all", max_det=None,
    theme="cyanotype", paper_mode=False, ax=None, save_path=None,
)
```

Plot the precision-recall curve for a single category at a fixed IoU threshold, with an F1 peak annotation.

| Parameter | Type | Description |
|-----------|------|-------------|
| `coco_eval` | `COCOeval` | Must have `run()` called first. |
| `cat_id` | `int` | Category ID to plot. |
| `iou_thr` | `float` | IoU threshold. Default `0.5`. |
| `area_rng` | `str` | Area range. Default `"all"`. |
| `max_det` | <code>int &#124; None</code> | Max detections. Default: last entry in `params.max_dets`. |

---

## `pr_curve_top_n`

```python
pr_curve_top_n(
    coco_eval, *,
    cat_ids=None, top_n=10, iou_thr=0.5, area_rng="all", max_det=None,
    theme="cyanotype", paper_mode=False, ax=None, save_path=None,
)
```

Plot precision-recall curves for multiple categories on one axes. When `cat_ids` is omitted, selects the top `top_n` categories by AP automatically.

| Parameter | Type | Description |
|-----------|------|-------------|
| `coco_eval` | `COCOeval` | Must have `run()` called first. |
| `cat_ids` | <code>list[int] &#124; None</code> | Categories to plot. Default: top `top_n` by AP. |
| `top_n` | `int` | Number of top categories when `cat_ids` is omitted. Default `10`. |
| `iou_thr` | `float` | IoU threshold. Default `0.5`. |
| `area_rng` | `str` | Area range. Default `"all"`. |
| `max_det` | <code>int &#124; None</code> | Max detections. Default: last entry in `params.max_dets`. |

---

## `pr_curve`

```python
pr_curve(
    coco_eval, *,
    iou_thrs=None, cat_id=None, cat_ids=None,
    iou_thr=None, top_n=10, area_rng="all", max_det=None,
    theme="cyanotype", paper_mode=False, ax=None, save_path=None,
)
```

Convenience dispatcher — inspects the arguments and calls the appropriate named function. Prefer calling the named functions directly for clarity.

- No `cat_id` or `cat_ids` → `pr_curve_iou_sweep`
- `cat_id` set → `pr_curve_by_category`
- `cat_ids` or `iou_thr` set → `pr_curve_top_n`

| Parameter | Type | Description |
|-----------|------|-------------|
| `coco_eval` | `COCOeval` | Must have `run()` called first. |
| `iou_thrs` | <code>list[float] &#124; None</code> | IoU thresholds to plot (IoU sweep mode). |
| `cat_id` | <code>int &#124; None</code> | Single category to plot. |
| `cat_ids` | <code>list[int] &#124; None</code> | Categories to compare. |
| `iou_thr` | <code>float &#124; None</code> | Fixed IoU for single/multi-category modes. Default 0.50. |
| `top_n` | `int` | Top N categories by AP. Default 10. |
| `area_rng` | `str` | Area range: `"all"`, `"small"`, `"medium"`, `"large"`. |
| `max_det` | <code>int &#124; None</code> | Max detections. Default: last in params. |

---

## `confusion_matrix`

```python
confusion_matrix(
    cm_dict, *,
    normalize=True, top_n=None,
    group_by=None, cat_groups=None,
    theme="cyanotype", paper_mode=False, ax=None, save_path=None,
)
```

Plot a confusion matrix heatmap.

| Parameter | Type | Description |
|-----------|------|-------------|
| `cm_dict` | `dict` | Output of `coco_eval.confusion_matrix()`. |
| `normalize` | `bool` | Row-normalize values. Default `True`. |
| `top_n` | <code>int &#124; None</code> | Show only top N categories by confusion mass. Auto-set to 25 when K > 30. |
| `group_by` | <code>str &#124; None</code> | `"supercategory"` to aggregate by group. |
| `cat_groups` | <code>dict &#124; None</code> | Group name → list of category names. Required with `group_by`. |

---

## `top_confusions`

```python
top_confusions(
    cm_dict, *,
    top_n=20,
    theme="cyanotype", paper_mode=False, ax=None, save_path=None,
)
```

Plot the top N misclassifications as horizontal bars. Shows "ground truth → prediction" pairs sorted by count.

| Parameter | Type | Description |
|-----------|------|-------------|
| `cm_dict` | `dict` | Output of `coco_eval.confusion_matrix()`. |
| `top_n` | `int` | Number of confusions to show. Default 20. |

---

## `per_category_ap`

```python
per_category_ap(
    results_dict, *,
    top_n=20, bottom_n=5,
    theme="cyanotype", paper_mode=False, ax=None, save_path=None,
)
```

Plot per-category AP as horizontal bars with a mean AP reference line.

| Parameter | Type | Description |
|-----------|------|-------------|
| `results_dict` | `dict` | Output of `coco_eval.results(per_class=True)`. |
| `top_n` | `int` | Top categories to show. Default 20. |
| `bottom_n` | `int` | Bottom categories to show. Default 5. |

---

## `tide_errors`

```python
tide_errors(
    tide_dict, *,
    theme="cyanotype", paper_mode=False, ax=None, save_path=None,
)
```

Plot TIDE error breakdown as horizontal bars.

| Parameter | Type | Description |
|-----------|------|-------------|
| `tide_dict` | `dict` | Output of `coco_eval.tide_errors()`. |

---

## `reliability_diagram`

```python
reliability_diagram(
    cal_or_eval, *,
    n_bins=10, iou_threshold=0.5,
    theme="cyanotype", paper_mode=False, ax=None, save_path=None,
)
```

Plot a reliability diagram — predicted confidence vs actual accuracy per bin, with a perfect calibration diagonal and gap overlay.

| Parameter | Type | Description |
|-----------|------|-------------|
| `cal_or_eval` | <code>dict &#124; COCOeval</code> | Either the output of `ev.calibration()` (a dict) or a `COCOeval` instance. If a `COCOeval` is passed, `calibration()` is called automatically. |
| `n_bins` | `int` | Number of bins (only used when `cal_or_eval` is a `COCOeval`). Default `10`. |
| `iou_threshold` | `float` | IoU threshold (only used when `cal_or_eval` is a `COCOeval`). Default `0.5`. |

```python
ev = COCOeval(coco_gt, coco_dt, "bbox")
ev.evaluate()

# From a calibration dict
cal = ev.calibration(n_bins=15)
fig, ax = reliability_diagram(cal)

# Or directly from a COCOeval
fig, ax = reliability_diagram(ev, n_bins=15)
```

---

## `comparison_bar`

```python
comparison_bar(
    compare_result: dict, *,
    theme: str = "cyanotype",
    paper_mode: bool = False,
    ax=None,
    save_path: str | Path | None = None,
) -> tuple[Figure, Axes]
```

Grouped bar chart comparing all metrics between two models. When the compare result includes bootstrap CIs, error bars are drawn on the model B bars.

```python
from hotcoco import compare
from hotcoco.plot import comparison_bar

result = compare(ev_a, ev_b, n_bootstrap=1000)
fig, ax = comparison_bar(result, save_path="comparison.png")
```

---

## `category_deltas`

```python
category_deltas(
    compare_result: dict, *,
    top_k: int = 20,
    theme: str = "cyanotype",
    paper_mode: bool = False,
    ax=None,
    save_path: str | Path | None = None,
) -> tuple[Figure, Axes]
```

Horizontal bar chart of per-category AP deltas (B − A), sorted by magnitude. Green bars are improvements, red bars are regressions. Shows `top_k` categories from each end.

```python
from hotcoco import compare
from hotcoco.plot import category_deltas

result = compare(ev_a, ev_b)
fig, ax = category_deltas(result, top_k=10, save_path="deltas.png")
```

---

## `report`

```python
report(
    coco_eval, *,
    save_path,
    gt_path=None,
    dt_path=None,
    title="COCO Evaluation Report",
)
```

Generate a publication-quality single-page PDF report. Requires `pip install hotcoco[plot]`.

The report contains:

- **Header** — title and timestamp
- **Run context** — GT/DT file paths, eval params, and dataset statistics (images, annotations, categories, detections)
- **Summary metrics** — AP and AR tables with a PR-curve panel and KPI tiles
- **Per-category AP** — bar chart sorted descending, three columns

The metric rows adapt automatically to the evaluation mode:

| Mode | AP rows | AR rows |
|------|---------|---------|
| `bbox` / `segm` | AP AP50 AP75 APs APm APl | AR1 AR10 AR100 ARs ARm ARl |
| `keypoints` | AP AP50 AP75 APm APl | AR AR50 AR75 ARm ARl |
| LVIS | AP AP50 AP75 APs APm APl APr APc APf | AR@300 ARs@300 ARm@300 ARl@300 |

| Parameter | Type | Description |
|-----------|------|-------------|
| `coco_eval` | `COCOeval` | Must have `run()` called first. |
| `save_path` | <code>str &#124; Path</code> | Output PDF path. |
| `gt_path` | <code>str &#124; None</code> | Ground-truth JSON path shown in the run context block. |
| `dt_path` | <code>str &#124; None</code> | Detections JSON path shown in the run context block. |
| `title` | `str` | Report title shown in the header. Default `"COCO Evaluation Report"`. |

Returns `None`. Raises on I/O error or if `run()` was not called first.

---

## Themes

Two built-in themes:

| Theme | Character |
|-------|-----------|
| `"cyanotype"` | Default. Silver-paper background, neutral chrome, 10-color palette led by Prussian blue. |
| `"cyanotype-dark"` | The same ten hues on a graphite background, lifted to hold against a dark ground. For dark notebooks, dark slides, and dark documentation. |

Pass `paper_mode=True` to set figure and axes backgrounds to white, keeping all other theme colors intact. Useful when embedding plots in LaTeX documents or PowerPoint slides. It is rejected on `"cyanotype-dark"` — a white background would erase that theme's chrome and leave unreadable text.

```python
# Academic paper
fig, ax = pr_curve(ev, paper_mode=True, save_path="pr.pdf")

# Dark ground, for slides and dark docs
fig, ax = per_category_ap(results, theme="cyanotype-dark", save_path="ap.png")
```

Use the `style()` context manager to apply a theme to your own matplotlib code:

```python
from hotcoco.plot import style

with style(paper_mode=True):
    fig, ax = plt.subplots()
    ax.plot(recall, precision)
    fig.savefig("custom.pdf")
```

## Color palette

The `cyanotype` theme constants are available for custom plots:

```python
from hotcoco.plot import SERIES_COLORS, CHROME, SEQUENTIAL, eval_colors
```

- `SERIES_COLORS` — 10 data series colors (Prussian, Madder, Olive, Viridian, Plum, Ochre, Cerulean, Vermilion, Fern, Graphite)
- `SERIES_COLORS_DARK` — the same ten, lifted for dark grounds
- `EVAL_COLORS`, `EVAL_COLORS_DARK` — the TP/FP/FN semantics, deliberately outside the series palette
- `CHROME` — non-data element colors (text, label, tick, grid, spine, background)
- `SEQUENTIAL` — 3-stop colormap for heatmaps (silver paper → Prussian → deep navy)
- `SEQUENTIAL_DARK` — the inverted ramp for dark grounds
- `CHROME_DARK` — dark-ground text/label/tick/grid/spine values

### `eval_colors`

```python
eval_colors(theme="cyanotype") -> dict[str, str]
```

Return the TP/FP/FN colors for the ground `theme` renders against — `EVAL_COLORS`
for a light theme, `EVAL_COLORS_DARK` for a dark one. Prefer this to naming either
constant directly: the two are not interchangeable, and picking the wrong one gives
an unreadable figure rather than an off-brand one, since `#B24A2E` disappears into
`#141415` and `#F0A050` fails contrast on `#F5F5F4`. Raises `ValueError` for an
unknown theme name.

```python
from hotcoco.plot import eval_colors

eval_colors("cyanotype")["fp"]       # '#B24A2E'
eval_colors("cyanotype-dark")["fp"]  # '#F0A050'
```
