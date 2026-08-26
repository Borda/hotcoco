---
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

# hotcoco

<p class="hero-tagline">
Evaluation that tells you why.
</p>

<p class="hero-sub">
A perception evaluation toolkit in pure Rust, built for Python. Detection metrics, model diagnostics, and dataset tools from one install — and a drop-in pycocotools replacement on the way in, up to 36× faster.
</p>

<div class="hero-actions" markdown>

[Get Started](getting-started/installation.md){ .md-button .md-button--primary }
[API Reference](api/coco.md){ .md-button }
[Open Notebook](https://github.com/derekallman/hotcoco/blob/main/examples/coco_evaluation_101.ipynb){ .md-button }

</div>

</div>

## Quick start

```bash
pip install hotcoco
```

=== "Python"

    ```python
    from hotcoco import COCO, COCOeval

    coco_gt = COCO("instances_val2017.json")
    coco_dt = coco_gt.load_res("detections.json")

    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.run()
    ```

=== "Drop-in replacement"

    ```python
    from hotcoco import init_as_pycocotools
    init_as_pycocotools()

    # All pycocotools imports now resolve to hotcoco
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    ```

=== "CLI"

    ```bash
    coco eval --gt instances_val2017.json --dt detections.json --iou-type bbox
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

<div class="feature-grid" markdown>

<div class="feature-card" markdown>
<strong>Eval in under a second</strong>
<p>Up to 36× faster than pycocotools. Eval goes from a bottleneck to background noise.</p>
</div>

<div class="feature-card" markdown>
<strong>Drops into your stack</strong>
<p>All 34 metrics match pycocotools to the limit of double precision. <code>init_as_pycocotools()</code> patches imports in place for Detectron2, mmdetection, and RF-DETR — no code changes.</p>
</div>

<div class="feature-card" markdown>
<strong>Answers, not just a score</strong>
<p>TIDE error attribution, confusion matrices, confidence calibration, per-image label errors. Find out <em>why</em> your model falls short, not just by how much.</p>
</div>

<div class="feature-card" markdown>
<strong>One engine underneath</strong>
<p>Similarity kernels, matchers, and metric functions are public and composable. Every family reports in one shape, with provenance saying whether a number is leaderboard-comparable.</p>
</div>

</div>

## Performance

Bbox evaluation on COCO val2017 runs in **0.14s** against 5.11s for pycocotools — 36× faster. At Objects365 scale (80k images, 1.2M detections), it finishes in 18s where pycocotools takes 721s, using half the memory.

<figure markdown>
![Grouped bar chart of evaluation wall clock for bbox, segm and keypoints across the three libraries](assets/benchmark-speed.png#only-light)
![Grouped bar chart of evaluation wall clock for bbox, segm and keypoints across the three libraries](assets/benchmark-speed-dark.png#only-dark)
<figcaption>COCO val2017, 36,781 detections. Linear axis — the bars are to scale.</figcaption>
</figure>

All 34 metrics — 12 bbox, 12 segm, 10 keypoints — match pycocotools to the limit of double precision, and a hypothesis-based fuzzer separately checks ~10,000 generated datasets.

See [Benchmarks](benchmarks.md) for the full tables, hardware, and parity verification.

## Why the model misses

AP tells you *how much* your model misses. It doesn't tell you *why*. hotcoco ships the
diagnostics that do — TIDE error attribution, confusion matrices, calibration curves, and
per-image breakdowns — as first-class outputs rather than a separate tool.

<figure markdown>
![Row-normalized confusion matrix showing which COCO categories get mistaken for each other](assets/confusion-matrix.png#only-light)
![Row-normalized confusion matrix showing which COCO categories get mistaken for each other](assets/confusion-matrix-dark.png#only-dark)
<figcaption>Which categories your model actually confuses, and how much of the loss is
background rather than a mix-up. See the <a href="guide/diagnostics/">diagnostics guide</a>.</figcaption>
</figure>

Point it at a dataset with no detections at all and it becomes a browser — an annotated
grid you can scan for labeling problems. See the [dataset browser](guide/browse.md).

## One engine

hotcoco is layered rather than monolithic. Similarity kernels and matchers live in
`primitives`, the metric math lives in `metrics`, and a family driver composes them —
`detection` today, panoptic and tracking next. Nothing is locked behind an evaluator:

```python
import numpy as np
from hotcoco import metrics

scores  = np.array([0.9, 0.8, 0.7, 0.6])
matched = np.array([True, False, True, True])

metrics.average_precision(scores, matched, num_gt=4)  # 0.6287
metrics.calibration_error(scores, matched)            # (ece, mce)
```

`COCOeval` calls those same functions, so a number you derive by hand and a number
hotcoco prints cannot drift apart. The layering is enforced, not merely intended — the
test suite fails the build on a second IoU formula, a second matcher, or a call that
crosses a layer boundary the wrong way.

Every evaluation reports in one shape. `ev.report()` returns metrics, per-class and
per-group breakdowns, plottable curves, and a `provenance` field stating whether the
number is comparable to a published leaderboard or a hotcoco extension — so code that
renders a detection report renders a panoptic or tracking one unchanged.

Detection is the family that ships today. See the
[roadmap](https://github.com/derekallman/hotcoco/blob/main/ROADMAP.md) for what follows.
