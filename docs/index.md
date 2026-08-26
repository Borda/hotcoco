---
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

# hotcoco

<p class="hero-tagline">
Fast enough for every epoch, lean enough for every dataset.
</p>

<p class="hero-sub">
A drop-in replacement for pycocotools that doesn't become the bottleneck — in your training loop or at foundation model scale.
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
<strong>Your metrics, unchanged</strong>
<p>All 34 metrics match pycocotools to the limit of double precision. Your AP scores don't budge.</p>
</div>

<div class="feature-card" markdown>
<strong>More than a metric</strong>
<p>TIDE error breakdown, confusion matrix, per-category AP, confidence calibration, and publication-quality plots built in. Find out <em>why</em> your model falls short, not just by how much.</p>
</div>

<div class="feature-card" markdown>
<strong>Already works with your stack</strong>
<p><code>init_as_pycocotools()</code> patches imports in-place. Detectron2, mmdetection, RF-DETR — no code changes.</p>
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

## Not just a number

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
