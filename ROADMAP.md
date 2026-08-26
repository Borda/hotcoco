# Roadmap

What's planned, in rough priority order. This page is forward-looking only — for
what has already shipped, see the [CHANGELOG](CHANGELOG.md).

hotcoco is a perception evaluation toolkit built as one engine with a family of metric
drivers on top. **Detection ships today** — COCO, LVIS, and Open Images over bbox,
segmentation, keypoints, and oriented boxes. **Panoptic is next, then tracking**, each a
sibling of detection on the same `primitives` → `metrics` layering, each landing as an
additive minor release. That ladder is what orders everything below.

## Near term

- **HTML evaluation report** — one self-contained HTML file per eval run:
  headline metrics, provenance, PR curves, and error impact at a glance, with
  interactive depth (sortable per-category table, hover detail) below the fold.
  `ev.report().to_html(path)` plus a CLI flag; print CSS covers casual PDF
  needs. Charts move from Plotly to vendored Observable Plot, browse's
  dashboard tab migrates to the same templates, and the matplotlib PDF report
  retires once the HTML report ships.

## Medium term

- **Panoptic segmentation (PQ)** — PQ = SQ × RQ for unified "stuff" + "things"
  evaluation, with per-class, things, and stuff breakdowns and panoptic PNG
  format support.
- **Open-vocabulary detection** — a guide to evaluating open-vocabulary
  detector outputs (Grounding DINO, OWL-ViT, YOLO-World) with hotcoco. OV-LVIS
  is federated LVIS AP over rare categories, which hotcoco already computes;
  the near-term work is documentation, with a `grounding` metric family
  (Recall@k over box–phrase matches, RefCOCO-style accuracy) to follow only if
  demand shows up. Caption metrics (CIDEr, SPICE) and model-in-the-loop metrics
  (CLIPScore, FID) stay out of scope — their values depend on a checkpoint, so
  no parity claim is possible.

## Later

- **Multi-object tracking metrics** — HOTA, MOTA, IDF1, with Track AP (TAO) as
  the natural entry point. All-or-nothing scope: a partial MOT implementation
  has no value.
- **Streaming evaluation** — chunked evaluation for datasets that don't fit in
  memory. Slots in once real users hit memory limits at Objects365/LVIS scale.
- **Ecosystem backends** — a FiftyOne evaluation backend surfacing TIDE errors
  and confusion matrices in its UI; `MeanAveragePrecision(backend="hotcoco")`
  for torchmetrics via a setuptools entry point; a Hugging Face `evaluate`
  metric module.
- **Browse enhancements** — model A/B overlay toggle, failure clustering by
  TIDE error type, PR-curve click-through, aggregate → category → image
  drill-down.
- **CrowdPose** — crowded-scene keypoint evaluation with the modified OKS
  crowd factor.
- **Per-sequence video breakdowns** — per-clip metric trends for video object
  detection.
