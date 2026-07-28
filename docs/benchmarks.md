# Benchmarks

## Feature comparison

| Feature | pycocotools | faster-coco-eval | hotcoco |
|---------|-------------|------------------|---------|
| **Installation** | Prebuilt wheels available | Prebuilt wheels available | Prebuilt wheels — `pip install` just works |
| **Metric parity** | Reference | Exact | All 34 metrics exact to float precision (≤3.7e-14) |
| **LVIS evaluation** | No | Yes — via `lvis_style=True` flag | Yes — 13 metrics, `LVISeval` class, `init_as_lvis()` |
| **TIDE error analysis** | No | No | Yes — 6 error types, ΔAP per type |
| **Confusion matrix** | No | No | Yes — cross-category, configurable threshold |
| **F-scores** | No | No | Yes — F-beta at any β |
| **Per-class AP** | Manual only | Yes — via `extended_metrics` | Built-in via `get_results(per_class=True)` |
| **Dataset operations** | No | No | Yes — filter, merge, split, sample, stats |
| **Format conversion** | No | No | Yes — COCO ↔ YOLO, VOC, CVAT, DOTA |
| **PyTorch integration** | Via torchvision | Yes — TorchVision compatible | Yes — `CocoDetection`, `CocoEvaluator` |
| **Rust API** | No | No | Yes — native crate on crates.io |
| **CLI** | No | No | Yes — `coco` (Python) + `coco-eval` (Rust) |
| **Results export** | No | No | Yes — JSON with params + metrics + per-class |
| **Memory at scale** | 24 GB committed on O365 | 30 GB committed on O365 | 8 GB committed on O365 |
| **Python versions** | 3.9+ | 3.7+ | 3.9+ |
| **License** | BSD | Apache 2.0 | MIT |

## Speed benchmarks

**Hardware:** Apple M1 MacBook Air, 16 GB RAM
**Dataset:** COCO val2017 — 5,000 images
**Detections:** 36,781 synthetic (seed=42; AP scores are not meaningful)
**Timing:** Wall clock time, single run
**Versions:** pycocotools 2.0.11, faster-coco-eval 1.7.2, hotcoco 0.3.0

### Results (1x detections)

| Eval Type | pycocotools | faster-coco-eval | hotcoco |
|-----------|-------------|------------------|-----------|
| bbox      | 9.46s | 2.45s (3.9×) | **0.41s (23.0×)** |
| segm      | 9.16s | 4.36s (2.1×) | **0.49s (18.6×)** |
| keypoints | 2.62s | 1.78s (1.5×) | **0.21s (12.7×)** |

Speedups in parentheses are vs pycocotools.

### Results (10x detections)

Scaling detections by 10x (~368,000) to test behavior under higher load:

| Eval Type | pycocotools | faster-coco-eval | hotcoco |
|-----------|-------------|------------------|-----------|
| bbox      | 34.53s | 5.72s (6.0×) | **1.91s (18.0×)** |
| segm      | 39.91s | 11.91s (3.4×) | **3.42s (11.7×)** |
| keypoints | 16.93s | 16.28s (1.0×) | **1.76s (9.6×)** |

hotcoco scales better at higher detection counts due to multi-threaded evaluation.

### Objects365 scale benchmark

**Hardware:** Windows 11, AMD Ryzen 5 5600X, 16 GB RAM + swap
**Dataset:** Objects365 val — 80,000 images, 1.2M annotations, 365 categories
**Detections:** ~1.2M synthetic bbox (capped at 100/image, seed=42)
**Timing:** Wall clock time, single run
**Versions:** pycocotools 2.0.11, faster-coco-eval 1.7.2, hotcoco 0.3.0

| Library | Time | Peak RAM | Committed | Speedup |
|---------|------|----------|-----------|---------|
| pycocotools | 721.18s | 14.34 GB | 23.71 GB | baseline |
| faster-coco-eval | 250.90s | 14.57 GB | 29.96 GB | 2.9x |
| **hotcoco** | **18.32s** | **7.47 GB** | **8.11 GB** | **39.4x** |

Peak RAM is the peak working set (physical memory). Committed includes swap — both pycocotools and faster-coco-eval exceeded physical RAM and relied heavily on the pagefile, which significantly inflated their wall clock times. hotcoco completed within physical memory with minimal swap.

## Metric parity

**Reference:** pycocotools 2.0.11, numpy 2.4.3.
**Ground truth:** COCO val2017 — 5,000 images.
**Detections:** the published `instances_val2017_fake*_results.json` files from
[ppwwyyxx/cocoapi](https://github.com/ppwwyyxx/cocoapi), the same inputs pycocotools uses in its own
tests. Both are fetched by `just download-coco` — `data/` is not checked into the repository.

**All 34 metrics agree to within 3.7e-14** — floating-point noise, the last few
bits a `f64` can represent. Diffs below are raw measured differences, not rounded.

### Bounding box

| Metric | pycocotools | hotcoco | Diff |
|--------|-------------|---------|------|
| AP     | 0.57793065 | 0.57793065 | 3.02e-14 |
| AP50   | 0.86052720 | 0.86052720 | 8.88e-16 |
| AP75   | 0.60003745 | 0.60003745 | 1.04e-14 |
| APs    | 0.32723763 | 0.32723763 | 1.57e-14 |
| APm    | 0.70684507 | 0.70684507 | 3.63e-14 |
| APl    | 0.91751661 | 0.91751661 | 1.27e-14 |
| AR1    | 0.42708926 | 0.42708926 | 3.33e-16 |
| AR10   | 0.68690535 | 0.68690535 | 6.66e-16 |
| AR100  | 0.70127765 | 0.70127765 | 2.22e-16 |
| ARs    | 0.43712612 | 0.43712612 | 0.00e+00 |
| ARm    | 0.80637778 | 0.80637778 | 3.33e-16 |
| ARl    | 0.95956720 | 0.95956720 | 5.55e-16 |

Every bbox metric agrees to within 3.7e-14 — the limit of double precision. The
threshold grids are constructed to match `numpy.linspace` bit-for-bit, which
removed the last systematic source of divergence here.

### Segmentation

| Metric | pycocotools | hotcoco | Diff |
|--------|-------------|---------|------|
| AP     | 0.65763117 | 0.65763117 | 1.11e-14 |
| AP50   | 0.92315461 | 0.92315461 | 8.77e-15 |
| AP75   | 0.70141134 | 0.70141134 | 6.55e-15 |
| APs    | 0.46056290 | 0.46056290 | 9.49e-15 |
| APm    | 0.77182113 | 0.77182113 | 1.19e-14 |
| APl    | 0.93431919 | 0.93431919 | 6.11e-15 |
| AR1    | 0.45457448 | 0.45457448 | 4.44e-16 |
| AR10   | 0.74556101 | 0.74556101 | 1.78e-15 |
| AR100  | 0.76167772 | 0.76167772 | 1.89e-15 |
| ARs    | 0.54570783 | 0.54570783 | 0.00e+00 |
| ARm    | 0.85891625 | 0.85891625 | 0.00e+00 |
| ARl    | 0.98103170 | 0.98103170 | 4.44e-16 |

Exact. Segmentation was the last family carrying a real residual (AP ~1e-5);
it came from polygon rasterization, where the reference's C compiler fuses
`s*t+ys` into a single FMA and Rust does not. Matching that arithmetic closed it.

### Keypoints

| Metric | pycocotools | hotcoco | Diff |
|--------|-------------|---------|------|
| AP     | 0.41255451 | 0.41255451 | 1.67e-15 |
| AP50   | 0.60631206 | 0.60631206 | 1.11e-16 |
| AP75   | 0.42916428 | 0.42916428 | 1.11e-16 |
| APm    | 0.40337197 | 0.40337197 | 1.39e-15 |
| APl    | 0.88304294 | 0.88304294 | 1.44e-15 |
| AR     | 0.76642003 | 0.76642003 | 1.11e-16 |
| AR50   | 0.97481108 | 0.97481108 | 0.00e+00 |
| AR75   | 0.80636020 | 0.80636020 | 0.00e+00 |
| ARm    | 0.62190658 | 0.62190658 | 0.00e+00 |
| ARl    | 0.96335935 | 0.96335935 | 0.00e+00 |

Keypoint metrics are exact. Note that keypoint evaluation reports `AR`, `AR50`, and
`AR75` — there is no small area range and no maxDets sweep, so the `AR1`/`AR10`/`AR100`
of bbox and segm do not apply.

## Methodology

- **Wall clock time** includes file I/O, evaluation, and accumulation. Excludes Python import time.
- **Detections are synthetic** — generated from GT annotations with a fixed seed (`seed=42`), so AP scores are meaningless but detection count and format are representative of real model output. Fixed seed means results are identical across runs.
- **Only detections are scaled** for the 10x benchmark — ground truth annotations are unchanged.
- Benchmark scripts are in `scripts/` at the repo root.

## Reproducing the benchmarks

You'll need the COCO val2017 annotation files and a working hotcoco build — see the [installation page](getting-started/installation.md) for setup. Then:

```bash
just bench                                        # speed benchmark (1x)
uv run python scripts/bench.py --scale 10        # 10x stress test
just parity                                       # metric parity vs pycocotools
uv run python scripts/bench_objects365.py        # O365 scale (requires O365 annotations)
```
