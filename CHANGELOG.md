# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- **Open Images group-of boxes now follow the Challenge protocol.** A group-of box
  is worth exactly one ground truth: the best-scoring detection inside it is a true
  positive, surplus detections inside it are ignored, and an undetected group-of box
  is a single false negative. Previously they were ignored entirely, contributing to
  neither the numerator nor the denominator.

  Both behaviours are real published protocols — the old one is the Open Images
  **V2** detection metric, the new one is the **Challenge** metric (2018/2019), i.e.
  TensorFlow's `group_of_weight = 1.0` and what FiftyOne implements. **OID AP values
  will change.** If you need V2 semantics, open an issue; the two differ by a single
  parameter and an enum is easy to add.

- **Open Images AP now uses VOC 2010 all-points integration.** The protocol says
  detections are "evaluated as in the PASCAL VOC 2010 protocol", and both reference
  implementations follow it — TensorFlow's `compute_average_precision` and
  FiftyOne's `_compute_AP`, the latter alongside a separate 101-point path for its
  COCO evaluation. hotcoco was reusing COCO's 101-point recall grid, which quantizes
  the result by up to ~`1/101` per class. **OID AP values change**; COCO and LVIS are
  untouched and keep the 101-point grid.

- **Open Images is now parity-checked against the TensorFlow Object Detection API.**
  `scripts/parity_oid.py` compares 70 cases — mAP *and* per-class AP — against
  frozen output from `OpenImagesDetectionEvaluator(group_of_weight=1.0)`. Worst
  difference **1.11e-16**, one ulp. Runs in CI; needs neither network nor
  TensorFlow. Regenerate fixtures with `just gen-oid-fixtures`.

  Open Images stays `Provenance::Extension`, but the warning now names the real gap:
  the challenge's non-exhaustive image-level-label rule is not implemented (it needs
  per-image label data COCO JSON cannot carry). It previously claimed no reference
  implementation existed, which was untrue.

### Fixed

- **Open Images group-of matching used IoU instead of IoA.** The protocol says a
  detection is inside a group-of box when intersection divided by the *detection's*
  area exceeds 0.5. hotcoco forced the plain-IoU formula for every OID ground truth,
  so a detection smaller than the group box — an individual object inside a cluster,
  the normal case — was never absorbed and leaked out as a false positive. An 80×80
  detection wholly inside a 200×200 group-of box scores IoA 1.00 but IoU 0.16, and
  was counted as an error.

  The tests that should have caught this used geometry below the threshold, so the
  group-of code path never ran; they passed on interpolation instead. They now assert
  the measure, and place the false positive before full recall where AP can see it.

- **Group-of tie-breaking disagreed with the reference.** When a detection sat
  inside two overlapping group-of boxes at equal IoA — which containment makes
  common, since it saturates at 1.0 — hotcoco credited the later box and left the
  earlier one permanently unmatched, converting a true positive into a miss. The
  reference's `np.argmax` takes the first maximum. The rule now lives in
  `primitives::greedy::best_above_floor` so a second caller cannot re-derive it.
  Found by the new Open Images parity check.

- **`AccumulatedEval` gained `ap_all_points`**, the exact area under the precision
  envelope, shaped and indexed like `recall`. `precision` holds that same envelope
  sampled at the 101 recall thresholds. `AccumulatedEval` is now `#[non_exhaustive]`
  so later fields are additive — done while still pre-1.0, when it is free.

- **`EvalImg` gained `gt_in_denominator`** (`gtInDenominator` in Python) and is now
  `#[non_exhaustive]`. `gt_ignore` means "excluded from matching", which for group-of
  boxes is no longer the same as "excluded from the recall denominator". Code
  computing `num_gt` should read the new field.

### Added

- **`EvalResults` carries `provenance`.** It reaches the CLI's `--json`, the PDF
  report, and `ev.results()` in Python - the artifacts users archive and come
  back to. Previously the marker existed only on `EvalReport`, which nothing that
  writes a file uses, so comparability died with the process.


- **`hotcoco.metrics` and `hotcoco.primitives` — the functional layer.** Metric
  functions you can call on plain arrays, with no evaluator, no dataset, and no COCO
  JSON:

  ```python
  from hotcoco import metrics, primitives

  ap = metrics.average_precision(scores, matched, num_gt=len(ground_truths))
  ece, mce = metrics.calibration_error(scores, matched)
  cm = metrics.confusion_matrix(gt_labels, dt_labels, num_classes=80)
  rows, cols = primitives.lsap(similarity, maximize=True)
  ```

  This is the shape `sklearn.metrics` and `torchmetrics.functional` use. Before 1.0
  every one of these lived only as a `COCOeval` method, so scoring anything that had
  not been through the COCO pipeline meant reimplementing it.

  The two namespaces split by what a function *produces*: `primitives` produces matches
  and similarities (`lsap`, `bbox_iou`, `mask_iou`), `metrics` produces numbers from
  matches (`average_precision`, `precision_recall_curve`, `calibration_curve`,
  `calibration_error`, `confusion_matrix`). `tests/architecture.rs` enforces the
  boundary: `metrics` may not import a family driver, and `primitives` may not import
  `metrics`.

  **Purely additive.** Every `COCOeval` method still exists and returns exactly what it
  did — the methods are now thin adapters that marshal `eval_imgs` into arrays and call
  these functions. Verified byte-identical on val2017 bbox/segm/keypoints plus LVIS and
  boundary fixtures, including bootstrap confidence intervals.

  Bootstrap CIs (`metrics::bootstrap::bootstrap_ci`) and greedy matching
  (`primitives::greedy::greedy_match`) are Rust-only for now — see
  [the API reference](https://derekallman.github.io/hotcoco/api/metrics/) for why.

- `hotcoco.detection` — the first metric-family namespace, exposing `COCOeval`,
  `Params`, `Hierarchy`, and `compare`. Panoptic, tracking, and concepts follow the same
  shape, so code evaluating several families reads consistently:
  `from hotcoco import detection, panoptic`.

  **Purely additive.** `hotcoco.detection.COCOeval` *is* `hotcoco.COCOeval` — the same
  object under the family name. Top-level names are permanent compatibility guarantees,
  not deprecated aliases; if you are replacing `pycocotools`, keep importing from the
  top level. The LVIS helpers stay top-level too, since they exist to mirror
  `lvis-api`'s import paths.

- `hotcoco::report::EvalReport` — the shape every metric family reports in. Carries headline `metrics`, nested `per_class` and
  `per_group` breakdowns, renderable `curves`, and the producing `params`, so a renderer
  that can draw a detection result will be able to draw a panoptic or tracking one
  unchanged.

  `Provenance` (`ParityVerified` / `Extension` / `UserComposed`) records whether numbers
  may be compared against a leaderboard. Family drivers set it and it survives
  serialization, so a deserialized report renders as what it actually is. Oriented-box
  evaluation reports `Extension`: it is a real metric, but no reference implementation
  exists for it to be standard against.

- `COCOeval.report()` in Python and `COCOeval::report()` in Rust assemble one. `curves` holds the aggregate precision-recall
  curve per IoU threshold (`pr@0.50` …), meaned over categories at `area="all"` and the
  largest `max_dets`, plus the shared `rec_thrs` axis — the slice a chart actually draws.
  The full `T×R×K×A×M` tensor (~1M floats on COCO) stays reachable via `accumulated()`
  rather than being copied into the report.

- `EvalParams` is now exported. It was `pub` and appeared as a public field of
  `EvalResults`, but its module was private, so downstream code could receive the type
  and never name it.

- `just semver` — checks the public Rust API against the last published release via
  `cargo-semver-checks`. The 1.0 reorganisation moves nearly every type between modules
  while promising the crate-root paths keep resolving, and this is the mechanical proof
  of that rather than re-reading `lib.rs` by hand.

- `primitives::greedy::coco_match_floor` — the canonical spelling of pycocotools'
  `min(t, 1 - 1e-10)` match floor, so the detection lineage has one definition of the
  clamp instead of a literal repeated at each call site.

### Changed

- **`COCOeval::results()` and the Python result dicts are now byte-stable.** They
  used `HashMap`, so three identical runs produced three different key orders and
  anything archived, hashed, or diffed in CI churned for no reason. `EvalResults`
  is `BTreeMap` throughout and the PyO3 dict builders sort before inserting, since
  Python dicts preserve insertion order. `report::EvalReport` already did this.

- **`reference_deviations()` covers four more ways a run stops being comparable.**
  It now flags non-default area-range **bounds** (not just labels), `rec_thrs`,
  `use_cats=False`, and custom `kpt_oks_sigmas`. Each changes what the metric
  means while leaving its name intact, and each previously reported
  `parity_verified`.

  The area-range case was the sharpest: the check compared labels only, and the
  Python `params.areaRng` setter deliberately *preserves* labels — so the one path
  a caller takes to redefine "small" was the one the guard could not see.

  The early return for modes without a checked reference is now an exhaustive
  `match` rather than a default-allow `!=`, so a future `EvalMode` cannot inherit
  `parity_verified` by omission.

- **`summarize()` selects the nearest IoU threshold rather than every threshold
  within 1e-9.** A params list holding two thresholds that close silently reported
  their average under the name of one of them.

- **Open Images: detections absorbed by a group-of box are now ignored, not counted
  as true positives.** This changes Open Images AP values.

  A group-of GT carries no false-negative penalty, so it never enters the recall
  denominator. Crediting every detection that overlapped one as a true positive grew
  the numerator against a denominator that could not grow, and `recall` was unbounded
  above 1.0 — measured at 4.0 on a one-image case with three detections on a single
  group-of box. Absorbed detections now score as neither true positives nor false
  positives, which is what the reference `OpenImagesChallengeEvaluator` does and what
  keeps the metric coherent.

  Open Images reports only AP@0.5, and AP was unaffected by the old behavior, so most
  users will see no change in the summary line. `AccumulatedEval.recall` is public,
  however, and was wrong for any Open Images run with a detection on a group-of box.

  Found by an auditability sweep, not by a parity test — Open Images has no reference
  implementation checked against, which is why `report()` marks it
  `Provenance::Extension`. The one test covering this path used detections at IoU 0.16
  against the group-of box and never reached it; it now asserts the absorption itself
  alongside `recall <= 1.0`.

- **`primitives` narrowed to the matching kernels; the metric functions moved to
  `metrics`.** `primitives` now holds `sim`, `greedy`, and `assign` — the three kernels
  that produce matches. `primitives::counts` moved to `metrics::counts`, because
  computing AP from match flags is scoring, not matching.

  `EvalReport` and `Provenance` live in their own top-level `hotcoco::report` — they
  are the cross-family output *contract*, a peer of `metrics` and `primitives`
  rather than a member of either, and `metrics` is free functions over arrays. The
  crate-root `hotcoco::EvalReport` and `hotcoco::Provenance` paths are unchanged.

- The detection analysis methods are now adapters over `metrics`. `COCOeval::calibration`,
  `confusion_matrix`, and `compare` used to each own their own math; the math now lives
  in `metrics::calibration`, `metrics::confusion`, and `metrics::bootstrap`, and the
  methods marshal `eval_imgs` into arrays and call it. Return types and values are
  unchanged. `bootstrap_ci` takes the statistic as a closure, so it computes intervals
  for any resampled quantity rather than only detection metric deltas.

  Detection keeps what is genuinely detection-shaped: TIDE's Cls/Loc/Both/Dupe/Bkg
  taxonomy is about box localization versus classification, and `image_diagnostics`
  reports per-image fields. Both stay in `detection`.

- `detection::metrics` (the internal `MetricDef` catalog) is now `detection::catalog`, so
  it no longer collides with the crate-level `metrics`. It was already private.

- **The Rust `eval` module is now `detection`.** `eval` was its name while detection was
  the only metric family in the crate; it is now one family beside the panoptic,
  tracking, and concepts families that follow.

  **The Python API is completely unaffected.** `hotcoco.COCOeval`, `init_as_pycocotools()`,
  and the whole `pycocotools`/LVIS drop-in surface are permanent compatibility
  guarantees.

- Two relocations for the same reason:
  - `hotcoco::hierarchy` → `hotcoco::detection::hierarchy`. The Open Images label
    hierarchy is consumed only by the detection family's GT/DT expansion, so its
    top-level placement wrongly implied it was a cross-family primitive.
  - `hotcoco::healthcheck` → `hotcoco::quality::healthcheck`, joined by `COCO::stats`
    and the `SummaryStats`/`CategoryStats`/`DatasetStats` DTOs, which moved out of
    `hotcoco::types`. Dataset quality is its own tier, distinct from the schema
    (`types` says what a COCO file may contain) and from the metrics engine
    (`detection` scores predictions against one).

- `eval/types.rs` is dissolved. It held types for four unrelated concerns plus two
  sibling features' public types, so "where is `EvalImg` defined?" answered "in a junk
  drawer" rather than "in the matching stage". Each type now lives next to the stage
  that produces it: `EvalImg`/`EvalImgContext`/`IouMatrix` → `eval/matching.rs`,
  `AccumulatedEval`/`EvalShape` → `eval/accumulate.rs`, `ConfusionMatrix` →
  `eval/confusion.rs`, `TideErrors` → `eval/tide.rs`, and `EvalMode` plus the LVIS
  `FreqGroup`/`FreqGroups` buckets → a new `eval/mode.rs`. Every public path is
  unchanged — `hotcoco::EvalImg` and `hotcoco::eval::EvalImg` both still resolve.

- The analysis layer no longer reaches into `COCOeval`'s private state. TIDE read the
  whole-dataset `ious` similarity cache directly, and `compare`/`slice`/`report` read
  `freq_groups`; both now go through driver-private accessors. `cell_ious(img_id,
  cat_id)` deliberately returns **one cell rather than the map** — the 0.5 primitives
  contract review identified that cache as the likeliest route by which retention leaks
  into a shared contract, which would foreclose the recompute-instead-of-retain lever
  the tracking family needs. A new `similarity_cache_stays_driver_private` conformance
  test in `tests/architecture.rs` fails the build if anything outside the driver touches
  the field.

- Per-image matching moved out of `eval/evaluate.rs` into a new `eval/matching.rs`, and
  the 229-line `evaluate_img_static` is now four named steps over two explicit views:
  `gather_gt` (load + apply the mode-dependent ignore rules + partition
  non-ignored-first), `gather_dt` (load + score-descending order + `max_det` cap),
  `reordered_iou` (build the flat matrix the matcher's contract expects), and
  `match_cell` (invoke `greedy_match`, translate indices back to annotation ids, run
  the Open Images group-of pass). `evaluate.rs` is now purely the outer driver —
  parameter resolution, sparse-pair collection, and the parallel fan-out — with no
  matching math of its own. Behavior unchanged.

- `eval/summarize.rs` is split by responsibility. It was 861 lines doing five unrelated
  jobs — the metric catalog, the reduction, output formatting, pipeline orchestration,
  and `f_scores` — which made "where is AP75 defined?" and "where is it computed?" the
  same unhelpful answer. Now: `eval/metrics.rs` owns the catalog (pure `MetricDef`
  configuration, no evaluation data), `eval/summarize.rs` owns only the reduction
  (accumulated arrays + definitions → numbers, 127 lines), and `eval/report.rs` owns
  presentation (`summarize_lines`, `summarize`, `metric_keys`, `get_results`,
  `f_scores`, `print_results`, `results`). `run()` moved to the `COCOeval` facade in
  `eval/mod.rs`, where the other pipeline entry points live. No public path changed and
  no behavior changed.

- TIDE's false-positive classifier is now a free `classify_fp` function over an
  explicit `FpEvidence` struct, with `ErrType` promoted out of the function body.
  The tidecv priority order (`Loc > Cls > Dupe > Bkg > Both`) is a parity contract,
  and it previously lived as an inline `else` block inside a nested loop with a
  function-local enum — so it could not be read or tested on its own. It now carries
  the priority table in its docs and has unit tests covering every variant, each
  precedence pair, and the inclusive interval bounds. Behavior is unchanged.
  `ErrType::as_str` also replaces a second, separate enumeration of the variants in
  the aggregation step.

### Removed

- **Five pre-1.0 Rust *module* paths, with no compatibility aliases.** `cargo-semver-checks`
  reports 47 removed paths, and every one is under these five:

  | Gone | Use |
  |---|---|
  | `hotcoco::eval::*` | `hotcoco::detection::*` |
  | `hotcoco::hierarchy` | `hotcoco::detection::hierarchy` |
  | `hotcoco::healthcheck` | `hotcoco::quality::healthcheck` |
  | `hotcoco::types::{SummaryStats, CategoryStats, DatasetStats}` | `hotcoco::quality::*` |
  | `hotcoco::primitives::counts` | `hotcoco::metrics::counts` |

  **No crate-root path was removed.** `hotcoco::COCOeval`, `hotcoco::EvalImg`,
  `hotcoco::Hierarchy`, `hotcoco::HealthReport`, `hotcoco::SummaryStats` and the rest
  resolve exactly as they did in 0.5.0, so code using them needs no edit — which is
  most code, since the module paths are the verbose form.

  0.x is pre-release under SemVer ("anything MAY change at any time"), so these paths
  carried no stability promise. An earlier draft of 1.0 kept them as deprecated
  aliases; that was dropped because it committed the crate to carrying a compatibility
  tier through all of 1.x and removing it at 2.0 — a multi-year obligation to a Rust
  module surface with no known consumer, when the crate-root paths already absorb the
  moves. A compile error naming the new path is a better migration experience than a
  silent alias.

  **Python is entirely unaffected** and always was: these are Rust module paths.
  `hotcoco.COCOeval`, `hotcoco.mask`, `init_as_pycocotools()`, and the
  `pycocotools`/LVIS drop-in surface are permanent.

### Fixed

- **`ev.params.imgIds = [...]` was a silent no-op.** The `params` getter cloned
  into a fresh object on every access, so pycocotools' canonical configuration
  idiom - the one in its own demo - mutated a temporary and evaluation proceeded
  over the whole dataset anyway. `params` is now a persistent object, reconciled
  with the evaluator by a single `with_params` helper that `evaluate()`, `run()`
  and `summarize()` all route through: pulled in before, pushed back after, so a
  caller reading `ev.params.imgIds` sees the resolved list as pycocotools leaves
  it.

  One helper rather than one call site, because the first attempt patched
  `evaluate()` alone and left two holes: `run()` - the path the docstring points
  LVIS/Detectron2/MMDetection users at - still ignored `params` entirely, and a
  mutation *after* `evaluate()` stayed invisible to `summarize()`, which then
  reported `parity_verified` for an off-reference configuration. That is the
  silent downgrade `reference_deviations()` exists to prevent, arriving from the
  other direction.

- **Comparability warnings are now real Python warnings.** `summarize()` writes
  them with `eprintln!`, which goes to file descriptor 2 and bypasses
  `sys.stderr` - invisible in a Jupyter cell, invisible to `capsys`, uncatchable
  by `warnings.catch_warnings`. Notebook users are the primary audience and never
  saw them. `COCOeval::reference_deviations()` is public for the same reason:
  `Provenance` is one bit, and the reason is what a caller can act on.

- **`ImageSummary.ap` had no assertion anywhere in the repo**, despite surfacing
  in `coco eval --diagnostics`, the browse viewer, and the dashboard. Now pinned
  by closed-form cases, including the `n_gt == 0` convention that is deliberately
  the opposite of TIDE's.

- **`load_res()` rejects NaN detection scores.** Ranking sorts with
  `partial_cmp(..).unwrap_or(Equal)`, which is not transitive once NaN is present:
  the sort does not panic, it produces an arbitrary order, and AP becomes a
  function of the sort implementation. Loading from a *file* was already safe —
  `sanitize_non_finite` rewrites bare `NaN` to `null` — so this closes the
  programmatic path.

- **`compare()` paired per-category results by position instead of category id.**
  When two evaluators carried different category lists — different GT files, or the
  same file filtered differently — category *i* of model A was compared against
  whatever model B evaluated in slot *i*, and reported under A's name. `compare()`
  validates `eval_mode` and `iou_type` but never the category lists, so nothing
  caught it. Both sides are now keyed by category id, and the iteration covers the
  union so a category only one side evaluated is reported rather than silently
  dropped.

  Per-category deltas also treated a category missing from one side as a swing of up
  to 1.0 rather than as no evidence, so a category absent from B sorted to the top of
  the "worst regressions first" table. They now use the same `metric_delta` the
  summary metrics and bootstrap CIs use.

  Invisible to the previous tests, which all compared a model to itself — the case
  where positional and id-keyed lookup are indistinguishable.

- **`COCOeval::calibration()` now rejects detection scores outside `[0, 1]`.**
  Previously they were silently accepted and produced a meaningless result.

  Binning buckets by `score * n_bins` and clamps the *index*, not the score, so an
  out-of-range value saturates into an end bin and carries its raw magnitude into
  that bin's mean — a model exporting raw logits got a calibration error above 1.0
  with nothing to indicate why. `[0, 1]` was already a documented precondition;
  it is now enforced, with an error naming the offending score, how many
  detections are affected, and what to do about it.

- **Polygon rasterization now reproduces the reference's floating-point
  arithmetic, making segmentation parity exact.** All 12 segmentation metrics moved
  from ~1e-5 to ~1e-14 on COCO val2017.

  `maskApi.c` writes `(int)(ys+s*t+.5)`, and both clang and gcc default to
  `-ffp-contract=fast` — so every shipped pycocotools wheel fuses `s*t+ys` into a
  single FMA, one rounding where the unfused form has two. Rust never contracts
  implicitly, so the plain expression was a *more accurate* computation that
  disagreed with the reference. `mask::fr_poly` now uses `f64::mul_add`.

  It bites only at a boundary — with `s = -5/6`, `t = 57`, `ys = 75` the product
  lands a hair either side of `-47.5`, so the two forms round to 28 and 27 and the
  polygon differs by one pixel. Two of 400 random polygons hit it. But COCO
  ground-truth segmentations *are* polygons, so this path builds every segm GT
  mask, and one pixel was the whole residual.

- **`mask.frPyObjects` produced empty masks for bounding-box input.** It routed
  every coordinate list to the polygon rasterizer, which returns an empty RLE below
  three points — so a 4-element box was read as a 2-point polygon and rasterized to
  nothing, silently. It now dispatches on entry length the way pycocotools does:
  exactly 4 values is a box, more than 4 is a flattened polygon. Entries shorter
  than 4 raise instead of returning a blank mask.

  hotcoco accepts boxes as either a list of lists or a numpy array; pycocotools
  requires an array and raises `TypeError` on the list form. Being more permissive
  is safe — nothing that worked against pycocotools changes.

- **`mask.iou` and `mask.bbox_iou` rejected integer `iscrowd`.** COCO JSON stores
  `iscrowd` as `0`/`1`, so the idiomatic
  `maskUtils.iou(dt, gt, [a["iscrowd"] for a in anns])` — and any numpy array —
  raised `TypeError` against a `Vec<bool>` parameter. Both now accept bools, ints,
  and numpy arrays of either, matching what the crate already does when
  deserializing annotations. Non-numeric entries still raise, naming the type.

- **The adversarial corpus is tracked and actually runs.** 18 hand-curated edge
  cases (all-crowd, area boundaries, 1x1 images, boxes outside the frame) were
  gitignored, so they verified nothing on any machine but the author's, and
  nothing iterated them. `scripts/test_adversarial.py` runs them under pytest and
  in CI, comparing per-(image, category) matching *decisions* against
  pycocotools annotation by annotation — a check metrics cannot replace, since
  two detections swapped between images can leave AP identical to fifteen decimal
  places. Level 2 also runs unconditionally now; it used to be gated on level 1
  having already failed, which made it unreachable in exactly the case it is good
  at.

- **A pinned val2017 baseline** (`scripts/fixtures/val2017_expected.json`),
  produced by pycocotools rather than by hotcoco. `parity.py` checks it alongside
  the live comparison, which catches what the live run structurally cannot:
  hotcoco and the reference drifting *together*, as a pycocotools upgrade that
  silently changed a metric would.

- **The fuzzer checks invariants, not only parity.** It was purely differential,
  so its ~10,000 generated datasets only ever exercised surfaces pycocotools also
  computes — leaving Open Images, oriented boxes, LVIS frequency groups, TIDE,
  calibration, and the confusion matrix with no fuzz coverage at all, which are
  precisely the surfaces marked `Provenance::Extension` because no reference
  exists.

- **New: `scripts/parity_mask.py`,** a differential test of every `hotcoco.mask`
  operation against `pycocotools.mask` — encode, decode, round-trip, area, toBbox,
  iou across crowd modes, merge, the string codec, and `frPyObjects` for both
  polygons and boxes. Compared bit-for-bit, since RLE is integer run lengths and
  masks are `uint8`. The RLE codec was previously covered only transitively through
  segmentation AP, which never reached `merge`, `frPyObjects`, or the string codec —
  all three of the bugs above were in that gap.

- **The default IoU and recall grids now match `numpy.linspace` bit-for-bit.**
  This shifts metric values in the last few decimal places, closer to
  pycocotools.

  pycocotools builds both grids with `np.linspace`; hotcoco built them as
  `0.5 + 0.05 * i` and `i / 100.0`. Those disagree with numpy by one ulp at 2 of
  the 10 IoU thresholds and 10 of the 101 recall thresholds, because numpy
  computes a single `step` once and multiplies where the other forms round twice
  or divide exactly.

  One ulp is not harmless on the recall grid. `recall = tp / num_gt` is a ratio of
  small integers, so it lands exactly on a grid point routinely — at
  `num_gt = 20, tp = 7` it equalled the old `rec_thrs[35]` bit-for-bit while
  sitting strictly below numpy's, so the two-pointer scan stopped one detection
  early and reported a different precision there. Neither grid is more *correct*,
  which is exactly why matching the reference is free. `params::linspace` is now
  the single owner of both, pinned against captured numpy bit patterns.

- **Property tests over `primitives` and `metrics`.** Randomised coverage of the
  matcher contract (injectivity, output agreement, threshold clearance, phase-2
  eligibility), bbox IoU algebra, PR-curve well-formedness, `f_beta` bounds,
  confusion marginals, and calibration binning — roughly 80k generated cases,
  each verified to fail against an injected violation. These check hotcoco
  against its own stated contracts rather than against a reference, which is the
  only kind of check available on Open Images and oriented boxes, where no
  reference exists. The Open Images recall bug above was found this way.

- **Corrected the `primitives::greedy` note on exact-duplicate matching.** It
  claimed identical geometry yields exactly `1.0`, and concluded that unclamped
  callers therefore still match duplicates at `t == 1.0`. The intersection extent
  is computed as `(x + w) - x`, which does not round-trip to `w` in binary
  floating point, so a box against itself gives `0.9999999999999993`. pycocotools
  computes it the same way — the kernel is right and unchanged; the note was
  wrong, and backwards: the clamp is what makes duplicate matching work at
  `t == 1.0`, not a redundancy. For sub-pixel geometry even the clamp is not
  enough. Both regimes are now pinned by property tests.

- **`from lvis import LVISEval` now works after `init_as_lvis()`.** lvis-api spells the
  class `LVISEval` with a capital E, and that is what Detectron2 and MMDetection import.
  hotcoco exported only `LVISeval`, following pycocotools' `COCOeval` — so `init_as_lvis()`
  registered a `lvis` module that the canonical import could not use. Both spellings now
  exist and are the same object. Found by the 1.0 third-party-consumer smoke test, which
  is now a regression test.

- `scripts/parity_tide.py` can now fail. It exited 0 in both failure modes: when a ΔAP
  exceeded tolerance it printed "Some values exceed tolerance" and fell through, and
  when `tidecv` was absent it printed a skip notice and exited 0 after comparing
  hotcoco's numbers against nothing. Both now exit non-zero, and `tidecv` joins the dev
  extras beside the other reference implementations so it is installed rather than
  silently missing.

- `import hotcoco.mask` now works. It raised `ModuleNotFoundError` while
  `from hotcoco import mask` succeeded, because PyO3's `add_submodule` makes a submodule
  reachable as an attribute without registering it in `sys.modules`. Anyone migrating
  from `import pycocotools.mask` writes the failing form — and `init_as_pycocotools()`
  masked the problem, since it registers `pycocotools.mask` explicitly and so always
  worked.

- Detection matching now applies pycocotools' match floor. pycocotools starts each
  detection's search at `min(t, 1 - 1e-10)` rather than at `t`; hotcoco compared
  against the raw threshold, so a detection whose IoU fell in `[1 - 1e-10, 1.0)`
  matched in pycocotools but not in hotcoco. The clamp is inert for every threshold
  below 1.0, so the default `0.50:0.05:0.95` sweep and every published metric are
  unchanged — it is observable only at `iou_thr == 1.0`. On COCO val2017 at
  `iou_thr = 1.0` this closed a real divergence: 7,688 match decisions before,
  13,724 after, which is exactly pycocotools' count. AP is unaffected on that data
  because the affected pairs sit in ignored partitions, but on non-ignored geometry
  the difference is material (AP 0.25 → 1.0 on a two-image regression fixture).

  The policy is now settled and documented as a table in `primitives::greedy`:
  the detection `evaluate()` path (including the Open Images group-of pass) clamps;
  TIDE does not, because its parity contract is *tidecv* rather than pycocotools;
  and the confusion matrix, per-image diagnostics, and calibration do not, because
  they are hotcoco-native analysis over a user-chosen threshold. `greedy_match`
  itself still adds no epsilon of its own — the clamp remains caller-applied.

## [0.5.0] - 2026-07-26

### Added

- `COCOeval.eval` now includes the `params` and `date` keys, matching pycocotools'
  dict exactly (`params`, `counts`, `date`, `precision`, `recall`, `scores` in that
  order). `params` is the `Params` object used for evaluation; `date` uses
  pycocotools' `'%Y-%m-%d %H:%M:%S'` format. Closes a Tier-1 drop-in gap vs
  pycocotools. The numeric arrays are unchanged, so all metrics still match.
- `primitives::sim` now owns every similarity kernel: the `bbox_iou`, `mask_iou`, and
  `obb_iou` matrix kernels moved here from `mask`/`geometry`, joining `oks_matrix`.
  A new scalar `bbox_iou_pair` is the single definition of single-pair bbox IoU.
  `hotcoco::mask::iou`, `hotcoco::mask::bbox_iou`, and `hotcoco::geometry::obb_iou`
  still resolve — they are now re-exports, so the `pycocotools.mask` drop-in surface
  is unchanged.
- `primitives::counts::average_precision` — the mechanical AP core (sort → classify →
  cumsum → interpolate → mean), now shared by TIDE and per-image diagnostics.
- `SimKind` gained `From<IouType>`, so detection dispatches on the geometry axis.
- `coco-eval` gained subcommands: `coco-eval eval …` and `coco-eval completions <shell>`.
  The bare form (`coco-eval --gt … --dt …`) is unchanged and still supported.
- Architecture conformance tests (`crates/hotcoco/tests/architecture.rs`) that fail the
  build if the IoU formula, the parallelism threshold, or greedy matching is duplicated
  outside `primitives/`.
- `[tool.pyright]` config for the Python package, scoped to `python/` (excluding the
  untyped `scripts/` and vendored `external/`) at `basic` strictness with
  `pythonVersion = "3.9"` to match ruff's `target-version`. It resolves the compiled
  `hotcoco` extension through `.venv`, so it type-checks call sites against the
  hand-written `__init__.pyi` — catching the signature drift that
  `scripts/test_stubs.py` cannot see, since that test checks name coverage only.
- Python lint is now enforced instead of merely available. The pre-commit hook runs
  `ruff format --check` and `ruff check` whenever Python files are staged (and fails
  loudly if `uv` is missing rather than skipping), and CI gained a `python-lint` job.
  `just py-lint` and `just py-fmt-check` had existed for a while but gated nothing,
  which is how the tree accumulated 48 ruff errors.
- `just setup` now installs the `rust-analyzer` rustup component. Because the toolchain
  is pinned and the component was never installed for the pinned version, editors and
  LSP clients had no Rust code intelligence in this repo — silently, because
  `~/.cargo/bin/rust-analyzer` is a rustup proxy: `which` resolved it while every spawn
  failed with "Unknown binary". It is deliberately *not* listed in `rust-toolchain.toml`,
  since CI installs that file's components and the setup action's `components:` input
  only adds and cannot subtract — listing it there would make all five CI jobs download
  an editor backend they never use. Re-run `just setup` after a channel bump; switching
  channels drops any component not in the toolchain file.

### Changed

- Upgraded `pyo3` and `numpy` from 0.28 to 0.29, which clears the RUSTSEC-2026-0176 (OOB read in `PyList`/`PyTuple` iterators) and RUSTSEC-2026-0177 (missing `Sync` bound on `PyCFunction::new_closure`) security advisories. The corresponding `deny.toml` ignores have been removed. No public Python API changes; the binding sources compiled unchanged against the 0.29 API.
- The confusion matrix now uses the shared `primitives::greedy::greedy_match` instead of
  its own greedy loop, so every matcher in the crate shares one tie-breaking rule.
  Verified byte-identical on val2017 across 160 threshold/max-det/min-score
  configurations, including at the match boundary.
- `primitives::greedy` documents the pycocotools threshold-epsilon clamp
  (`min(t, 1 - 1e-10)`) as **caller-owned**. It is inert below `t = 1.0` and no caller
  applies it today, so behavior is unchanged; the difference at `t = 1.0` is now
  written down rather than implicit.
- The release workflow only triggers on true version tags (`v[0-9]+.[0-9]+.[0-9]+*`),
  and `cargo publish` failures now fail the release instead of being downgraded to a
  warning — only an already-published version is tolerated.
- The Python tree is now clean under `just py-lint` and `just py-fmt-check`, which had
  drifted to 48 ruff errors across 13 unformatted files because neither the pre-commit
  hook nor CI runs them. Beyond formatting, this removed five unused imports, three dead
  local variables, and one unresolvable annotation (`plot/core.py` annotated a return as
  `"np.ndarray"` while importing numpy only inside the function body — now declared
  under `TYPE_CHECKING`, so the annotation resolves without making numpy a hard import).
- The dashboard confusion matrix shows the raw count alongside the normalized rate in
  its hover, completing what the code already intended — the raw matrix was being read
  and discarded under a comment reading "Hover text with counts". A rate alone cannot
  distinguish one stray detection from a systematic confusion.
- `ruff` is pinned to `>=0.15,<0.16` instead of `>=0.4`. Lint results are now a gate
  (pre-commit and CI), so an unconstrained ruff could fail a PR that changed no code.

### Fixed

- `shapely` was never declared as a dependency, so `scripts/fuzz_obb_parity.py` — the
  OBB IoU fuzz harness the `/parity` skill offers on request — failed at collection with
  `ModuleNotFoundError` for anyone whose venv did not happen to have it. It is now in the
  `dev` extra, and all 9 OBB parity tests pass.
- `import hotcoco` raised `TypeError: unsupported operand type(s) for |` on Python 3.9,
  the oldest version declared by `requires-python` and served by the single `abi3-py39`
  wheel. `python/hotcoco/__init__.py` and `python/hotcoco/_style.py` used PEP 604
  `X | None` unions in annotations that Python evaluates at runtime (function
  signatures, and an annotated attribute assignment), which requires 3.10. Both files
  now carry `from __future__ import annotations`, so the annotations are never
  evaluated. Reproduced and verified fixed on CPython 3.9.6.
- CI now runs the Python smoke test on a `["3.9", "3.12"]` matrix instead of 3.12 only.
  The declared support floor had never been exercised, which is why the import failure
  above shipped.
- Bumped `crossbeam-epoch` (→0.9.20), `rand` (→0.9.5), and `quick-xml` (→0.41) to clear RUSTSEC-2026-0204, -0097, -0194, and -0195 security advisories.
- `coco-eval --completions <shell>` works standalone. It previously required `--gt` and
  `--dt`, which clap validated before the completions branch ran — so the flag could
  never generate a completion script on its own.
- `MIN_PARALLEL_WORK`, the threshold at which IoU kernels switch to rayon, was defined
  twice with different values (1024 in `mask`, 1000 in `geometry`) under a comment
  claiming they matched. There is now one constant.
- The internal path-dependency pins in `hotcoco-cli` and `hotcoco-pyo3` had lagged at
  `0.4.0` since the 0.4.1 release. crates.io resolves these for published builds, so a
  stale pin publishes against an older API than was tested. They now live in
  `[workspace.dependencies]` next to `[workspace.package] version`, and the members
  inherit them — so the mismatch is visible in one file instead of hidden in two.
- `primitives::assign::lsap` reported an infeasible cost matrix (a row with no finite
  entry) as a bare index-out-of-bounds panic in release builds, because the check was a
  `debug_assert!`. It now asserts with a message naming the cause, as scipy does.
- Six README links pointed at documentation pages that do not exist and returned 404:
  TIDE errors, confusion matrix, F-scores, and logging metrics are sections of
  `guide/evaluation/`; format conversion is a section of `guide/datasets/`; the PyTorch
  integrations page is at `api/integrations/`.
- The Rust install snippet in `docs/getting-started/installation.md` still suggested
  `hotcoco = "0.3"`.
- The Objects365 sentence in `README.md` gave `39×` and `14×` in parentheses two lines
  after stating that parenthesized speedups are versus pycocotools, though the `14×` is
  versus faster-coco-eval. Both figures are unchanged; the baselines are now named.

## [0.4.1] - 2026-07-23

### Fixed

- COCO JSON loading now tolerates non-standard `NaN`, `Infinity`, and `-Infinity` float tokens. Python's `json` module emits and accepts these by default, so files produced by pycocotools/numpy pipelines frequently contain them even though they are invalid JSON; serde_json previously rejected such files with an `expected value` error. `COCO::new` and `load_res` now normalize non-finite tokens to `null` (matching serde's own float serialization), which becomes `None` on `Option<f64>` fields such as `area` and `score`. String values that merely contain the substrings `NaN`/`Infinity` are left untouched. As part of this, `COCO::new` reads the file via `serde_json::from_slice` instead of `from_reader`.

## [0.4.0] - 2026-04-06

### Added

- CVAT for Images 1.1 format conversion — `COCO.to_cvat(output_path)` exports to a single CVAT XML file with `<box>` and `<polygon>` elements; `COCO.from_cvat(cvat_path)` imports CVAT XML back to COCO format with polygon segmentation support (shoelace area, bbox from vertex extents); `coco convert --from coco --to cvat` and `--from cvat --to coco` CLI support; `<polyline>`, `<points>`, `<cuboid>` elements skipped
- Pascal VOC format conversion — `COCO.to_voc(output_dir)` exports to VOC XML annotations (`Annotations/*.xml` + `labels.txt`); `COCO.from_voc(voc_dir)` imports VOC XML back to COCO format; `coco convert --from coco --to voc` and `--from voc --to coco` CLI support; COCO `iscrowd` maps to VOC `<difficult>`, VOC `<difficult>` dropped on import; integer-pixel round-trip within ≤1px
- Confidence calibration analysis — `COCOeval.calibration(n_bins=10, iou_threshold=0.5)` computes Expected Calibration Error (ECE), Maximum Calibration Error (MCE), per-bin accuracy vs confidence breakdown, and per-category ECE; measures how well predicted confidence scores align with actual detection accuracy
- `coco eval --calibration` CLI flag with `--cal-bins` and `--cal-iou-thr` options; formatted table output with per-bin breakdown and top-10 worst-calibrated categories; included in `--json` output
- `hotcoco.plot.reliability_diagram()` — matplotlib reliability diagram showing predicted confidence vs actual accuracy per bin, with perfect calibration diagonal, gap overlay, and ECE/MCE annotation; accepts either a calibration dict or COCOeval instance
- `CalibrationResult` and `CalibrationBin` Rust types exported from crate root
- `coco.browse()` dataset browser rewritten: replaced Gradio with FastAPI + HTMX + Jinja2 + vanilla JS Canvas; sidebar with multi-select category filter and shuffle; infinite-scroll thumbnail grid with server-side annotated thumbnails; lightbox with full-resolution canvas overlay for bbox/segmentation/keypoint annotations; hover-to-highlight syncs canvas and annotation sidebar; scroll-to-zoom and drag-to-pan; keyboard navigation (arrow keys, Escape); responsive layout adapts from 400px to 1400px+ viewports; works inline in Jupyter IFrames
- `coco.browse(dt=...)` — detection overlay: GT solid bboxes, DT dashed bboxes, confidence scores on labels; Sources toggle (GT/DT) and Min Score slider for filtering detections
- `coco explore --dt <results.json>` CLI flag — enables detection overlay from the command line
- Eval-aware browse: `coco.browse(dt=..., iou_type="bbox", iou_thr=0.5)` auto-runs evaluation and colors detections as TP (green), FP (red), FN (blue); "Category | Eval" toggle switches between standard and eval coloring; hover highlights matched DT↔GT pairs with connecting line; eval badges on annotation sidebar items
- `coco.browse(eval=coco_eval)` — accept pre-computed `COCOeval` for advanced users who want custom evaluation settings
- `coco.browse(slices={"daytime": [1,2,3], ...})` — sliced browsing with per-slice AP display; accepts dict or path to JSON file
- `coco explore --iou-type`, `--iou-thr`, `--no-eval`, `--slices` CLI flags for eval-aware browsing
- Gallery eval badges: TP/FP/FN count chips on thumbnail cards when eval data is available
- Gallery eval sorting: "Worst first", "Most FP", "Most FN" sort options; eval filter: "Has FP", "Has FN", "Has errors", "Perfect only"
- Interactive IoU threshold slider in sidebar — re-indexes cached eval data without re-evaluation
- Category hierarchy tree view: supercategory groupings with expand/collapse, group-level check/uncheck, keyboard navigation; flat/tree toggle persisted in localStorage
- `python/hotcoco/eval_index.py` — `build_eval_index(coco_eval, iou_thr)` extracts per-annotation TP/FP/FN status from `eval_imgs` at any IoU threshold
- `coco.browse(port=7860)` — new `port` parameter for custom server port
- `python/hotcoco/server.py` — new FastAPI server module with `create_app()`, `run_server()`, and `start_server_background()` for Jupyter
- `python/hotcoco/static/` — new static assets: `style.css` (responsive dark theme), `overlay.js` (Canvas annotation renderer), `htmx.min.js` (vendored HTMX 2.0.4)
- `python/hotcoco/templates/` — new Jinja2 templates: `base.html`, `index.html`, `partials/gallery.html`, `partials/detail.html`
- `COCO(annotation_file, image_dir=...)` — new `image_dir` constructor arg and settable attribute; propagated through `filter`, `split`, `sample`, and `load_res`
- `--json` flag on every `coco` subcommand (`eval`, `stats`, `healthcheck`, `filter`, `merge`, `split`, `sample`, `convert`) — writes a single JSON object to stdout; intended for CI/CD pipelines, dashboards, and shell scripts; stderr and exit codes are unchanged; errors also emit JSON when the flag is active
- `coco eval --json` suppresses the Rust-side metrics table (via fd-level stdout redirect) and returns `{metrics, params, tide?, slices?, healthcheck?}` — optional keys only present when their flags are passed
- `docs/cli.md` — new "JSON output mode" section with CI gating example and JSON error format; `--json` row added to every subcommand flags table; JSON output shape documented for `eval`
- Model comparison — `hotcoco.compare(eval_a, eval_b)` computes per-metric deltas, per-category AP differences, and optional bootstrap confidence intervals for statistical significance; `n_bootstrap` resamples images with replacement and re-accumulates in parallel (rayon); result includes `metric_keys`, `metrics_a`, `metrics_b`, `deltas`, `ci`, `per_category`
- `COCOeval.metric_keys()` — returns metric names in canonical display order for the current evaluation mode; single source of truth for metric ordering across all Python consumers
- `coco compare` CLI subcommand — `--gt`, `--dt-a`, `--dt-b`, `--bootstrap`, `--seed`, `--confidence`, `--name-a`, `--name-b`, `--json` flags; formatted table with deltas, CIs, and per-category breakdown
- `hotcoco.plot.comparison_bar()` — grouped bar chart comparing two models, with CI error bars from bootstrap
- `hotcoco.plot.category_deltas()` — horizontal bar chart of per-category AP deltas sorted by magnitude (green=improvement, red=regression)
- `ComparisonResult`, `CompareOpts`, `BootstrapCI`, `CategoryDelta` Rust types exported from crate root
- Per-image diagnostics & label error detection — `COCOeval.image_diagnostics(iou_thr=0.5, score_thr=0.5)` computes per-annotation TP/FP/FN classification, per-image F1 and AP scores, error profiles, and automatically flags suspected label errors (wrong_label: cross-category GT mislabels; missing_annotation: high-confidence FPs with no nearby GT)
- `coco eval --diagnostics` CLI flag with `--diag-iou-thr` and `--diag-score-thr` options; compact summary output with F1 distribution and top label error categories
- `ImageDiagnostics`, `AnnotationIndex`, `ImageSummary`, `LabelError`, `DtStatus`, `GtStatus`, `ErrorProfile`, `LabelErrorType` Rust types exported from crate root
- Eval dashboard — `/dashboard` route in browse server with KPI tiles (AP, AP50, AP75, AR100), IoU-sweep PR curves, per-category AP leaderboard with expand/collapse, confusion matrix heatmap (click cell → gallery), TIDE error breakdown, calibration reliability diagram (ECE/MCE), per-image F1 histogram colored by error profile, and suspected label errors table (click row → image); all charts Plotly.js with dark theme matching browse UI; Gallery↔Dashboard nav pills in sidebar; dashboard data cached after first compute
- `python/hotcoco/dashboard.py` — Plotly chart generation module: `build_dashboard()`, `kpi_tiles()`, `chart_pr_curves()`, `chart_per_category_ap()`, `chart_confusion_matrix()`, `chart_tide_errors()`, `chart_calibration()`, `chart_f1_distribution()`, `label_errors_table()`
- `python/hotcoco/templates/dashboard.html` — dashboard template with sidebar metadata, KPI row, chart grid, and label errors table
- Dashboard responsive layout — 5 breakpoints (1400px max-width, 1000px single-column charts, 768px toolbar mode with inline metadata, 480px compact with hidden chart hints and 3-column TIDE, 350px+ ultra-narrow); matches gallery responsive behavior at all widths
- Browse UI: HTMX loading spinner on gallery container (`hx-indicator`); error toast for failed HTMX requests (`htmx:responseError`, `htmx:sendError`) with 4-second auto-dismiss
- Browse UI: `@media (prefers-reduced-motion: reduce)` — disables all CSS animations and transitions for users with motion sensitivity preferences
- Browse UI: `:focus-visible` ring on custom range slider thumbs for keyboard accessibility
- Browse UI: `title` attributes on eval badges in annotation sidebar ("True positive", "False positive", "False negative") for screen reader and tooltip accessibility
- Oriented bounding box (OBB) evaluation — `IouType::Obb` / `iou_type="obb"` for rotated detection tasks (aerial imagery, document analysis, scene text); `Annotation.obb` field as `[cx, cy, w, h, angle]` (radians); rotated IoU via Sutherland-Hodgman polygon clipping in new `geometry` module; same 12 AP/AR metrics as bbox; `load_res` auto-computes `area` and axis-aligned `bbox` from OBB
- DOTA format conversion — `coco_to_dota()` exports OBB annotations to DOTA text format (one `.txt` per image, 8-point polygon corners); `dota_to_coco()` imports DOTA text files with auto-category discovery and corner-to-OBB reconstruction; `DotaStats` type exported from crate root
- `crates/hotcoco/src/geometry.rs` — new computational geometry module with `obb_to_corners()`, `obb_iou()` (rayon-parallelized D×G matrix), and Sutherland-Hodgman polygon clipping internals
- OBB visualization in browse — rotated rectangle overlays on canvas (lightbox) and PIL thumbnails (gallery); OBB-aware hover hit-testing via point-in-convex-polygon; eval coloring (TP/FP/FN) and match connector lines work with OBB annotations; dashed outlines for DT, solid for GT
- `scripts/fuzz_obb_parity.py` — hypothesis-based OBB IoU parity fuzzer using Shapely (GEOS) as reference implementation; 200 random cases + 8 deterministic known-value tests
- `python/hotcoco/_style.py` — new zero-dependency terminal styling module with `green()`, `red()`, `yellow()`, `dim()` color helpers, `status()` / `error()` / `warning()` output helpers, `Timer` context manager, and `Spinner` (delayed-start braille spinner, 100ms threshold, no-op in non-TTY/Jupyter)
- `COCOeval.summarize_lines()` (Rust) / `ev.summary_lines()` (Python) — returns metric summary as `Vec<String>` / `list[str]` without printing; `summarize()` now delegates to it
- Styled CLI output — all `coco` subcommands show colored status lines on stderr (green action verbs, dimmed file paths and timing); `NO_COLOR` env var respected; color works in Jupyter, spinners disabled in non-TTY
- Rust CLI (`coco-eval`) — `anstyle`/`anstream` colored output, `indicatif` braille spinners, elapsed timing on all operations; uses `summarize_lines()` for metrics output
- `_style.section(title, params)` — prints a section header with green title and dim `(params)`; used by all analysis outputs (TIDE, calibration, diagnostics, slices, compare)
- `_table(columns, rows, footer)` helper in `cli.py` — auto-aligned table with `─` separators; replaces 5 hand-built table implementations
- Type stubs — `python/hotcoco/__init__.pyi` and `py.typed` marker ship with the package; full coverage of COCO, COCOeval, Params, Hierarchy, mask, and compare APIs; enables autocomplete and type checking in VS Code, PyCharm, etc.
- `scripts/test_stubs.py` — stub coverage test verifying every public name in the runtime module has a corresponding stub entry; wired into `just test`
- `text_signature` on all mask functions and `compare()` — `help()` and IPython `?` now show real parameter names instead of `(*args, **kwargs)`
- `deny.toml` — cargo-deny configuration for dependency auditing (security advisories, license compliance, duplicate crate detection)
- Cold Brew design system — unified visual theme across all surfaces: browse UI (`style.css`), docs site (`extra.css`, `zensical.toml`), matplotlib plots (`theme.py`), Plotly dashboard (`dashboard.py`); canonical spec at `.claude/skills/theme-factory/themes/cold-brew.md`
- `"cold-brew"` matplotlib theme — 10-color infographic-optimized chart palette (warm/cool alternation), espresso-cream chrome, DM Sans bundled font family; added as default for all `hotcoco.plot` functions
- `python/hotcoco/_fonts/` — bundled DM Sans static font instances (Regular 400, Medium 500, Bold 700) extracted from variable font; auto-registered by matplotlib on import
- `hotcoco.plot.reliability_diagram()` — gap bars now render in both directions: solid fill for overconfident bins (accuracy < confidence), diagonal hatching for underconfident bins (accuracy > confidence)
- Title/subtitle positioning — `_place_title_and_subtitle()` helper computes dynamic figure-fraction spacing from actual figure height; reserves layout rect so titles never overlap axes on any figure size
- `Rle::new(h, w, counts)` constructor — validates that counts sum to `h * w` via `debug_assert` (zero overhead in release builds); provides a safe construction path for external callers

### Changed

- Default matplotlib theme changed from `"warm-slate"` to `"cold-brew"` in all `hotcoco.plot` functions and `style()` context manager
- Browse UI accent shifted from warm caramel (`#d4a574`) to Dusty Steel (`#8694A8`); background surfaces updated to espresso tones; GT badge uses Dusty Clay (`#A8806E`), DT badge uses Dusty Steel
- Dashboard Plotly colorway updated to 10-color infographic palette; confusion matrix midpoint shifted from gold to steel
- Docs site fonts changed to DM Sans (body) and JetBrains Mono (code); accent colors updated to Dusty Steel / Deep Steel
- `zensical.toml` font stack updated: `text = "DM Sans"`, `code = "JetBrains Mono"`
- Pre-commit hook clippy step changed from `-p hotcoco -p hotcoco-cli` to `--workspace`, matching CI; removed redundant `cargo check -p hotcoco-pyo3` step (now covered by workspace clippy); hook reduced from 4 steps to 3
- Rust edition 2021 → 2024, MSRV 1.74 → 1.85, workspace resolver 2 → 3
- PyO3 0.23 → 0.28, numpy crate 0.23 → 0.28 — `PyObject` → `Py<PyAny>`, `allow_threads` → `detach`, `downcast` → `cast`, `#[pyclass(from_py_object)]` for Clone types
- `[workspace.lints]` — centralized clippy/rust lint configuration across all 3 crates; `unsafe_code = "forbid"`, `unwrap_used = "warn"`, `dbg_macro = "deny"`, `clippy::pedantic` with targeted allows
- Cargo profiles — `dist-release` (LTO + codegen-units=1 + strip) for PyPI wheels, `profiling` (release + debug symbols) for flamegraphs, `opt-level = 1` for dev/test profiles
- GIL released during heavy computation — `py.detach()` wraps `evaluate()`, `accumulate()`, `run()`, `confusion_matrix()`, `tide_errors()`, `calibration()`, `slice_by()`, `image_diagnostics()`, `compare()`; enables concurrent Python workloads
- Library `.unwrap()` calls replaced with `.expect()` or safe alternatives; `#[allow(clippy::unwrap_used)]` in test modules only

- `crates/hotcoco/src/convert.rs` split into `convert/mod.rs` (shared types) + `convert/yolo.rs` + `convert/voc.rs` + `convert/cvat.rs` submodules; public API unchanged
- `quick-xml` 0.37 added as dependency for XML read/write in VOC conversion
- `ConvertError` gains `XmlError(String)` variant for XML parsing/writing failures
- `coco convert` CLI `--from`/`--to` choices extended from `{coco, yolo}` to `{coco, yolo, voc}`
- Metric display ordering is now derived from Rust `MetricDef` arrays everywhere; removed all hardcoded Python-side metric name lists from `cli.py`, `report.py`, `plots.py`, and parity scripts; `COCOeval.metric_keys()` and `ComparisonResult.metric_keys` are the canonical sources; `report.py` splits AP/AR by key prefix instead of maintaining parallel lists
- `python/hotcoco/__init__.py` — replaced `from .hotcoco import *` with explicit imports; added `__all__` listing all 13 public names
- `scripts/helpers.py` — extracted `suppress_stdout()`, keypoint constants (`COCO_KEYPOINT_NAMES`, `COCO_SKELETON`, `COCO_KPT_OKS_SIGMAS`), and path constants (`WORKSPACE`, `DATA_DIR`) shared across `parity.py`, `bench.py`, `test_parity.py`, `fuzz_parity.py`
- `crates/hotcoco/src/error.rs` — new unified `Error` enum with typed `#[from]` variants for `io::Error`, `serde_json::Error`, `ConvertError`, and a catch-all `Other(String)`; replaces `Box<dyn Error>`, bare `String`, and `io::Error` returns across 13 functions in `coco.rs`, `mask.rs`, and `eval/` submodules
- `crates/hotcoco-pyo3/src/lib.rs` — added `to_pyerr()` helper that maps `hotcoco::Error` variants to appropriate Python exceptions (`PyIOError`, `PyValueError`, `PyRuntimeError`); replaces 14 inline `.map_err(...)` calls
- `crates/hotcoco-pyo3/src/convert.rs` — added `opt!` and `req!` macros for extracting optional/required fields from Python dicts; replaces ~20 repetitions of the `dict.get_item("key")?.map(|v| v.extract()).transpose()?` pattern
- `COCOeval` fields `eval_imgs`, `eval`, `stats` changed from `pub` to `pub(crate)` with public getter methods: `eval_imgs()`, `accumulated()`, `stats()`; Python API unchanged (still `@property` getters)
- Browse UI: overlay toggles (Boxes/Segments/Keypoints/GT/DT/Eval) moved from bottom of image panel into the lightbox header bar for better visibility and space efficiency
- Browse UI: unified IoU threshold and Min Score sliders to use the same stacked layout (label + value on top, slider below) via shared `.range-slider` CSS class, eliminating ~70 lines of duplicate vendor-prefixed slider styling
- Browse UI: fixed segmentation mask misalignment on first lightbox open — replaced single `requestAnimationFrame` with double-rAF to guarantee layout is settled after `display:none → flex` transition
- Browse UI: fixed "ANNOTATIONS" header bounce during arrow-key navigation by adding `contain: size layout` on `.lightbox-card` and stable dimensions on `.info-panel-header`
- Browse UI: removed `scale(0.98)` from lightbox slide-up animation — was distorting `getBoundingClientRect()` measurements during the animation
- Browse UI: keyboard hints condensed from `← → navigate Esc close` to `← → Esc`
- Browse UI: canvas now fills the entire container (not just the image rect), preventing zoom clipping at edges
- Browse UI: gallery infinite scroll sentinel uses `hx-swap="outerHTML"` instead of `afterend` to prevent blank grid cells
- `coco explore` CLI — added `--iou-type`, `--iou-thr`, `--no-eval`, `--slices` flags; prints TP/FP/FN summary on startup when eval is active
- `pip install hotcoco[browse]` optional extra — now pulls in `fastapi>=0.100`, `uvicorn>=0.20`, `jinja2>=3.1`, `Pillow>=8.0` (previously required `gradio>=4.0`)
- `coco explore` CLI — removed `--share` flag (Gradio-specific); added `--dt` flag
- `coco.browse()` return type changed from `gr.Blocks` to `None`; use `create_app()` from `hotcoco.server` for advanced control
- `python/hotcoco/browse.py` — removed Gradio-specific code (`build_app`, `render_annotated_image`, `_require_gradio`, `_build_theme`, `_CSS`); added `prepare_annotation_data()` for client-side canvas rendering
- `python/hotcoco/cli.py` — extracted `_load_res()` helper to deduplicate error handling
- Documentation updated: `docs/guide/browse.md`, `docs/api/coco.md`, `docs/cli.md`, `README.md` — all Gradio references removed
- Internal: removed `Box<dyn Iterator>` in `COCO::get_ann_ids` — extracted filter closure, eliminated heap allocation and dynamic dispatch
- Internal: removed unnecessary `Vec::clone()` in `tide_errors()` (borrowed slices) and `confusion_matrix()` (`Cow<[u64]>` avoids allocation when params are already set)
- Internal: added `IouMatrix` type alias for `Vec<Vec<f64>>` in eval module, removed `#[allow(clippy::type_complexity)]`
- Internal: pre-allocated `Vec`s in `accumulate()` with capacity hints based on total detection count, eliminating repeated reallocations
- Internal: added `#[inline]` to hot mask functions (`area`, `to_bbox`, `intersection_area`)
- `examples/coco_evaluation_101.ipynb` — restructured and expanded: 5-act narrative (Getting Started → Understand Your Model → Compare & Slice → Dataset Tools → Integration); added sections for confusion matrix, calibration, per-image diagnostics, model comparison, sliced evaluation, dataset operations, interactive browse, and publication plots with inline visualisations; richer 4-image synthetic dataset for diagnostic demos
- `docs/getting-started/quickstart.md` — updated notebook description to reflect new content
- `coco --help` and subcommand help: new description and epilog examples on top-level parser and `eval`/`healthcheck` subparsers; `--gt`/`--dt`/`--slices` help text improved; `stats` one-liner updated
- `scripts/test_parity.py` renamed to `scripts/fuzz_parity.py` — clarifies that this is the slow hypothesis-based fuzzer (`just fuzz`), distinct from `scripts/test_parity.py` (the fast CI regression suite, `just test`)
- `scripts/fixtures/adversarial/` added to `.gitignore` and removed from tracking — hypothesis-generated fixtures are ephemeral outputs, not source files; the directory is recreated locally by running `just fuzz`
- `docs/stylesheets/extra.css` — full docs theme redesign: custom CSS variable palettes for light (stone-cream) and dark (cool charcoal) modes; all 12 `--md-code-hl-*` syntax token colors set to a warm editorial palette (dusty steel blue keywords, sage strings, clay numbers, plum functions); admonition type overrides (note/info/warning/tip) with flat tinted backgrounds and no title-bar box artifact; hero pill buttons, feature card lift-on-hover, warm-tinted shadows throughout
- `zensical.toml` — docs theme: `primary`/`accent` palette entries switched to `"custom"`; `navigation.tabs` added to features (top-level sections move to tab bar, freeing sidebar width); `[project.theme.font]` added with `text = "Nunito"` and `code = "IBM Plex Mono"`; color palette shifted from saturated warm-brown to desaturated gray-brown (`#4A4540`) with dusty slate blue accent (`#6B7E9A`) for a cooler, less heavy feel; logo icon updated to `lucide/coffee`
- Internal: extracted named constants `AREA_SMALL` (32²), `AREA_LARGE` (96²), `KPT_OKS_SIGMAS` and `default_iou_thrs()` helper in `params.rs`; replaced inline literals in `Params::new()` and deduplicated the IoU range formula between `params.rs` and `summarize.rs`
- Browse server (`server.py`) and CLI (`cli.py`) now use `COCOeval.image_diagnostics()` instead of the Python-side `build_eval_index()`; `eval_index.py` reduced to a backward-compatible thin wrapper
- Per-image AP in `image_diagnostics()` uses the shared `precision_recall_curve` from `accumulate.rs` (monotone precision correction, O(N+R) two-pointer scan) instead of a separate O(N×R) implementation
- Browse UI: extracted 367 lines of inline JavaScript from `index.html` into `static/gallery.js` (IIFE module, browser-cacheable) and 20 lines from `dashboard.html` into `static/dashboard.js`
- `python/hotcoco/cli.py` — all raw ANSI escape codes (`\033[91m`, etc.) replaced with `_style` module helpers; all `print("error: ...", file=sys.stderr)` calls replaced with `error()` helper; `cmd_eval --json` uses `ev.summary_lines()` instead of `dup2`/`devnull` fd-level stdout suppression; `cmd_stats` and `cmd_merge` use shared `_load_coco()` helper (removes redundant `ImportError` guards)
- `crates/hotcoco-cli` — added `anstyle`, `anstream`, `indicatif` dependencies; `clap` feature `color` enabled
- Unified CLI analysis output formatting — TIDE, calibration, diagnostics, slices, and compare tables all use `_table()` helper with consistent `─` separators, 2-space indent, and `section()` headers; sub-section labels use `dim()` for parenthetical qualifiers; diagnostics tip line dimmed
- Removed `Eval type: bbox` line from `summarize()` output — redundant with status line and `--iou-type` flag
- Browse UI: `overlay.js` restructured — scattered module-level variables consolidated into `_cache`/`_ui` namespaces; all `var` replaced with `const`/`let`; magic numbers extracted into named constants (`DASH_SEGMENT`, `MIN_FONT_SIZE`, `BASE_KPT_RADIUS`, etc.); JSDoc added to `drawOverlays()` documenting the 4-pass rendering pipeline
- Browse UI: checkbox styling DRYed — shared base selector for all 3 variants (category filter, overlay toggles, tree group) with per-variant size/position overrides; reduces ~90 lines of duplicated CSS
- Browse UI: `metric_fmt` Jinja2 filter added for consistent metric formatting across templates; replaces scattered `"%.3f" | format()` patterns in dashboard and gallery
- `ConvertError` — replaced manual `Display`/`Error`/`From<io::Error>` impls with `thiserror` derive macros, matching the crate's `Error` enum pattern
- `mask.rs` — `transpose_mask` doc comment clarified: now specifies `(h, w)` for row→column and `(w, h)` for the reverse, instead of claiming the operation is its own inverse
- CI — added `cargo-deny` dependency audit step (ubuntu-only); test step scoped to `-p hotcoco -p hotcoco-cli` (excludes cdylib)
- Pre-commit hook step numbering corrected from `[1/4]` to `[1/3]` (matches actual 3-step hook)
- `_typos.toml` — new typos-cli configuration: `en-us` locale, domain abbreviations allow-listed (`nd`, `obb`, `oks`, etc.), data/fixture dirs excluded
- `python/hotcoco/cli.py` — reformatted with ruff (line length compliance); `cmd_merge` import moved to function scope
- Docs — American English spelling throughout (`behaviour` → `behavior`, `normalised` → `normalized`, `maximises` → `maximizes`, `summarisation` → `summarization`, `Randomise` → `Randomize`); benchmark versions updated to 0.3.0
- Browse UI: `nav_query | safe` in `detail.html` replaced with JSON-encoded `<script type="application/json">` data element parsed in `overlay.js`, eliminating a potential XSS surface from string interpolation
- Browse UI: canvas `getBoundingClientRect()` result cached in `_cache.canvasRect` (updated on resize), avoiding forced layout reflow on every mousemove during hover hit-testing
- Browse UI: redundant `syncCatViews()` calls removed from per-checkbox handlers (`onCatChange`, `onTreeChildChange`, `toggleGroupCheck`); cross-view sync now runs only on view switch via `setCatView()`
- Browse server: inline HTML error strings in `server.py` replaced with `partials/error.html` template; `image_id` no longer duplicated inside `annotation_json` (dead `nav` key removed)
- `rand` 0.8 → 0.9 — `gen_range` → `random_range`, `small_rng` feature dropped (default in 0.9); `SmallRng::seed_from_u64` produces different sequences for the same seed (affects `split()`, `sample()`, bootstrap — dataset utilities only, no parity impact)
- `quick-xml` 0.37 → 0.39 — `BytesText::unescape()` → `decode()` in CVAT and VOC converters; no functional change for ASCII label names
- Internal: `img_cat_to_anns` HashMap pre-allocated with `reserve(n_anns)` in `create_index()`, eliminating rehashes during annotation indexing

### Fixed

- `accumulate.rs` — recall initialization moved into early-return path when `nd == 0` (GT exists but no detections); previously recall was written unconditionally then overwritten, now it's set once at the correct point
- `tide.rs` — replaced hardcoded `/ 101.0` with `/ rec_thrs.len() as f64` so TIDE ΔAP computation adapts if recall threshold count ever changes
- `mask.rs` — `iou()` and `bbox_iou()` now validate that `iscrowd` length matches `gt` length, raising `ValueError` with a clear message instead of panicking on out-of-bounds access
- `lib.rs` — browse `slice_by` callable that returns `None` now skips the image instead of raising a type error on extract
- `hotcoco-pyo3/src/lib.rs` — 13 clippy lints fixed: redundant closures replaced with method references (`String::as_str`, `str::to_lowercase`, `<[f64]>::to_vec`), `.map().unwrap_or()` → `.map_or()`, missing semicolons on unit-returning setter delegates
- Browse UI: `mouseup` and `touchstart` event listeners in `overlay.js` accumulated on every lightbox open (memory leak); now stored in module-level refs and cleaned up before re-attaching in `initOverlay()`
- Browse server: unhandled exceptions returned FastAPI's default JSON error instead of themed HTML; added `@app.exception_handler(Exception)` with styled error page and `logging.exception()` for 500s; `/detail/{id}` 404 now returns themed HTML instead of plain text
- Browse server: broad `except Exception: pass` on slice metric computation replaced with `except (KeyError, ValueError, TypeError)` and `logger.warning()` to avoid silently hiding bugs

## [0.3.0] - 2026-03-16

### Changed

- Internal: replaced `is_lvis: bool` with `eval_mode: EvalMode` enum (`Coco | Lvis | OpenImages`) across all evaluation branch points; `EvalParams.is_lvis` serialized field renamed to `eval_mode` (string: `"coco"`, `"lvis"`, `"openimages"`); no behavior change — prepares for Open Images evaluation support
- `hotcoco.plot` internal refactor: new `PlotData` dataclass (`python/hotcoco/plot/data.py`) centralises eval extraction from `COCOeval`, exposes `area_idx`, `max_det_idx`, `nearest_iou_idx` helpers and cat-name lookup; all plot functions now consume `PlotData` instead of reaching into `COCOeval` internals directly
- `hotcoco.plot` figure saving: increased output DPI from 150 to 200; added `bbox_inches="tight"` to prevent label clipping on save
- `hotcoco.plot.confusion_matrix`: colorbar now uses `make_axes_locatable` for proportional sizing; normalized matrix clamped to `[0, 1]` with `vmax=1.0`; PR-curve plots switched to `layout="compressed"` for tighter axis packing
- `_annotate_bars` internal helper removed in favour of native `ax.bar_label`

### Added

- Open Images evaluation mode (`oid_style=True` / `COCOeval::new_oid()`): single AP@IoU=0.5, `is_group_of` ignore semantics (no FN penalty), group-of second-pass multi-match, iscrowd re-matching disabled
- `Hierarchy` type — category hierarchy for GT/DT expansion with three construction methods: `from_parent_map`, `from_categories` (supercategory fields), `from_oid_json` / `from_file` / `from_dict` (OID JSON format)
- `Annotation.is_group_of: Option<bool>` field (`#[serde(default)]`) — Open Images group-of flag
- `Params.expand_dt: bool` flag — opt-in DT expansion up the hierarchy (default `false`; GT is always expanded)
- `Hierarchy` Python class with `from_file`, `from_dict`, `from_parent_map`, `ancestors`, `children`, `parent` methods
- `docs/guide/evaluation.md` — "Open Images evaluation" section covering hierarchy, group-of, detection expansion, and the single AP metric
- `docs/api/hierarchy.md` — API reference for the `Hierarchy` class
- `docs/api/cocoeval.md` — updated constructor docs with `oid_style` and `hierarchy` parameters; Rust `new_oid()` constructor
- `docs/api/params.md` — `expand_dt` parameter documented
- `docs/api/plot.md` — documented `pr_curve_iou_sweep`, `pr_curve_by_category`, and `pr_curve_top_n`; these three functions were in `__all__` and importable but had no API reference entries; `pr_curve` section updated to describe it as a convenience dispatcher and to prefer calling the named functions directly
- `docs/guide/masks.md` — warning admonition: `mask.encode()` returns `counts` as `bytes`; must decode to UTF-8 string before passing to `load_res()` or storing in a COCO JSON file
- `docs/getting-started/quickstart.md` — bbox format warning: COCO uses `[x, y, width, height]`, not `[x1, y1, x2, y2]`; silent failure if wrong format is passed; includes conversion snippet
- `docs/guide/evaluation.md` — "Key concepts" subsection before the metrics table with plain-language definitions of AP, AR, IoU, and area ranges (with pixel-scale reference); RLE format explanation and conversion snippet in the Segmentation section
- `docs/guide/pytorch.md` — tip noting that standard torchvision models (Faster R-CNN, RetinaNet, FCOS, etc.) output XYXY boxes and that `CocoEvaluator` converts to XYWH automatically
- `docs/api/cocoeval.md` — `per_class=True` example output showing `"AP/person"`, `"AP/car"` keys in `results()` entry
- `docs/api/coco.md` — Pillow dependency note in `from_yolo()` parameters table
- `docs/cli.md` — `--output / -o` flag documented in `coco-eval` section
- `docs/getting-started/troubleshooting.md` — category ID mismatch diagnostic: how to detect and fix mismatched IDs between GT and DT files
- `hotcoco.plot.report()` — single-page PDF evaluation report: run context block, mode-aware metrics table (correct rows for bbox/segm, keypoints, and LVIS), PR curves at IoU 0.50/0.75/mean, F1 peak tile, and per-category AP chart; "hotcoco" brand mark in header
- `coco eval --report <path>` — saves a PDF evaluation report as a side-output of eval; `--lvis` and `--title` flags added to `coco eval`; requires `pip install hotcoco[plot]`
- `hotcoco.plot` module — publication-quality matplotlib plots for evaluation results: `pr_curve`, `confusion_matrix`, `top_confusions`, `per_category_ap`, `tide_errors`
- `theme` and `paper_mode` parameters on all plot functions — `theme` selects one of three built-in palettes (`"warm-slate"`, `"scientific-blue"`, `"ember"`); `paper_mode=True` forces white figure/axes backgrounds for LaTeX or PowerPoint embedding
- `hotcoco.plot.style(theme, paper_mode)` context manager — apply any theme to custom matplotlib code outside of hotcoco plot functions
- Bundled Inter font (Medium + Bold) in `python/hotcoco/_fonts/` for consistent typography across platforms
- `plot` optional dependency group: `pip install hotcoco[plot]` (matplotlib >= 3.5)
- `docs/guide/plotting.md` — user guide with examples for all 5 plot types, unstyled mode, and subplot composition
- `docs/api/plot.md` — API reference for all plot functions and color palette constants
- Shell completions for `coco-eval` (Rust) — `coco-eval --completions <bash|zsh|fish|elvish|powershell>` prints a completion script to stdout; powered by `clap_complete`
- Shell completions for `coco` (Python) — `pip install "hotcoco[completions]"` enables tab completion via `argcomplete`; `# PYTHON_ARGCOMPLETE_OK` magic comment added to CLI entrypoint
- `docs/getting-started/troubleshooting.md` — covers import conflicts with pycocotools, numpy version issues, detection format mistakes (XYXY vs XYWH, missing fields, unknown image IDs), RLE pitfalls, and all-`-1` metric diagnosis
- `docs/guide/pytorch.md` — full guide for `CocoDetection` and `CocoEvaluator`: transforms, distributed training, multi-iou-type evaluation, migration from torchvision
- `docs/guide/frameworks.md` — Detectron2, MMDetection, RF-DETR integration via `init_as_pycocotools()`; Ultralytics `save_json` workflow; LVIS-based pipeline drop-in via `init_as_lvis()`
- Feature comparison table in `docs/benchmarks.md` — hotcoco vs pycocotools vs faster-coco-eval across installation, parity, LVIS, TIDE, confusion matrix, dataset ops, PyTorch integration, CLI, memory, and license
- `scripts/download_coco.py` — downloads COCO val2017 annotations and generates deterministic parity result files; replaces the old untracked `data/gen_*.py` scripts
- `scripts/download_o365.py` — downloads Objects365 validation annotations from HuggingFace (moved from gitignored `data/`, now tracked)
- `just download-coco`, `just download-o365`, `just download-all` recipes in `Justfile`
- Benchmark data section in `docs/getting-started/installation.md` — one-command setup for COCO val2017 and Objects365 benchmark data via `just download-coco` / `just download-o365`
- Rust examples: `crates/hotcoco/examples/basic_eval.rs` and `custom_params.rs` — runnable end-to-end evaluation examples with `cargo run --example`
- Notebook link surfaced in quickstart "Next steps" and index hero actions
- "Troubleshooting", "PyTorch Integration", and "Framework Integrations" added to `zensical.toml` nav
- `ConfusionMatrix.cat_names` / `confusion_matrix()` dict now includes `"cat_names"` — category names parallel to `cat_ids`, eliminating a manual `load_cats` lookup after computing a confusion matrix
- `EvalResults.hotcoco_version` — records the library version that produced the results file; included in the `results()` dict and saved JSON
- `TideErrors` now derives `Serialize` (Rust) — can be serialized directly with `serde_json`
- Dataset healthcheck — 4-layer validation (structural, quality, distribution, GT/DT compatibility) for COCO annotation files; `coco.healthcheck()` and `coco.healthcheck(dt)` in Python, `healthcheck()` / `healthcheck_compatibility()` in Rust, `coco healthcheck` CLI subcommand
- `--healthcheck` flag on `coco eval` — runs healthcheck before evaluation and prints errors/warnings to stderr
- Sliced evaluation — `COCOeval.slice_by(slices)` re-accumulates metrics for named image-ID subsets (indoor/outdoor, day/night) without recomputing IoU; `--slices <json>` flag on `coco eval` CLI

### Fixed

- `hotcoco.plot.report()`: table caption underline was too far below the caption text; moved from `y=0.0` to `y=0.3` (axes coordinates)
- `hotcoco.plot.report()`: floating-point values in the metrics table, per-category AP table, and PR-curve legend were right-aligned; now left-aligned
- `hotcoco.plot.report()`: PR-curve legend labels now lead with the numeric value (e.g. `0.456  AP50`) so stacked values align correctly regardless of label width
- `_annotate_f1_peak`: guard against all-NaN precision arrays that caused `ValueError` from `nanargmax`
- `evaluate_img_static` (eval/evaluate.rs): detection-side area-ignore flags were not applied when a (image, category) pair had detections but no GT annotations — `dt_ignore_flags` was initialized to all-`false` and only populated inside the `if let Some(iou_mat)` branch, so DTs with area outside the area range were silently treated as false positives instead of being ignored; fixed by initializing `dt_ignore_flags` from `dt_area_ignore` unconditionally; affected APm/APl/APs for images with zero GT for a given category
- `docs/benchmarks.md` feature comparison table: four inaccurate cells corrected — pycocotools Installation changed from "Requires C compiler" to "Prebuilt wheels available (Python 3.9+)"; pycocotools Python versions changed from "3.7+" to "3.9+"; faster-coco-eval License changed from "BSD" to "Apache 2.0"; faster-coco-eval PyTorch changed from "No" to "Yes — TorchVision compatible"
- `docs/guide/results.md` per-category AP Python example: was indexing `ev.stats[0]` (the scalar overall AP) for every category in the loop, printing the same number for every class; fixed to index the precision array by category (`precision[:, :, i, 0, 2]`); promoted `get_results(per_class=True)` as the recommended approach
- `docs/guide/datasets.md` area range comment: `area_rng=[1024.0, 9216.0]` covers medium objects only (32²–96² px²), not "medium-to-large"
- README removed incorrect claim that hotcoco works as a drop-in for Ultralytics YOLO — Ultralytics implements its own internal metrics and does not use pycocotools or faster-coco-eval

### Changed

- Summary table alignment widened from 18 to 22 characters so "Average Precision (AP)" and "Average Recall (AR)" align at the `@` sign across all rows
- Sliced evaluation table uses fixed-width columns (14 chars) with `_overall` values aligned to the integer part of slice metric values for vertical readability
- Healthcheck imbalance label now shows actual category names and counts (e.g., `person: 11,004 / toaster: 9`) instead of a bare ratio
- `accumulate_impl` and `summarize_impl` extracted as `pub(super)` pure functions; `summarize_impl` now accepts `&[MetricDef]` to avoid redundant `build_metric_defs` calls across `summarize()`, `slice_by()`, and `metric_keys()`
- `docs/index.md` feature card updated from "Just pip install / No Cython, no compiler" to "More than a metric / TIDE error breakdown, confusion matrix, per-category AP, and publication-quality plots" — installation ease is no longer a unique differentiator since pycocotools now ships prebuilt wheels; analysis toolkit is the clearer differentiator
- `README.md` opening expanded with a paragraph calling out the diagnostic toolkit (TIDE error breakdown, confusion matrix, per-category AP, F-scores, publication-quality plots with PDF report) as features pycocotools and faster-coco-eval don't have
- Consolidated repo layout: single root `pyproject.toml` (maturin `manifest-path` pattern); Python package source moved from `crates/hotcoco-pyo3/python/` to root `python/`; all scripts moved from `crates/hotcoco-pyo3/data/` to root `scripts/`
- `Justfile` added at repo root with `build`, `test`, `parity`, `bench`, `lint`, `fmt`, `fmt-check`, `download-coco`, `download-o365`, `download-all` recipes — replaces ad-hoc `uv run python ...` invocations
- `EvalResults::to_json_string()` renamed to `to_json()` for consistency with Rust naming conventions

## [0.2.0] - 2026-03-11

### Added

- Objects365 benchmark results (80k images, 365 categories, ~1.2M detections): hotcoco **39×** vs pycocotools and **14×** vs faster-coco-eval; peak committed RAM 8 GB vs 24–30 GB for alternatives
- `bench_objects365.py` now includes pycocotools as a third runner; Windows support (`peak_wset` + pagefile for memory measurement, `.exe` binary name); `_bench_python_runner` shared helper; process-tree memory tracking via psutil
- `COCOeval.results(per_class=False)` — return serializable evaluation results as a dict; `save_results(path, per_class=False)` writes the same structure as pretty-printed JSON
- `coco-eval --output / -o <path>` — CLI flag to write evaluation results JSON after evaluation (always includes per-category AP)
- `AreaRange` struct in `hotcoco::params` (re-exported from crate root) — replaces the two parallel `area_rng` / `area_rng_lbl` vecs in `Params` with a single `Vec<AreaRange { label, range }>`
- `Params::area_range_idx(label) -> Option<usize>` — label-based lookup helper; eliminates all positional `unwrap_or(0)` fallbacks
- `FreqGroup` enum (`Rare` / `Common` / `Frequent`) and `FreqGroups` struct in `hotcoco::eval::types` — named fields replace the implicit `[Vec<usize>; 3]` index convention for LVIS frequency groups
- `MetricDef.name` field — `metric_keys()` is now derived from the same `Vec<MetricDef>` that drives `summarize()`, eliminating the parallel-list sync risk; `metrics_lvis()` brings LVIS into the unified `MetricDef` path
- `EvalShape` re-exported from the crate root for Rust users who need to index into `AccumulatedEval.precision`/`recall` arrays directly
- `CONTRIBUTING.md` — contributor guide covering build setup, pre-commit hook, parity workflow, and PR process
- `CODE_OF_CONDUCT.md` — Contributor Covenant
- `SECURITY.md` — vulnerability disclosure policy
- `.github/ISSUE_TEMPLATE/` — bug report and feature request templates
- `.github/pull_request_template.md` — PR checklist with parity output section
- `examples/coco_evaluation_101.ipynb` — Jupyter notebook: quickstart, per-class AP, F-scores, TIDE error analysis, drop-in replacement, and experiment logging
- `docs/benchmarks.md` — "Reproducing the benchmarks" section with step-by-step clone, build, data setup, and benchmark commands
- CI, PyPI, Crates.io, and MIT license badges in `README.md`
- `COCO(dict)` — constructor now accepts an in-memory dataset dict in addition to a file path or `None`
- `COCOeval.f_scores(beta=1.0)` — compute F-beta scores after `accumulate()`; for each (IoU threshold, category) finds the confidence operating point that maximises F-beta, then averages across categories; returns `{"F1": ..., "F150": ..., "F175": ...}` (key prefix reflects beta value); supports arbitrary beta for precision/recall trade-off weighting
- `get_results(prefix, per_class)` — optional `prefix` parameter prepends a path to all metric keys (e.g. `"val/bbox/AP"`), and `per_class=True` adds per-category AP entries keyed as `"AP/{cat_name}"`; returns a flat dict ready for `wandb.log()`, `mlflow.log_metrics()`, or any experiment tracker
- `IouType` now implements `Display` and `FromStr` traits
- `mask.frPyObjects(seg, h, w)` — pycocotools-compatible unified entry point: accepts a list of polygon coord lists, a single uncompressed RLE dict, or a list of uncompressed RLE dicts; returns the same type as input (single dict or list of dicts)
- `mask.encode` now accepts 3-D `(H, W, N)` arrays and returns a list of N RLE dicts (pycocotools batch encoding)
- `mask.decode` now accepts a list of RLE dicts and returns a `(H, W, N)` Fortran-order array (pycocotools batch decoding)
- `mask.area` and `mask.to_bbox` / `mask.toBbox` now accept a single dict or a list of dicts, matching pycocotools batch semantics
- camelCase aliases `frPoly`, `frBbox`, `toBbox` in `hotcoco.mask` matching pycocotools naming
- `mask.iou` now returns a numpy float64 ndarray instead of a nested list
- `COCO.to_yolo(output_dir)` — export a COCO dataset to YOLO label format; writes one `<stem>.txt` per image with normalized `class_idx cx cy w h` lines plus `data.yaml`; crowd and no-bbox annotations are skipped; returns a stats dict with `images`, `annotations`, `skipped_crowd`, `missing_bbox`
- `COCO.from_yolo(yolo_dir, images_dir=None)` — load a YOLO label directory as a COCO dataset; reads `data.yaml` for the category list; if `images_dir` is given, Pillow reads image dimensions from disk (requires `pip install Pillow`)
- `hotcoco::convert::coco_to_yolo` / `yolo_to_coco` — Rust functions backing the above; `YoloStats` and `ConvertError` types re-exported from crate root
- `coco convert --from coco --to yolo --input <json> --output <dir>` / `--from yolo --to coco --input <dir> --output <json> [--images-dir <dir>]` — CLI subcommand for format conversion
- `coco eval --tide` — print TIDE error decomposition after standard metrics; `--tide-pos-thr` and `--tide-bg-thr` control the IoU thresholds (defaults: 0.5 and 0.1)
- `COCOeval.tide_errors(pos_thr=0.5, bg_thr=0.1)` — TIDE error decomposition (Bolya et al., ECCV 2020); classifies every FP into six mutually exclusive types (Loc, Cls, Dupe, Bkg, Both, Miss) and reports ΔAP — the AP gain from eliminating each type; requires `evaluate()` first; priority order matches tidecv (Loc > Cls > Dupe > Bkg > Both); Bkg/Both/Dupe ΔAP uses suppression (not flip-to-TP) for correct curve behaviour
- `TideErrors` Rust type with `delta_ap`, `counts`, `ap_base`, `pos_thr`, `bg_thr` fields
- `COCO.load_res()` now accepts three input formats: file path (`str`), list of annotation dicts (`list[dict]`), or a numpy float64 array of shape `(N, 6)` or `(N, 7)` with columns `[image_id, x, y, w, h, score[, category_id]]` — matches pycocotools `loadNumpyAnnotations` convention
- `COCO::load_res_anns(Vec<Annotation>)` — new Rust method for in-memory result loading without a filesystem round-trip
- `COCOeval.confusion_matrix(iou_thr=0.5, max_det=None, min_score=None)` — per-category confusion matrix with cross-category greedy matching; returns `(K+1)×(K+1)` numpy int64 array (rows = GT, cols = predicted, index K = background); standalone, no `evaluate()` needed; parallelised with rayon
- `ConfusionMatrix` Rust type with `.get(gt_idx, pred_idx)` and `.normalized()` methods
- LVIS federated evaluation — `COCOeval(..., lvis_style=True)` and `LVISeval` drop-in replacement for lvis-api `LVISEval`; 13 metrics (AP, AP50, AP75, APs/m/l, APr/c/f, AR@300, ARs/m/l@300); federated FP filtering via `neg_category_ids` / `not_exhaustive_category_ids`
- `init_as_lvis()` — `sys.modules` patch so `from lvis import LVIS, LVISEval, LVISResults` transparently resolves to hotcoco; enables drop-in use in Detectron2 and MMDetection LVIS pipelines
- `LVISResults`, `LVIS` Python aliases matching lvis-api conventions
- `COCOeval.run()`, `.get_results()`, `.print_results()` methods (used by lvis-api-style pipelines)
- `COCO.stats()` — dataset health-check statistics: annotation counts, image dimensions, area distributions, per-category breakdowns
- Dataset operations on `COCO`: `filter`, `merge` (classmethod), `split`, `sample`, `save`
- Python CLI (`coco`) with subcommands: `eval`, `stats`, `filter`, `merge`, `split`, `sample`

### Fixed

- `mask.area()` PyO3 binding now returns native `u64` instead of truncating to `u32`
- `get_results(per_class=True)` index misalignment when a category ID is missing from the GT dataset

### Changed

- Feature comparison table in `docs/benchmarks.md` corrected: faster-coco-eval installation (prebuilt wheels available), metric parity (exact vs pycocotools), LVIS support (`lvis_style=True`), per-class AP (`extended_metrics`), Python version floor (3.7+)
- Parity tolerance claim updated from flat "≤1e-4" to per-type breakdown: bbox ≤1e-4, segm ≤2e-4, keypoints exact
- Benchmark numbers in `README.md` and `docs/index.md` synced to current bench.py output (bbox 0.41s 23×, segm 0.49s 18.6×, kpts 0.21s 12.7×); corrected detection count from ~43,700 to 36,781
- Documentation: added paper citations for COCO eval (Lin et al. ECCV 2014), OKS (cocodataset.org), LVIS (Gupta et al. ECCV 2019), and TIDE (Bolya et al. ECCV 2020 arxiv); area range notation clarified to square pixels (px²); LVIS frequency definition corrected from instance count to training image count
- Pre-commit hook relocated from `hooks/pre-commit` to `.github/hooks/pre-commit` (standard location)
- `crates/hotcoco-pyo3/README.md` converted to a symlink to root `README.md` — always in sync, no manual copy needed
- `.gitignore` tightened: `data/` blanket exclusion replaced with targeted patterns so benchmark scripts and test fixtures are now tracked; `examples/*.ipynb` exempted from `*.ipynb` exclusion
- Deleted stale investigation and one-off run scripts from `data/`
- Simplified Rust internals: extracted shared helpers (`cross_category_iou`, `subset_by_img_ids`, `per_cat_ap`, `metric_keys`, `format_metric`), pre-sized HashMap allocations, pre-computed GT bbox coordinates in `bbox_iou` hot path
- `lvis` moved from runtime dependency to `dev` optional dependency; hotcoco implements the lvis-api interface natively and never imports `lvis` at runtime
- `mask.encode` signature changed: `h` and `w` parameters removed; dimensions are inferred from the array shape. Accepts both Fortran-order (pycocotools convention) and C-order arrays.
- All RLE-returning mask functions (`encode`, `decode`, `merge`, `fr_poly`, `fr_bbox`, `rle_from_string`) now return pycocotools format `{"size": [h, w], "counts": b"..."}` instead of the previous internal format `{"h": h, "w": w, "counts": [ints]}`
- `py_to_rle` now accepts `bytes` counts (pycocotools format) in addition to `str` and `list[int]`
- `integrations.py` segm path simplified — no longer manually converts RLE format; `mask.encode` now returns coco format directly
- Eval internals: split `eval.rs` (2500 lines) into 8 focused submodules — `accumulate`, `evaluate`, `iou`, `summarize`, `tide`, `confusion`, `types`, `mod`; no API change
- Eval performance: greedy matching now uses a linear scan instead of pre-sorted index vectors, eliminating 2×D `Vec` allocations per (image, category) pair; faster for typical COCO (≤5 GTs/cat); `precision_recall_curve` extracted as a shared kernel reused by both `accumulate` and `tide_errors`
- Eval performance: flat IoU matrix, OKS single-pass accumulation, direct index tracking (no HashMaps), area_rng HashMap in accumulate — 4–26% faster depending on dataset scale
- Mask performance: rayon sequential fallback for small D×G (`MIN_PARALLEL_WORK = 1024`), intersection_area early exit, fr_poly allocation reduction — biggest impact on segm (10% on val2017)
- PyO3 error handling: `.unwrap()` → proper `PyValueError` with descriptive messages in convert.rs and mask.rs
- PyO3 safety: mask decode/encode use safe numpy array construction (no unsafe `PyArray2::new()`)
- `tide_errors()` returns `Result<TideErrors, String>` instead of panicking on precondition failure

### Removed

- `hotcoco.loggers` module (`log_wandb`, `log_mlflow`, `log_tensorboard`) — replaced by the `prefix`/`per_class` parameters on `get_results()`, which produce logger-ready dicts without framework-specific wrappers

## [0.1.0] - 2025-06-15

### Added

- Pure Rust COCO API — dataset loading, indexing, querying (bbox, segmentation, keypoints)
- Full evaluation pipeline with all 12 AP/AR metrics (10 for keypoints)
- Pure Rust RLE encoding/decoding (no C FFI)
- Rayon-based parallel evaluation
- CLI tool (`hotcoco-cli`) with `--no-cats` flag
- PyO3 Python bindings (`hotcoco` package) with numpy interop
- `init_as_pycocotools()` drop-in replacement via `sys.modules` patching
- camelCase aliases for pycocotools API compatibility
- `eval_imgs` and `eval` properties on COCOeval
- MkDocs documentation site with GitHub Actions deployment
- Performance optimizations: fused intersection, analytical `fr_bbox`, pre-computed indexing, in-place precision interpolation (11-26x faster than pycocotools)

### Fixed

- Zero-length RLE run handling in `intersection_area` and `merge_two`
- `iscrowd` vs `gt_ignore` matching bug in evaluation
- RLE string delta encoding parity with maskApi.c
- Segmentation and keypoints metric parity with pycocotools
