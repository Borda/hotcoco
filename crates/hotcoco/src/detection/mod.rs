//! The detection metric family.
//!
//! [`COCOeval`] is the driver: a faithful port of `pycocotools/cocoeval.py`'s
//! `evaluate` → `accumulate` → `summarize` lifecycle for bbox, segm, keypoint
//! and oriented-box geometry, plus the two protocol variants that share it —
//! LVIS federated evaluation ([`COCOeval::new_lvis`]) and Open Images
//! ([`COCOeval::new_oid`]).
//!
//! Layered on the same evaluated cells are the analysis methods, which are
//! adapters rather than metric implementations: they decide which detections
//! count, marshal them into flat arrays, and call
//! [`crate::metrics`]. TIDE's error taxonomy ([`TideErrors`]) and per-image
//! diagnostics ([`ImageDiagnostics`]) are the exceptions that stay here, being
//! genuinely detection-shaped.

mod accumulate;
mod calibration;
mod catalog;
mod compare;
mod confusion;
mod diagnostics;
mod evaluate;
pub mod expand;
pub mod hierarchy;
mod iou;
mod matching;
mod mode;
mod report;
mod results;
pub mod slice;
mod summarize;
mod tide;

pub use accumulate::{AccumulatedEval, EvalShape};
pub use calibration::CalibrationResult;
pub use catalog::MetricDef;
pub use compare::{CategoryDelta, CompareOpts, ComparisonResult, compare};
pub use confusion::ConfusionMatrix;
pub use diagnostics::{
    AnnotationIndex, DtStatus, ErrorProfile, GtStatus, ImageDiagnostics, ImageSummary, LabelError,
    LabelErrorType,
};
pub use matching::EvalImg;
pub use mode::{EvalMode, FreqGroup};
pub use results::{EvalParams, EvalResults};
pub use slice::{SliceResult, SlicedResults};
pub use tide::TideErrors;

use std::borrow::Cow;
use std::collections::HashMap;

use crate::coco::COCO;
use crate::detection::hierarchy::Hierarchy;
use crate::params::{IouType, Params};
use mode::FreqGroups;

/// COCO evaluation engine.
///
/// Computes AP and AR metrics for bbox, segmentation, and keypoint predictions.
/// Also supports LVIS federated evaluation via [`COCOeval::new_lvis`].
///
/// The standard workflow is three steps:
///
/// ```no_run
/// # use hotcoco::{COCO, COCOeval, params::IouType};
/// # fn main() -> hotcoco::error::Result<()> {
/// # let coco_gt = COCO::new(std::path::Path::new("gt.json"))?;
/// # let coco_dt = coco_gt.load_res(std::path::Path::new("dt.json"))?;
/// let mut ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
/// ev.evaluate();   // per-image IoU matching
/// ev.accumulate(); // aggregate into precision/recall curves
/// ev.summarize();  // print + store the summary metrics in ev.stats
/// # Ok(())
/// # }
/// ```
///
/// For LVIS, use [`run`](COCOeval::run) as a convenience:
///
/// ```no_run
/// # use hotcoco::{COCO, COCOeval, params::IouType};
/// # fn main() -> hotcoco::error::Result<()> {
/// # let coco_gt = COCO::new(std::path::Path::new("gt.json"))?;
/// # let coco_dt = coco_gt.load_res(std::path::Path::new("dt.json"))?;
/// let mut ev = COCOeval::new_lvis(coco_gt, coco_dt, IouType::Segm);
/// ev.run();
/// let results = ev.get_results(None, false); // BTreeMap<metric_name, f64>
/// # Ok(())
/// # }
/// ```
pub struct COCOeval {
    pub coco_gt: COCO,
    pub coco_dt: COCO,
    pub params: Params,
    pub(crate) eval_imgs: Vec<Option<EvalImg>>,
    ious: HashMap<(u64, u64), matching::IouMatrix>,
    /// Per-annotation RLEs for segm runs, rebuilt by each `evaluate()` (like
    /// `ious`) and `None` for every other geometry. See [`iou::SegmRles`].
    segm_rles: Option<iou::SegmRles>,
    pub(crate) eval: Option<AccumulatedEval>,
    pub(crate) stats: Option<Vec<f64>>,
    /// Evaluation mode (COCO, LVIS, or OpenImages).
    pub eval_mode: EvalMode,
    /// LVIS: k_indices bucketed by category frequency.
    /// Populated during `evaluate()` when `eval_mode == Lvis`.
    freq_groups: FreqGroups,
    /// Open Images: category hierarchy for GT/DT expansion.
    pub hierarchy: Option<Hierarchy>,
}

impl COCOeval {
    /// The one struct literal behind all three public constructors. What they
    /// differ in is a parameter here; everything else is the same empty
    /// pre-`evaluate()` state, so a field added later is initialized once.
    fn with_mode(
        coco_gt: COCO,
        coco_dt: COCO,
        params: Params,
        eval_mode: EvalMode,
        hierarchy: Option<Hierarchy>,
    ) -> Self {
        COCOeval {
            coco_gt,
            coco_dt,
            params,
            eval_imgs: Vec::new(),
            ious: HashMap::new(),
            segm_rles: None,
            eval: None,
            stats: None,
            eval_mode,
            freq_groups: FreqGroups::default(),
            hierarchy,
        }
    }

    /// Create a new COCOeval from ground truth and detection COCO objects.
    pub fn new(coco_gt: COCO, coco_dt: COCO, iou_type: IouType) -> Self {
        Self::with_mode(
            coco_gt,
            coco_dt,
            Params::new(iou_type),
            EvalMode::Coco,
            None,
        )
    }

    /// Per-image evaluation results (sparse — indexed by image position).
    pub fn eval_imgs(&self) -> &[Option<EvalImg>] {
        &self.eval_imgs
    }

    /// Accumulated precision/recall curves (set after `accumulate()`).
    pub fn accumulated(&self) -> Option<&AccumulatedEval> {
        self.eval.as_ref()
    }

    /// Summary statistics (set after `summarize()`).
    pub fn stats(&self) -> Option<&[f64]> {
        self.stats.as_deref()
    }

    /// Cached similarity matrix for one (image, category) cell, if `evaluate()`
    /// computed one.
    ///
    /// # Deliberately one cell at a time
    ///
    /// `self.ious` is a **whole-dataset** similarity cache. Exposing it as one
    /// would foreclose the memory lever the tracking family depends on: HOTA's
    /// second pass must be free to *recompute* similarity rather than retain it,
    /// because at MOT20 scale retention costs hundreds of megabytes per sequence
    /// per thread.
    ///
    /// So this accessor hands out one cell, never the map. The detection driver may
    /// cache as much as it likes; nothing outside it may learn that a whole-dataset
    /// cache exists. **Keep this driver-private** — it must not gain a `pub` variant,
    /// and it must not return `&HashMap<..>`. `pub(in crate::detection)`, not
    /// `pub(super)`: the visibility is the enforcement. `tests/architecture.rs`
    /// separately bans direct `.ious` access outside this module.
    pub(in crate::detection) fn cell_ious(
        &self,
        img_id: u64,
        cat_id: u64,
    ) -> Option<&matching::IouMatrix> {
        self.ious.get(&(img_id, cat_id))
    }

    /// The evaluated cells every whole-dataset analysis reads: `area = "all"` at
    /// the default per-image detection cap.
    ///
    /// TIDE, calibration and per-image diagnostics all want exactly this subset
    /// of `eval_imgs`. The `max_det` leg is inert today — `evaluate()` stamps one
    /// cap on every cell, taken from [`Params::max_det`](crate::Params::max_det)
    /// — but it is what keeps a future second cap per cell from making these
    /// analyses double-count every detection.
    ///
    /// Both legs come from `params`, so an evaluator re-configured after
    /// `evaluate()` yields nothing rather than a partial mixture.
    pub(in crate::detection) fn default_cells(&self) -> impl Iterator<Item = &EvalImg> {
        let area_rng = self.params.all_area_range();
        let max_det = self.params.max_det();
        self.eval_imgs
            .iter()
            .flatten()
            .filter(move |e| e.area_rng == area_rng && e.max_det == max_det)
    }

    /// The image and category ids this evaluation covers, without mutating anything.
    ///
    /// User-set `params` filters win; otherwise the ids come from `COCO`'s
    /// unfiltered getters, which return them sorted — the order the whole
    /// evaluation is keyed on. Borrowed when `params` already holds them, so the
    /// common path allocates nothing.
    ///
    /// **Non-mutating** on purpose, because its two callers differ: `evaluate()`
    /// writes the answer back into `params`, while `confusion_matrix()` is a
    /// `&self` method that must cover the same ids without a prior `evaluate()`
    /// and without touching state.
    pub(in crate::detection) fn resolved_ids(&self) -> (Cow<'_, [u64]>, Cow<'_, [u64]>) {
        let img_ids = if self.params.img_ids.is_empty() {
            Cow::Owned(self.coco_gt.get_img_ids(&[], &[]))
        } else {
            Cow::Borrowed(self.params.img_ids.as_slice())
        };
        let cat_ids = if self.params.cat_ids.is_empty() {
            Cow::Owned(self.coco_gt.get_cat_ids(&[], &[], &[]))
        } else {
            Cow::Borrowed(self.params.cat_ids.as_slice())
        };
        (img_ids, cat_ids)
    }

    /// LVIS frequency-group buckets, populated during `evaluate()` in LVIS mode.
    ///
    /// Driver-private: the analysis layer re-aggregates over these, but they are an
    /// implementation detail of federated evaluation rather than public surface.
    pub(in crate::detection) fn freq_groups(&self) -> &FreqGroups {
        &self.freq_groups
    }

    /// Create a new COCOeval configured for LVIS federated evaluation.
    ///
    /// LVIS uses federated annotation — each image is only exhaustively labeled
    /// for a subset of categories. This constructor sets `max_dets=300` and enables
    /// federated filtering so unmatched detections on unlabeled or unchecked categories
    /// are not penalized as false positives.
    ///
    /// Behavior controlled by per-image GT fields:
    /// - `neg_category_ids`: categories confirmed absent → unmatched DTs count as FP.
    /// - `not_exhaustive_category_ids`: categories not fully checked → unmatched DTs ignored.
    ///
    /// Produces 13 metrics: AP, AP50, AP75, APs, APm, APl, APr (rare), APc (common),
    /// APf (frequent), AR@300, ARs@300, ARm@300, ARl@300.
    pub fn new_lvis(coco_gt: COCO, coco_dt: COCO, iou_type: IouType) -> Self {
        let mut params = Params::new(iou_type);
        params.max_dets = vec![300];

        Self::with_mode(coco_gt, coco_dt, params, EvalMode::Lvis, None)
    }

    /// Run the full evaluation pipeline in one call: `evaluate` → `accumulate` → `summarize`.
    ///
    /// Equivalent to calling the three methods in sequence. Primarily used with LVIS
    /// pipelines such as Detectron2 and MMDetection that expect a single `run()` entry point.
    pub fn run(&mut self) {
        self.evaluate();
        self.accumulate();
        self.summarize();
    }

    /// Create a new COCOeval configured for Open Images detection evaluation.
    ///
    /// OID uses a single IoU threshold (0.5), one area range ("all"), and
    /// `max_dets=100`. If a [`Hierarchy`] is provided, GT annotations are expanded
    /// up the hierarchy during `evaluate()`. Set `params.expand_dt = true` to
    /// also expand detections.
    pub fn new_oid(coco_gt: COCO, coco_dt: COCO, hierarchy: Option<Hierarchy>) -> Self {
        let mut params = Params::new(IouType::Bbox);
        params.iou_thrs = vec![0.5];
        params.area_ranges = vec![crate::AreaRange {
            label: "all".to_string(),
            range: [0.0, 1e10],
        }];
        params.max_dets = vec![100];

        Self::with_mode(coco_gt, coco_dt, params, EvalMode::OpenImages, hierarchy)
    }
}
