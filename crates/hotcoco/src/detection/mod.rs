//! COCO evaluation engine — faithful port of `pycocotools/cocoeval.py`.
//!
//! Implements evaluate, accumulate, and summarize for bbox, segm, and keypoint evaluation.

pub(super) mod accumulate;
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
pub use compare::{CategoryDelta, CompareOpts, ComparisonResult, compare};
pub use confusion::ConfusionMatrix;
pub use diagnostics::{
    AnnotationIndex, DtStatus, ErrorProfile, GtStatus, ImageDiagnostics, ImageSummary, LabelError,
    LabelErrorType,
};
pub use matching::EvalImg;
pub use mode::EvalMode;
pub use results::{EvalParams, EvalResults};
pub use slice::{SliceResult, SlicedResults};
pub use tide::TideErrors;

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
/// ```rust,ignore
/// let mut ev = COCOeval::new(coco_gt, coco_dt, IouType::Bbox);
/// ev.evaluate();   // per-image IoU matching
/// ev.accumulate(); // aggregate into precision/recall curves
/// ev.summarize();  // print + store the summary metrics in ev.stats
/// ```
///
/// For LVIS, use [`run`](COCOeval::run) as a convenience:
///
/// ```rust,ignore
/// let mut ev = COCOeval::new_lvis(coco_gt, coco_dt, IouType::Segm);
/// ev.run();
/// let results = ev.get_results(None, false); // HashMap<metric_name, f64>
/// ```
pub struct COCOeval {
    pub coco_gt: COCO,
    pub coco_dt: COCO,
    pub params: Params,
    pub(crate) eval_imgs: Vec<Option<EvalImg>>,
    ious: HashMap<(u64, u64), matching::IouMatrix>,
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
    /// Create a new COCOeval from ground truth and detection COCO objects.
    pub fn new(coco_gt: COCO, coco_dt: COCO, iou_type: IouType) -> Self {
        COCOeval {
            coco_gt,
            coco_dt,
            params: Params::new(iou_type),
            eval_imgs: Vec::new(),
            ious: HashMap::new(),
            eval: None,
            stats: None,
            eval_mode: EvalMode::Coco,
            freq_groups: FreqGroups::default(),
            hierarchy: None,
        }
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
    /// `self.ious` is a **whole-dataset** similarity cache, and the 0.5 primitives
    /// contract review flagged it as the single most likely route by which retention
    /// leaks into a shared contract. If it ever became a primitive-level or
    /// `EvalReport`-level "similarity cache" type, it would foreclose the memory lever
    /// the tracking family depends on — HOTA's second pass must be free to *recompute*
    /// similarity rather than retain it, because at MOT20 scale retention costs
    /// hundreds of megabytes per sequence per thread.
    ///
    /// So this accessor hands out one cell, never the map. The detection driver may
    /// cache as much as it likes; nothing outside it may learn that a whole-dataset
    /// cache exists. **Keep this driver-private** — it must not gain a `pub` variant,
    /// and it must not return `&HashMap<..>`. See CRATE-STRUCTURE.md item 13.
    /// `pub(in crate::detection)`, not `pub(super)`: the visibility is the enforcement.
    pub(in crate::detection) fn cell_ious(
        &self,
        img_id: u64,
        cat_id: u64,
    ) -> Option<&matching::IouMatrix> {
        self.ious.get(&(img_id, cat_id))
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
    /// Behaviour controlled by per-image GT fields:
    /// - `neg_category_ids`: categories confirmed absent → unmatched DTs count as FP.
    /// - `not_exhaustive_category_ids`: categories not fully checked → unmatched DTs ignored.
    ///
    /// Produces 13 metrics: AP, AP50, AP75, APs, APm, APl, APr (rare), APc (common),
    /// APf (frequent), AR@300, ARs@300, ARm@300, ARl@300.
    pub fn new_lvis(coco_gt: COCO, coco_dt: COCO, iou_type: IouType) -> Self {
        let mut params = Params::new(iou_type);
        params.max_dets = vec![300];

        COCOeval {
            coco_gt,
            coco_dt,
            params,
            eval_imgs: Vec::new(),
            ious: HashMap::new(),
            eval: None,
            stats: None,
            eval_mode: EvalMode::Lvis,
            freq_groups: FreqGroups::default(),
            hierarchy: None,
        }
    }

    /// Run the full evaluation pipeline in one call: `evaluate` → `accumulate` → `summarize`.
    ///
    /// Equivalent to calling the three methods in sequence. Primarily used with LVIS
    /// pipelines (e.g. Detectron2 / MMDetection) that expect a single `run()` entry point.
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

        COCOeval {
            coco_gt,
            coco_dt,
            params,
            eval_imgs: Vec::new(),
            ious: HashMap::new(),
            eval: None,
            stats: None,
            eval_mode: EvalMode::OpenImages,
            freq_groups: FreqGroups::default(),
            hierarchy,
        }
    }
}
