//! The summary-metric catalog: which metrics each evaluation mode reports,
//! in canonical display order.
//!
//! Pure configuration — every function here is a `MetricDef` constructor and
//! none of them touch evaluation data. The reduction that turns these
//! definitions into numbers lives in [`super::summarize`]; the formatting that
//! turns those numbers into output lives in [`super::report`].

use crate::params::{IouType, Params};

use super::EvalMode;
use super::mode::FreqGroup;

/// Definition of a single summary metric (one row in the COCO output table).
///
/// The catalog entry behind every headline number: what the metric is called,
/// which axis slices it reads, and — for LVIS frequency buckets — which group of
/// categories it averages. [`COCOeval::metric_defs`](super::COCOeval::metric_defs)
/// hands out the list for the current mode, in the same order as
/// [`metric_keys`](super::COCOeval::metric_keys) and `stats`.
///
/// Public so a renderer can label a metric from its definition rather than by
/// parsing its name. Deriving `IoU=0.50 | area=small | maxDets=100` back out of
/// the string `"APs"` is exactly the re-derivation this type exists to prevent.
#[derive(Debug, Clone)]
pub struct MetricDef {
    /// Short metric name, e.g. "AP", "AP50", "ARs". Used as the key in `get_results()`.
    pub name: &'static str,
    /// true = Average Precision, false = Average Recall.
    pub ap: bool,
    /// Specific IoU threshold, or None to average over all thresholds.
    pub iou_thr: Option<f64>,
    /// Area range label to filter by (e.g. "all", "small", "medium", "large").
    pub area_lbl: &'static str,
    /// Maximum detections per image for this metric.
    pub max_det: usize,
    /// LVIS frequency-group AP. When `Some(_)`, all other fields are unused;
    /// value is mean per-category AP for that frequency bucket.
    pub freq_group: Option<FreqGroup>,
}

impl MetricDef {
    /// An Average Precision row, averaged over the whole IoU sweep.
    const fn ap(name: &'static str, area_lbl: &'static str, max_det: usize) -> Self {
        MetricDef {
            name,
            ap: true,
            iou_thr: None,
            area_lbl,
            max_det,
            freq_group: None,
        }
    }

    /// An Average Recall row, averaged over the whole IoU sweep.
    const fn ar(name: &'static str, area_lbl: &'static str, max_det: usize) -> Self {
        MetricDef {
            name,
            ap: false,
            iou_thr: None,
            area_lbl,
            max_det,
            freq_group: None,
        }
    }

    /// Pin this row to a single IoU threshold — `AP50`, `AR75`.
    const fn at(mut self, iou_thr: f64) -> Self {
        self.iou_thr = Some(iou_thr);
        self
    }

    /// Make this row an LVIS frequency-bucket AP. Every other axis goes unused;
    /// the value is the mean per-category AP over that bucket.
    const fn freq(mut self, group: FreqGroup) -> Self {
        self.freq_group = Some(group);
        self
    }
}

pub(super) fn metrics_bbox_segm(max_d: usize, max_d_s: usize, max_d_m: usize) -> Vec<MetricDef> {
    vec![
        MetricDef::ap("AP", "all", max_d),
        MetricDef::ap("AP50", "all", max_d).at(0.5),
        MetricDef::ap("AP75", "all", max_d).at(0.75),
        MetricDef::ap("APs", "small", max_d),
        MetricDef::ap("APm", "medium", max_d),
        MetricDef::ap("APl", "large", max_d),
        MetricDef::ar("AR1", "all", max_d_s),
        MetricDef::ar("AR10", "all", max_d_m),
        MetricDef::ar("AR100", "all", max_d),
        MetricDef::ar("ARs", "small", max_d),
        MetricDef::ar("ARm", "medium", max_d),
        MetricDef::ar("ARl", "large", max_d),
    ]
}

pub(super) fn metrics_kp(max_d: usize) -> Vec<MetricDef> {
    vec![
        MetricDef::ap("AP", "all", max_d),
        MetricDef::ap("AP50", "all", max_d).at(0.5),
        MetricDef::ap("AP75", "all", max_d).at(0.75),
        MetricDef::ap("APm", "medium", max_d),
        MetricDef::ap("APl", "large", max_d),
        MetricDef::ar("AR", "all", max_d),
        MetricDef::ar("AR50", "all", max_d).at(0.5),
        MetricDef::ar("AR75", "all", max_d).at(0.75),
        MetricDef::ar("ARm", "medium", max_d),
        MetricDef::ar("ARl", "large", max_d),
    ]
}

pub(super) fn metrics_lvis(max_d: usize) -> Vec<MetricDef> {
    vec![
        MetricDef::ap("AP", "all", max_d),
        MetricDef::ap("AP50", "all", max_d).at(0.5),
        MetricDef::ap("AP75", "all", max_d).at(0.75),
        MetricDef::ap("APs", "small", max_d),
        MetricDef::ap("APm", "medium", max_d),
        MetricDef::ap("APl", "large", max_d),
        MetricDef::ap("APr", "all", max_d).freq(FreqGroup::Rare),
        MetricDef::ap("APc", "all", max_d).freq(FreqGroup::Common),
        MetricDef::ap("APf", "all", max_d).freq(FreqGroup::Frequent),
        MetricDef::ar("AR@300", "all", max_d),
        MetricDef::ar("ARs@300", "small", max_d),
        MetricDef::ar("ARm@300", "medium", max_d),
        MetricDef::ar("ARl@300", "large", max_d),
    ]
}

/// Resolve max_dets into (default, small, medium) triple.
///
/// Positions are read from a sorted view so the triple is independent of the
/// caller's ordering — pycocotools sorts `maxDets` before its positional reads,
/// and `Params::max_det()` (the `default` here) is order-insensitive by
/// definition.
fn resolve_max_dets(params: &Params) -> (usize, usize, usize) {
    let default = params.max_det();
    let (small, med) = if params.max_dets.len() >= 3 {
        let mut sorted = params.max_dets.clone();
        sorted.sort_unstable();
        (sorted[0], sorted[1])
    } else {
        (default, default)
    };
    (default, small, med)
}

/// Open Images metrics: single AP at IoU=0.5.
fn metrics_oid(max_d: usize) -> Vec<MetricDef> {
    vec![MetricDef::ap("AP", "all", max_d).at(0.5)]
}

/// Build the MetricDef vec for the current evaluation mode.
pub(super) fn build_metric_defs(params: &Params, eval_mode: EvalMode) -> Vec<MetricDef> {
    let (max_d, max_d_s, max_d_m) = resolve_max_dets(params);
    match eval_mode {
        EvalMode::Lvis => metrics_lvis(max_d),
        EvalMode::OpenImages => metrics_oid(max_d),
        EvalMode::Coco => {
            if params.iou_type == IouType::Keypoints {
                metrics_kp(max_d)
            } else {
                metrics_bbox_segm(max_d, max_d_s, max_d_m)
            }
        }
    }
}
