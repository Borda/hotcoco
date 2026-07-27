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
pub(super) struct MetricDef {
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

pub(super) fn metrics_bbox_segm(max_d: usize, max_d_s: usize, max_d_m: usize) -> Vec<MetricDef> {
    vec![
        MetricDef {
            name: "AP",
            ap: true,
            iou_thr: None,
            area_lbl: "all",
            max_det: max_d,
            freq_group: None,
        },
        MetricDef {
            name: "AP50",
            ap: true,
            iou_thr: Some(0.5),
            area_lbl: "all",
            max_det: max_d,
            freq_group: None,
        },
        MetricDef {
            name: "AP75",
            ap: true,
            iou_thr: Some(0.75),
            area_lbl: "all",
            max_det: max_d,
            freq_group: None,
        },
        MetricDef {
            name: "APs",
            ap: true,
            iou_thr: None,
            area_lbl: "small",
            max_det: max_d,
            freq_group: None,
        },
        MetricDef {
            name: "APm",
            ap: true,
            iou_thr: None,
            area_lbl: "medium",
            max_det: max_d,
            freq_group: None,
        },
        MetricDef {
            name: "APl",
            ap: true,
            iou_thr: None,
            area_lbl: "large",
            max_det: max_d,
            freq_group: None,
        },
        MetricDef {
            name: "AR1",
            ap: false,
            iou_thr: None,
            area_lbl: "all",
            max_det: max_d_s,
            freq_group: None,
        },
        MetricDef {
            name: "AR10",
            ap: false,
            iou_thr: None,
            area_lbl: "all",
            max_det: max_d_m,
            freq_group: None,
        },
        MetricDef {
            name: "AR100",
            ap: false,
            iou_thr: None,
            area_lbl: "all",
            max_det: max_d,
            freq_group: None,
        },
        MetricDef {
            name: "ARs",
            ap: false,
            iou_thr: None,
            area_lbl: "small",
            max_det: max_d,
            freq_group: None,
        },
        MetricDef {
            name: "ARm",
            ap: false,
            iou_thr: None,
            area_lbl: "medium",
            max_det: max_d,
            freq_group: None,
        },
        MetricDef {
            name: "ARl",
            ap: false,
            iou_thr: None,
            area_lbl: "large",
            max_det: max_d,
            freq_group: None,
        },
    ]
}

pub(super) fn metrics_kp(max_d: usize) -> Vec<MetricDef> {
    vec![
        MetricDef {
            name: "AP",
            ap: true,
            iou_thr: None,
            area_lbl: "all",
            max_det: max_d,
            freq_group: None,
        },
        MetricDef {
            name: "AP50",
            ap: true,
            iou_thr: Some(0.5),
            area_lbl: "all",
            max_det: max_d,
            freq_group: None,
        },
        MetricDef {
            name: "AP75",
            ap: true,
            iou_thr: Some(0.75),
            area_lbl: "all",
            max_det: max_d,
            freq_group: None,
        },
        MetricDef {
            name: "APm",
            ap: true,
            iou_thr: None,
            area_lbl: "medium",
            max_det: max_d,
            freq_group: None,
        },
        MetricDef {
            name: "APl",
            ap: true,
            iou_thr: None,
            area_lbl: "large",
            max_det: max_d,
            freq_group: None,
        },
        MetricDef {
            name: "AR",
            ap: false,
            iou_thr: None,
            area_lbl: "all",
            max_det: max_d,
            freq_group: None,
        },
        MetricDef {
            name: "AR50",
            ap: false,
            iou_thr: Some(0.5),
            area_lbl: "all",
            max_det: max_d,
            freq_group: None,
        },
        MetricDef {
            name: "AR75",
            ap: false,
            iou_thr: Some(0.75),
            area_lbl: "all",
            max_det: max_d,
            freq_group: None,
        },
        MetricDef {
            name: "ARm",
            ap: false,
            iou_thr: None,
            area_lbl: "medium",
            max_det: max_d,
            freq_group: None,
        },
        MetricDef {
            name: "ARl",
            ap: false,
            iou_thr: None,
            area_lbl: "large",
            max_det: max_d,
            freq_group: None,
        },
    ]
}

pub(super) fn metrics_lvis(max_d: usize) -> Vec<MetricDef> {
    vec![
        MetricDef {
            name: "AP",
            ap: true,
            iou_thr: None,
            area_lbl: "all",
            max_det: max_d,
            freq_group: None,
        },
        MetricDef {
            name: "AP50",
            ap: true,
            iou_thr: Some(0.5),
            area_lbl: "all",
            max_det: max_d,
            freq_group: None,
        },
        MetricDef {
            name: "AP75",
            ap: true,
            iou_thr: Some(0.75),
            area_lbl: "all",
            max_det: max_d,
            freq_group: None,
        },
        MetricDef {
            name: "APs",
            ap: true,
            iou_thr: None,
            area_lbl: "small",
            max_det: max_d,
            freq_group: None,
        },
        MetricDef {
            name: "APm",
            ap: true,
            iou_thr: None,
            area_lbl: "medium",
            max_det: max_d,
            freq_group: None,
        },
        MetricDef {
            name: "APl",
            ap: true,
            iou_thr: None,
            area_lbl: "large",
            max_det: max_d,
            freq_group: None,
        },
        MetricDef {
            name: "APr",
            ap: true,
            iou_thr: None,
            area_lbl: "all",
            max_det: max_d,
            freq_group: Some(FreqGroup::Rare),
        },
        MetricDef {
            name: "APc",
            ap: true,
            iou_thr: None,
            area_lbl: "all",
            max_det: max_d,
            freq_group: Some(FreqGroup::Common),
        },
        MetricDef {
            name: "APf",
            ap: true,
            iou_thr: None,
            area_lbl: "all",
            max_det: max_d,
            freq_group: Some(FreqGroup::Frequent),
        },
        MetricDef {
            name: "AR@300",
            ap: false,
            iou_thr: None,
            area_lbl: "all",
            max_det: max_d,
            freq_group: None,
        },
        MetricDef {
            name: "ARs@300",
            ap: false,
            iou_thr: None,
            area_lbl: "small",
            max_det: max_d,
            freq_group: None,
        },
        MetricDef {
            name: "ARm@300",
            ap: false,
            iou_thr: None,
            area_lbl: "medium",
            max_det: max_d,
            freq_group: None,
        },
        MetricDef {
            name: "ARl@300",
            ap: false,
            iou_thr: None,
            area_lbl: "large",
            max_det: max_d,
            freq_group: None,
        },
    ]
}

/// Resolve max_dets into (default, small, medium) triple.
fn resolve_max_dets(params: &Params) -> (usize, usize, usize) {
    let default = *params.max_dets.last().unwrap_or(&100);
    let small = if params.max_dets.len() >= 3 {
        params.max_dets[0]
    } else {
        default
    };
    let med = if params.max_dets.len() >= 3 {
        params.max_dets[1]
    } else {
        default
    };
    (default, small, med)
}

/// Open Images metrics: single AP at IoU=0.5.
fn metrics_oid(max_d: usize) -> Vec<MetricDef> {
    vec![MetricDef {
        name: "AP",
        ap: true,
        iou_thr: Some(0.5),
        area_lbl: "all",
        max_det: max_d,
        freq_group: None,
    }]
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
