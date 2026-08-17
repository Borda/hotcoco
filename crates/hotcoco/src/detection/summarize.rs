//! The summary reduction: accumulated eval arrays + metric definitions -> numbers.
//!
//! This module computes and nothing else. The catalog of *which* metrics exist
//! is [`super::catalog`]; turning the resulting numbers into printed lines,
//! result maps, or DTOs is [`super::report`].

use std::collections::{BTreeMap, HashSet};

use crate::params::Params;

use super::EvalMode;
use super::accumulate::{AccumulatedEval, EvalGrouping, accumulate_impl};
use super::catalog::MetricDef;
use super::mode::FreqGroups;

/// Mean of `count` values summing to `sum`, or the `-1.0` "not computed" sentinel.
///
/// The sole producer of the sentinel documented on
/// [`metrics::is_computed`](crate::metrics::is_computed), which is how every
/// consumer reads it back.
pub(super) fn mean_or_missing(sum: f64, count: usize) -> f64 {
    if count == 0 { -1.0 } else { sum / count as f64 }
}

/// Mean of the values that were actually computed, or the `-1.0` sentinel.
///
/// [`mean_or_missing`]'s companion: that function owns what an empty mean *is*,
/// this one owns **which values are allowed into it** — averaging a `-1.0` in
/// treats "not computed" as a real score of minus one.
///
/// Folds `(sum, count)` in visit order rather than collecting, so the caller's
/// iteration order is the summation order and the last bit of every AP is
/// reproducible.
pub(super) fn mean_of_valid(values: impl Iterator<Item = f64>) -> f64 {
    let (sum, count) = values
        .filter(|&v| crate::metrics::is_computed(v))
        .fold((0.0f64, 0usize), |(s, c), v| (s + v, c + 1));
    mean_or_missing(sum, count)
}

/// B minus A, treating a metric missing from either side as no evidence.
///
/// The subtraction counterpart of [`mean_or_missing`]: `-1.0` is "not computed",
/// so subtracting it manufactures a swing of up to 1.0 out of missing data. The
/// comparison point estimate, its bootstrap CIs, and the per-slice deltas all
/// route through here so they agree on that.
#[inline]
pub(super) fn metric_delta(a: f64, b: f64) -> f64 {
    if a >= 0.0 && b >= 0.0 { b - a } else { 0.0 }
}

/// Metric names paired with their values, in catalog order.
///
/// `BTreeMap` rather than `HashMap`: these maps are serialized and iterated by
/// callers, and key order is part of what makes a saved comparison or slice
/// diffable.
pub(super) fn stats_to_map(metric_keys: &[&str], stats: &[f64]) -> BTreeMap<String, f64> {
    metric_keys
        .iter()
        .zip(stats.iter())
        .map(|(&k, &v)| (k.to_string(), v))
        .collect()
}

/// Re-accumulate an evaluated `COCOeval` over an image subset and summarize it.
///
/// The single entry point for re-summarization — `compare`, its bootstrap
/// statistic, and both halves of `slice_by`. `img_filter` of `None` means the
/// full dataset. Both halves of the result are returned because callers need
/// different parts: comparison reads the accumulated eval for per-category AP,
/// slicing and bootstrapping only the stats.
///
/// The evaluator arrives inside the [`EvalGrouping`] rather than beside it, so a
/// grouping bucketed under one evaluator's categories cannot be accumulated under
/// another's params.
pub(super) fn accumulate_and_summarize(
    grouping: &EvalGrouping<'_>,
    img_filter: Option<&HashSet<u64>>,
    metrics: &[MetricDef],
) -> (AccumulatedEval, Vec<f64>) {
    let ev = grouping.eval();
    let acc = accumulate_impl(grouping, img_filter);
    let stats = summarize_impl(&acc, &ev.params, ev.eval_mode, ev.freq_groups(), metrics);
    (acc, stats)
}

/// The AP samples for one `(t, k, a, m)` cell, `-1.0` sentinels already dropped.
///
/// Which integration applies is a property of the *mode*, not of the caller. COCO
/// and LVIS average the precision envelope over the 101 recall thresholds, so a
/// cell yields `r` samples; Open Images takes the exact area under that same
/// envelope, so it yields one. Every AP path routes through here, so the two
/// integrations cannot split.
///
/// Returns an iterator rather than filling an out-param so callers keep the shape
/// that suits them: `summarize_impl` extends a shared `Vec`, `per_cat_ap_static`
/// folds into a running `(sum, count)` with no allocation at all.
fn ap_samples(
    eval: &AccumulatedEval,
    eval_mode: EvalMode,
    t_idx: usize,
    k_idx: usize,
    a_idx: usize,
    m_idx: usize,
) -> impl Iterator<Item = f64> + '_ {
    let all_points = eval_mode == EvalMode::OpenImages;
    let n = if all_points { 1 } else { eval.shape.r };
    (0..n)
        .map(move |r_idx| {
            if all_points {
                eval.ap_all_points[eval.recall_idx(t_idx, k_idx, a_idx, m_idx)]
            } else {
                eval.precision[eval.precision_idx(t_idx, r_idx, k_idx, a_idx, m_idx)]
            }
        })
        .filter(|&v| crate::metrics::is_computed(v))
}

pub(super) fn per_cat_ap_static(
    eval: &AccumulatedEval,
    params: &Params,
    eval_mode: EvalMode,
) -> Vec<f64> {
    let a_idx = params.all_area_idx();
    // `max_det_idx`, not `shape.m - 1` — see `Params::max_det_idx`.
    let m_idx = params.max_det_idx();
    (0..eval.shape.k)
        .map(|k_idx| {
            mean_of_valid(
                (0..eval.shape.t)
                    .flat_map(|t_idx| ap_samples(eval, eval_mode, t_idx, k_idx, a_idx, m_idx)),
            )
        })
        .collect()
}

/// Pure computation of summary statistics from accumulated eval data.
///
/// Returns one `f64` per metric in the same order as the MetricDef vec for the
/// current evaluation mode.
pub(super) fn summarize_impl(
    eval: &AccumulatedEval,
    params: &Params,
    eval_mode: EvalMode,
    freq_groups: &FreqGroups,
    metrics: &[MetricDef],
) -> Vec<f64> {
    let summarize_stat = |ap: bool, iou_thr: Option<f64>, area_lbl: &str, max_det: usize| -> f64 {
        // A missing area label or max-det setting degrades to the `-1.0` "not
        // computed" sentinel, like the missing-IoU-threshold branch below.
        // Falling back to index 0 would report the "all" slice under a per-size
        // metric's name — a plausible wrong number with nothing to flag it.
        let Some(a_idx) = params.area_range_idx(area_lbl) else {
            return -1.0;
        };
        let Some(m_idx) = params.max_dets.iter().position(|&d| d == max_det) else {
            return -1.0;
        };

        let t_indices: Vec<usize> = if let Some(thr) = iou_thr {
            // `Params::iou_thr_idx` owns this lookup: a single-threshold metric
            // like AP50 means exactly one slice of the IoU axis, never the
            // average of every threshold within tolerance.
            params.iou_thr_idx(thr).map(|i| vec![i]).unwrap_or_default()
        } else {
            (0..eval.shape.t).collect()
        };

        // Folded rather than collected: materializing the samples costs up to
        // T×K×R f64 (~646 KB on COCO) purely to take their mean, twice per
        // bootstrap resample. The two branches are separate iterators because the
        // AP branch yields R samples per (t, k) cell and the AR branch yields one;
        // both visit (t, k) in the same order, so the summation sequence — and
        // therefore the last bit of every AP — is fixed.
        if ap {
            mean_of_valid(t_indices.iter().flat_map(|&t_idx| {
                (0..eval.shape.k)
                    .flat_map(move |k_idx| ap_samples(eval, eval_mode, t_idx, k_idx, a_idx, m_idx))
            }))
        } else {
            mean_of_valid(t_indices.iter().flat_map(|&t_idx| {
                (0..eval.shape.k)
                    .map(move |k_idx| eval.recall[eval.recall_idx(t_idx, k_idx, a_idx, m_idx)])
            }))
        }
    };

    // LVIS only — it is the one mode with frequency-group metrics. Other modes
    // leave this empty and the `freq_group` arm below never fires for them.
    let per_cat_ap: Vec<f64> = if eval_mode == EvalMode::Lvis {
        per_cat_ap_static(eval, params, eval_mode)
    } else {
        Vec::new()
    };

    metrics
        .iter()
        .map(|m| match m.freq_group {
            // Same `(sum, count)` fold as `summarize_stat`, over the categories
            // in this frequency bucket.
            Some(fg) => mean_of_valid(
                freq_groups
                    .get(fg)
                    .iter()
                    .filter_map(|&k| per_cat_ap.get(k).copied()),
            ),
            None => summarize_stat(m.ap, m.iou_thr, m.area_lbl, m.max_det),
        })
        .collect()
}
