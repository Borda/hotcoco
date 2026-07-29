//! The summary reduction: accumulated eval arrays + metric definitions -> numbers.
//!
//! This module computes and nothing else. The catalog of *which* metrics exist
//! is [`super::metrics`]; turning the resulting numbers into printed lines,
//! result maps, or DTOs is [`super::report`].

use crate::params::Params;

use super::EvalMode;
use super::accumulate::AccumulatedEval;
use super::catalog::MetricDef;
use super::mode::FreqGroups;

/// Per-category mean AP as a free function (for use by `summarize_impl` and `slice_by`).
/// Mean of `count` values summing to `sum`, or the `-1.0` "not computed" sentinel.
///
/// The crate's most load-bearing convention, in one place. `-1.0` means a metric
/// was not computable for this configuration — no ground truth in an area range,
/// a category absent from the split — and it is *not* a low score: `report()`
/// filters on it before emitting a per-class metric, and
/// [`max_f_beta`](crate::metrics::counts::max_f_beta) skips it. It was spelled out
/// at five sites across two modules; change the sentinel or the validity test at
/// four of them and a category silently reports `-1.0` as a real score.
pub(super) fn mean_or_missing(sum: f64, count: usize) -> f64 {
    if count == 0 { -1.0 } else { sum / count as f64 }
}

/// The AP samples for one `(t, k, a, m)` cell, `-1.0` sentinels already dropped.
///
/// Which integration applies is a property of the *mode*, not of the caller. COCO
/// and LVIS average the precision envelope over the 101 recall thresholds, so a
/// cell yields `r` samples; Open Images takes the exact area under that same
/// envelope, so it yields one. Every AP path routes through here — `summarize_impl`
/// and `per_cat_ap_static`, the latter also serving `report` and `compare` — so a
/// mode check at only some of them cannot silently split the two integrations.
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
        .filter(|&v| v >= 0.0)
}

pub(super) fn per_cat_ap_static(
    eval: &AccumulatedEval,
    params: &Params,
    eval_mode: EvalMode,
) -> Vec<f64> {
    let a_idx = params.all_area_idx();
    let m_idx = eval.shape.m - 1;
    (0..eval.shape.k)
        .map(|k_idx| {
            let (sum, count) = (0..eval.shape.t)
                .flat_map(|t_idx| ap_samples(eval, eval_mode, t_idx, k_idx, a_idx, m_idx))
                .fold((0.0, 0usize), |(s, c), v| (s + v, c + 1));
            mean_or_missing(sum, count)
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
        let a_idx = params.area_range_idx(area_lbl).unwrap_or(0);
        let m_idx = params
            .max_dets
            .iter()
            .position(|&d| d == max_det)
            .unwrap_or(0);

        let t_indices: Vec<usize> = if let Some(thr) = iou_thr {
            // `Params::iou_thr_idx` owns this lookup — a single-threshold metric
            // like AP50 means one slice of the IoU axis, and taking every
            // threshold within tolerance silently reported their average.
            params.iou_thr_idx(thr).map(|i| vec![i]).unwrap_or_default()
        } else {
            (0..eval.shape.t).collect()
        };

        let mut vals = Vec::with_capacity(t_indices.len() * eval.shape.k * eval.shape.r);
        for &t_idx in &t_indices {
            for k_idx in 0..eval.shape.k {
                if ap {
                    vals.extend(ap_samples(eval, eval_mode, t_idx, k_idx, a_idx, m_idx));
                } else {
                    let idx = eval.recall_idx(t_idx, k_idx, a_idx, m_idx);
                    let v = eval.recall[idx];
                    if v >= 0.0 {
                        vals.push(v);
                    }
                }
            }
        }

        mean_or_missing(vals.iter().sum(), vals.len())
    };

    let per_cat_ap = if eval_mode == EvalMode::Lvis || eval_mode == EvalMode::OpenImages {
        Some(per_cat_ap_static(eval, params, eval_mode))
    } else {
        None
    };

    let freq_group_ap = |indices: &[usize]| -> f64 {
        let per_cat = per_cat_ap.as_deref().unwrap_or(&[]);
        let valid: Vec<f64> = indices
            .iter()
            .filter_map(|&k| {
                let v = per_cat[k];
                if v >= 0.0 { Some(v) } else { None }
            })
            .collect();
        mean_or_missing(valid.iter().sum(), valid.len())
    };

    metrics
        .iter()
        .map(|m| {
            if let Some(fg) = m.freq_group {
                freq_group_ap(freq_groups.get(fg))
            } else {
                summarize_stat(m.ap, m.iou_thr, m.area_lbl, m.max_det)
            }
        })
        .collect()
}
