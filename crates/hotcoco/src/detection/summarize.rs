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

pub(super) fn per_cat_ap_static(eval: &AccumulatedEval, params: &Params) -> Vec<f64> {
    let a_idx = params.all_area_idx();
    let m_idx = eval.shape.m - 1;
    (0..eval.shape.k)
        .map(|k_idx| {
            let mut sum = 0.0;
            let mut count = 0_usize;
            for t_idx in 0..eval.shape.t {
                for r_idx in 0..eval.shape.r {
                    let idx = eval.precision_idx(t_idx, r_idx, k_idx, a_idx, m_idx);
                    let v = eval.precision[idx];
                    if v >= 0.0 {
                        sum += v;
                        count += 1;
                    }
                }
            }
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
            // The *nearest* threshold within tolerance, not every threshold within
            // it. A single-threshold metric like AP50 means one slice of the IoU
            // axis; collecting all matches meant that a params list holding two
            // thresholds within 1e-9 of each other silently reported their average
            // under the name of one of them. Picking the nearest is deterministic
            // and degenerates to the same answer in every well-formed config.
            //
            // The tolerance itself is not a fudge: `thr` is a caller-supplied
            // `f64` compared against a grid built by `params::linspace`, so exact
            // equality would fail on values that are 0.5 in every sense that
            // matters.
            params
                .iou_thrs
                .iter()
                .enumerate()
                .filter(|&(_, &t)| (t - thr).abs() < 1e-9)
                .min_by(|&(_, &a), &(_, &b)| {
                    (a - thr)
                        .abs()
                        .partial_cmp(&(b - thr).abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| vec![i])
                .unwrap_or_default()
        } else {
            (0..eval.shape.t).collect()
        };

        let mut vals = Vec::new();
        for &t_idx in &t_indices {
            for k_idx in 0..eval.shape.k {
                if ap {
                    for r_idx in 0..eval.shape.r {
                        let idx = eval.precision_idx(t_idx, r_idx, k_idx, a_idx, m_idx);
                        let v = eval.precision[idx];
                        if v >= 0.0 {
                            vals.push(v);
                        }
                    }
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
        Some(per_cat_ap_static(eval, params))
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
