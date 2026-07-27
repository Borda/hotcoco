//! The summary reduction: accumulated eval arrays + metric definitions -> numbers.
//!
//! This module computes and nothing else. The catalog of *which* metrics exist
//! is [`super::metrics`]; turning the resulting numbers into printed lines,
//! result maps, or DTOs is [`super::report`].

use crate::params::Params;

use super::EvalMode;
use super::metrics::MetricDef;
use super::types::{AccumulatedEval, FreqGroups};

/// Per-category mean AP as a free function (for use by `summarize_impl` and `slice_by`).
pub(super) fn per_cat_ap_static(eval: &AccumulatedEval, params: &Params) -> Vec<f64> {
    let a_idx = params.area_range_idx("all").unwrap_or(0);
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
            if count == 0 { -1.0 } else { sum / count as f64 }
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
            params
                .iou_thrs
                .iter()
                .enumerate()
                .filter(|&(_, &t)| (t - thr).abs() < 1e-9)
                .map(|(i, _)| i)
                .collect()
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

        if vals.is_empty() {
            -1.0
        } else {
            vals.iter().sum::<f64>() / vals.len() as f64
        }
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
        if valid.is_empty() {
            -1.0
        } else {
            valid.iter().sum::<f64>() / valid.len() as f64
        }
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
