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

/// Mean of the values that were actually computed, or the `-1.0` sentinel.
///
/// [`mean_or_missing`]'s companion, and the other half of the same convention:
/// that function owns what an empty mean *is*, this one owns **which values are
/// allowed into it**. Five sites spelled the pair out by hand — per-category AP,
/// both branches of `summarize_stat`, the LVIS frequency buckets, and
/// `report()`'s PR curves — and each was one edit away from averaging a `-1.0`
/// in as if it were a real score of minus one.
///
/// Takes an iterator and folds `(sum, count)` in visit order rather than
/// collecting: the caller's iteration order *is* the summation order, so the last
/// bit of every AP is whatever the hand-written loop produced. The filter is
/// [`metrics::is_computed`](crate::metrics), the crate's one sentinel predicate.
pub(super) fn mean_of_valid(values: impl Iterator<Item = f64>) -> f64 {
    let (sum, count) = values
        .filter(|&v| crate::metrics::is_computed(v))
        .fold((0.0f64, 0usize), |(s, c), v| (s + v, c + 1));
    mean_or_missing(sum, count)
}

/// B minus A, treating a metric missing from either side as no evidence.
///
/// The subtraction counterpart of [`mean_or_missing`], and it lives beside it for
/// the same reason: `-1.0` is "not computed for this configuration", so
/// subtracting it manufactures a swing of up to 1.0 out of missing data. The
/// comparison point estimate, its bootstrap CIs, and the per-slice deltas must all
/// agree on that, which is why it is one function — the slice path had its own
/// inlined copy, free to drift from the one `compare()` uses.
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
/// The `accumulate_impl` → `summarize_impl` pair takes six arguments across the
/// two calls, five of which are fields of the same evaluator; it was spelled out
/// at every re-summarization site (`compare`, its bootstrap statistic, and both
/// halves of `slice_by`). One of those forgetting `freq_groups` or passing the
/// *other* evaluator's params is a wrong number with nothing to catch it.
///
/// `img_filter` of `None` means the full dataset. Both halves of the result are
/// returned because callers need different parts: comparison reads the
/// accumulated eval for per-category AP, slicing and bootstrapping only the stats.
///
/// The evaluator arrives inside the [`EvalGrouping`] rather than beside it: every
/// caller here re-summarizes the *same* evaluator many times over different image
/// subsets, the grouping is invariant across those, and pairing it with a
/// different evaluator's params would be a wrong number with nothing to catch it.
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
        .filter(|&v| crate::metrics::is_computed(v))
}

pub(super) fn per_cat_ap_static(
    eval: &AccumulatedEval,
    params: &Params,
    eval_mode: EvalMode,
) -> Vec<f64> {
    let a_idx = params.all_area_idx();
    // `max_det_idx`, not `shape.m - 1`. The two agree on the sorted default
    // `[1, 10, 100]` and diverge on anything else, and this vector feeds
    // per-class AP in `report()`, `get_results(per_class = true)` and
    // `compare()` — so `max_dets = [100, 10, 1]` reported every class at
    // `max_det = 1` beside a headline `AP` computed at 100.
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
        // computed" sentinel, exactly like the missing-IoU-threshold branch
        // below. Falling back to index 0 instead reported the "all" slice (or
        // an arbitrary M slot) under a per-size metric's name — a plausible
        // wrong number with nothing to flag it.
        let Some(a_idx) = params.area_range_idx(area_lbl) else {
            return -1.0;
        };
        let Some(m_idx) = params.max_dets.iter().position(|&d| d == max_det) else {
            return -1.0;
        };

        let t_indices: Vec<usize> = if let Some(thr) = iou_thr {
            // `Params::iou_thr_idx` owns this lookup — a single-threshold metric
            // like AP50 means one slice of the IoU axis, and taking every
            // threshold within tolerance silently reported their average.
            params.iou_thr_idx(thr).map(|i| vec![i]).unwrap_or_default()
        } else {
            (0..eval.shape.t).collect()
        };

        // Folded rather than collected, in the same (t, k, r) visit order the Vec
        // was filled and summed in — so the addition sequence, and therefore the
        // last bit of every AP, is unchanged. The Vec held up to T×K×R f64
        // (~646 KB on COCO) purely to take its mean, twice per bootstrap resample.
        //
        // The two branches are separate iterators rather than one loop with an
        // `if` inside, because the AP branch yields R samples per (t, k) cell and
        // the AR branch yields one. Both visit (t, k) in the same order the loop
        // did.
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

    // LVIS only. Open Images computed this vector and threw it away — its single
    // metric has no frequency group — and the `unwrap_or(&[])` fallback below
    // then indexed an empty slice, so any future mode with a frequency metric and
    // no per-category AP would have panicked rather than degraded.
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
