//! Count aggregation and metric formulas.
//!
//! This slice provides the **detection rank-based PR accumulator**
//! ([`precision_recall_curve`]) — the pycocotools/PASCAL-VOC precision-at-fixed-
//! recall computation shared by detection's accumulate, TIDE, and diagnostics
//! paths — and [`average_precision`], the mechanical AP core layered on it
//! (sort → classify → cumsum → interpolate → mean).
//!
//! # Empty-set conventions stay at the call site
//!
//! What AP *means* when there is no ground truth is a per-metric decision, not a
//! mechanical one: TIDE reports `0.0` (a vacuous corpus AP), while per-image
//! diagnostics reports `1.0` (an empty image with nothing predicted is
//! legitimately perfect). These are different metrics, not drift, so the
//! primitive takes no policy flag — callers guard `num_gt == 0` themselves and
//! document why.
//!

/// Precision interpolated at fixed recall thresholds, from cumulative TP/FP.
///
/// `tp_cum` and `fp_cum` must already be cumulative (prefix-summed) over
/// detections sorted by score descending. Returns:
/// - the final recall achieved (`tp_cum[nd-1] / num_gt`);
/// - for each recall threshold that is reached, `(threshold_idx, precision,
///   detection_ptr)`, where `detection_ptr` is the rank at which the threshold is
///   first met (lets the caller recover the corresponding sorted score).
///
/// Unreachable recall thresholds are omitted. Precision is made monotonically
/// non-increasing right-to-left before sampling (PASCAL VOC interpolation),
/// matching pycocotools.
pub fn precision_recall_curve(
    tp_cum: &[f64],
    fp_cum: &[f64],
    num_gt: usize,
    rec_thrs: &[f64],
) -> (f64, Vec<(usize, f64, usize)>) {
    let nd = tp_cum.len();
    if nd == 0 || num_gt == 0 {
        return (0.0, vec![]);
    }

    let num_gt_f = num_gt as f64;

    // Recall and precision at each detection rank.
    let mut rc = vec![0.0f64; nd];
    let mut pr = vec![0.0f64; nd];
    for d in 0..nd {
        rc[d] = tp_cum[d] / num_gt_f;
        let total = tp_cum[d] + fp_cum[d];
        pr[d] = if total > 0.0 { tp_cum[d] / total } else { 0.0 };
    }

    let final_recall = rc[nd - 1];

    // Make precision monotonically non-increasing from right to left (VOC interp).
    for d in (0..nd.saturating_sub(1)).rev() {
        pr[d] = pr[d].max(pr[d + 1]);
    }

    // Two-pointer scan: map pr onto fixed recall thresholds.
    let mut result = Vec::with_capacity(rec_thrs.len());
    let mut rc_ptr = 0;
    for (r_idx, &rec_thr) in rec_thrs.iter().enumerate() {
        while rc_ptr < nd && rc[rc_ptr] < rec_thr {
            rc_ptr += 1;
        }
        if rc_ptr < nd {
            result.push((r_idx, pr[rc_ptr], rc_ptr));
        }
    }

    (final_recall, result)
}

/// Average precision over `rec_thrs`, from per-detection match flags.
///
/// The mechanical AP core: sort by score descending → classify each detection as
/// TP/FP (skipping ignored ones) → cumulative sum → interpolate onto `rec_thrs`
/// via [`precision_recall_curve`] → mean. Thresholds beyond the achieved recall
/// contribute zero, matching pycocotools' 101-point convention.
///
/// `scores`, `matched`, and `ignored` (when supplied) are parallel arrays over
/// detections in any order; `ignored = None` means no detection is ignored. The
/// sort is stable, so callers whose input is already score-descending keep their
/// tie order.
///
/// Returns `0.0` when there are no detections or no ground truth — but see the
/// [module note](self) on empty-set conventions: a caller that wants a different
/// answer for `num_gt == 0` must guard before calling.
pub fn average_precision(
    scores: &[f64],
    matched: &[bool],
    ignored: Option<&[bool]>,
    num_gt: usize,
    rec_thrs: &[f64],
) -> f64 {
    let nd = scores.len();
    if nd == 0 || num_gt == 0 || rec_thrs.is_empty() {
        return 0.0;
    }

    let mut order: Vec<usize> = (0..nd).collect();
    order.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut tp_cum = Vec::with_capacity(nd);
    let mut fp_cum = Vec::with_capacity(nd);
    let (mut tp, mut fp) = (0.0f64, 0.0f64);
    for &i in &order {
        if !ignored.is_some_and(|ig| ig[i]) {
            if matched[i] {
                tp += 1.0;
            } else {
                fp += 1.0;
            }
        }
        tp_cum.push(tp);
        fp_cum.push(fp);
    }

    let (_, curve) = precision_recall_curve(&tp_cum, &fp_cum, num_gt, rec_thrs);
    curve.iter().map(|&(_, prec, _)| prec).sum::<f64>() / rec_thrs.len() as f64
}

/// Average precision by the VOC 2010 "all-points" rule — the exact area under the
/// interpolated precision-recall curve, with no recall grid.
///
/// This is the integration COCO does *not* use. COCO samples the same envelope at
/// [`crate::params::default_rec_thrs`]'s 101 points and averages, which quantizes
/// the result: a class with 2 ground truths and 1 true positive scores 0.504950 on
/// the grid against an exact 0.500000. The error is bounded by roughly `1/101` per
/// class, so it matters most where classes have few instances.
///
/// Open Images specifies this rule — "evaluated as in the PASCAL VOC 2010
/// protocol" — and both reference implementations follow it: TensorFlow's
/// `object_detection.utils.metrics.compute_average_precision` and FiftyOne's
/// `_compute_AP`. Notably FiftyOne uses the 101-point grid for its COCO evaluation
/// and this rule for Open Images, so the split is deliberate, not an oversight.
///
/// Takes cumulative counts because that is what the caller already has; deriving
/// them here would duplicate the score-ordering the accumulator has done.
/// `tp_cum` and `fp_cum` must be in score-descending order and the same length.
/// Returns `0.0` for an empty curve or `num_gt == 0`.
///
/// Runs in one reverse pass with no allocation. The reference builds padded
/// `recall`/`precision` arrays first, but both sentinels turn out to be inert: the
/// leading `precision = 0` is never a summation term, and the trailing
/// `(recall = 1, precision = 0)` contributes `(1 - max_recall) * 0`. They collapse
/// into the loop bounds and the initial envelope value. Sweeping right to left also
/// means the envelope is simply the running maximum, so it needs no second pass.
pub fn average_precision_all_points(tp_cum: &[f64], fp_cum: &[f64], num_gt: usize) -> f64 {
    let nd = tp_cum.len();
    if nd == 0 || num_gt == 0 {
        return 0.0;
    }

    let n = num_gt as f64;
    let mut ap = 0.0;
    // Best precision at this recall or beyond. Starts at 0 — the reference's
    // trailing sentinel, which nothing to the right can beat.
    let mut envelope = 0.0f64;

    for i in (0..nd).rev() {
        let denom = tp_cum[i] + fp_cum[i];
        let precision = if denom > 0.0 { tp_cum[i] / denom } else { 0.0 };
        envelope = envelope.max(precision);

        // Divide before subtracting, as the reference does — it builds the recall
        // array first and differences it, so matching that order keeps the
        // arithmetic bit-comparable.
        let recall_prev = if i == 0 { 0.0 } else { tp_cum[i - 1] / n };
        // A step where recall does not move contributes exactly zero, so the
        // reference's explicit filter on that is unnecessary here.
        ap += (tp_cum[i] / n - recall_prev) * envelope;
    }

    ap
}

/// The F-beta score for one precision/recall pair.
///
/// `beta` weights recall relative to precision: `beta = 1` is the harmonic mean
/// (F1), `beta > 1` favors recall, `beta < 1` favors precision. Returns `0.0`
/// when both inputs are zero, where the formula is otherwise `0/0`.
pub fn f_beta(precision: f64, recall: f64, beta: f64) -> f64 {
    let beta2 = beta * beta;
    let denom = beta2 * precision + recall;
    if denom < f64::EPSILON {
        return 0.0;
    }
    (1.0 + beta2) * precision * recall / denom
}

/// The best F-beta achievable anywhere on a precision-recall curve.
///
/// `precisions[i]` is the precision at `recalls[i]`; the pair is the curve
/// [`precision_recall_curve`] produces. Sweeping it answers "how good could this
/// model be at its best operating point?", which is what an F-score reports —
/// unlike AP, which averages over the whole curve.
///
/// Entries with negative precision are skipped: `-1.0` is the crate's
/// "not computed for this configuration" sentinel, not a real low score.
/// Returns `None` when no entry is valid, so callers pick their own convention
/// for an undefined score rather than inheriting one.
pub fn max_f_beta(precisions: &[f64], recalls: &[f64], beta: f64) -> Option<f64> {
    let n = precisions.len().min(recalls.len());
    let mut best = f64::NEG_INFINITY;
    for i in 0..n {
        if precisions[i] < 0.0 {
            continue;
        }
        best = best.max(f_beta(precisions[i], recalls[i], beta));
    }
    (best > f64::NEG_INFINITY).then_some(best)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// The shape guarantees `precision_recall_curve` makes to its callers.
    ///
    /// `report()`'s PR curves and [`max_f_beta`] both read this output directly,
    /// and both assume it is a well-formed curve rather than an arbitrary bag of
    /// points. VOC interpolation makes precision non-increasing in `r_idx`, and
    /// the two-pointer scan advances monotonically, so `detection_ptr` is
    /// non-decreasing and every emitted threshold is genuinely reached.
    #[test]
    fn precision_recall_curve_is_well_formed() {
        let mut rng = StdRng::seed_from_u64(0xC0_1174);
        let rec_thrs = crate::params::default_rec_thrs();

        for case in 0..5000 {
            let nd = rng.random_range(1..=40);
            let num_gt = rng.random_range(1..=25);

            // Cumulative TP/FP over score-descending detections: each step adds
            // one to exactly one of them, or to neither when ignored.
            let (mut tp_cum, mut fp_cum) = (Vec::with_capacity(nd), Vec::with_capacity(nd));
            let (mut tp, mut fp) = (0.0f64, 0.0f64);
            for _ in 0..nd {
                match rng.random_range(0..3) {
                    0 => tp += 1.0,
                    1 => fp += 1.0,
                    _ => {} // ignored: contributes to neither
                }
                tp_cum.push(tp);
                fp_cum.push(fp);
            }

            let (final_recall, curve) = precision_recall_curve(&tp_cum, &fp_cum, num_gt, &rec_thrs);
            let ctx = format!("case {case}: nd={nd} num_gt={num_gt}");

            // The recall a curve reports is the recall its last detection achieves.
            assert!(
                (final_recall - tp_cum[nd - 1] / num_gt as f64).abs() < 1e-12,
                "{ctx}: final_recall {final_recall} disagrees with tp_cum/num_gt"
            );

            let mut prev_r_idx: Option<usize> = None;
            let mut prev_precision = f64::INFINITY;
            let mut prev_ptr = 0usize;

            for &(r_idx, precision, ptr) in &curve {
                assert!(r_idx < rec_thrs.len(), "{ctx}: r_idx {r_idx} out of range");
                assert!(ptr < nd, "{ctx}: detection_ptr {ptr} out of range");
                assert!(
                    (0.0..=1.0).contains(&precision),
                    "{ctx}: precision {precision} outside [0,1]"
                );

                if let Some(prev) = prev_r_idx {
                    assert!(r_idx > prev, "{ctx}: r_idx went {prev} -> {r_idx}");
                    assert!(
                        precision <= prev_precision + 1e-12,
                        "{ctx}: precision rose {prev_precision} -> {precision} at r_idx {r_idx}"
                    );
                    assert!(
                        ptr >= prev_ptr,
                        "{ctx}: detection_ptr went backwards {prev_ptr} -> {ptr}"
                    );
                }

                // An emitted threshold must actually be reached by that detection.
                assert!(
                    tp_cum[ptr] / num_gt as f64 >= rec_thrs[r_idx] - 1e-12,
                    "{ctx}: r_idx {r_idx} emitted at ptr {ptr} which does not reach it"
                );

                prev_r_idx = Some(r_idx);
                prev_precision = precision;
                prev_ptr = ptr;
            }

            // Thresholds are emitted exactly while they remain reachable, so the
            // curve is a prefix of the grid.
            let reachable = rec_thrs.iter().filter(|&&t| final_recall >= t).count();
            assert_eq!(
                curve.len(),
                reachable,
                "{ctx}: emitted {} points for {reachable} reachable thresholds \
                 (final_recall {final_recall})",
                curve.len()
            );
        }
    }

    /// All-points AP, derived by hand rather than recorded from our own output.
    ///
    /// Each case is small enough to integrate on paper, which is the point: the
    /// end-to-end check against TensorFlow lives in `scripts/parity_oid.py`, and
    /// this pins the arithmetic so a failure there localises to the reference
    /// rather than to this function.
    #[test]
    fn all_points_ap_matches_hand_derived_values() {
        // 2 GT, 1 found. Envelope is precision 1.0 over recall [0, 0.5], then 0.
        //   AP = (0.5 - 0.0) * 1.0 + (1.0 - 0.5) * 0.0 = 0.5
        assert_eq!(average_precision_all_points(&[1.0], &[0.0], 2), 0.5);

        // 2 GT, both found, no false positives — precision 1.0 across the board.
        //   AP = 0.5 * 1.0 + 0.5 * 1.0 = 1.0
        assert_eq!(
            average_precision_all_points(&[1.0, 2.0], &[0.0, 0.0], 2),
            1.0
        );

        // 1 GT, a false positive ranked above the true positive.
        //   raw:      recall [0.0, 1.0], precision [0.0, 0.5]
        //   envelope: precision 0.5 everywhere to the left of full recall
        //   AP = (1.0 - 0.0) * 0.5 = 0.5
        assert_eq!(
            average_precision_all_points(&[0.0, 1.0], &[1.0, 1.0], 1),
            0.5
        );

        // The quantisation this function exists to avoid: the 101-point grid
        // reports 51/101 for the first case above, not 0.5.
        let grid = average_precision(&[0.9], &[true], None, 2, &crate::params::default_rec_thrs());
        assert!((grid - 51.0 / 101.0).abs() < 1e-12);
        assert!(
            (grid - 0.5).abs() > 1e-3,
            "the two integrations must actually differ"
        );

        // Degenerate inputs agree with the empty-set convention in the module note.
        assert_eq!(average_precision_all_points(&[], &[], 5), 0.0);
        assert_eq!(average_precision_all_points(&[1.0], &[0.0], 0), 0.0);
    }

    /// `f_beta` is a weighted harmonic mean, so it is bounded by its inputs and
    /// collapses to them when they agree.
    #[test]
    fn f_beta_algebraic_properties() {
        let mut rng = StdRng::seed_from_u64(0xFBE7A);

        for case in 0..20000 {
            let p: f64 = rng.random_range(0.0..=1.0);
            let r: f64 = rng.random_range(0.0..=1.0);
            let beta: f64 = rng.random_range(0.1..=5.0);

            let f = f_beta(p, r, beta);
            let ctx = format!("case {case}: p={p} r={r} beta={beta}");

            assert!((0.0..=1.0).contains(&f), "{ctx}: f_beta {f} outside [0,1]");
            // A mean cannot exceed its largest input nor fall below its smallest.
            assert!(f <= p.max(r) + 1e-12, "{ctx}: f_beta {f} above max(p,r)");
            assert!(f >= p.min(r) - 1e-12, "{ctx}: f_beta {f} below min(p,r)");

            // Equal inputs collapse to that value for every beta — the weighting
            // has nothing left to trade off.
            let equal = f_beta(p, p, beta);
            assert!(
                (equal - p).abs() < 1e-12,
                "{ctx}: f_beta(p, p, beta) = {equal}, expected {p}"
            );

            // max_f_beta is a maximum over the curve, so it dominates every point.
            if let Some(best) = max_f_beta(&[p], &[r], beta) {
                assert!(
                    (best - f).abs() < 1e-12,
                    "{ctx}: max over one point != that point"
                );
            }
        }
    }

    #[test]
    fn f_beta_at_one_is_the_harmonic_mean() {
        assert!((f_beta(0.5, 0.5, 1.0) - 0.5).abs() < 1e-12);
        // Harmonic mean of 1.0 and 0.5 is 2/3.
        assert!((f_beta(1.0, 0.5, 1.0) - 2.0 / 3.0).abs() < 1e-12);
        // Both zero would be 0/0; defined as 0.
        assert_eq!(f_beta(0.0, 0.0, 1.0), 0.0);
    }

    #[test]
    fn beta_shifts_the_weight_between_precision_and_recall() {
        // High precision, low recall. beta < 1 favors precision, so scores higher.
        let (p, r) = (0.9, 0.3);
        assert!(f_beta(p, r, 0.5) > f_beta(p, r, 1.0));
        assert!(f_beta(p, r, 2.0) < f_beta(p, r, 1.0));
    }

    #[test]
    fn max_f_beta_sweeps_the_curve_for_the_best_point() {
        // Best F1 is at the middle point: f_beta(0.6, 0.6) = 0.6.
        let precisions = [1.0, 0.6, 0.2];
        let recalls = [0.1, 0.6, 0.9];
        let best = max_f_beta(&precisions, &recalls, 1.0).expect("a valid point exists");
        assert!((best - 0.6).abs() < 1e-12);
    }

    #[test]
    fn max_f_beta_skips_the_missing_data_sentinel() {
        // -1.0 means "not computed", not "precision of -1".
        assert_eq!(max_f_beta(&[-1.0, -1.0], &[0.5, 0.5], 1.0), None);
        let best = max_f_beta(&[-1.0, 0.5], &[0.1, 0.5], 1.0).expect("one valid point");
        assert!((best - 0.5).abs() < 1e-12);
    }

    #[test]
    fn empty_or_no_gt_is_zero() {
        assert_eq!(precision_recall_curve(&[], &[], 5, &[0.5]), (0.0, vec![]));
        assert_eq!(
            precision_recall_curve(&[1.0], &[0.0], 0, &[0.5]),
            (0.0, vec![])
        );
    }

    #[test]
    fn perfect_detections_precision_one() {
        // 4 TPs, no FPs, 4 GTs => recall reaches 1.0, precision 1.0 throughout.
        let tp = [1.0, 2.0, 3.0, 4.0];
        let fp = [0.0, 0.0, 0.0, 0.0];
        let (final_recall, curve) = precision_recall_curve(&tp, &fp, 4, &[0.0, 0.5, 1.0]);
        assert!((final_recall - 1.0).abs() < 1e-12);
        assert_eq!(curve.len(), 3);
        for (_, p, _) in &curve {
            assert!((p - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn unreachable_recall_thresholds_omitted() {
        // 1 TP among 4 GTs => max recall 0.25; thresholds above are dropped.
        let tp = [1.0, 1.0];
        let fp = [0.0, 1.0];
        let (final_recall, curve) = precision_recall_curve(&tp, &fp, 4, &[0.1, 0.25, 0.5, 1.0]);
        assert!((final_recall - 0.25).abs() < 1e-12);
        // only the 0.1 and 0.25 thresholds are reachable
        assert_eq!(curve.iter().map(|c| c.0).collect::<Vec<_>>(), vec![0, 1]);
    }

    #[test]
    fn voc_interpolation_makes_precision_monotone() {
        // Raw precision dips then recovers; interpolation lifts the dip to the
        // later higher value. Ranks: tp=[1,1,2], fp=[0,1,1] => pr=[1, .5, .667],
        // recall=[.33,.33,.67]. After right-to-left max: [1, .667, .667].
        let tp = [1.0, 1.0, 2.0];
        let fp = [0.0, 1.0, 1.0];
        let (_, curve) = precision_recall_curve(&tp, &fp, 3, &[0.5]);
        // recall 0.5 first met at rank 2 (recall .667); interpolated precision .667
        assert_eq!(curve.len(), 1);
        let (_, p, ptr) = curve[0];
        assert_eq!(ptr, 2);
        assert!((p - 2.0 / 3.0).abs() < 1e-12);
    }
}
