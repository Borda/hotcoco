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
//! The broader count vocabulary the plan envisions here — a `GroupKey` unifying
//! the class axis and structs for TP/FP/FN, TPA/FPA/FNA, IDSW, per-track
//! coverage, and matched-similarity sums (MOTP/LocA/SQ) — is intentionally NOT
//! built yet. Those serve tracking/panoptic/concepts, which don't exist; adding
//! them now would shape the contract around detection alone (the plan's risk #2)
//! and introduce enum variants nothing matches on. They land additively with the
//! family that needs them.

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

#[cfg(test)]
mod tests {
    use super::*;

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
