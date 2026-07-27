//! COCO greedy matching — the detection matcher as a first-class, documented stage.
//!
//! This is pycocotools' per-image detection→ground-truth assignment, lifted out
//! of the eval monolith so the detection driver and its LVIS/OID variants share
//! one matcher. The match *choices* are public contract (surfaced through
//! `evalImgs` `dtMatches`/`gtMatches`), so the algorithm — score-descending
//! iteration, best-IoU selection, the two ignore phases, crowd re-match, and tie
//! behavior — is reproduced exactly.
//!
//! The caller (driver) owns everything COCO-specific: computing the ignore/crowd
//! flags, ordering detections and ground-truths, building the IoU matrix, and
//! translating the returned indices back into annotation ids. This module owns
//! only the assignment algorithm.
//!
//! # What this matcher is *not* for
//!
//! It looks like a general IoU-threshold matcher; it is not. It is the
//! **detection lineage** matcher (COCO/LVIS/OID/TIDE), and its score-descending
//! greediness is a pycocotools compatibility requirement, not an optimality
//! claim — it does not maximize total similarity.
//!
//! - **Tracking-lineage metrics** (CLEAR, Identity, HOTA) must use
//!   [`crate::primitives::assign::lsap`]: TrackEval solves an optimal assignment
//!   per frame, and a greedy approximation silently changes IDSW/IDF1.
//! - **Panoptic quality** needs no solver at all — an IoU > 0.5 match between
//!   non-overlapping segments is provably unique.
//!
//! Reaching for this matcher outside detection is the likeliest way for the two
//! lineages to collide.
//!
//! # Threshold-epsilon policy: caller-owned
//!
//! pycocotools does not use the raw threshold as its match floor — it starts each
//! detection's search at `min(t, 1 - 1e-10)` (`evaluateImg`: `iou = min([t,
//! 1-1e-10])`). That clamp is **the caller's to apply**: [`greedy_match`] compares
//! against exactly the `iou_thrs` it is handed and adds no epsilon of its own, so
//! a family whose thresholds mean something other than COCO's is not silently
//! given COCO's fudge factor.
//!
//! The clamp is inert for every threshold `< 1.0`, so it never fires on COCO's
//! default `0.5:0.05:0.95` sweep, nor on the single-threshold analysis callers.
//! It is observable only at `t == 1.0`, where pycocotools still matches a pair
//! whose IoU lies in `[1 - 1e-10, 1.0)` and an unclamped caller does not. No
//! caller in this crate clamps today; that difference against pycocotools at
//! `t == 1.0` is a known, deliberate carry-over of v0.4.x behavior, to be
//! settled with the 1.0 detection driver rather than changed mid-0.5.

/// Per-threshold greedy match results, indexed `[T]` over IoU thresholds.
pub struct GreedyMatches {
    /// `[T][D]`: for each threshold and detection (caller's score-descending
    /// order), the matched ground-truth index (caller's GT order) or `None`.
    pub dt_gt: Vec<Vec<Option<usize>>>,
    /// `[T][G]`: whether each ground-truth was matched at each threshold.
    pub gt_matched: Vec<Vec<bool>>,
}

/// Greedy-match detections to ground-truths, pycocotools-exact.
///
/// # Caller ordering contract
/// - Detections are ordered **score-descending** — matching iterates in this order.
/// - Ground-truths are partitioned **non-ignored first**: indices
///   `[0, num_gt_not_ignored)` are non-ignored, `[num_gt_not_ignored, g)` ignored.
/// - `iou_flat` is row-major `[D*G]` (`iou_flat[di * g + gi]`) in that ordering.
///
/// `gt_rematchable` and `gt_phase2_eligible` are length `g`.
///
/// The matrix is flat (single allocation) rather than `sim`'s nested `[D][G]`
/// because the matching loop is `T×D×G` and benefits from contiguous access.
/// This doesn't conflict with the `sim` kernels' `[D][G]` output: a matcher
/// always sits behind a reorder step (detections score-descending, GTs
/// non-ignored-first), and that step is where the reordered flat matrix is
/// produced — sim's raw output is never fed in directly.
///
/// # Algorithm (per IoU threshold, per detection in score order)
/// Phase 1 scans non-ignored GTs for the highest-IoU available match `>= thr`.
/// Only if phase 1 finds nothing does phase 2 scan the ignored GTs. A GT already
/// matched is skipped unless `gt_rematchable[gi]` (crowd GTs, which multiple
/// detections may match). Phase 2 additionally skips any GT with
/// `gt_phase2_eligible[gi] == false` (e.g. OID group-of, matched in a separate
/// driver pass). Among equal IoUs the later GT index wins — matching
/// pycocotools' `>=` update rule, which is observable through `evalImgs`.
pub fn greedy_match(
    iou_flat: &[f64],
    d: usize,
    g: usize,
    num_gt_not_ignored: usize,
    gt_rematchable: &[bool],
    gt_phase2_eligible: &[bool],
    iou_thrs: &[f64],
) -> GreedyMatches {
    let t = iou_thrs.len();
    let mut dt_gt = vec![vec![None; d]; t];
    let mut gt_matched = vec![vec![false; g]; t];

    for (ti, &iou_thr) in iou_thrs.iter().enumerate() {
        for (di, dt_slot) in dt_gt[ti].iter_mut().enumerate() {
            let base = di * g;
            let mut best_iou = iou_thr;
            let mut best_gi: Option<usize> = None;

            // Phase 1: non-ignored GTs — highest-IoU available match.
            for gi in 0..num_gt_not_ignored {
                if gt_matched[ti][gi] && !gt_rematchable[gi] {
                    continue;
                }
                let iou_val = iou_flat[base + gi];
                if iou_val >= best_iou {
                    best_iou = iou_val;
                    best_gi = Some(gi);
                }
            }

            // Phase 2: ignored GTs — only if phase 1 found no match.
            if best_gi.is_none() {
                for gi in num_gt_not_ignored..g {
                    if !gt_phase2_eligible[gi] {
                        continue;
                    }
                    if gt_matched[ti][gi] && !gt_rematchable[gi] {
                        continue;
                    }
                    let iou_val = iou_flat[base + gi];
                    if iou_val >= best_iou {
                        best_iou = iou_val;
                        best_gi = Some(gi);
                    }
                }
            }

            if let Some(gi) = best_gi {
                *dt_slot = Some(gi);
                gt_matched[ti][gi] = true;
            }
        }
    }

    GreedyMatches { dt_gt, gt_matched }
}

#[cfg(test)]
mod tests {
    use super::*;

    // No crowd, all GTs eligible for phase 2.
    fn simple(iou_flat: &[f64], d: usize, g: usize, num_ni: usize, thrs: &[f64]) -> GreedyMatches {
        greedy_match(
            iou_flat,
            d,
            g,
            num_ni,
            &vec![false; g],
            &vec![true; g],
            thrs,
        )
    }

    #[test]
    fn matches_highest_iou_above_threshold() {
        // 1 DT, 2 non-ignored GTs; GT1 has higher IoU.
        let m = simple(&[0.6, 0.9], 1, 2, 2, &[0.5]);
        assert_eq!(m.dt_gt[0][0], Some(1));
        assert_eq!(m.gt_matched[0], vec![false, true]);
    }

    #[test]
    fn below_threshold_is_no_match() {
        let m = simple(&[0.4, 0.49], 1, 2, 2, &[0.5]);
        assert_eq!(m.dt_gt[0][0], None);
    }

    #[test]
    fn score_order_gives_earlier_dt_first_pick() {
        // 2 DTs (score-desc), 1 GT. DT0 (first) takes it; DT1 gets nothing.
        let m = simple(&[0.9, 0.8], 2, 1, 1, &[0.5]);
        assert_eq!(m.dt_gt[0][0], Some(0));
        assert_eq!(m.dt_gt[0][1], None);
    }

    #[test]
    fn phase1_preferred_over_better_ignored_gt() {
        // g=2: gi0 non-ignored (IoU 0.6), gi1 ignored (IoU 0.99). Phase 1 finds
        // gi0, so phase 2 never runs even though gi1 has higher IoU.
        let m = simple(&[0.6, 0.99], 1, 2, 1, &[0.5]);
        assert_eq!(m.dt_gt[0][0], Some(0));
    }

    #[test]
    fn falls_back_to_ignored_gt_when_no_phase1_match() {
        // gi0 non-ignored but below threshold (0.4); gi1 ignored at 0.8.
        let m = simple(&[0.4, 0.8], 1, 2, 1, &[0.5]);
        assert_eq!(m.dt_gt[0][0], Some(1));
    }

    #[test]
    fn crowd_gt_rematched_by_multiple_dts() {
        // 2 DTs, 1 ignored crowd GT (rematchable). Both DTs match it.
        let m = greedy_match(&[0.9, 0.8], 2, 1, 0, &[true], &[true], &[0.5]);
        assert_eq!(m.dt_gt[0][0], Some(0));
        assert_eq!(m.dt_gt[0][1], Some(0));
    }

    #[test]
    fn non_rematchable_gt_taken_only_once() {
        let m = greedy_match(&[0.9, 0.8], 2, 1, 0, &[false], &[true], &[0.5]);
        assert_eq!(m.dt_gt[0][0], Some(0));
        assert_eq!(m.dt_gt[0][1], None);
    }

    #[test]
    fn phase2_ineligible_gt_is_skipped() {
        // gi0 non-ignored below threshold; gi1 ignored at 0.9 but phase2-ineligible.
        let m = greedy_match(
            &[0.4, 0.9],
            1,
            2,
            1,
            &[false, false],
            &[true, false],
            &[0.5],
        );
        assert_eq!(m.dt_gt[0][0], None);
    }

    #[test]
    fn equal_iou_later_index_wins() {
        // Two non-ignored GTs with identical IoU; pycocotools' `>=` picks the last.
        let m = simple(&[0.7, 0.7], 1, 2, 2, &[0.5]);
        assert_eq!(m.dt_gt[0][0], Some(1));
    }
}
