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
