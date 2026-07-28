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
//! # Threshold-epsilon policy: caller-owned, with one canonical clamp
//!
//! pycocotools does not use the raw threshold as its match floor — it starts each
//! detection's search at `min(t, 1 - 1e-10)` (`evaluateImg`: `iou = min([t,
//! 1-1e-10])`). That clamp is **the caller's to apply**: [`greedy_match`] compares
//! against exactly the `iou_thrs` it is handed and adds no epsilon of its own, so
//! a family whose thresholds mean something other than COCO's is not silently
//! given COCO's fudge factor.
//!
//! The clamp is inert for every threshold `< 1.0`, so it never fires on COCO's
//! default `0.5:0.05:0.95` sweep. It is observable only at `t == 1.0`, where
//! pycocotools still matches a pair whose IoU lies in `[1 - 1e-10, 1.0)` and an
//! unclamped caller does not.
//!
//! **Settled at 1.0** (this paragraph replaces the v0.4.x "known carry-over"
//! note). Callers that claim pycocotools matching semantics apply
//! [`coco_match_floor`]; callers implementing a different reference do not:
//!
//! | Caller | Clamped? | Why |
//! |---|---|---|
//! | the detection `evaluate()` path (incl. the OID group-of pass) | **yes** | pycocotools drop-in parity; both matching phases in one `evaluate()` must share a floor or `t == 1.0` is internally incoherent |
//! | TIDE (`pos_thr`/`bg_thr`) | no | parity contract is *tidecv*, not pycocotools — clamping would diverge from that reference |
//! | confusion matrix, per-image diagnostics, calibration | no | hotcoco-native analysis with a user-chosen threshold; COCO's fudge factor is not implied |
//!
//! Identical geometry yields `1.0` to within a few ulp — but **not** exactly
//! `1.0`. The intersection width is computed as `(x + w) - x`, which does not
//! round-trip to `w` in binary floating point: `[94.13, 88.47, 21.53, 46.14]`
//! against itself gives `0.9999999999999993`. pycocotools computes it the same
//! way, so this is the reference's arithmetic rather than a defect here.
//!
//! The consequence is the reverse of what this note used to claim. For boxes at
//! least a pixel on a side the drift is bounded around `1e-13`, so a *clamped*
//! caller still matches exact duplicates at `t == 1.0` — three orders of
//! magnitude clear of the `1 - 1e-10` floor. An **unclamped** caller comparing
//! against a raw `1.0` may not. That makes the clamp a reason to apply the floor
//! at `t == 1.0`, not evidence it is unnecessary.
//!
//! The floor is not a universal rescue: for sub-pixel geometry (a side ~1e-5
//! against a coordinate ~1e2) the subtraction keeps almost none of the extent's
//! significand and the drift reaches ~1.5e-9, below the floor. No detection
//! dataset contains boxes that small and no standard sweep reaches `t == 1.0`,
//! so this bounds the guarantee rather than breaking anything in practice.
//!
//! Both regimes are pinned by `sim::tests::bbox_iou_algebraic_properties` and
//! `sim::tests::self_iou_degrades_for_subpixel_boxes`.

/// pycocotools' match floor for an IoU threshold: `min(t, 1 - 1e-10)`.
///
/// The canonical definition of the clamp described in the [module
/// docs][self#threshold-epsilon-policy-caller-owned-with-one-canonical-clamp].
/// It exists so the detection lineage has **one** spelling of the epsilon rather
/// than a literal repeated at each call site; [`greedy_match`] deliberately does
/// not apply it for you.
///
/// ```
/// # use hotcoco::primitives::greedy::coco_match_floor;
/// assert_eq!(coco_match_floor(0.5), 0.5);       // inert below 1.0
/// assert_eq!(coco_match_floor(1.0), 1.0 - 1e-10);
/// ```
#[inline]
pub fn coco_match_floor(iou_thr: f64) -> f64 {
    iou_thr.min(1.0 - 1e-10)
}

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
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// Every property the matcher's contract guarantees, over random inputs.
    ///
    /// Each detection family reaches its numbers through this function, so a
    /// violation here is a wrong metric everywhere at once — and the fixtures
    /// above are all 2x1 and 2x2. Randomising the shape, the crowd flags, the
    /// phase-2 mask and the threshold list is what exercises the interactions
    /// between them.
    ///
    /// The IoU grid deliberately mixes continuous values with a coarse
    /// quantised set: exact ties are where the `>=` update rule (later GT index
    /// wins) is observable, and they essentially never occur under pure
    /// continuous sampling.
    #[test]
    fn greedy_match_contract_random() {
        let mut rng = StdRng::seed_from_u64(0x6DEED1);

        for case in 0..5000 {
            let d = rng.random_range(1..=6);
            let g = rng.random_range(1..=6);
            let num_ni = rng.random_range(0..=g);

            let quantised = rng.random_bool(0.5);
            let iou_flat: Vec<f64> = (0..d * g)
                .map(|_| {
                    if quantised {
                        // 0.0, 0.25, 0.5, 0.75, 1.0 — collides constantly.
                        rng.random_range(0..=4) as f64 / 4.0
                    } else {
                        rng.random_range(0.0..=1.0)
                    }
                })
                .collect();

            let rematchable: Vec<bool> = (0..g).map(|_| rng.random_bool(0.25)).collect();
            let phase2: Vec<bool> = (0..g).map(|_| rng.random_bool(0.75)).collect();

            let mut thrs: Vec<f64> = (0..rng.random_range(1..=4))
                .map(|_| rng.random_range(0.0..=1.0))
                .collect();
            thrs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let m = greedy_match(&iou_flat, d, g, num_ni, &rematchable, &phase2, &thrs);
            let ctx = format!("case {case}: d={d} g={g} num_ni={num_ni} thrs={thrs:?}");

            assert_eq!(m.dt_gt.len(), thrs.len(), "{ctx}");
            assert_eq!(m.gt_matched.len(), thrs.len(), "{ctx}");

            for (ti, &thr) in thrs.iter().enumerate() {
                assert_eq!(m.dt_gt[ti].len(), d, "{ctx}");
                assert_eq!(m.gt_matched[ti].len(), g, "{ctx}");

                let mut claimed = vec![0usize; g];
                for di in 0..d {
                    let Some(gi) = m.dt_gt[ti][di] else { continue };

                    assert!(gi < g, "{ctx}: gt index {gi} out of range");

                    // A recorded match must clear the bar it was matched at.
                    assert!(
                        iou_flat[di * g + gi] >= thr,
                        "{ctx}: dt {di} matched gt {gi} at IoU {} < {thr}",
                        iou_flat[di * g + gi]
                    );

                    // Phase 2 is the only route to an ignored GT, and it honours
                    // the eligibility mask.
                    if gi >= num_ni {
                        assert!(
                            phase2[gi],
                            "{ctx}: dt {di} matched phase-2-ineligible gt {gi}"
                        );
                    }

                    claimed[gi] += 1;
                }

                // Injectivity: a GT is claimed once, unless it is rematchable —
                // crowd regions, which absorb any number of detections.
                for gi in 0..g {
                    if claimed[gi] > 1 {
                        assert!(
                            rematchable[gi],
                            "{ctx}: gt {gi} claimed {} times but is not rematchable",
                            claimed[gi]
                        );
                    }
                    // The two outputs are one fact in two shapes. `detection`'s
                    // confusion adapter reads both halves specifically to avoid a
                    // second source of truth, which makes this a contract.
                    assert_eq!(
                        m.gt_matched[ti][gi],
                        claimed[gi] > 0,
                        "{ctx}: gt_matched[{gi}] disagrees with dt_gt"
                    );
                }
            }
        }
    }

    /// Raising the IoU threshold cannot increase the number of **true-positive
    /// eligible** matches — those to non-ignored ground truths. This is what
    /// underwrites AP@0.5 >= AP@0.75 downstream.
    ///
    /// The obvious stronger claim — that the *total* match count is monotone —
    /// is **false**, and the counterexample is instructive rather than exotic.
    /// Phase 1 is preferred over phase 2, so raising the threshold can evict a
    /// detection out of phase 1 and into phase 2, freeing the non-ignored GT it
    /// was holding for a later detection. With `num_gt_not_ignored = 1` and
    ///
    /// ```text
    ///        G0     G1     G2          thresholds 0.42 and 0.70
    ///   D0  0.50   0.75   0.25         phase2 eligible: G1, G2
    ///   D1  0.75   0.50   0.50
    ///   D2  0.00   0.25   0.75
    /// ```
    ///
    /// the low threshold matches 2 (D0->G0 blocks D1, which takes G2 and blocks
    /// D2) while the high threshold matches 3 (D0 cannot reach G0, so it takes
    /// G1, leaving G0 for D1 and G2 for D2). Both give **one** TP-eligible match,
    /// which is why the property has to be stated over that subset.
    ///
    /// Established empirically over 200k random cases rather than proved; the
    /// loop here is smaller so the suite stays fast.
    #[test]
    fn tp_eligible_matches_are_monotone_in_threshold() {
        let mut rng = StdRng::seed_from_u64(0xA11CE);

        for case in 0..20000 {
            let d = rng.random_range(1..=5);
            let g = rng.random_range(1..=5);
            let num_ni = rng.random_range(0..=g);
            let quantised = rng.random_bool(0.5);
            let iou: Vec<f64> = (0..d * g)
                .map(|_| {
                    if quantised {
                        rng.random_range(0..=4) as f64 / 4.0
                    } else {
                        rng.random_range(0.0..=1.0)
                    }
                })
                .collect();
            let rematchable: Vec<bool> = (0..g).map(|_| rng.random_bool(0.2)).collect();
            let phase2: Vec<bool> = (0..g).map(|_| rng.random_bool(0.8)).collect();

            let mut thrs: Vec<f64> = (0..2).map(|_| rng.random_range(0.0..=1.0)).collect();
            thrs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let m = greedy_match(&iou, d, g, num_ni, &rematchable, &phase2, &thrs);
            let tp_at = |ti: usize| {
                m.dt_gt[ti]
                    .iter()
                    .flatten()
                    .filter(|&&gi| gi < num_ni)
                    .count()
            };

            assert!(
                tp_at(1) <= tp_at(0),
                "case {case}: raising the threshold {:?} -> {:?} grew TP-eligible \
                 matches {} -> {} (d={d} g={g} num_ni={num_ni}) iou={iou:?}",
                thrs[0],
                thrs[1],
                tp_at(0),
                tp_at(1),
            );
        }
    }

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
