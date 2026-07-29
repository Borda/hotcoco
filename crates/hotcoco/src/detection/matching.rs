//! Per-image matching: one (image, category, area range, max_det) cell of the
//! evaluation.
//!
//! [`evaluate.rs`](super::evaluate) is the outer driver — it resolves parameters,
//! collects the sparse (image, category) pairs, and fans out across them. This
//! module is what each of those fan-out slots runs, and it owns no control flow
//! above the single cell.
//!
//! The assignment algorithm itself is not here either: that is
//! [`crate::primitives::greedy::greedy_match`]. What lives here is everything
//! COCO-specific *around* it — deciding which ground truths are ignored, ordering
//! detections by score, reordering the IoU matrix to match, and translating the
//! matcher's indices back into annotation ids. The matcher's contract requires the
//! caller to own exactly this, so this module is that caller.
//!
//! Naming note: `matching`, not `match` — the latter is a reserved word and
//! `mod match;` does not compile without `r#` escaping.

use std::collections::HashMap;

use crate::coco::COCO;
use crate::params::{IouType, Params};
use crate::types::Annotation;

use super::EvalMode;

/// Ground truths for one cell, partitioned non-ignored-first.
///
/// The matcher's contract requires that partition: indices
/// `[0, num_not_ignored)` are non-ignored and the rest are ignored. Fields
/// suffixed `_sorted` are in that partitioned order; `anns` and `iou_indices`
/// stay in load order, and [`order`](Self::order) maps between them.
struct GtView<'a> {
    anns: Vec<&'a Annotation>,
    /// Partitioned position -> index into `anns`.
    order: Vec<usize>,
    /// Index into `anns` -> column in the cell's IoU matrix.
    iou_indices: Vec<usize>,
    ignore_sorted: Vec<bool>,
    /// Whether each GT counts toward the recall denominator. Differs from
    /// `!ignore_sorted` only for Open Images group-of boxes, which are held out
    /// of matching but still counted. See `gather_gt`.
    in_denominator_sorted: Vec<bool>,
    iscrowd_sorted: Vec<bool>,
    /// Open Images only; empty otherwise. Guarded by `is_oid` at every use.
    is_group_of_sorted: Vec<bool>,
    num_not_ignored: usize,
    /// Number of ids returned before annotation lookup, which can drop entries.
    /// The skip conditions below test the *raw* count, so it must be kept.
    raw_count: usize,
}

impl GtView<'_> {
    fn len(&self) -> usize {
        self.anns.len()
    }

    /// Annotation id at a partitioned position.
    fn id_at(&self, sorted_idx: usize) -> u64 {
        self.anns[self.order[sorted_idx]].id
    }

    /// Annotation ids in partitioned order — the order `EvalImg` reports.
    fn sorted_ids(&self) -> Vec<u64> {
        (0..self.len()).map(|gi| self.id_at(gi)).collect()
    }
}

/// Detections for one cell, score-descending and truncated to `max_det`.
struct DtView<'a> {
    anns: Vec<&'a Annotation>,
    /// Position -> row in the cell's IoU matrix.
    iou_indices: Vec<usize>,
    scores: Vec<f64>,
    /// Detection falls outside this cell's area range.
    area_ignore: Vec<bool>,
}

impl DtView<'_> {
    fn len(&self) -> usize {
        self.anns.len()
    }
}

/// Per-threshold match bookkeeping — the payload of an [`EvalImg`].
struct MatchOutcome {
    dt_matches: Vec<Vec<u64>>,
    gt_matches: Vec<Vec<u64>>,
    dt_matched: Vec<Vec<bool>>,
    gt_matched: Vec<Vec<bool>>,
    dt_ignore: Vec<Vec<bool>>,
}

/// Load ground truths and decide which of them this cell ignores.
///
/// Ignore rules are mode-dependent: Open Images ignores group-of boxes and does
/// not care about `iscrowd`; COCO/LVIS ignore crowds, and keypoint evaluation
/// additionally ignores annotations with no labelled keypoints.
fn gather_gt<'a>(
    ctx: &EvalImgContext<'a>,
    gt_ids: &[u64],
    area_rng: [f64; 2],
    is_kp: bool,
    is_oid: bool,
) -> GtView<'a> {
    let (iou_indices, anns): (Vec<usize>, Vec<&Annotation>) = gt_ids
        .iter()
        .enumerate()
        .filter_map(|(iou_idx, &id)| Some((iou_idx, ctx.coco_gt.get_ann(id)?)))
        .unzip();

    // `ignore` governs *matching*; `in_denominator` governs the *recall
    // denominator*. They are complements of each other in every mode but Open
    // Images, where a group-of box is held out of matching (the second pass in
    // `match_cell` absorbs it instead) yet still counts as one ground truth,
    // because the protocol scores an undetected group-of box as a single false
    // negative. COCO's single `gtIgnore` cannot express "not matchable here" and
    // "counted" at once, so the two are computed together and kept apart.
    let (ignore, in_denominator): (Vec<bool>, Vec<bool>) = anns
        .iter()
        .map(|ann| {
            let a = ann.area.unwrap_or(0.0);
            let area_ignore = a < area_rng[0] || a > area_rng[1];
            if is_oid {
                (
                    ann.is_group_of.unwrap_or(false) || area_ignore,
                    !area_ignore,
                )
            } else {
                let mut ignore = ann.iscrowd || area_ignore;
                if is_kp {
                    ignore = ignore || ann.num_keypoints.unwrap_or(0) == 0;
                }
                (ignore, !ignore)
            }
        })
        .unzip();

    // Stable sort on the ignore flag: non-ignored first, load order preserved
    // within each partition. Tie order is observable through `evalImgs`.
    let mut order: Vec<usize> = (0..anns.len()).collect();
    order.sort_by_key(|&i| ignore[i] as u8);

    let ignore_sorted: Vec<bool> = order.iter().map(|&i| ignore[i]).collect();
    let in_denominator_sorted: Vec<bool> = order.iter().map(|&i| in_denominator[i]).collect();
    let iscrowd_sorted: Vec<bool> = order.iter().map(|&i| anns[i].iscrowd).collect();
    let is_group_of_sorted: Vec<bool> = if is_oid {
        order
            .iter()
            .map(|&i| anns[i].is_group_of.unwrap_or(false))
            .collect()
    } else {
        Vec::new()
    };
    let num_not_ignored = ignore_sorted.iter().filter(|&&x| !x).count();

    GtView {
        anns,
        order,
        iou_indices,
        ignore_sorted,
        in_denominator_sorted,
        iscrowd_sorted,
        is_group_of_sorted,
        num_not_ignored,
        raw_count: gt_ids.len(),
    }
}

/// Load detections, order them score-descending, and cap at `max_det`.
///
/// The cap is applied *after* sorting, so it keeps the highest-scoring
/// detections rather than the first-loaded ones.
fn gather_dt<'a>(
    ctx: &EvalImgContext<'a>,
    dt_ids: &[u64],
    area_rng: [f64; 2],
    max_det: usize,
) -> DtView<'a> {
    let mut with_iou_idx: Vec<(usize, &Annotation)> = dt_ids
        .iter()
        .enumerate()
        .filter_map(|(iou_idx, &id)| Some((iou_idx, ctx.coco_dt.get_ann(id)?)))
        .collect();
    with_iou_idx.sort_by(|a, b| {
        b.1.score
            .unwrap_or(0.0)
            .partial_cmp(&a.1.score.unwrap_or(0.0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    with_iou_idx.truncate(max_det);

    let (iou_indices, anns): (Vec<usize>, Vec<&Annotation>) = with_iou_idx.into_iter().unzip();
    let (scores, area_ignore): (Vec<f64>, Vec<bool>) = anns
        .iter()
        .map(|ann| {
            let a = ann.area.unwrap_or(0.0);
            (ann.score.unwrap_or(0.0), a < area_rng[0] || a > area_rng[1])
        })
        .unzip();

    DtView {
        anns,
        iou_indices,
        scores,
        area_ignore,
    }
}

/// Reorder the cell's IoU matrix into the flat row-major `[D*G]` layout the
/// matcher expects, with rows in score order and columns non-ignored-first.
///
/// This reorder step is exactly where the matcher's flat layout is produced —
/// the `sim` kernels' nested `[D][G]` output is never fed in directly.
fn reordered_iou(iou_mat: &IouMatrix, dt: &DtView<'_>, gt: &GtView<'_>) -> Vec<f64> {
    let (d, g) = (dt.len(), gt.len());
    let mut flat = vec![0.0_f64; d * g];
    for di in 0..d {
        let dt_row = dt.iou_indices[di];
        for (gi_sorted, &gi_orig) in gt.order.iter().enumerate() {
            let gt_col = gt.iou_indices[gi_orig];
            if dt_row < iou_mat.len() && gt_col < iou_mat[dt_row].len() {
                flat[di * g + gi_sorted] = iou_mat[dt_row][gt_col];
            }
        }
    }
    flat
}

/// Run the matcher over one cell and translate its indices back to annotation ids.
fn match_cell(
    ctx: &EvalImgContext<'_>,
    gt: &GtView<'_>,
    dt: &DtView<'_>,
    iou_matrix: Option<&IouMatrix>,
    is_oid: bool,
) -> MatchOutcome {
    let (d, g) = (dt.len(), gt.len());
    let num_iou_thrs = ctx.params.iou_thrs.len();

    let mut dt_matches = vec![vec![0u64; d]; num_iou_thrs];
    let mut gt_matches = vec![vec![0u64; g]; num_iou_thrs];
    let mut dt_matched = vec![vec![false; d]; num_iou_thrs];
    // Seeded from the area-ignore flags so unmatched detections carry the right
    // ignore status even when there is no IoU data at all.
    let mut dt_ignore: Vec<Vec<bool>> = (0..num_iou_thrs).map(|_| dt.area_ignore.clone()).collect();

    let Some(iou_mat) = iou_matrix else {
        // No detections and/or no ground truths: nothing matched.
        return MatchOutcome {
            dt_matches,
            gt_matches,
            dt_matched,
            gt_matched: vec![vec![false; g]; num_iou_thrs],
            dt_ignore,
        };
    };

    let iou_flat = reordered_iou(iou_mat, dt, gt);

    // The per-GT policy flags encode the mode: crowd GTs are re-matchable
    // (COCO/LVIS only); under OID `iscrowd` is irrelevant and group-of GTs are
    // held out of the fallback phase, to be matched in the separate pass below.
    let gt_rematchable: Vec<bool> = (0..g).map(|gi| !is_oid && gt.iscrowd_sorted[gi]).collect();
    // `is_oid` short-circuits, so `is_group_of_sorted` (empty unless OID, length
    // g when OID) is only indexed when the index is in bounds.
    let gt_phase2_eligible: Vec<bool> = (0..g)
        .map(|gi| !(is_oid && gt.is_group_of_sorted[gi]))
        .collect();

    // Both matching phases share `ctx.match_floors` — pycocotools' clamped
    // thresholds. See the policy table in `primitives::greedy`.
    let m = crate::primitives::greedy::greedy_match(
        &iou_flat,
        d,
        g,
        gt.num_not_ignored,
        &gt_rematchable,
        &gt_phase2_eligible,
        ctx.match_floors,
    );

    // Translate matched indices into annotation ids + ignore flags. Unmatched
    // detections keep the area-ignore status they were seeded with.
    for t_idx in 0..num_iou_thrs {
        for (di, dt_ann) in dt.anns.iter().enumerate() {
            if let Some(gi) = m.dt_gt[t_idx][di] {
                dt_matches[t_idx][di] = gt.id_at(gi);
                gt_matches[t_idx][gi] = dt_ann.id;
                dt_matched[t_idx][di] = true;
                // A detection matched to an ignored GT is itself ignored.
                dt_ignore[t_idx][di] = gt.ignore_sorted[gi];
            }
        }
    }
    let mut gt_matched = m.gt_matched;

    // Open Images second pass — group-of boxes.
    //
    // The protocol (https://storage.googleapis.com/openimages/web/evaluation.html):
    //
    //   "If at least one detection is inside group-of box a single True Positive
    //    is scored. ... Multiple correct detections inside the same group-of box
    //    is still count as a single True Positive. Otherwise, the group-of box is
    //    counted as a single False Negative."
    //
    // So a group-of box is worth exactly one ground truth: the best-scoring
    // detection inside it becomes a true positive, every other detection inside it
    // is ignored (neither TP nor FP), and if nothing is inside it the box is a
    // miss. This is the Open Images *Challenge* metric, equivalently TensorFlow's
    // `group_of_weight = 1.0`, and it is what FiftyOne implements unconditionally.
    //
    // "Inside" is IoA, not IoU — see the note in `iou.rs` where group-of GT
    // columns are flagged crowd so `sim` selects intersection-over-detection-area.
    //
    // Group-of GTs are excluded from both greedy phases (`gt_phase2_eligible`),
    // so this is their only matching route. That exclusion is load-bearing: an IoA
    // column saturates at 1.0 for any detection inside the region, so a group-of
    // box left in phase 1 would outbid the real object a detection is sitting on
    // and turn that object into a false negative.
    //
    // Detections arrive score-descending, so the first one to claim a given box is
    // the highest-scoring one — the same choice TF makes with
    // `scores_group_of[gt_id] = max(scores_group_of[gt_id], scores[i])`.
    //
    // `is_group_of_sorted` doubles as the candidate mask: under OID `ignore` is
    // `is_group_of || area_ignore`, so every group-of box is ignored and therefore
    // already sorted into the `[num_not_ignored, g)` tail. A separate eligibility
    // vector would only restate that invariant.
    //
    // The guard skips the whole pass for cells with no group-of GT — the common
    // case, since group-of is a minority annotation — which otherwise costs a full
    // `d x g` scan per threshold for a guaranteed-empty result.
    if is_oid && gt.is_group_of_sorted.iter().any(|&x| x) {
        for (t_idx, &iou_thr) in ctx.match_floors.iter().enumerate() {
            for di in 0..d {
                if dt_matched[t_idx][di] {
                    continue;
                }
                // Best enclosing group-of box. The reference does
                // `np.argmax(ioa, axis=1)` then tests the threshold, which is the
                // same selection and the same first-wins tie-break that
                // `best_above_floor` owns — see its docs for why the tie matters.
                let row = &iou_flat[di * g..(di + 1) * g];
                let Some(gi) = crate::primitives::greedy::best_above_floor(
                    row,
                    &gt.is_group_of_sorted,
                    iou_thr,
                ) else {
                    continue;
                };

                dt_matches[t_idx][di] = gt.id_at(gi);
                dt_matched[t_idx][di] = true;
                // `gt_matched` *is* the "already credited" flag: group-of boxes are
                // excluded from both greedy phases, so it is false on entry here and
                // only this loop ever sets it. A separate `credited` vector would be
                // a second copy of the same bit, free to drift from the one
                // `EvalImg` reports.
                if gt_matched[t_idx][gi] {
                    // The box already has its true positive; absorb this one.
                    dt_ignore[t_idx][di] = true;
                } else {
                    // First (highest-scoring) detection inside this box scores it.
                    dt_ignore[t_idx][di] = false;
                    gt_matches[t_idx][gi] = dt.anns[di].id;
                    gt_matched[t_idx][gi] = true;
                }
            }
        }
    }

    MatchOutcome {
        dt_matches,
        gt_matches,
        dt_matched,
        gt_matched,
        dt_ignore,
    }
}

/// Evaluate a single image+category cell.
///
/// `not_exhaustive_cat` — when true (LVIS mode), unmatched detections are ignored
/// rather than counted as false positives.
///
/// Returns `None` for cells with nothing to report, which is what keeps
/// `evalImgs` sparse.
pub(super) fn evaluate_img(
    ctx: &EvalImgContext<'_>,
    img_id: u64,
    cat_id: u64,
    area_rng: [f64; 2],
    max_det: usize,
    not_exhaustive_cat: bool,
) -> Option<EvalImg> {
    use super::COCOeval;

    let gt_ids = COCOeval::get_anns_static(ctx.coco_gt, ctx.params, img_id, cat_id);
    let dt_ids = COCOeval::get_anns_static(ctx.coco_dt, ctx.params, img_id, cat_id);
    if gt_ids.is_empty() && dt_ids.is_empty() {
        return None;
    }

    let is_kp = ctx.params.iou_type == IouType::Keypoints;
    let is_oid = ctx.eval_mode == EvalMode::OpenImages;

    let gt = gather_gt(ctx, gt_ids, area_rng, is_kp, is_oid);
    let dt = gather_dt(ctx, dt_ids, area_rng, max_det);

    let mut outcome = match_cell(ctx, &gt, &dt, ctx.ious.get(&(img_id, cat_id)), is_oid);

    // LVIS: on a not-exhaustively-labelled category, unmatched detections are
    // ignored instead of penalised as false positives.
    if not_exhaustive_cat {
        for t_idx in 0..ctx.params.iou_thrs.len() {
            for di in 0..dt.len() {
                if !outcome.dt_matched[t_idx][di] {
                    outcome.dt_ignore[t_idx][di] = true;
                }
            }
        }
    }

    // Nothing non-ignored on either side means this cell contributes nothing —
    // but only skip it when there were no ground-truth ids at all, matching the
    // original condition. Note both tests use the *raw* id counts.
    let has_content = gt.num_not_ignored > 0 || dt.area_ignore.iter().any(|&ignored| !ignored);
    if !has_content && gt.raw_count == 0 {
        return None;
    }

    Some(EvalImg {
        image_id: img_id,
        category_id: cat_id,
        area_rng,
        max_det,
        dt_ids: dt.anns.iter().map(|a| a.id).collect(),
        gt_ids: gt.sorted_ids(),
        dt_matches: outcome.dt_matches,
        gt_matches: outcome.gt_matches,
        dt_matched: outcome.dt_matched,
        gt_matched: outcome.gt_matched,
        dt_scores: dt.scores,
        gt_ignore: gt.ignore_sorted,
        gt_in_denominator: gt.in_denominator_sorted,
        dt_ignore: outcome.dt_ignore,
    })
}

/// D×G IoU matrix (row-major: dt.len() rows, gt.len() columns).
pub(in crate::detection) type IouMatrix = Vec<Vec<f64>>;

/// Per-image, per-category evaluation result.
///
/// `#[non_exhaustive]`: evaluation families added later (panoptic, tracking) will
/// need fields here, and this keeps that additive rather than breaking. Construct
/// via evaluation, not by struct literal.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct EvalImg {
    pub image_id: u64,
    pub category_id: u64,
    pub area_rng: [f64; 2],
    pub max_det: usize,
    /// Detection annotation IDs (sorted by score descending, truncated to max_det)
    pub dt_ids: Vec<u64>,
    /// Ground truth annotation IDs (sorted: non-ignored first, then ignored)
    pub gt_ids: Vec<u64>,
    /// Matched GT annotation id per IoU threshold: `dt_matches[t][d]` = GT id, or **0 as a
    /// sentinel for unmatched**. Do not use a non-zero check for presence — `Annotation.id`
    /// defaults to 0, so a real GT can have id=0. Use `dt_matched[t][d]` instead.
    pub dt_matches: Vec<Vec<u64>>,
    /// Matched DT annotation id per IoU threshold: `gt_matches[t][g]` = DT id, or **0 as a
    /// sentinel for unmatched**. Same caveat as `dt_matches`. Use `gt_matched[t][g]` instead.
    pub gt_matches: Vec<Vec<u64>>,
    /// Whether each detection was matched at each IoU threshold. Authoritative presence check;
    /// avoids the id=0 sentinel ambiguity in `dt_matches`.
    pub dt_matched: Vec<Vec<bool>>,
    /// Whether each GT was matched at each IoU threshold. Authoritative presence check;
    /// avoids the id=0 sentinel ambiguity in `gt_matches`.
    pub gt_matched: Vec<Vec<bool>>,
    /// Detection scores
    pub dt_scores: Vec<f64>,
    /// Whether each GT is ignored *for matching*
    pub gt_ignore: Vec<bool>,
    /// Whether each GT counts toward the recall denominator.
    ///
    /// Equal to `!gt_ignore` in every mode except Open Images, where a group-of
    /// box is held out of matching yet still counts as one ground truth — the
    /// protocol scores an undetected group-of box as a single false negative.
    /// Consumers computing `num_gt` must read this, not `gt_ignore`.
    pub gt_in_denominator: Vec<bool>,
    /// Whether each detection is ignored per IoU threshold
    pub dt_ignore: Vec<Vec<bool>>,
}

impl EvalImg {
    /// How many ground truths in this cell count toward recall.
    ///
    /// Use this rather than counting `!gt_ignore`. The two agree in every mode but
    /// Open Images, where a group-of box is excluded from matching yet still counts
    /// as one ground truth — see [`gt_in_denominator`](Self::gt_in_denominator).
    /// Having the rule in a method rather than repeated at each call site is what
    /// stops the next consumer from reaching for the wrong field.
    pub fn num_gt_in_denominator(&self) -> usize {
        self.gt_in_denominator.iter().filter(|&&x| x).count()
    }

    /// Whether ground truth `gi` is a *scored* miss when unmatched.
    ///
    /// The false-negative counterpart of [`num_gt_in_denominator`](Self::num_gt_in_denominator):
    /// a ground truth that counts in the denominator and went unmatched is a miss.
    /// Consumers tallying false negatives should ask this instead of `!gt_ignore`,
    /// or they will disagree with the recall the same evaluation reports.
    pub fn counts_as_miss(&self, gi: usize) -> bool {
        self.gt_in_denominator.get(gi).copied().unwrap_or(false)
    }
}

/// Read-only context shared across all [`COCOeval::evaluate_img_static`] calls
/// within a single [`COCOeval::evaluate`] invocation.
///
/// Grouping these shared references avoids passing them individually to every
/// call and removes the `#[allow(clippy::too_many_arguments)]` suppressor.
pub(super) struct EvalImgContext<'a> {
    pub(super) coco_gt: &'a COCO,
    pub(super) coco_dt: &'a COCO,
    pub(super) params: &'a Params,
    pub(super) ious: &'a HashMap<(u64, u64), IouMatrix>,
    pub(super) eval_mode: super::EvalMode,
    /// `params.iou_thrs` with pycocotools' match floor applied
    /// ([`crate::primitives::greedy::coco_match_floor`]). Resolved once per
    /// `evaluate()` rather than per image-category pair: this is read inside a
    /// rayon fan-out over every (category, area range, image) tuple, so deriving
    /// it at the call site would allocate a short `Vec` hundreds of thousands of
    /// times per evaluation.
    pub(super) match_floors: &'a [f64],
}
