use std::collections::{BTreeMap, HashMap, HashSet};

use rayon::prelude::*;
use serde::Serialize;

use super::COCOeval;
use super::matching::EvalImg;

/// TIDE false-positive error types, named as in tidecv.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum ErrType {
    Cls,
    Loc,
    Both,
    Dupe,
    Bkg,
}

impl ErrType {
    /// The key this error type is reported under. Single source of the spelling —
    /// the aggregation must not re-enumerate the variants.
    pub(super) fn as_str(self) -> &'static str {
        match self {
            ErrType::Cls => "Cls",
            ErrType::Loc => "Loc",
            ErrType::Both => "Both",
            ErrType::Dupe => "Dupe",
            ErrType::Bkg => "Bkg",
        }
    }
}

/// Every false-positive error type, in report order.
///
/// The one enumeration of the five. Counts, per-category ΔAP vectors, the
/// per-type oracle runs and the output map are all driven from this array, and
/// `err as usize` is the index into anything sized by it — which holds because
/// the array is in declaration order. `"Miss"` is not here: it is counted from
/// ground truths, not from detections, and is added to the output separately.
const FP_TYPES: [ErrType; 5] = [
    ErrType::Cls,
    ErrType::Loc,
    ErrType::Both,
    ErrType::Dupe,
    ErrType::Bkg,
];

/// What [`classify_fp`] needs to know about one false-positive detection.
///
/// Gathering this is the caller's job (it needs the per-image IoU views and the
/// cross-category scan); deciding what it *means* is the parity contract below.
pub(super) struct FpEvidence {
    /// Highest IoU with any same-class GT (0.0 if there are none).
    pub(super) max_same_iou: f64,
    /// Highest IoU with any *different*-class GT (0.0 if there are none).
    pub(super) max_cross_iou: f64,
    /// Some same-class GT with IoU >= `pos_thr` was already matched by a
    /// higher-scoring detection.
    pub(super) best_same_gt_matched: bool,
}

/// Classify a false positive into a TIDE error type.
///
/// **This is the tidecv parity contract**, extracted so it is readable and
/// testable on its own: the priority order is load-bearing, and reordering two
/// arms silently changes every published TIDE number.
///
/// Priority matches tidecv (`BoxError > ClassError > DuplicateError >
/// BackgroundError > OtherError`):
///
/// | Order | Type | Condition |
/// |---|---|---|
/// | 1 | `Loc`  | same-class max IoU in `[bg_thr, pos_thr]` — the upper bound is inclusive and is what excludes `Dupe`, whose same-class IoU exceeds `pos_thr` |
/// | 2 | `Cls`  | cross-class max IoU >= `pos_thr` (and `Loc` did not fire) |
/// | 3 | `Dupe` | a same-class GT at IoU >= `pos_thr` is already matched by a higher-scoring TP |
/// | 4 | `Bkg`  | max IoU with any GT <= `bg_thr` (same-class is already below `bg_thr` here, so only cross-class needs checking) |
/// | 5 | `Both` | fallthrough — cross-class IoU in `(bg_thr, pos_thr)` |
pub(super) fn classify_fp(ev: &FpEvidence, pos_thr: f64, bg_thr: f64) -> ErrType {
    if ev.max_same_iou >= bg_thr && ev.max_same_iou <= pos_thr {
        ErrType::Loc
    } else if ev.max_cross_iou >= pos_thr {
        ErrType::Cls
    } else if ev.best_same_gt_matched {
        ErrType::Dupe
    } else if ev.max_cross_iou <= bg_thr {
        ErrType::Bkg
    } else {
        ErrType::Both
    }
}

impl COCOeval {
    /// Compute average precision from per-detection matched/ignored flags.
    ///
    /// Uses the same 101-point interpolation as [`accumulate`](COCOeval::accumulate),
    /// via [`crate::metrics::counts::average_precision`].
    ///
    /// Returns `0.0` when `num_gt == 0`: TIDE's ΔAP compares corpus-level APs, so
    /// a category with no ground truth contributes a vacuous `0.0`. (Per-image
    /// diagnostics deliberately uses the opposite convention — see the
    /// [`counts`](crate::metrics::counts) module note.)
    pub(super) fn compute_ap_from_matched(
        scores: &[f64],
        matched: &[bool],
        ignored: &[bool],
        num_gt: usize,
        rec_thrs: &[f64],
    ) -> f64 {
        crate::metrics::counts::average_precision(scores, matched, Some(ignored), num_gt, rec_thrs)
    }

    /// Decompose detection errors into TIDE error types.
    ///
    /// Requires [`evaluate`](COCOeval::evaluate) to have been called first.
    ///
    /// Returns a [`TideErrors`] with ΔAP values and counts for six error types:
    ///
    /// | Error | Meaning |
    /// |-------|---------|
    /// | `Cls`  | Wrong class, correct location (IoU ≥ `pos_thr` with other-class GT) |
    /// | `Loc`  | Right class, poor localization (`bg_thr` ≤ IoU < `pos_thr`) |
    /// | `Both` | Wrong class AND poor localization |
    /// | `Dupe` | Duplicate — correct class GT already claimed by higher-scoring TP |
    /// | `Bkg`  | Pure background (IoU < `bg_thr` with all GTs) |
    /// | `Miss` | Undetected GT (false negative) |
    pub fn tide_errors(&self, pos_thr: f64, bg_thr: f64) -> crate::error::Result<TideErrors> {
        if self.eval_imgs.is_empty() {
            return Err("tide_errors() requires evaluate() to be called first".into());
        }

        let cat_ids = &self.params.cat_ids;
        let iou_type = self.params.iou_type;
        // Built once for the whole run, not per image — see
        // `cross_category_pairs`'s ordering note.
        let cat_slots = Self::cat_slots(cat_ids);

        // `pos_thr` is an analysis threshold, not a metric name, so it snaps to
        // the nearest grid point rather than requiring an exact match.
        let t_idx = self.params.nearest_iou_thr_idx(pos_thr);

        let coco_gt = &self.coco_gt;
        let coco_dt = &self.coco_dt;
        // `tide_errors` requires `evaluate()` first (checked above), so on a segm
        // run this cache is populated and the cross-category matrices below skip
        // re-rasterizing every polygon.
        let segm_rles = self.segm_rles.as_ref();

        // --- Cross-category IoU pass ---
        // For each image, compute max IoU between each DT annotation
        // and any GT annotation of a *different* category.
        let img_ids = &self.params.img_ids;

        // Returns: img_id → (dt_ann_id → (max_cross_iou, argmax_cross_gt_ann_id))
        // argmax_cross_gt_ann_id is u64::MAX when there are no cross-class GTs.
        let cross_iou_per_img: HashMap<u64, HashMap<u64, (f64, u64)>> = img_ids
            .par_iter()
            .map(|&img_id| {
                let mut dt_max_cross: HashMap<u64, (f64, u64)> = HashMap::new();

                // All non-crowd GTs and all DTs in the image, tagged with their
                // category slot. `None`: TIDE takes every detection in index
                // order — it scores each against its own `eval_imgs` entry, so it
                // needs neither a score floor nor a cap of its own.
                let (gt_pairs, dt_pairs) =
                    Self::cross_category_pairs(coco_gt, coco_dt, &cat_slots, img_id, None);

                if dt_pairs.is_empty() || gt_pairs.is_empty() {
                    for &(_, ann_id) in &dt_pairs {
                        dt_max_cross.insert(ann_id, (0.0, u64::MAX));
                    }
                    return (img_id, dt_max_cross);
                }

                // Compute cross-category IoU matrix [D × G]
                let dt_ids: Vec<u64> = dt_pairs.iter().map(|&(_, ann_id)| ann_id).collect();
                let gt_ids: Vec<u64> = gt_pairs.iter().map(|&(_, ann_id)| ann_id).collect();
                let iou_matrix = Self::cross_category_iou(
                    &dt_ids, &gt_ids, coco_dt, coco_gt, iou_type, segm_rles,
                );

                // For each DT, find max IoU with any *other-category* GT and record that GT's id
                for (di, &(dt_cat_idx, dt_ann_id)) in dt_pairs.iter().enumerate() {
                    let mut max_cross = 0.0f64;
                    let mut argmax_cross_gt_ann_id = u64::MAX;
                    let row = &iou_matrix[di * gt_pairs.len()..(di + 1) * gt_pairs.len()];
                    for (gi, &(gt_cat_idx, gt_ann_id)) in gt_pairs.iter().enumerate() {
                        if gt_cat_idx != dt_cat_idx && row[gi] > max_cross {
                            max_cross = row[gi];
                            argmax_cross_gt_ann_id = gt_ann_id;
                        }
                    }
                    dt_max_cross.insert(dt_ann_id, (max_cross, argmax_cross_gt_ann_id));
                }

                (img_id, dt_max_cross)
            })
            .collect();

        // Per-category accumulated data for ΔAP computation
        struct CatData {
            scores: Vec<f64>,
            matched: Vec<bool>,
            ignored: Vec<bool>,
            // Error type for each FP DT (None = TP or ignored)
            fp_types: Vec<Option<ErrType>>,
            num_gt: usize,
        }

        impl CatData {
            /// Permute every parallel array into score-descending order, once.
            ///
            /// Each category is scored eight ways below (a baseline, five
            /// per-error-type fixes, and the FP/FN oracles), and every one of them
            /// used to re-sort the same detections inside
            /// [`average_precision`](crate::metrics::counts::average_precision) —
            /// 3285 sorts of up to 22k elements on Objects365. Ranking once here
            /// lets those calls use the presorted entry point.
            ///
            /// Bit-identical because the comparator and the stability are the
            /// same: this is exactly the permutation `average_precision` computes,
            /// and stably sorting an already-sorted array is the identity.
            fn rank_by_score_desc(&mut self) {
                let mut order: Vec<usize> = (0..self.scores.len()).collect();
                order.sort_by(|&a, &b| {
                    self.scores[b]
                        .partial_cmp(&self.scores[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                self.scores = order.iter().map(|&i| self.scores[i]).collect();
                self.matched = order.iter().map(|&i| self.matched[i]).collect();
                self.ignored = order.iter().map(|&i| self.ignored[i]).collect();
                self.fp_types = order.iter().map(|&i| self.fp_types[i]).collect();
            }
        }

        let mut cat_data: HashMap<u64, CatData> = HashMap::new();
        // Tallied as integers indexed by `err as usize`, then named once at the
        // end: the previous form allocated a `String` per false positive purely
        // to look up a counter, on a path that runs over every detection.
        let mut fp_counts = [0u64; FP_TYPES.len()];
        let mut miss_total = 0u64;

        // GTs that have a Loc or Cls FP DT "targeting" them — these are not Miss errors.
        // A Loc DT targets the same-class GT with highest IoU in [bg_thr, pos_thr).
        // A Cls DT targets the cross-class GT with highest IoU >= pos_thr.
        // Collected across all categories so cross-category Cls coverage is captured.
        let mut covered_gt_ann_ids: HashSet<u64> = HashSet::new();

        // Pre-filter once; both passes below use the same cells.
        // `default_cells` owns the (area = "all", default max_det) predicate.
        let matching_eval_imgs: Vec<&EvalImg> = self.default_cells().collect();

        // --- Process each eval_img at (target_area_rng, max_det) ---
        for eval_img in &matching_eval_imgs {
            let img_id = eval_img.image_id;
            let cat_id = eval_img.category_id;
            let d = eval_img.dt_ids.len();

            // Annotation id → its row/column in the cell's IoU matrix, which is
            // indexed by *original* (JSON-order) position within the cell.
            //
            // A linear scan, not a `HashMap`: `d` and `g` are single digits in
            // almost every cell, so building two hash tables per cell — hashing
            // every id, allocating twice — cost more than the handful of integer
            // compares it saved. Annotation ids are unique, so `position` and a
            // map lookup return the same answer.
            let dt_orig_ids = coco_dt.get_ann_ids_for_img_cat(img_id, cat_id);
            let gt_orig_ids = coco_gt.get_ann_ids_for_img_cat(img_id, cat_id);
            let orig_pos = |ids: &[u64], id: u64| ids.iter().position(|&x| x == id);
            // Sorted GT position → column, resolved once per cell. The scan below
            // is (detections × GTs), so resolving it there would repeat the lookup
            // once per pair.
            let gt_sorted_to_orig: Vec<Option<usize>> = eval_img
                .gt_ids
                .iter()
                .map(|&id| orig_pos(gt_orig_ids, id))
                .collect();

            let same_iou_mat = self.cell_ious(img_id, cat_id);
            let cross_map = cross_iou_per_img.get(&img_id);

            let entry = cat_data.entry(cat_id).or_insert_with(|| CatData {
                scores: Vec::new(),
                matched: Vec::new(),
                ignored: Vec::new(),
                fp_types: Vec::new(),
                num_gt: 0,
            });

            entry.num_gt += eval_img.num_gt_in_denominator();

            // Classify each DT
            for di in 0..d {
                let dt_ann_id = eval_img.dt_ids[di];
                let is_matched = eval_img.dt_matched[(t_idx, di)];
                let is_ignored = eval_img.dt_ignore[(t_idx, di)];

                let fp_type = if is_matched || is_ignored {
                    None
                } else {
                    // FP: classify by priority (matches tidecv: Loc > Cls > Dupe > Bkg > Both)
                    let (max_cross_iou, argmax_cross_gt_ann_id) = cross_map
                        .and_then(|m| m.get(&dt_ann_id))
                        .copied()
                        .unwrap_or((0.0, u64::MAX));

                    // Same-class IoU: max IoU to any same-class GT; track argmax GT and Dupe
                    let mut max_same_iou = 0.0f64;
                    let mut argmax_same_gt_ann_id = u64::MAX;
                    let mut best_same_gt_matched = false;
                    if let Some(iou_mat) = same_iou_mat {
                        if let Some(di_orig) = orig_pos(dt_orig_ids, dt_ann_id) {
                            // One row borrow per detection. An out-of-range row
                            // reads as all-zero, as the bounds test it replaces did.
                            let row: &[f64] = iou_mat.get(di_orig).map_or(&[], Vec::as_slice);
                            for (gi_sorted, &gi_orig) in gt_sorted_to_orig.iter().enumerate() {
                                let Some(gi_orig) = gi_orig else {
                                    continue;
                                };
                                let iou = row.get(gi_orig).copied().unwrap_or(0.0);
                                if iou > max_same_iou {
                                    max_same_iou = iou;
                                    argmax_same_gt_ann_id = eval_img.gt_ids[gi_sorted];
                                }
                                if iou >= pos_thr && eval_img.gt_matched[(t_idx, gi_sorted)] {
                                    best_same_gt_matched = true;
                                }
                            }
                        }
                    }

                    // The tidecv priority order lives in `classify_fp` — see its
                    // docs for the table. Everything above this line is evidence
                    // gathering; the decision itself is the parity contract.
                    let err = classify_fp(
                        &FpEvidence {
                            max_same_iou,
                            max_cross_iou,
                            best_same_gt_matched,
                        },
                        pos_thr,
                        bg_thr,
                    );

                    // Track which GTs are "covered" (not Miss) by Loc or Cls FP DTs.
                    // Only Loc and Cls errors can be fixed to produce a TP for their target GT;
                    // Bkg/Both/Dupe fixes suppress the DT rather than turning it into a TP.
                    match err {
                        ErrType::Loc => {
                            if argmax_same_gt_ann_id != u64::MAX {
                                covered_gt_ann_ids.insert(argmax_same_gt_ann_id);
                            }
                        }
                        ErrType::Cls if argmax_cross_gt_ann_id != u64::MAX => {
                            covered_gt_ann_ids.insert(argmax_cross_gt_ann_id);
                        }
                        _ => {}
                    }

                    Some(err)
                };

                entry.scores.push(eval_img.dt_scores[di]);
                entry.matched.push(is_matched);
                entry.ignored.push(is_ignored);
                entry.fp_types.push(fp_type);
            }
        }

        // Aggregate FP error type counts
        for data in cat_data.values() {
            for err in data.fp_types.iter().flatten() {
                fp_counts[*err as usize] += 1;
            }
        }

        // --- Count Miss errors (second pass, after all FP types are classified) ---
        // A GT is Miss only if it is unmatched, non-ignored, AND not covered by any Loc/Cls FP DT.
        // Cross-category Cls coverage requires the second pass (a dog DT may cover a cat GT).
        let mut cat_miss_counts: HashMap<u64, usize> = HashMap::new();
        // tidecv's FalseNeg oracle counts *every* unmatched in-denominator GT;
        // the covered/uncovered split below only narrows Miss.
        let mut cat_fn_counts: HashMap<u64, usize> = HashMap::new();
        for eval_img in &matching_eval_imgs {
            let g = eval_img.gt_ids.len();
            let mut n_miss = 0usize;
            let mut n_fn = 0usize;
            for gi in 0..g {
                // `counts_as_miss`, not `!gt_ignore`: `num_gt` above already
                // counts an Open Images group-of box, so excluding it here
                // would make Miss disagree with the denominator it is a
                // fraction of, inside this one function.
                if eval_img.gt_matched[(t_idx, gi)] || !eval_img.counts_as_miss(gi) {
                    continue;
                }
                n_fn += 1;
                if !covered_gt_ann_ids.contains(&eval_img.gt_ids[gi]) {
                    n_miss += 1;
                }
            }
            miss_total += n_miss as u64;
            *cat_miss_counts.entry(eval_img.category_id).or_insert(0) += n_miss;
            *cat_fn_counts.entry(eval_img.category_id).or_insert(0) += n_fn;
        }

        // --- ΔAP computation ---
        // Rank once per category; every AP below then reads the presorted entry
        // point instead of re-sorting the same detections eight times.
        for data in cat_data.values_mut() {
            data.rank_by_score_desc();
        }

        /// One category's ΔAP contributions, in report order.
        struct CatDeltas {
            baseline: f64,
            /// Indexed like [`FP_TYPES`].
            fp_types: [f64; FP_TYPES.len()],
            miss: f64,
            fp: f64,
            fn_oracle: f64,
        }

        let rec_thrs = &self.params.rec_thrs;
        let cat_data = &cat_data;

        // Fanned out over categories. `par_iter().map(..).collect()` is an
        // *indexed* collect, so the results come back in `cat_ids` order and the
        // means below sum in exactly the sequence the sequential loop did —
        // no float reordering.
        let per_cat: Vec<Option<CatDeltas>> = cat_ids
            .par_iter()
            .map(|&cat_id| {
                let data = match cat_data.get(&cat_id) {
                    Some(d) if d.num_gt > 0 => d,
                    _ => return None,
                };

                let ranked_ap = |matched: &[bool], ignored: &[bool], num_gt: usize| -> f64 {
                    crate::metrics::counts::average_precision_ranked(
                        matched,
                        Some(ignored),
                        num_gt,
                        rec_thrs,
                    )
                };

                let baseline = ranked_ap(&data.matched, &data.ignored, data.num_gt);

                // Fix one FP error type.
                // Cls and Loc: flip FP → TP (the DT would have been correct if the error were
                // fixed). Bkg, Both, Dupe: suppress the DT (set ignored=true), matching tidecv's
                // fix()→None behavior where these errors produce no corrected TP.
                let fix_fp = |fix_type: ErrType| -> f64 {
                    let mut fixed_matched = data.matched.clone();
                    let mut fixed_ignored = data.ignored.clone();
                    for (i, fp_type) in data.fp_types.iter().enumerate() {
                        if *fp_type != Some(fix_type) {
                            continue;
                        }
                        match fix_type {
                            ErrType::Cls | ErrType::Loc => fixed_matched[i] = true,
                            ErrType::Bkg | ErrType::Both | ErrType::Dupe => fixed_ignored[i] = true,
                        }
                    }
                    ranked_ap(&fixed_matched, &fixed_ignored, data.num_gt)
                };

                let mut fp_types = [0.0f64; FP_TYPES.len()];
                for (slot, &err) in fp_types.iter_mut().zip(FP_TYPES.iter()) {
                    *slot = fix_fp(err) - baseline;
                }

                // FP: tidecv's FalsePositiveError oracle — perfect precision
                // without affecting recall. Every false positive is scored out of
                // existence; nothing is converted into a TP, unlike the per-type
                // fixes above, so this is *not* the union of the five.
                let fp = {
                    let fixed_ignored: Vec<bool> = data
                        .ignored
                        .iter()
                        .zip(&data.fp_types)
                        .map(|(&ig, fp_type)| ig || fp_type.is_some())
                        .collect();
                    ranked_ap(&data.matched, &fixed_ignored, data.num_gt) - baseline
                };

                // FN: tidecv's FalseNegativeError oracle — perfect recall without
                // affecting precision. Every unmatched in-denominator GT leaves
                // the denominator; detections are untouched. A superset of Miss,
                // which drops only the GTs no Loc/Cls fix could recover.
                let fn_count = cat_fn_counts.get(&cat_id).copied().unwrap_or(0);
                debug_assert!(
                    fn_count <= data.num_gt,
                    "FN count exceeds the GT denominator it was counted from"
                );
                let fn_oracle = ranked_ap(
                    &data.matched,
                    &data.ignored,
                    data.num_gt.saturating_sub(fn_count),
                ) - baseline;

                // Fix Miss: inject fake TPs for unmatched GTs.
                //
                // The sorting entry point, deliberately: the injected scores are
                // 2.0, which sits above any real confidence in practice but is not
                // *guaranteed* to — nothing rejects a score above 2.0 — and the old
                // behavior was to sort the concatenation. One sort per category
                // rather than eight is already the win.
                let miss_count = cat_miss_counts.get(&cat_id).copied().unwrap_or(0);
                let miss = if miss_count > 0 {
                    let mut fixed_scores = Vec::with_capacity(data.scores.len() + miss_count);
                    let mut fixed_matched = Vec::with_capacity(data.matched.len() + miss_count);
                    let mut fixed_ignored = Vec::with_capacity(data.ignored.len() + miss_count);
                    for _ in 0..miss_count {
                        fixed_scores.push(2.0);
                        fixed_matched.push(true);
                        fixed_ignored.push(false);
                    }
                    fixed_scores.extend_from_slice(&data.scores);
                    fixed_matched.extend_from_slice(&data.matched);
                    fixed_ignored.extend_from_slice(&data.ignored);
                    Self::compute_ap_from_matched(
                        &fixed_scores,
                        &fixed_matched,
                        &fixed_ignored,
                        data.num_gt,
                        rec_thrs,
                    ) - baseline
                } else {
                    0.0
                };

                Some(CatDeltas {
                    baseline,
                    fp_types,
                    miss,
                    fp,
                    fn_oracle,
                })
            })
            .collect();

        let mut baseline_aps: Vec<f64> = Vec::new();
        // One per-category delta vector per FP type, indexed the same way
        // `fp_counts` is — `FP_TYPES[i]` is what `d_fp_types[i]` measures.
        let mut d_fp_types: [Vec<f64>; FP_TYPES.len()] = Default::default();
        let mut d_miss: Vec<f64> = Vec::new();
        let mut d_fp: Vec<f64> = Vec::new();
        let mut d_fn: Vec<f64> = Vec::new();

        for cat in per_cat.into_iter().flatten() {
            baseline_aps.push(cat.baseline);
            for (deltas, v) in d_fp_types.iter_mut().zip(cat.fp_types) {
                deltas.push(v);
            }
            d_miss.push(cat.miss);
            d_fp.push(cat.fp);
            d_fn.push(cat.fn_oracle);
        }

        let mean_ap = |v: &[f64]| -> f64 {
            if v.is_empty() {
                0.0
            } else {
                v.iter().sum::<f64>() / v.len() as f64
            }
        };

        let ap_base = mean_ap(&baseline_aps);
        let miss_mean = mean_ap(&d_miss);

        let mut delta_ap: BTreeMap<String, f64> = FP_TYPES
            .iter()
            .zip(d_fp_types.iter())
            .map(|(err, deltas)| (err.as_str().to_string(), mean_ap(deltas)))
            .collect();
        delta_ap.insert("Miss".to_string(), miss_mean);
        delta_ap.insert("FP".to_string(), mean_ap(&d_fp));
        delta_ap.insert("FN".to_string(), mean_ap(&d_fn));

        // The counters, named once. `"Miss"` is the extra key: it is a ground
        // truth tally, not one of `FP_TYPES`.
        let mut counts: BTreeMap<String, u64> = FP_TYPES
            .iter()
            .zip(fp_counts)
            .map(|(err, n)| (err.as_str().to_string(), n))
            .collect();
        counts.insert("Miss".to_string(), miss_total);

        Ok(TideErrors {
            delta_ap,
            counts,
            ap_base,
            pos_thr,
            bg_thr,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const POS: f64 = 0.5;
    const BG: f64 = 0.1;

    fn ev(max_same_iou: f64, max_cross_iou: f64, best_same_gt_matched: bool) -> FpEvidence {
        FpEvidence {
            max_same_iou,
            max_cross_iou,
            best_same_gt_matched,
        }
    }

    fn classify(same: f64, cross: f64, dupe: bool) -> ErrType {
        classify_fp(&ev(same, cross, dupe), POS, BG)
    }

    #[test]
    fn each_error_type_is_reachable() {
        // Loc:  same-class IoU inside [bg, pos]
        assert_eq!(classify(0.3, 0.0, false), ErrType::Loc);
        // Cls:  no Loc, but a cross-class GT is well overlapped
        assert_eq!(classify(0.0, 0.9, false), ErrType::Cls);
        // Dupe: same-class GT above pos_thr already taken by a higher-scoring TP
        assert_eq!(classify(0.9, 0.0, true), ErrType::Dupe);
        // Bkg:  overlaps nothing
        assert_eq!(classify(0.0, 0.0, false), ErrType::Bkg);
        // Both: cross-class IoU strictly inside (bg, pos)
        assert_eq!(classify(0.0, 0.3, false), ErrType::Both);
    }

    #[test]
    fn loc_outranks_cls_and_dupe() {
        // A DT can satisfy Loc *and* Cls; tidecv reports Loc.
        assert_eq!(classify(0.3, 0.9, false), ErrType::Loc);
        // Loc also wins over a would-be Dupe.
        assert_eq!(classify(0.3, 0.0, true), ErrType::Loc);
    }

    #[test]
    fn cls_outranks_dupe_and_bkg() {
        assert_eq!(classify(0.9, 0.9, true), ErrType::Cls);
        // Cross-class overlap above pos_thr beats the "overlaps nothing" reading.
        assert_eq!(classify(0.0, 0.5, false), ErrType::Cls);
    }

    #[test]
    fn dupe_outranks_bkg() {
        // Same-class IoU above pos_thr, so Loc's upper bound excludes it; the
        // already-matched flag is then what distinguishes Dupe from Bkg.
        assert_eq!(classify(0.9, 0.0, true), ErrType::Dupe);
        assert_eq!(classify(0.9, 0.0, false), ErrType::Bkg);
    }

    /// The `Loc` window is closed at both ends. tidecv uses `>=` and `<=`, and
    /// the inclusive upper bound is specifically what keeps a detection sitting
    /// exactly on `pos_thr` out of `Dupe`.
    #[test]
    fn loc_window_is_inclusive_at_both_ends() {
        assert_eq!(classify(BG, 0.0, false), ErrType::Loc);
        assert_eq!(classify(POS, 0.0, true), ErrType::Loc);
        // Just outside the window on either side is not Loc.
        assert_ne!(classify(BG - 1e-9, 0.0, false), ErrType::Loc);
        assert_ne!(classify(POS + 1e-9, 0.0, true), ErrType::Loc);
    }

    /// `Bkg`'s bound is also inclusive: cross-class IoU exactly at `bg_thr` is
    /// background, and anything strictly above it falls through to `Both`.
    #[test]
    fn bkg_upper_bound_is_inclusive() {
        assert_eq!(classify(0.0, BG, false), ErrType::Bkg);
        assert_eq!(classify(0.0, BG + 1e-9, false), ErrType::Both);
    }

    #[test]
    fn as_str_covers_every_variant() {
        for (err, key) in [
            (ErrType::Cls, "Cls"),
            (ErrType::Loc, "Loc"),
            (ErrType::Both, "Both"),
            (ErrType::Dupe, "Dupe"),
            (ErrType::Bkg, "Bkg"),
        ] {
            assert_eq!(err.as_str(), key);
        }
    }

    /// `err as usize` indexes the count tally and the per-type delta vectors, so
    /// it must agree with the position in [`FP_TYPES`]. Reordering the enum
    /// declaration without reordering the array would silently file every `Loc`
    /// under `Cls`.
    #[test]
    fn fp_types_are_indexed_by_discriminant() {
        for (i, &err) in FP_TYPES.iter().enumerate() {
            assert_eq!(err as usize, i, "{} is out of order", err.as_str());
        }
    }
}

/// TIDE error decomposition for object detection.
///
/// Produced by [`super::COCOeval::tide_errors`]. Each ΔAP value measures how much
/// average AP would improve if all errors of that type were fixed.
#[derive(Debug, Clone, Serialize)]
pub struct TideErrors {
    /// ΔAP for each error type (fixing all errors of that type).
    /// Keys: `"Cls"`, `"Loc"`, `"Both"`, `"Dupe"`, `"Bkg"`, `"Miss"`, `"FP"`, `"FN"`.
    ///
    /// `"FP"` and `"FN"` are tidecv's *special* oracles, not sums of the five
    /// types: `"FP"` suppresses every false positive (perfect precision,
    /// recall untouched); `"FN"` removes every unmatched in-denominator GT
    /// from the denominator (perfect recall, precision untouched). `"FN"`
    /// covers a superset of the GTs behind `"Miss"`, which drops only those
    /// no Loc/Cls fix could recover.
    pub delta_ap: BTreeMap<String, f64>,
    /// Count of each error type across all categories and images.
    /// Keys: `"Cls"`, `"Loc"`, `"Both"`, `"Dupe"`, `"Bkg"`, `"Miss"`.
    pub counts: BTreeMap<String, u64>,
    /// Baseline AP at `pos_thr` (mean over categories with GT).
    pub ap_base: f64,
    /// IoU threshold for TP/FP classification.
    pub pos_thr: f64,
    /// Background IoU threshold for Loc/Both/Bkg discrimination.
    pub bg_thr: f64,
}
