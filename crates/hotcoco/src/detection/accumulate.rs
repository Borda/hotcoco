use std::collections::{HashMap, HashSet};

use rayon::prelude::*;

use super::COCOeval;
use super::EvalMode;
use super::matching::EvalImg;

/// `eval_imgs` bucketed by (category slot, area-range slot).
///
/// The bucketing depends only on `params` and the evaluated cells — never on which
/// images an accumulation happens to cover — so it is built once and reused across
/// every filtered re-accumulation. `compare()` runs two per bootstrap resample and
/// `slice_by()` one per slice; rebuilding the two id→index maps and walking all
/// ~400k cells inside each of those was the same answer recomputed hundreds of
/// times. The image filter is applied where it actually varies: in the per-item
/// loop of [`accumulate_impl`].
///
/// It carries the evaluator it was built from rather than leaving the caller to
/// pass a matching `params`/`eval_mode` alongside it, so a grouping bucketed
/// under one evaluator's category list cannot be accumulated under another's.
pub(super) struct EvalGrouping<'a> {
    ev: &'a COCOeval,
    /// Indexed `k_idx * a + a_idx`, each bucket in `eval_imgs` order. Each cell
    /// carries the dense image slot its `image_id` resolves to — see
    /// [`image_mask`](Self::image_mask).
    grouped: Vec<Vec<(&'a EvalImg, u32)>>,
    /// Number of area ranges — the stride of `grouped`.
    a: usize,
    /// Distinct `image_id` -> dense slot, over every cell in `grouped`.
    img_slots: HashMap<u64, u32>,
}

impl<'a> EvalGrouping<'a> {
    /// Bucket an evaluated `COCOeval`'s cells.
    pub(super) fn build(ev: &'a COCOeval) -> Self {
        let params = &ev.params;
        let k = if params.use_cats {
            params.cat_ids.len()
        } else {
            1
        };
        let a = params.area_ranges.len();

        // Build category_id → k_idx mapping for grouping eval_imgs.
        let cat_id_to_k_idx: HashMap<u64, usize> = if params.use_cats {
            params
                .cat_ids
                .iter()
                .enumerate()
                .map(|(i, &id)| (id, i))
                .collect()
        } else {
            std::iter::once((u64::MAX, 0usize)).collect()
        };

        // Build area_range → index lookup using bit-exact f64 keys (avoids linear search).
        // range values are copied verbatim from params, so bit-exact equality is safe.
        let area_rng_to_idx: HashMap<[u64; 2], usize> = params
            .area_ranges
            .iter()
            .enumerate()
            .map(|(i, ar)| ([ar.range[0].to_bits(), ar.range[1].to_bits()], i))
            .collect();

        // Group eval_imgs by (k_idx, a_idx) — O(eval_imgs) once.
        let mut grouped: Vec<Vec<(&EvalImg, u32)>> = vec![Vec::new(); k * a];
        let mut img_slots: HashMap<u64, u32> = HashMap::new();
        for eval in ev.eval_imgs.iter().flatten() {
            if let Some(&k_idx) = cat_id_to_k_idx.get(&eval.category_id) {
                let a_key = [eval.area_rng[0].to_bits(), eval.area_rng[1].to_bits()];
                let a_idx = match area_rng_to_idx.get(&a_key).copied() {
                    Some(idx) => idx,
                    None => continue, // skip eval results with area ranges not in current params
                };
                let next = img_slots.len() as u32;
                let slot = *img_slots.entry(eval.image_id).or_insert(next);
                grouped[k_idx * a + a_idx].push((eval, slot));
            }
        }

        EvalGrouping {
            ev,
            grouped,
            a,
            img_slots,
        }
    }

    /// The evaluator this grouping was built from.
    pub(super) fn eval(&self) -> &'a COCOeval {
        self.ev
    }

    /// A dense "is this image in scope" bitmap, indexed by the slot each cell carries.
    ///
    /// Built once per accumulation instead of probing a `HashSet<u64>` once per
    /// cell per work item. The membership test runs `K × A × M × cells` times —
    /// on COCO val that is ~1.2M hash lookups per `accumulate`, and `compare()`
    /// pays it again for every one of its bootstrap resamples.
    ///
    /// `None` means the whole dataset, which is an all-true mask rather than a
    /// branch at every cell.
    pub(super) fn image_mask(&self, img_filter: Option<&HashSet<u64>>) -> Vec<bool> {
        let n = self.img_slots.len();
        let Some(filter) = img_filter else {
            return vec![true; n];
        };
        let mut mask = vec![false; n];
        for (img_id, &slot) in &self.img_slots {
            if filter.contains(img_id) {
                mask[slot as usize] = true;
            }
        }
        mask
    }

    fn cell(&self, k_idx: usize, a_idx: usize) -> &[(&'a EvalImg, u32)] {
        &self.grouped[k_idx * self.a + a_idx]
    }
}

/// Accumulate per-image eval results into precision/recall arrays.
///
/// When `img_filter` is `Some`, only eval_imgs whose `image_id` is in the set
/// are included. Pass `None` to include all images (standard behavior).
///
/// `eval_mode` decides only whether [`AccumulatedEval::ap_all_points`] is filled.
/// Open Images is the one mode that reads it, and computing it costs about as much
/// again as the gridded curve beside it — measured at +32% on `accumulate` for COCO
/// val2017 bbox — so every other mode gets the `-1.0` "not computed" sentinel
/// rather than paying for a value it discards. `compare::bootstrap_ci` calls this
/// once per resample, which multiplies the saving.
pub(super) fn accumulate_impl(
    grouping: &EvalGrouping<'_>,
    img_filter: Option<&HashSet<u64>>,
) -> AccumulatedEval {
    let params = &grouping.eval().params;
    let want_all_points = grouping.eval().eval_mode == EvalMode::OpenImages;
    let t = params.iou_thrs.len();
    let r = params.rec_thrs.len();
    let k = if params.use_cats {
        params.cat_ids.len()
    } else {
        1
    };
    let a = params.area_ranges.len();
    let m = params.max_dets.len();

    // One dense in-scope bitmap for the whole accumulation, replacing a
    // `HashSet<u64>` probe per cell per work item.
    let img_mask = grouping.image_mask(img_filter);

    // Work items are `(k_idx, a_idx)` with the M axis **inside**, not the full
    // `(k, a, m)` product. Everything the max-detection settings share — which
    // cells are in scope, and `num_gt`, which does not depend on the cap — is
    // resolved once per item instead of once per `m`. On COCO that is three
    // passes over the cell list collapsed into one, and the `num_gt == 0`
    // short-circuit now skips all three M slots together.
    //
    // The output stays disjoint per `m`, so the merge below is unchanged and no
    // floating-point sum is reassociated.
    let work_items: Vec<(usize, usize)> = (0..k)
        .flat_map(|k_idx| (0..a).map(move |a_idx| (k_idx, a_idx)))
        .collect();

    /// Intermediate results from a single (category, area_range) work item.
    /// Each field is a list of (flat_index, value) pairs to write into the output arrays.
    #[derive(Default)]
    struct AccResult {
        /// Whether this work item had ground truth, and therefore needs the 0.0
        /// zero-fill applied across every M slot in the merge. The merge writes
        /// the zeros before the real writes, which overwrite them — same
        /// indices, same order. (Per-item, not per-slot: `num_gt` does not
        /// depend on the max-det cap, so all M slots fill together.)
        filled: bool,
        precision_writes: Vec<(usize, f64)>,
        /// `(flat_index, max_recall, all_points_ap)`. The AP rides along with the
        /// recall it was computed from rather than in a parallel vector, so the two
        /// cannot be written at different indices or one forgotten on an early
        /// return. Carries `-1.0` when the mode does not want it.
        recall_writes: Vec<(usize, f64, f64)>,
        scores_writes: Vec<(usize, f64)>,
    }

    let shape = EvalShape { t, r, k, a, m };

    let results: Vec<AccResult> = work_items
        .par_iter()
        .map(|&(k_idx, a_idx)| {
            // Materialized once for every `m`, rather than filtered per `m`.
            let evals: Vec<&EvalImg> = grouping
                .cell(k_idx, a_idx)
                .iter()
                .filter(|&&(_, slot)| img_mask[slot as usize])
                .map(|&(e, _)| e)
                .collect();

            // Independent of `max_det` — the cap truncates detections, never
            // ground truth — so it is summed once for the whole M axis.
            let num_gt: usize = evals.iter().map(|e| e.num_gt_in_denominator()).sum();
            if num_gt == 0 {
                return AccResult::default();
            }

            let mut out = AccResult {
                filled: true,
                precision_writes: Vec::with_capacity(m * t * r),
                recall_writes: Vec::with_capacity(m * t),
                scores_writes: Vec::with_capacity(m * t * r),
            };

            // Buffers reused across the M axis and the threshold sweep.
            let mut all_dt_scores: Vec<f64> = Vec::new();
            let mut all_dt_matched: Vec<Vec<bool>> = vec![Vec::new(); t];
            let mut all_dt_ignore: Vec<Vec<bool>> = vec![Vec::new(); t];
            let mut sorted_scores: Vec<f64> = Vec::new();
            let (mut tp, mut fp) = (Vec::new(), Vec::new());
            let mut pr_scratch = crate::metrics::counts::PrCurveScratch::default();
            let mut curve: Vec<(usize, f64, usize)> = Vec::new();

            for m_idx in 0..m {
                let max_det = params.max_dets[m_idx];

                all_dt_scores.clear();
                for v in all_dt_matched.iter_mut().chain(all_dt_ignore.iter_mut()) {
                    v.clear();
                }

                for eval_img in &evals {
                    let nd = eval_img.dt_scores.len().min(max_det);

                    all_dt_scores.extend_from_slice(&eval_img.dt_scores[..nd]);
                    for t_idx in 0..t {
                        all_dt_matched[t_idx]
                            .extend_from_slice(&eval_img.dt_matched.row(t_idx)[..nd]);
                        all_dt_ignore[t_idx]
                            .extend_from_slice(&eval_img.dt_ignore.row(t_idx)[..nd]);
                    }
                }

                // Sort by score descending
                let mut inds: Vec<usize> = (0..all_dt_scores.len()).collect();
                inds.sort_by(|&a, &b| {
                    all_dt_scores[b]
                        .partial_cmp(&all_dt_scores[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                let nd = inds.len();

                if nd == 0 {
                    // GT exists but no detections — recall and AP are 0.0, not -1.0
                    // "missing". The metric *is* computable here and the answer is that
                    // nothing was found; reporting "not computed" would drop the
                    // category from the mean and quietly raise mAP.
                    for t_idx in 0..t {
                        let recall_idx = shape.recall_idx(t_idx, k_idx, a_idx, m_idx);
                        let ap = if want_all_points { 0.0 } else { -1.0 };
                        out.recall_writes.push((recall_idx, 0.0, ap));
                    }
                    continue;
                }

                // Hoist sorted_scores outside the threshold loop (identical across thresholds)
                sorted_scores.clear();
                sorted_scores.extend(inds.iter().map(|&i| all_dt_scores[i]));

                for t_idx in 0..t {
                    // `metrics::counts` owns the TP/FP classification and its
                    // running sums.
                    crate::metrics::counts::cumulative_tp_fp(
                        inds.iter().copied(),
                        &all_dt_matched[t_idx],
                        Some(&all_dt_ignore[t_idx]),
                        &mut tp,
                        &mut fp,
                    );

                    let final_recall = crate::metrics::counts::precision_recall_curve_into(
                        &tp,
                        &fp,
                        num_gt,
                        &params.rec_thrs,
                        &mut pr_scratch,
                        &mut curve,
                    );

                    // The all-points AP is the exact area under the same envelope the
                    // grid samples. It has to be computed here, where `tp`/`fp` are
                    // already score-ordered and cumulative — it cannot be recovered
                    // from the 101 samples afterwards.
                    let all_points_ap = if want_all_points {
                        crate::metrics::counts::average_precision_all_points(&tp, &fp, num_gt)
                    } else {
                        -1.0
                    };
                    let recall_idx = shape.recall_idx(t_idx, k_idx, a_idx, m_idx);
                    out.recall_writes
                        .push((recall_idx, final_recall, all_points_ap));

                    for &(r_idx, pr_val, rc_ptr) in &curve {
                        let p_idx = shape.precision_idx(t_idx, r_idx, k_idx, a_idx, m_idx);
                        out.precision_writes.push((p_idx, pr_val));
                        out.scores_writes.push((p_idx, sorted_scores[rc_ptr]));
                    }
                }
            }

            out
        })
        .collect();

    // Merge results into output arrays
    let total = t * r * k * a * m;
    let mut precision = vec![-1.0f64; total];
    let mut scores = vec![-1.0f64; total];
    let total_recall = t * k * a * m;
    let mut recall = vec![-1.0f64; total_recall];
    let mut ap_all_points = vec![-1.0f64; total_recall];

    // `work_items` and `results` are index-parallel: rayon's indexed `collect`
    // preserves order, which is what lets the zero-fill recover each result's
    // (k, a) coordinates without carrying them in the struct.
    for (&(k_idx, a_idx), result) in work_items.iter().zip(results) {
        // Initialize precision and scores to 0.0 (distinct from -1.0, which means
        // "no data"). This ensures categories with GT but no matches show 0 AP,
        // not "missing". Only recall thresholds reached by actual detections get
        // overwritten below — unreachable thresholds stay at 0.0.
        if result.filled {
            for m_idx in 0..m {
                for t_idx in 0..t {
                    for r_idx in 0..r {
                        let p_idx = shape.precision_idx(t_idx, r_idx, k_idx, a_idx, m_idx);
                        precision[p_idx] = 0.0;
                        scores[p_idx] = 0.0;
                    }
                }
            }
        }
        for (idx, val) in result.precision_writes {
            precision[idx] = val;
        }
        for (idx, rec, ap) in result.recall_writes {
            recall[idx] = rec;
            ap_all_points[idx] = ap;
        }
        for (idx, val) in result.scores_writes {
            scores[idx] = val;
        }
    }

    AccumulatedEval {
        precision,
        recall,
        ap_all_points,
        scores,
        shape,
    }
}

impl COCOeval {
    /// Accumulate per-image results into precision/recall arrays.
    pub fn accumulate(&mut self) {
        // Scoped so the grouping's borrow of `self` ends before `self.eval` is written.
        let eval = accumulate_impl(&EvalGrouping::build(self), None);
        self.eval = Some(eval);
    }
}

/// Array dimensions of an accumulated evaluation result.
///
/// Precision and scores have shape `[T x R x K x A x M]`;
/// recall has shape `[T x K x A x M]`.
#[derive(Debug, Clone, Copy)]
pub struct EvalShape {
    /// Number of IoU thresholds (T).
    pub t: usize,
    /// Number of recall thresholds (R).
    pub r: usize,
    /// Number of categories (K).
    pub k: usize,
    /// Number of area ranges (A).
    pub a: usize,
    /// Number of max-detection limits (M).
    pub m: usize,
}

impl EvalShape {
    /// Flat index into `precision` (or `scores`) for 5-D coordinates.
    pub fn precision_idx(&self, t: usize, r: usize, k: usize, a: usize, m: usize) -> usize {
        ((((t * self.r + r) * self.k + k) * self.a + a) * self.m) + m
    }

    /// Flat index into `recall` for 4-D coordinates.
    pub fn recall_idx(&self, t: usize, k: usize, a: usize, m: usize) -> usize {
        (((t * self.k + k) * self.a + a) * self.m) + m
    }
}

/// Accumulated evaluation results across all images.
///
/// Precision and scores are stored as flat 5-D arrays with shape `[T x R x K x A x M]`.
/// Recall is a flat 4-D array with shape `[T x K x A x M]`. Values of -1.0 indicate
/// that no data was available for that combination (e.g. a category with no GT instances).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AccumulatedEval {
    /// Interpolated precision at each (iou_thr, recall_thr, category, area_range, max_det).
    pub precision: Vec<f64>,
    /// Maximum recall at each (iou_thr, category, area_range, max_det).
    pub recall: Vec<f64>,
    /// VOC 2010 all-points AP — same shape and indexing as
    /// [`recall`](Self::recall), so `recall_idx` works for both. The exact area
    /// under the precision envelope that [`precision`](Self::precision) holds
    /// sampled at 101 points; see
    /// [`average_precision_all_points`](crate::metrics::counts::average_precision_all_points)
    /// for why Open Images wants the former. `-1.0` in every other mode, which
    /// does not compute it.
    pub ap_all_points: Vec<f64>,
    /// Detection score at each precision threshold, same shape as `precision`.
    pub scores: Vec<f64>,
    /// Array dimensions — use to interpret the flat precision/recall/scores vectors.
    pub shape: EvalShape,
}

impl AccumulatedEval {
    /// Flat index into `precision` (or `scores`) for 5-D coordinates.
    pub fn precision_idx(&self, t: usize, r: usize, k: usize, a: usize, m: usize) -> usize {
        self.shape.precision_idx(t, r, k, a, m)
    }

    /// Flat index into `recall` for 4-D coordinates.
    pub fn recall_idx(&self, t: usize, k: usize, a: usize, m: usize) -> usize {
        self.shape.recall_idx(t, k, a, m)
    }
}
