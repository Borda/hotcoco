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
        // A few runs per thread: enough to balance, few enough that allocating
        // `k * a` buckets per run stays noise. Tests pass their own to put run
        // boundaries where they want them.
        let chunk_len = ev
            .eval_imgs
            .len()
            .div_ceil(4 * rayon::current_num_threads())
            .max(1);
        Self::build_chunked(ev, chunk_len)
    }

    /// [`build`](Self::build) with the cells walked in runs of `chunk_len`.
    ///
    /// The result does not depend on `chunk_len` — bucket contents and order are
    /// the same for any value; only the internal image-slot numbering differs.
    fn build_chunked(ev: &'a COCOeval, chunk_len: usize) -> Self {
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

        // Area-range keys as bit-exact f64 pairs. The range values are copied verbatim
        // from `params`, so bit equality is the right test. A linear scan over the
        // handful of ranges beats hashing a 16-byte key per cell; `rposition` makes
        // a range listed twice fill its last slot only.
        let area_keys: Vec<[u64; 2]> = params
            .area_ranges
            .iter()
            .map(|ar| [ar.range[0].to_bits(), ar.range[1].to_bits()])
            .collect();

        // Group eval_imgs by (k_idx, a_idx) — one pass over the cells, in parallel.
        //
        // The walk is memory-bound: `Option<EvalImg>` is 360 bytes and COCO-scale
        // runs have ~1.5M of them, so one core reads ~560 MB just to see three ids
        // per cell. The cells are cut into a few contiguous runs per thread; each
        // run keeps its own buckets, in cell order, and the runs are concatenated in
        // order below — so every bucket ends up in `eval_imgs` order exactly as a
        // sequential walk would leave it. That order feeds the stable score sort and
        // decides ties, so it is part of the output, not an implementation detail.
        // Explicit chunks rather than rayon's adaptive splitting: every run allocates
        // `k * a` buckets, so runs must stay few.
        //
        // `evaluate()` writes `area_ranges.len()` consecutive cells per (image,
        // category) pair, pairs sorted by image then category, so consecutive cells
        // almost always share their category and their image. The two memos turn one
        // hash lookup per cell into one per pair (category) and one per image. They
        // are keyed on the id a cell carries, never on its position, so a `params`
        // reconfigured between `evaluate()` and `accumulate()` still resolves every
        // cell correctly — the layout only makes the memo hit.
        //
        // Image slots are assigned per run first (an index into `imgs`), then
        // remapped to global slots once the runs are back together. Slot numbers are
        // internal — `image_mask` is the only reader — so which run saw an image
        // first does not matter, only that every cell of one image shares a slot.
        struct Run<'a> {
            /// `k_idx * a + a_idx` → cells in this run, in cell order; the `u32` is an
            /// index into `imgs`.
            buckets: Vec<Vec<(&'a EvalImg, u32)>>,
            /// Image ids in first-seen order. A repeat is only possible when an image's
            /// cells are not contiguous, and the remap below tolerates it.
            imgs: Vec<u64>,
        }
        let runs: Vec<Run<'a>> = ev
            .eval_imgs
            .par_chunks(chunk_len)
            .map(|cells| {
                let mut run = Run {
                    buckets: vec![Vec::new(); k * a],
                    imgs: Vec::new(),
                };
                let mut last_cat: Option<(u64, Option<usize>)> = None;
                let mut last_img: Option<(u64, u32)> = None;
                for eval in cells.iter().flatten() {
                    let k_idx = match last_cat {
                        Some((id, k_idx)) if id == eval.category_id => k_idx,
                        _ => {
                            let k_idx = cat_id_to_k_idx.get(&eval.category_id).copied();
                            last_cat = Some((eval.category_id, k_idx));
                            k_idx
                        }
                    };
                    let Some(k_idx) = k_idx else {
                        continue;
                    };
                    let a_key = [eval.area_rng[0].to_bits(), eval.area_rng[1].to_bits()];
                    let Some(a_idx) = area_keys.iter().rposition(|key| *key == a_key) else {
                        continue; // skip eval results with area ranges not in current params
                    };
                    let local = match last_img {
                        Some((id, local)) if id == eval.image_id => local,
                        _ => {
                            let local = run.imgs.len() as u32;
                            run.imgs.push(eval.image_id);
                            last_img = Some((eval.image_id, local));
                            local
                        }
                    };
                    run.buckets[k_idx * a + a_idx].push((eval, local));
                }
                run
            })
            .collect();

        // Global slots, and one run-local → global table per run.
        let mut img_slots: HashMap<u64, u32> = HashMap::new();
        let remaps: Vec<Vec<u32>> = runs
            .iter()
            .map(|run| {
                run.imgs
                    .iter()
                    .map(|&img_id| {
                        let next = img_slots.len() as u32;
                        *img_slots.entry(img_id).or_insert(next)
                    })
                    .collect()
            })
            .collect();

        // Concatenate the runs bucket by bucket, in run order.
        let grouped: Vec<Vec<(&EvalImg, u32)>> = (0..k * a)
            .into_par_iter()
            .map(|b| {
                let total: usize = runs.iter().map(|run| run.buckets[b].len()).sum();
                let mut bucket = Vec::with_capacity(total);
                for (run, remap) in runs.iter().zip(&remaps) {
                    bucket.extend(
                        run.buckets[b]
                            .iter()
                            .map(|&(eval, local)| (eval, remap[local as usize])),
                    );
                }
                bucket
            })
            .collect();

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

            // One gather and one sort per work item, shared by every `m`.
            //
            // pycocotools concatenates each cell's `dtScores[0:maxDet]` and
            // mergesorts (stably) per `m`. A stable sort of the per-cell-truncated
            // concatenation equals the stable sort of the concatenation truncated
            // at the *largest* cap, filtered to `rank_in_cell < maxDet`: filtering
            // preserves relative order, and ties break on concatenation position,
            // which the filter also preserves. So gather at `Params::max_det()`
            // once, sort once, and derive each `m` by a stable filter. Cells may hold more detections than the
            // *current* cap when `params.max_dets` shrank between `evaluate()` and
            // `accumulate()`, so the gather truncates to the current cap and never
            // trusts the stored length.
            //
            // The identity needs a strict weak order on scores, which `partial_cmp`
            // gives for every finite value (and for `-0.0` versus `0.0`, which
            // compare equal and keep input order). A NaN score breaks it: the
            // comparator below treats NaN as equal to everything, so the sorted
            // order — and therefore which detections a filtered slot keeps —
            // depends on the input sequence. `load_res` rejects NaN scores;
            // `COCO::from_dataset` does not, so NaN ranking is undefined.
            let cap = params.max_det();
            let mut all_dt_scores: Vec<f64> = Vec::new();
            // Position of each gathered detection inside its cell's score-descending
            // list — the per-cell truncation index that `dtScores[0:maxDet]` applies.
            let mut rank_in_cell: Vec<usize> = Vec::new();
            let mut all_dt_matched: Vec<Vec<bool>> = vec![Vec::new(); t];
            let mut all_dt_ignore: Vec<Vec<bool>> = vec![Vec::new(); t];
            for eval_img in &evals {
                let nd = eval_img.dt_scores.len().min(cap);
                all_dt_scores.extend_from_slice(&eval_img.dt_scores[..nd]);
                rank_in_cell.extend(0..nd);
                for t_idx in 0..t {
                    all_dt_matched[t_idx].extend_from_slice(&eval_img.dt_matched.row(t_idx)[..nd]);
                    all_dt_ignore[t_idx].extend_from_slice(&eval_img.dt_ignore.row(t_idx)[..nd]);
                }
            }

            // Sort by score descending — stable, ties keep concatenation order.
            let mut order: Vec<usize> = (0..all_dt_scores.len()).collect();
            order.sort_by(|&a, &b| {
                all_dt_scores[b]
                    .partial_cmp(&all_dt_scores[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Buffers reused across the M axis and the threshold sweep.
            let mut filtered: Vec<usize> = Vec::new();
            let (mut tp, mut fp) = (Vec::new(), Vec::new());
            let mut pr_scratch = crate::metrics::counts::PrCurveScratch::default();
            let mut curve: Vec<(usize, f64, usize)> = Vec::new();

            for m_idx in 0..m {
                let max_det = params.max_dets[m_idx];

                // Stable filter of the shared order == per-`m` sort (see above).
                let inds: &[usize] = if max_det >= cap {
                    &order
                } else {
                    filtered.clear();
                    filtered.extend(order.iter().copied().filter(|&i| rank_in_cell[i] < max_det));
                    &filtered
                };

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

                for t_idx in 0..t {
                    // `metrics::counts` owns the TP/FP classification and the
                    // curve. The all-points AP is the exact area under the same
                    // envelope the grid samples and needs the cumulative `tp`/`fp`
                    // arrays, score-ordered — it cannot be recovered from the 101
                    // samples afterwards — so Open Images materializes them and
                    // reads the curve from them. Every other mode discards that
                    // AP and takes the fused kernel, which produces the same
                    // curve without the two arrays.
                    let (final_recall, all_points_ap) = if want_all_points {
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
                        let ap =
                            crate::metrics::counts::average_precision_all_points(&tp, &fp, num_gt);
                        (final_recall, ap)
                    } else {
                        let final_recall =
                            crate::metrics::counts::precision_recall_curve_of_order_into(
                                inds.iter().copied(),
                                &all_dt_matched[t_idx],
                                Some(&all_dt_ignore[t_idx]),
                                num_gt,
                                &params.rec_thrs,
                                &mut pr_scratch,
                                &mut curve,
                            );
                        (final_recall, -1.0)
                    };
                    let recall_idx = shape.recall_idx(t_idx, k_idx, a_idx, m_idx);
                    out.recall_writes
                        .push((recall_idx, final_recall, all_points_ap));

                    for &(r_idx, pr_val, rc_ptr) in &curve {
                        let p_idx = shape.precision_idx(t_idx, r_idx, k_idx, a_idx, m_idx);
                        out.precision_writes.push((p_idx, pr_val));
                        out.scores_writes.push((p_idx, all_dt_scores[inds[rc_ptr]]));
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
/// that no data was available for that combination — a category with no GT instances, for example.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::COCO;
    use crate::params::{AreaRange, IouType};
    use crate::types::{Annotation, Category, Dataset, Image};

    /// A sequential, unmemoized grouping walk: three hash lookups per cell. The
    /// oracle the production build is checked against — the production code
    /// shares none of its lookups, so a memo keyed on the wrong id or a run
    /// concatenated out of order shows up as a difference.
    ///
    /// Returns, per `k_idx * a + a_idx` bucket, the cells in order as
    /// `(cell address, image_id)`; slot numbers are not compared (they are
    /// internal), only that they are consistent — see `assert_slots_consistent`.
    fn reference_grouping(ev: &COCOeval) -> Vec<Vec<(*const EvalImg, u64)>> {
        let params = &ev.params;
        let k = if params.use_cats {
            params.cat_ids.len()
        } else {
            1
        };
        let a = params.area_ranges.len();
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
        let area_rng_to_idx: HashMap<[u64; 2], usize> = params
            .area_ranges
            .iter()
            .enumerate()
            .map(|(i, ar)| ([ar.range[0].to_bits(), ar.range[1].to_bits()], i))
            .collect();
        let mut grouped = vec![Vec::new(); k * a];
        for eval in ev.eval_imgs.iter().flatten() {
            let Some(&k_idx) = cat_id_to_k_idx.get(&eval.category_id) else {
                continue;
            };
            let a_key = [eval.area_rng[0].to_bits(), eval.area_rng[1].to_bits()];
            let Some(&a_idx) = area_rng_to_idx.get(&a_key) else {
                continue;
            };
            grouped[k_idx * a + a_idx].push((std::ptr::from_ref(eval), eval.image_id));
        }
        grouped
    }

    fn area(label: &str, lo: f64, hi: f64) -> AreaRange {
        AreaRange {
            label: label.into(),
            range: [lo, hi],
        }
    }

    /// Six images × three categories. Most (image, category) pairs carry both a
    /// ground truth and detections; a few carry only one side, so some cells are
    /// `None` and some pairs exist only through detections. Boxes step in size so
    /// the area ranges below split them.
    fn make_eval(use_cats: bool) -> COCOeval {
        let images: Vec<Image> = (1..=6)
            .map(|id| Image {
                id,
                file_name: format!("{id}.jpg"),
                width: 640,
                height: 640,
                ..Default::default()
            })
            .collect();
        let categories: Vec<Category> = [1u64, 2, 3]
            .iter()
            .map(|&id| Category {
                id,
                name: format!("c{id}"),
                ..Default::default()
            })
            .collect();
        let bbox = |img: u64, cat: u64| {
            let side = 10.0 * (1 + img + 2 * cat) as f64;
            [5.0 * img as f64, 5.0 * cat as f64, side, side]
        };
        let mut gt_anns = Vec::new();
        let mut dt_anns = Vec::new();
        let mut next = 1u64;
        for img in 1..=6u64 {
            for cat in 1..=3u64 {
                let b = bbox(img, cat);
                // Image 5 has no ground truth for category 2; image 6 has no
                // detections for category 3.
                if !(img == 5 && cat == 2) {
                    gt_anns.push(Annotation {
                        id: next,
                        image_id: img,
                        category_id: cat,
                        bbox: Some(b),
                        area: Some(b[2] * b[3]),
                        ..Default::default()
                    });
                    next += 1;
                }
                if !(img == 6 && cat == 3) {
                    for (j, score) in [0.9, 0.6].iter().enumerate() {
                        let shifted = [b[0] + 2.0 * j as f64, b[1], b[2], b[3]];
                        dt_anns.push(Annotation {
                            id: next,
                            image_id: img,
                            category_id: cat,
                            bbox: Some(shifted),
                            area: Some(shifted[2] * shifted[3]),
                            score: Some(*score),
                            ..Default::default()
                        });
                        next += 1;
                    }
                }
            }
        }
        let dataset = |annotations| Dataset {
            info: None,
            images: images.clone(),
            annotations,
            categories: categories.clone(),
            licenses: vec![],
        };
        let gt = COCO::from_dataset(dataset(gt_anns));
        let dt = COCO::from_dataset(dataset(dt_anns));
        let mut ev = COCOeval::new(gt, dt, IouType::Bbox);
        ev.params.use_cats = use_cats;
        ev.params.area_ranges = vec![
            area("all", 0.0, 1e10),
            area("small", 0.0, 2500.0),
            area("medium", 2500.0, 6400.0),
            area("large", 6400.0, 1e10),
        ];
        ev.evaluate();
        ev
    }

    fn assert_same_grouping(ev: &COCOeval, chunk_len: usize, label: &str) {
        let expected = reference_grouping(ev);
        let got = EvalGrouping::build_chunked(ev, chunk_len);
        assert_eq!(got.grouped.len(), expected.len(), "{label}: bucket count");
        for (b, (got_bucket, want_bucket)) in got.grouped.iter().zip(&expected).enumerate() {
            let got_cells: Vec<(*const EvalImg, u64)> = got_bucket
                .iter()
                .map(|&(eval, _)| (std::ptr::from_ref(eval), eval.image_id))
                .collect();
            assert_eq!(
                got_cells, *want_bucket,
                "{label}, chunk_len {chunk_len}: bucket {b} differs from the sequential walk"
            );
        }
        assert_slots_consistent(&got, label);
    }

    /// Every cell of one image carries one slot, distinct images carry distinct
    /// slots, and `image_mask` selects exactly the cells of the requested images.
    fn assert_slots_consistent(g: &EvalGrouping<'_>, label: &str) {
        let mut slot_of: HashMap<u64, u32> = HashMap::new();
        let mut img_of: HashMap<u32, u64> = HashMap::new();
        for &(eval, slot) in g.grouped.iter().flatten() {
            assert_eq!(
                *slot_of.entry(eval.image_id).or_insert(slot),
                slot,
                "{label}: image {} carries two slots",
                eval.image_id
            );
            assert_eq!(
                *img_of.entry(slot).or_insert(eval.image_id),
                eval.image_id,
                "{label}: slot {slot} carries two images"
            );
        }
        let keep: HashSet<u64> = [2u64, 5].into_iter().collect();
        let mask = g.image_mask(Some(&keep));
        for &(eval, slot) in g.grouped.iter().flatten() {
            assert_eq!(
                mask[slot as usize],
                keep.contains(&eval.image_id),
                "{label}: image_mask disagrees with the filter for image {}",
                eval.image_id
            );
        }
    }

    /// Chunk lengths that put run boundaries in the middle of an image, in the
    /// middle of a pair's area-range block, at one cell, and nowhere.
    const CHUNKS: [usize; 5] = [1, 3, 7, 16, usize::MAX];

    #[test]
    fn grouping_matches_sequential_walk_under_reconfigured_params() {
        let mut ev = make_eval(true);
        assert!(
            ev.eval_imgs.iter().any(Option::is_none) && ev.eval_imgs.iter().any(Option::is_some),
            "fixture must produce both empty and filled cells"
        );
        for chunk_len in CHUNKS {
            assert_same_grouping(&ev, chunk_len, "as evaluated");
        }

        // Category subset, reordered: cells of category 2 must be skipped and the
        // remaining two land in swapped slots.
        ev.params.cat_ids = vec![3, 1];
        for chunk_len in CHUNKS {
            assert_same_grouping(&ev, chunk_len, "cat_ids = [3, 1]");
        }

        // Area subset, reordered, one range listed twice: a repeated range fills its
        // last slot only, and the dropped ranges skip.
        ev.params.area_ranges = vec![
            area("large", 6400.0, 1e10),
            area("all", 0.0, 1e10),
            area("all again", 0.0, 1e10),
        ];
        for chunk_len in CHUNKS {
            assert_same_grouping(&ev, chunk_len, "areas reordered with a duplicate");
        }
        let g = EvalGrouping::build(&ev);
        assert_eq!(
            g.grouped.len(),
            ev.params.cat_ids.len() * ev.params.area_ranges.len()
        );
        for k_idx in 0..ev.params.cat_ids.len() {
            assert!(
                g.cell(k_idx, 1).is_empty() && !g.cell(k_idx, 2).is_empty(),
                "a repeated area range must fill its last slot only"
            );
        }

        // Nothing left in scope: every bucket empty, no slots.
        ev.params.area_ranges = vec![area("none", 1.0, 2.0)];
        let g = EvalGrouping::build(&ev);
        assert!(g.grouped.iter().all(Vec::is_empty));
        assert!(g.img_slots.is_empty());
    }

    #[test]
    fn grouping_matches_sequential_walk_without_categories() {
        let ev = make_eval(false);
        assert!(ev.eval_imgs.iter().any(Option::is_some));
        for chunk_len in CHUNKS {
            assert_same_grouping(&ev, chunk_len, "use_cats = false");
        }
        // The category axis collapses to one slot.
        let g = EvalGrouping::build(&ev);
        assert_eq!(g.grouped.len(), ev.params.area_ranges.len());
        assert!(
            g.cell(0, 0).len() >= 6,
            "every image lands in the 'all' bucket"
        );
    }
}
