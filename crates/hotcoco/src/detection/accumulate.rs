use std::collections::{HashMap, HashSet};

use rayon::prelude::*;

use crate::params::Params;

use super::COCOeval;
use super::matching::EvalImg;

/// Accumulate per-image eval results into precision/recall arrays.
///
/// When `img_filter` is `Some`, only eval_imgs whose `image_id` is in the set
/// are included. Pass `None` to include all images (standard behavior).
pub(super) fn accumulate_impl(
    eval_imgs: &[Option<EvalImg>],
    params: &Params,
    img_filter: Option<&HashSet<u64>>,
) -> AccumulatedEval {
    let t = params.iou_thrs.len();
    let r = params.rec_thrs.len();
    let k = if params.use_cats {
        params.cat_ids.len()
    } else {
        1
    };
    let a = params.area_ranges.len();
    let m = params.max_dets.len();

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
    // Replaces the old dense index formula k_actual * (a * N) + a_idx * N + img_idx,
    // which assumed a specific dense layout that no longer applies after the sparse refactor.
    let mut grouped: Vec<Vec<&EvalImg>> = vec![Vec::new(); k * a];
    for eval in eval_imgs.iter().flatten() {
        if let Some(filter) = img_filter {
            if !filter.contains(&eval.image_id) {
                continue;
            }
        }
        if let Some(&k_idx) = cat_id_to_k_idx.get(&eval.category_id) {
            let a_key = [eval.area_rng[0].to_bits(), eval.area_rng[1].to_bits()];
            let a_idx = match area_rng_to_idx.get(&a_key).copied() {
                Some(idx) => idx,
                None => continue, // skip eval results with area ranges not in current params
            };
            grouped[k_idx * a + a_idx].push(eval);
        }
    }

    // Build flat list of (k_idx, a_idx, m_idx) work items
    let work_items: Vec<(usize, usize, usize)> = (0..k)
        .flat_map(|k_idx| {
            (0..a).flat_map(move |a_idx| (0..m).map(move |m_idx| (k_idx, a_idx, m_idx)))
        })
        .collect();

    // Each work item produces a set of (index, value) writes for precision, recall, scores
    /// Intermediate results from a single (category, area_range, max_det) work item.
    /// Each field is a list of (flat_index, value) pairs to write into the output arrays.
    struct AccResult {
        precision_writes: Vec<(usize, f64)>,
        recall_writes: Vec<(usize, f64)>,
        scores_writes: Vec<(usize, f64)>,
    }

    let shape = EvalShape { t, r, k, a, m };

    let results: Vec<AccResult> = work_items
        .par_iter()
        .map(|&(k_idx, a_idx, m_idx)| {
            let max_det = params.max_dets[m_idx];

            let evals = &grouped[k_idx * a + a_idx];
            let total_dts: usize = evals.iter().map(|e| e.dt_scores.len().min(max_det)).sum();

            let mut all_dt_scores: Vec<f64> = Vec::with_capacity(total_dts);
            let mut all_dt_matched: Vec<Vec<bool>> =
                (0..t).map(|_| Vec::with_capacity(total_dts)).collect();
            let mut all_dt_ignore: Vec<Vec<bool>> =
                (0..t).map(|_| Vec::with_capacity(total_dts)).collect();
            let mut num_gt = 0usize;

            for eval_img in evals {
                let nd = eval_img.dt_scores.len().min(max_det);

                all_dt_scores.extend_from_slice(&eval_img.dt_scores[..nd]);
                for t_idx in 0..t {
                    all_dt_matched[t_idx].extend_from_slice(&eval_img.dt_matched[t_idx][..nd]);
                    all_dt_ignore[t_idx].extend_from_slice(&eval_img.dt_ignore[t_idx][..nd]);
                }

                num_gt += eval_img.gt_ignore.iter().filter(|&&x| !x).count();
            }

            let mut precision_writes = Vec::new();
            let mut recall_writes = Vec::new();
            let mut scores_writes = Vec::new();

            if num_gt == 0 {
                return AccResult {
                    precision_writes,
                    recall_writes,
                    scores_writes,
                };
            }

            // Initialize precision, recall, and scores to 0.0 (distinct from -1.0 which
            // means "no data"). This ensures categories with GT but no matches show 0 AP,
            // not "missing". Only recall thresholds reached by actual detections get
            // overwritten — unreachable thresholds stay at 0.0.
            for t_idx in 0..t {
                for r_idx in 0..r {
                    let p_idx = shape.precision_idx(t_idx, r_idx, k_idx, a_idx, m_idx);
                    precision_writes.push((p_idx, 0.0));
                    scores_writes.push((p_idx, 0.0));
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
                // GT exists but no detections — recall is 0.0 (not -1.0 "missing").
                for t_idx in 0..t {
                    let recall_idx = shape.recall_idx(t_idx, k_idx, a_idx, m_idx);
                    recall_writes.push((recall_idx, 0.0));
                }
                return AccResult {
                    precision_writes,
                    recall_writes,
                    scores_writes,
                };
            }

            // Hoist sorted_scores outside the threshold loop (identical across thresholds)
            let sorted_scores: Vec<f64> = inds.iter().map(|&i| all_dt_scores[i]).collect();

            // Pre-allocate TP/FP buffers reused across thresholds.
            let mut tp = vec![0.0f64; nd];
            let mut fp = vec![0.0f64; nd];

            for t_idx in 0..t {
                // Classify each detection (in score-sorted order) as TP, FP, or ignored.
                // Ignored detections contribute neither TP nor FP.
                for (out_idx, &src_idx) in inds.iter().enumerate() {
                    if all_dt_ignore[t_idx][src_idx] {
                        tp[out_idx] = 0.0;
                        fp[out_idx] = 0.0;
                    } else if all_dt_matched[t_idx][src_idx] {
                        tp[out_idx] = 1.0;
                        fp[out_idx] = 0.0;
                    } else {
                        tp[out_idx] = 0.0;
                        fp[out_idx] = 1.0;
                    }
                }

                // Cumulative sum: tp[d] = total TPs up to detection d.
                for d in 1..nd {
                    tp[d] += tp[d - 1];
                    fp[d] += fp[d - 1];
                }

                let (final_recall, curve) = crate::metrics::counts::precision_recall_curve(
                    &tp,
                    &fp,
                    num_gt,
                    &params.rec_thrs,
                );

                let recall_idx = shape.recall_idx(t_idx, k_idx, a_idx, m_idx);
                recall_writes.push((recall_idx, final_recall));

                for (r_idx, pr_val, rc_ptr) in curve {
                    let p_idx = shape.precision_idx(t_idx, r_idx, k_idx, a_idx, m_idx);
                    precision_writes.push((p_idx, pr_val));
                    scores_writes.push((p_idx, sorted_scores[rc_ptr]));
                }
            }

            AccResult {
                precision_writes,
                recall_writes,
                scores_writes,
            }
        })
        .collect();

    // Merge results into output arrays
    let total = t * r * k * a * m;
    let mut precision = vec![-1.0f64; total];
    let mut scores = vec![-1.0f64; total];
    let total_recall = t * k * a * m;
    let mut recall = vec![-1.0f64; total_recall];

    for result in results {
        for (idx, val) in result.precision_writes {
            precision[idx] = val;
        }
        for (idx, val) in result.recall_writes {
            recall[idx] = val;
        }
        for (idx, val) in result.scores_writes {
            scores[idx] = val;
        }
    }

    AccumulatedEval {
        precision,
        recall,
        scores,
        shape,
    }
}

impl COCOeval {
    /// Accumulate per-image results into precision/recall arrays.
    pub fn accumulate(&mut self) {
        self.eval = Some(accumulate_impl(&self.eval_imgs, &self.params, None));
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
pub struct AccumulatedEval {
    /// Interpolated precision at each (iou_thr, recall_thr, category, area_range, max_det).
    pub precision: Vec<f64>,
    /// Maximum recall at each (iou_thr, category, area_range, max_det).
    pub recall: Vec<f64>,
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
