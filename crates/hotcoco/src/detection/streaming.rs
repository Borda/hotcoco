//! Incremental (streaming) evaluation — candidate H.
//!
//! [`evaluate()`](super::COCOeval::evaluate) is a whole-dataset batch: it needs
//! every ground truth and detection loaded and indexed before the first
//! (image, category) cell can be matched, so `loadRes` + the `COCOeval`
//! constructor + `evaluate()` all sit on the critical path at the end of an
//! epoch. But matching is genuinely per-image — [`matching::gather_pair`] and
//! [`matching::evaluate_cell`] never read another image's annotations — so
//! nothing about the algorithm requires waiting for the last image before
//! matching the first one.
//!
//! [`StreamingEval`] exploits that: [`StreamingEval::add_image`] runs one
//! image's matching immediately, as its ground truth and detections become
//! available (during postprocessing, overlapped with other work), and
//! [`StreamingEval::finalize`] assembles the accumulated cells into an
//! ordinary [`COCOeval`] ready for `accumulate()` → `summarize()` → `report()`.
//! What moves off the critical path is `loadRes` and the per-image matching in
//! `evaluate()`; `accumulate()` itself is unchanged (and already the target of
//! candidates A1/A2).
//!
//! # Scope of this slice (v1)
//!
//! This ships the matching half of candidate H's design, not the compact
//! `~20 B/det` record format the plan sketched. That format assumes a
//! detection's matched status is shared across area ranges
//! (`matched_mask[T]`, `ignore_mask[T·A]`); it is not; [`matching::partition_gt`]
//! reorders ground truths *by area range*, and the greedy matcher is
//! order-sensitive, so which ground truth a detection matches genuinely varies
//! per range (a detection overlapping a small box at 0.6 and a large box at
//! 0.9 can match different boxes in `"small"` vs `"all"`). Building compact
//! records against that unproven equivalence would risk a silent parity
//! regression in a crate whose whole purpose is parity. So v1 keeps the full
//! per-cell [`EvalImg`] this module was already computing and defers
//! compaction to a follow-up PR that can use this one as its bit-identity
//! oracle.
//!
//! # Restrictions
//!
//! - **No Open Images.** [`StreamingEval::new`] returns an error for
//!   [`EvalMode::OpenImages`]: hierarchy expansion
//!   ([`super::expand::expand_annotations`]) needs the whole GT dataset up
//!   front, which is incompatible with per-image incremental evaluation.
//! - **The category axis is frozen at construction**, not resolved at
//!   `finalize()` as the plan's initial sketch suggested. Batch `evaluate()`
//!   reads `coco_gt.dataset.categories` before the first cell is matched (so a
//!   category with zero annotations still gets a `-1.0` slot instead of
//!   vanishing from the K axis) — reproducing that from an unbounded,
//!   growing category set discovered only at `finalize()` would need a second
//!   accumulation pass. Pass the full category list to `new()` instead.
//! - **`finalize()`'s `coco_gt`/`coco_dt` carry categories only, no
//!   annotations.** `accumulate()`/`summarize()`/`report()` — the per-epoch
//!   path this candidate targets — never read annotations from those fields
//!   (only `eval_imgs`, `params`, and category names for display), so leaving
//!   them empty is what keeps `finalize()` off the critical path: it does not
//!   pay to rebuild the annotation index candidate D's interim already made
//!   cheaper. But `confusion_matrix()`, `tide()`, `compare()`, and `slice_by()`
//!   *do* read real annotations from `coco_gt`/`coco_dt`, and will silently see
//!   an empty dataset on a streaming-finalized evaluator. Build those from a
//!   batch [`COCOeval::new`] instead.

use std::collections::{HashMap, HashSet};

use crate::coco::COCO;
use crate::params::Params;
use crate::primitives::sim::SimKind;
use crate::types::{Annotation, Category, Dataset, Image};

use super::iou::SegmRles;
use super::matching::{self, EvalImg};
use super::mode::FreqGroups;
use super::{COCOeval, EvalMode};

/// Incremental evaluator: feed images one at a time, get an ordinary
/// [`COCOeval`] back. See the [module docs](self) for scope and restrictions.
pub struct StreamingEval {
    params: Params,
    eval_mode: EvalMode,
    categories: Vec<Category>,
    /// `params.iou_thrs` with pycocotools' match floor applied, resolved once
    /// — see [`matching::EvalImgContext::match_floors`].
    match_floors: Vec<f64>,
    n_area_ranges: usize,
    /// One entry per (image, category) pair seen so far. Each value always has
    /// exactly `n_area_ranges` slots, in `params.area_ranges` order — the same
    /// per-pair chunk shape [`COCOeval::evaluate`](super::evaluate) writes.
    pairs: HashMap<(u64, u64), Vec<Option<EvalImg>>>,
}

impl StreamingEval {
    /// Start a new incremental evaluation.
    ///
    /// `params` is frozen from here on: `iou_thrs`, `area_ranges`, and
    /// `max_dets` must not change between `new()` and `finalize()`, and
    /// `use_cats`/`cat_ids` decide the K axis for good — see the
    /// [module docs](self). If `params.cat_ids` is empty and `use_cats` is
    /// true, it is filled from `categories` (sorted, deduplicated), matching
    /// what [`COCOeval::resolved_ids`](super::COCOeval::resolved_ids) would
    /// derive from a batch GT dataset carrying the same categories.
    ///
    /// `categories` should list every category the run will ever see —
    /// including ones with no annotations in any image, so they still get a
    /// K-axis slot (reporting `-1.0`) instead of silently vanishing.
    ///
    /// # Errors
    ///
    /// Returns an error for [`EvalMode::OpenImages`] — see the
    /// [module docs](self).
    pub fn new(
        mut params: Params,
        eval_mode: EvalMode,
        categories: Vec<Category>,
    ) -> crate::error::Result<Self> {
        if eval_mode == EvalMode::OpenImages {
            return Err(crate::error::Error::Other(
                "StreamingEval does not support Open Images: hierarchy expansion needs the \
                 whole GT dataset before the first image is evaluated, which is incompatible \
                 with per-image incremental evaluation. Use COCOeval::new_oid instead."
                    .to_string(),
            ));
        }

        if params.use_cats && params.cat_ids.is_empty() {
            let mut ids: Vec<u64> = categories.iter().map(|c| c.id).collect();
            ids.sort_unstable();
            ids.dedup();
            params.cat_ids = ids;
        }

        let match_floors = params
            .iou_thrs
            .iter()
            .map(|&t| crate::primitives::greedy::coco_match_floor(t))
            .collect();
        let n_area_ranges = params.area_ranges.len();

        Ok(StreamingEval {
            params,
            eval_mode,
            categories,
            match_floors,
            n_area_ranges,
            pairs: HashMap::new(),
        })
    }

    /// Match one image's ground truth against its detections, immediately.
    ///
    /// `image` supplies the id every annotation must reference via
    /// `image_id`, plus (in LVIS mode) `neg_category_ids` and
    /// `not_exhaustive_category_ids` — the per-image federated-annotation
    /// metadata `evaluate()` reads from `coco_gt.dataset.images` in the batch
    /// path. `gt_anns`/`dt_anns` need not carry ids assigned by a shared
    /// counter across images; only uniqueness *within* this image's lists
    /// matters, the same as `img_cat_to_anns` scoping in the batch index.
    ///
    /// A no-op if `params.img_ids` is non-empty and does not contain
    /// `image.id`, or if this image contributes no in-scope (image, category)
    /// pair (no annotations, or every category outside `params.cat_ids`).
    pub fn add_image(&mut self, image: &Image, gt_anns: &[Annotation], dt_anns: &[Annotation]) {
        if !self.params.img_ids.is_empty() && !self.params.img_ids.contains(&image.id) {
            return;
        }

        let is_lvis = self.eval_mode == EvalMode::Lvis;
        let neg_cats: HashSet<u64> = if is_lvis {
            image.neg_category_ids.iter().copied().collect()
        } else {
            HashSet::new()
        };
        let not_exhaustive: HashSet<u64> = if is_lvis {
            image.not_exhaustive_category_ids.iter().copied().collect()
        } else {
            HashSet::new()
        };

        let cats = self.sparse_cats_for_image(gt_anns, dt_anns, &neg_cats);
        if cats.is_empty() {
            return;
        }

        // Tiny, single-image COCOs — indexing cost scales with this image's
        // annotation count, not the dataset's, which is what keeps the work
        // here instead of one big rebuild at `finalize()`.
        let tiny_gt = COCO::from_dataset(Dataset {
            images: vec![image.clone()],
            annotations: gt_anns.to_vec(),
            categories: self.categories.clone(),
            ..Default::default()
        });
        let tiny_dt = COCO::from_dataset(Dataset {
            images: vec![image.clone()],
            annotations: dt_anns.to_vec(),
            categories: self.categories.clone(),
            ..Default::default()
        });

        let segm_rles = (SimKind::from(self.params.iou_type) == SimKind::Mask)
            .then(|| SegmRles::prepare(&tiny_gt, &tiny_dt, &self.params));

        let max_det = self.params.max_det();

        let mut iou_cache: HashMap<(u64, u64), matching::IouMatrix> = HashMap::new();
        for &cat_id in &cats {
            let m = COCOeval::compute_iou_static(
                &tiny_gt,
                &tiny_dt,
                &self.params,
                image.id,
                cat_id,
                self.eval_mode,
                segm_rles.as_ref(),
            );
            if !m.is_empty() {
                iou_cache.insert((image.id, cat_id), m);
            }
        }

        let ctx = matching::build_context(
            &tiny_gt,
            &tiny_dt,
            &self.params,
            &iou_cache,
            self.eval_mode,
            &self.match_floors,
        );

        for &cat_id in &cats {
            let Some(pair) = matching::gather_pair(&ctx, image.id, cat_id, max_det) else {
                continue;
            };
            let not_exhaustive_cat = is_lvis && not_exhaustive.contains(&cat_id);
            let slots: Vec<Option<EvalImg>> = self
                .params
                .area_ranges
                .iter()
                .map(|ar| matching::evaluate_cell(&ctx, &pair, ar.range, not_exhaustive_cat))
                .collect();
            self.pairs.insert((image.id, cat_id), slots);
        }
    }

    /// The (image, category) pairs this image contributes, sorted — the
    /// per-image restriction of [`COCOeval::collect_sparse_pairs`]'s sparse
    /// set: a category counts if it has a GT annotation here, or (LVIS) a DT
    /// annotation and either a GT annotation or a confirmed-negative label.
    fn sparse_cats_for_image(
        &self,
        gt_anns: &[Annotation],
        dt_anns: &[Annotation],
        neg_cats: &HashSet<u64>,
    ) -> Vec<u64> {
        let is_lvis = self.eval_mode == EvalMode::Lvis;
        let mut cats: Vec<u64> = if self.params.use_cats {
            let allowed: HashSet<u64> = self.params.cat_ids.iter().copied().collect();
            let gt_cats: HashSet<u64> = gt_anns
                .iter()
                .map(|a| a.category_id)
                .filter(|c| allowed.contains(c))
                .collect();
            let mut set = gt_cats.clone();
            for a in dt_anns {
                let c = a.category_id;
                if !allowed.contains(&c) {
                    continue;
                }
                if is_lvis {
                    if gt_cats.contains(&c) || neg_cats.contains(&c) {
                        set.insert(c);
                    }
                } else {
                    set.insert(c);
                }
            }
            set.into_iter().collect()
        } else if gt_anns.is_empty() && dt_anns.is_empty() {
            Vec::new()
        } else {
            vec![u64::MAX]
        };
        cats.sort_unstable();
        cats
    }

    /// Assemble every image seen so far into an ordinary [`COCOeval`], ready
    /// for `accumulate()` → `summarize()` → `report()`.
    ///
    /// `eval_imgs` is built in `(image_id, category_id)` order, each pair
    /// contributing exactly `area_ranges.len()` consecutive slots — bit-for-bit
    /// the layout [`COCOeval::evaluate`](super::COCOeval::evaluate) leaves,
    /// since its sparse-pairs collection sorts the same tuples the same way.
    /// `coco_gt`/`coco_dt` carry categories only — see the [module docs](self)
    /// restrictions before calling `confusion_matrix()`, `tide()`, `compare()`,
    /// or `slice_by()` on the result.
    pub fn finalize(self) -> COCOeval {
        let StreamingEval {
            params,
            eval_mode,
            categories,
            n_area_ranges,
            mut pairs,
            ..
        } = self;

        let mut keys: Vec<(u64, u64)> = pairs.keys().copied().collect();
        keys.sort_unstable();

        let mut eval_imgs = Vec::with_capacity(keys.len() * n_area_ranges);
        for key in &keys {
            let slots = pairs.remove(key).expect("key came from this map");
            debug_assert_eq!(slots.len(), n_area_ranges);
            eval_imgs.extend(slots);
        }

        let mut freq_groups = FreqGroups::default();
        if eval_mode == EvalMode::Lvis {
            let cat_id_to_k_idx: HashMap<u64, usize> = params
                .cat_ids
                .iter()
                .enumerate()
                .map(|(i, &id)| (id, i))
                .collect();
            for cat in &categories {
                if let Some(&k_idx) = cat_id_to_k_idx.get(&cat.id) {
                    match cat.frequency.as_deref() {
                        Some("r") => freq_groups.rare.push(k_idx),
                        Some("c") => freq_groups.common.push(k_idx),
                        Some("f") => freq_groups.frequent.push(k_idx),
                        _ => {}
                    }
                }
            }
        }

        let coco_gt = COCO::from_dataset(Dataset {
            categories,
            ..Default::default()
        });
        let coco_dt = COCO::from_dataset(Dataset::default());

        let mut ev = COCOeval::with_mode(coco_gt, coco_dt, params, eval_mode, None);
        ev.eval_imgs = eval_imgs;
        ev.freq_groups = freq_groups;
        ev
    }
}
