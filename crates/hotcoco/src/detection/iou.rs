use std::collections::HashMap;

use rayon::prelude::*;

use crate::coco::COCO;
use crate::params::Params;
use crate::primitives::sim::{self, SimKind};
use crate::types::Rle;

use super::{COCOeval, EvalMode};

/// Every in-scope annotation's mask, converted to RLE once per `evaluate()`.
///
/// pycocotools converts in `_prepare`, before the IoU loop; converting inside
/// the per-cell IoU computation instead put `fr_polys` — the 5×-upsampled
/// polygon rasterizer — at ~48% of all samples in a val2017 segm profile, and
/// re-paid it in every cross-category matrix `confusion_matrix` and `tide`
/// build. Rebuilt on each `evaluate()` call, exactly like the `ious` cache, so
/// it can never go stale relative to the datasets it was drawn from.
///
/// Annotations without a convertible mask (no segmentation *and* no bbox) are
/// simply absent; readers fall back to [`COCO::ann_to_rle`].
pub(super) struct SegmRles {
    // Private on purpose, mirroring the `ious` cache discipline ("the
    // visibility is the enforcement"): consumers get one RLE at a time via the
    // `*_rle_or_convert` helpers and can never iterate or retain the map.
    gt: HashMap<u64, Rle>,
    dt: HashMap<u64, Rle>,
}

impl SegmRles {
    /// The cache-miss policy, in one place: a hit clones the prepared RLE, a
    /// miss — or no cache at all — converts on the spot. Correctness never
    /// depends on what the cache happens to hold; a miss only costs speed.
    pub(super) fn gt_rle_or_convert(cache: Option<&Self>, coco_gt: &COCO, id: u64) -> Option<Rle> {
        if let Some(rle) = cache.and_then(|c| c.gt.get(&id)) {
            return Some(rle.clone());
        }
        coco_gt.ann_to_rle(coco_gt.get_ann(id)?)
    }

    /// Detection twin of [`gt_rle_or_convert`](Self::gt_rle_or_convert).
    pub(super) fn dt_rle_or_convert(cache: Option<&Self>, coco_dt: &COCO, id: u64) -> Option<Rle> {
        if let Some(rle) = cache.and_then(|c| c.dt.get(&id)) {
            return Some(rle.clone());
        }
        coco_dt.ann_to_rle(coco_dt.get_ann(id)?)
    }

    /// Convert every in-scope annotation, in parallel.
    ///
    /// Scope is delegated to [`COCO::get_ann_ids`] — the owner of "which
    /// annotations do these params cover" — rather than a third spelling of
    /// the img/cat filter, so a run filtered to a handful of images does not
    /// rasterize the whole dataset and the filter cannot drift from the one
    /// the evaluation itself uses.
    pub(super) fn prepare(coco_gt: &COCO, coco_dt: &COCO, params: &Params) -> Self {
        let cat_ids: &[u64] = if params.use_cats {
            &params.cat_ids
        } else {
            &[]
        };

        let convert = |coco: &COCO| -> HashMap<u64, Rle> {
            coco.get_ann_ids(&params.img_ids, cat_ids, None, None)
                .into_par_iter()
                .filter_map(|id| Some((id, coco.ann_to_rle(coco.get_ann(id)?)?)))
                .collect()
        };

        SegmRles {
            gt: convert(coco_gt),
            dt: convert(coco_dt),
        }
    }
}

/// Whether this ground truth's similarity column uses intersection-over-area.
///
/// IoA — intersection divided by the *detection's* area — is how both COCO and
/// Open Images express "this ground truth is a region, not an instance". They just
/// flag it differently: COCO with `iscrowd`, Open Images with `is_group_of`. The
/// protocol wording is "a detection is inside a group-of box if the area of
/// intersection of the detection and the box divided by the area of the detection
/// is greater than 0.5".
///
/// One function rather than the same branch in each of the three similarity
/// kernels: hardcoding `false` for Open Images made the group-of pass unreachable
/// for any detection smaller than the group box — the normal case — and it was
/// wrong in all three places at once, because there were three places.
fn uses_ioa(ann: &crate::types::Annotation, eval_mode: EvalMode) -> bool {
    if eval_mode == EvalMode::OpenImages {
        ann.is_group_of.unwrap_or(false)
    } else {
        ann.iscrowd
    }
}

impl COCOeval {
    /// Compute the IoU/OKS matrix for a given image and category.
    pub(super) fn compute_iou_static(
        coco_gt: &COCO,
        coco_dt: &COCO,
        params: &Params,
        img_id: u64,
        cat_id: u64,
        eval_mode: EvalMode,
        segm_rles: Option<&SegmRles>,
    ) -> Vec<Vec<f64>> {
        let gt_anns = Self::get_anns_static(coco_gt, params, img_id, cat_id);
        let dt_anns = Self::get_anns_static(coco_dt, params, img_id, cat_id);

        if gt_anns.is_empty() || dt_anns.is_empty() {
            return Vec::new();
        }

        // Dispatch on the geometry axis, not the eval-config one: `SimKind` is
        // what selects a kernel, and every family (detection here, tracking and
        // panoptic later) branches on the same four values. The helpers below do
        // marshaling only — reshaping annotations into the kernel's input types.
        match SimKind::from(params.iou_type) {
            SimKind::Mask => Self::compute_segm_iou_static(
                coco_gt, coco_dt, dt_anns, gt_anns, eval_mode, segm_rles,
            ),
            SimKind::Bbox => {
                Self::compute_bbox_iou_static(coco_gt, coco_dt, dt_anns, gt_anns, eval_mode)
            }
            SimKind::Oks => Self::compute_oks_static(coco_gt, coco_dt, params, dt_anns, gt_anns),
            SimKind::Obb => {
                Self::compute_obb_iou_static(coco_gt, coco_dt, dt_anns, gt_anns, eval_mode)
            }
        }
    }

    /// Get annotation IDs for an image, optionally filtered by category.
    pub(super) fn get_anns_static<'a>(
        coco: &'a COCO,
        params: &Params,
        img_id: u64,
        cat_id: u64,
    ) -> &'a [u64] {
        if params.use_cats {
            coco.get_ann_ids_for_img_cat(img_id, cat_id)
        } else {
            coco.get_ann_ids_for_img(img_id)
        }
    }

    /// Compute segmentation mask IoU by converting annotations to RLE and calling `sim::mask_iou`.
    ///
    /// RLEs come through [`SegmRles::dt_rle_or_convert`]/[`SegmRles::gt_rle_or_convert`],
    /// which own the cache-or-convert policy.
    pub(super) fn compute_segm_iou_static(
        coco_gt: &COCO,
        coco_dt: &COCO,
        dt_ids: &[u64],
        gt_ids: &[u64],
        eval_mode: EvalMode,
        segm_rles: Option<&SegmRles>,
    ) -> Vec<Vec<f64>> {
        let dt_rles: Vec<Rle> = dt_ids
            .iter()
            .filter_map(|&id| SegmRles::dt_rle_or_convert(segm_rles, coco_dt, id))
            .collect();
        let (gt_rles, iscrowd): (Vec<Rle>, Vec<bool>) = gt_ids
            .iter()
            .filter_map(|&id| {
                let ann = coco_gt.get_ann(id)?;
                let crowd = uses_ioa(ann, eval_mode);
                Some((SegmRles::gt_rle_or_convert(segm_rles, coco_gt, id)?, crowd))
            })
            .unzip();

        sim::mask_iou(&dt_rles, &gt_rles, &iscrowd)
    }

    /// Compute bounding box IoU by extracting bbox arrays and calling `sim::bbox_iou`.
    pub(super) fn compute_bbox_iou_static(
        coco_gt: &COCO,
        coco_dt: &COCO,
        dt_ids: &[u64],
        gt_ids: &[u64],
        eval_mode: EvalMode,
    ) -> Vec<Vec<f64>> {
        let dt_bbs: Vec<[f64; 4]> = dt_ids
            .iter()
            .filter_map(|&id| coco_dt.get_ann(id)?.bbox)
            .collect();
        let (gt_bbs, iscrowd): (Vec<[f64; 4]>, Vec<bool>) = gt_ids
            .iter()
            .filter_map(|&id| {
                let ann = coco_gt.get_ann(id)?;
                let crowd = uses_ioa(ann, eval_mode);
                Some((ann.bbox?, crowd))
            })
            .unzip();

        sim::bbox_iou(&dt_bbs, &gt_bbs, &iscrowd)
    }

    /// Compute OKS (Object Keypoint Similarity) between detection and GT keypoints.
    ///
    /// OKS = mean_k[ exp( -d_k^2 / (2 * s_k^2 * area) ) ] where d_k is the Euclidean
    /// distance for keypoint k, s_k is the per-keypoint sigma, and area is the GT area.
    /// Only visible GT keypoints contribute. When no GT keypoints are visible, distance
    /// is measured to the GT bounding box boundary instead.
    pub(super) fn compute_oks_static(
        coco_gt: &COCO,
        coco_dt: &COCO,
        params: &Params,
        dt_ids: &[u64],
        gt_ids: &[u64],
    ) -> Vec<Vec<f64>> {
        // The OKS math lives in the shared `primitives::sim::oks_matrix` kernel
        // (COCO-decoupled, matrix-shaped). Here we only marshal the annotations
        // into the kernel's flat-slice form. A missing `keypoints` field maps to
        // an empty slice, which the kernel skips — matching the previous
        // `None => continue` behavior that left that row/column zero.
        let gt_anns: Vec<_> = gt_ids
            .iter()
            .filter_map(|&id| coco_gt.get_ann(id))
            .collect();
        let dt_anns: Vec<_> = dt_ids
            .iter()
            .filter_map(|&id| coco_dt.get_ann(id))
            .collect();

        let gt: Vec<crate::primitives::sim::GtPose<'_>> = gt_anns
            .iter()
            .map(|a| crate::primitives::sim::GtPose {
                keypoints: a.keypoints.as_deref().unwrap_or(&[]),
                area: a.area.unwrap_or(0.0),
                bbox: a.bbox.unwrap_or([0.0; 4]),
            })
            .collect();
        let dt_keypoints: Vec<&[f64]> = dt_anns
            .iter()
            .map(|a| a.keypoints.as_deref().unwrap_or(&[]))
            .collect();

        crate::primitives::sim::oks_matrix(&dt_keypoints, &gt, &params.kpt_oks_sigmas)
    }

    /// Compute oriented bounding box IoU by extracting OBB arrays and calling `sim::obb_iou`.
    pub(super) fn compute_obb_iou_static(
        coco_gt: &COCO,
        coco_dt: &COCO,
        dt_ids: &[u64],
        gt_ids: &[u64],
        eval_mode: EvalMode,
    ) -> Vec<Vec<f64>> {
        let dt_obbs: Vec<[f64; 5]> = dt_ids
            .iter()
            .filter_map(|&id| coco_dt.get_ann(id)?.obb)
            .collect();
        let (gt_obbs, iscrowd): (Vec<[f64; 5]>, Vec<bool>) = gt_ids
            .iter()
            .filter_map(|&id| {
                let ann = coco_gt.get_ann(id)?;
                let crowd = uses_ioa(ann, eval_mode);
                Some((ann.obb?, crowd))
            })
            .unzip();

        sim::obb_iou(&dt_obbs, &gt_obbs, &iscrowd)
    }
}
