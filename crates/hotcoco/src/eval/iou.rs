use crate::coco::COCO;
use crate::params::Params;
use crate::primitives::sim::{self, SimKind};
use crate::types::Rle;

use super::{COCOeval, EvalMode};

impl COCOeval {
    /// Compute the IoU/OKS matrix for a given image and category.
    pub(super) fn compute_iou_static(
        coco_gt: &COCO,
        coco_dt: &COCO,
        params: &Params,
        img_id: u64,
        cat_id: u64,
        eval_mode: EvalMode,
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
            SimKind::Mask => {
                Self::compute_segm_iou_static(coco_gt, coco_dt, dt_anns, gt_anns, eval_mode)
            }
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
    pub(super) fn compute_segm_iou_static(
        coco_gt: &COCO,
        coco_dt: &COCO,
        dt_ids: &[u64],
        gt_ids: &[u64],
        eval_mode: EvalMode,
    ) -> Vec<Vec<f64>> {
        let dt_rles: Vec<Rle> = dt_ids
            .iter()
            .filter_map(|&id| {
                let ann = coco_dt.get_ann(id)?;
                coco_dt.ann_to_rle(ann)
            })
            .collect();
        let (gt_rles, iscrowd): (Vec<Rle>, Vec<bool>) = gt_ids
            .iter()
            .filter_map(|&id| {
                let ann = coco_gt.get_ann(id)?;
                // OID: iscrowd is irrelevant — always use standard IoU
                let crowd = if eval_mode == EvalMode::OpenImages {
                    false
                } else {
                    ann.iscrowd
                };
                Some((coco_gt.ann_to_rle(ann)?, crowd))
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
                // OID: iscrowd is irrelevant — always use standard IoU
                let crowd = if eval_mode == EvalMode::OpenImages {
                    false
                } else {
                    ann.iscrowd
                };
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
                let crowd = if eval_mode == EvalMode::OpenImages {
                    false
                } else {
                    ann.iscrowd
                };
                Some((ann.obb?, crowd))
            })
            .unzip();

        sim::obb_iou(&dt_obbs, &gt_obbs, &iscrowd)
    }
}
