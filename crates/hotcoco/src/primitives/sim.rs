//! Similarity kernels and the `SimKind` geometry axis.
//!
//! # Similarity contract
//!
//! Every kernel returns a dense `[D][G]` matrix (detections × ground-truths) of
//! similarities that are:
//! - in `[0, 1]`, higher = better,
//! - directly threshold-comparable (a match is `sim >= threshold`),
//! - crowd/ignore-aware **for the IoU kernels**: the per-column `iscrowd` flag
//!   switches that column's formula (see [intersection-over-area](#crowd-columns-are-intersection-over-area)),
//!   exactly as pycocotools does. [`oks_matrix`] is the documented exception — it
//!   takes no flags, because pycocotools handles keypoint crowd/ignore in
//!   `evaluateImg` (via `gtIg`) rather than in `computeOks`. Adding a parameter
//!   later would be a breaking change, so the asymmetry is stated rather than
//!   papered over.
//!
//! # Which argument the flags index
//!
//! The flag slice indexes the **second** argument, whatever role it plays —
//! written `gt` here because detection passes ground truth there. IoU is symmetric
//! in the non-crowd branch, so a family may swap the arguments to get a
//! `[GT][tracker]`-oriented matrix (TrackEval's convention), but then the flags
//! describe *that* second argument. [`oks_matrix`] is asymmetric and cannot be
//! swapped at all.
//!
//! # Crowd columns are intersection-over-area
//!
//! For a flagged column the formula is `intersection / area(first argument)` —
//! i.e. IoA, not IoU. Detection calls this "crowd", but the quantity is general:
//! it is exactly what MOT preprocessing needs for distractor suppression
//! (TrackEval's `do_ioa=True`). A tracking driver wanting IoA against ignore
//! regions should call `bbox_iou(dets, ignore_regions, &vec![true; n])` rather
//! than write its own.
//!
//! Distance-based similarities (the 3D seam) enter later via a normalization
//! adapter (`1 - d / d_max`); they are not part of this slice.
//!
//! # BYO matrix
//!
//! A family that computes similarity by some other means supplies its own
//! `[D][G]` matrix directly (a `&[Vec<f64>]`) instead of a [`SimKind`]. Such a
//! matrix must already satisfy the contract above, including pre-encoding any
//! crowd/ignore formula changes — kernels bake that in, BYO matrices must too.
//!
//! # Kinds vs. IoU types
//!
//! [`SimKind`] is the *geometry* axis (how similarity is computed). It is
//! deliberately distinct from [`crate::params::IouType`] (the eval-config axis):
//! `IouType::Segm` is computed with `SimKind::Mask`, `IouType::Keypoints` with
//! `SimKind::Oks`. Parsers accept `"segm"` as an alias for `"mask"`.
//!
//! # Where the math lives
//!
//! This module **defines** every similarity formula. [`crate::mask`] and
//! [`crate::geometry`] own only the *mechanics* the formulas are built from (the
//! RLE codec and `intersection_area`; polygon clipping and
//! `obb_intersection_area`) and re-export these kernels back under their historic
//! paths. Those re-exports are one-way path sugar — never the definition — so
//! `hotcoco::mask::iou` keeps resolving for the pycocotools drop-in surface while
//! an auditor asking "where is IoU computed?" bounces exactly once, to here.
//!
//! Because those historic paths are Tier-1 drop-in surface mirroring
//! `pycocotools.mask`, **the signatures of [`mask_iou`], [`bbox_iou`], and
//! [`obb_iou`] are effectively frozen**: buffer-reusing or batched variants must
//! be added under new names, never by changing these. This paragraph is the
//! canonical statement of that rule; other modules link here rather than restate it.

use std::fmt;
use std::str::FromStr;

use rayon::prelude::*;

use crate::geometry::{obb_intersection_area, obb_to_corners};
use crate::mask::{area as rle_area, intersection_area};
use crate::types::Rle;

/// Minimum D×G product before a kernel switches from sequential to parallel
/// (rayon). Below this threshold, thread dispatch overhead exceeds the
/// parallelism benefit. One constant for every kernel — the per-row work is
/// independent, so this only trades dispatch overhead, never results.
const MIN_PARALLEL_WORK: usize = 1024;

/// The shared IoU closing formula, identical across bbox/mask/OBB.
///
/// When the column is flagged, this is **intersection-over-area**: divide by
/// `dt_area` alone (a detection inside a crowd region is not penalized for the
/// region's extent). Otherwise it is standard intersection-over-union. See the
/// [module note](self#crowd-columns-are-intersection-over-area) — the IoA branch
/// is reusable beyond detection's crowd semantics.
#[inline]
fn iou_from_areas(inter: f64, dt_area: f64, gt_area: f64, gt_is_crowd: bool) -> f64 {
    if gt_is_crowd {
        if dt_area == 0.0 { 0.0 } else { inter / dt_area }
    } else {
        let union = dt_area + gt_area - inter;
        if union == 0.0 { 0.0 } else { inter / union }
    }
}

/// Run a per-detection-row kernel, going parallel only past [`MIN_PARALLEL_WORK`].
///
/// `pub(crate)` so a future kernel outside this module (panoptic, a
/// distance-derived kind) can obey the one-threshold rule that
/// `tests/architecture.rs` enforces, instead of being forced to redeclare the
/// constant and trip the guard.
#[inline]
pub(crate) fn rows<F>(d: usize, g: usize, compute_row: F) -> Vec<Vec<f64>>
where
    F: Fn(usize) -> Vec<f64> + Sync + Send,
{
    if d * g >= MIN_PARALLEL_WORK {
        (0..d).into_par_iter().map(compute_row).collect()
    } else {
        (0..d).map(compute_row).collect()
    }
}

/// Compute IoU between `dt` and `gt` RLE masks.
///
/// Returns a D×G matrix (row-major, `dt.len()` rows, `gt.len()` columns).
/// For `iscrowd[j] == true`, uses crowd IoU: intersection / area(dt) instead of
/// intersection / union.
pub fn mask_iou(dt: &[Rle], gt: &[Rle], iscrowd: &[bool]) -> Vec<Vec<f64>> {
    let d = dt.len();
    let g = gt.len();
    if d == 0 || g == 0 {
        return vec![vec![]; d];
    }

    let dt_areas: Vec<u64> = dt.iter().map(rle_area).collect();
    let gt_areas: Vec<u64> = gt.iter().map(rle_area).collect();

    rows(d, g, |i| {
        let dt_a = dt_areas[i] as f64;
        (0..g)
            .map(|j| {
                let inter = intersection_area(&dt[i], &gt[j]) as f64;
                iou_from_areas(inter, dt_a, gt_areas[j] as f64, iscrowd[j])
            })
            .collect()
    })
}

/// IoU between two axis-aligned bounding boxes, each `[x, y, w, h]`.
///
/// The scalar counterpart of [`bbox_iou`] — the one place a single-pair bbox IoU
/// is defined, so analysis code never hand-rolls it.
///
/// `#[inline]` is load-bearing, not decoration: [`bbox_iou`]'s inner loop only
/// auto-vectorizes because this body is inlined into it. Without inlining the
/// matrix kernel degrades to one call per pair, each copying two `[f64; 4]` by
/// value.
#[inline]
pub fn bbox_iou_pair(a: [f64; 4], b: [f64; 4], b_is_crowd: bool) -> f64 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = (a[0] + a[2]).min(b[0] + b[2]);
    let y2 = (a[1] + a[3]).min(b[1] + b[3]);
    let iw = (x2 - x1).max(0.0);
    let ih = (y2 - y1).max(0.0);
    iou_from_areas(iw * ih, a[2] * a[3], b[2] * b[3], b_is_crowd)
}

/// Compute bbox IoU between sets of bounding boxes.
///
/// Each bbox is `[x, y, w, h]`. Returns D×G matrix.
pub fn bbox_iou(dt: &[[f64; 4]], gt: &[[f64; 4]], iscrowd: &[bool]) -> Vec<Vec<f64>> {
    let d = dt.len();
    let g = gt.len();
    if d == 0 || g == 0 {
        return vec![vec![]; d];
    }

    rows(d, g, |i| {
        (0..g)
            .map(|j| bbox_iou_pair(dt[i], gt[j], iscrowd[j]))
            .collect()
    })
}

/// IoU between two pre-computed rotated rectangle corner sets.
///
/// `area_a` and `area_b` are the rectangle areas (w × h).
/// Returns 0.0 for zero-area boxes or non-overlapping boxes.
pub(crate) fn obb_iou_pair(
    corners_a: &[(f64, f64); 4],
    area_a: f64,
    corners_b: &[(f64, f64); 4],
    area_b: f64,
    b_is_crowd: bool,
) -> f64 {
    if area_a <= 0.0 || area_b <= 0.0 {
        return 0.0;
    }

    let inter_area = obb_intersection_area(corners_a, corners_b);
    if inter_area <= 0.0 {
        return 0.0;
    }

    iou_from_areas(inter_area, area_a, area_b, b_is_crowd)
}

/// Compute D×G IoU matrix for oriented bounding boxes.
///
/// `dt` contains detection OBBs `[cx, cy, w, h, angle]`, `gt` contains ground truth OBBs,
/// and `iscrowd` indicates whether each GT is a crowd annotation (one per GT).
///
/// When `iscrowd[j]` is true, IoU = intersection / dt_area (matching bbox crowd semantics).
pub fn obb_iou(dt: &[[f64; 5]], gt: &[[f64; 5]], iscrowd: &[bool]) -> Vec<Vec<f64>> {
    let d = dt.len();
    let g = gt.len();
    if d == 0 || g == 0 {
        return vec![vec![]; d];
    }

    // Pre-compute GT corners and areas (loop-invariant over DT rows).
    let gt_corners: Vec<[(f64, f64); 4]> = gt.iter().map(obb_to_corners).collect();
    let gt_areas: Vec<f64> = gt.iter().map(|b| b[2] * b[3]).collect();

    rows(d, g, |i| {
        let corners_a = obb_to_corners(&dt[i]);
        let area_a = dt[i][2] * dt[i][3];
        (0..g)
            .map(|j| obb_iou_pair(&corners_a, area_a, &gt_corners[j], gt_areas[j], iscrowd[j]))
            .collect()
    })
}

/// The geometry axis for similarity: which built-in kernel computes the matrix.
///
/// Each kind maps to a matrix kernel in this module: [`bbox_iou`], [`mask_iou`],
/// [`obb_iou`], [`oks_matrix`].
///
/// Marked `#[non_exhaustive]`: later families are expected to add kinds (a
/// panoptic kind, a distance-derived kind for 3D), and downstream code must not
/// be broken by that. Match with a `_` arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SimKind {
    /// Axis-aligned bounding boxes — [`bbox_iou`].
    Bbox,
    /// Segmentation masks (RLE) — [`mask_iou`].
    Mask,
    /// Oriented bounding boxes — [`obb_iou`].
    Obb,
    /// Object keypoint similarity (pose) — [`oks_matrix`].
    Oks,
}

impl SimKind {
    /// The canonical lowercase name (round-trips through [`FromStr`]).
    pub fn as_str(self) -> &'static str {
        match self {
            SimKind::Bbox => "bbox",
            SimKind::Mask => "mask",
            SimKind::Obb => "obb",
            SimKind::Oks => "oks",
        }
    }
}

impl fmt::Display for SimKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for SimKind {
    type Err = String;

    /// Parse a `SimKind`. Accepts `"segm"` as an alias for `"mask"` and
    /// `"keypoints"` as an alias for `"oks"`, so eval-config strings map cleanly
    /// onto the geometry axis.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "bbox" => Ok(SimKind::Bbox),
            "mask" | "segm" => Ok(SimKind::Mask),
            "obb" => Ok(SimKind::Obb),
            "oks" | "keypoints" => Ok(SimKind::Oks),
            _ => Err(format!(
                "Unknown sim kind: '{s}'. Expected 'bbox', 'mask' (alias 'segm'), \
                 'obb', or 'oks' (alias 'keypoints')"
            )),
        }
    }
}

/// Object Keypoint Similarity (OKS) matrix, `[D][G]` (detections × ground-truths).
///
/// A ground-truth pose instance — the typed GT unit for [`oks_matrix`],
/// analogous to `[f64; 4]` for bboxes or `Rle` for masks. Keeps the keypoints,
/// area, and bbox of one instance together instead of in caller-aligned
/// parallel arrays.
#[derive(Debug, Clone, Copy)]
pub struct GtPose<'a> {
    /// Flat keypoints `[x, y, v, x, y, v, ...]` (length `3*K`); empty = skip.
    pub keypoints: &'a [f64],
    /// Object area — the OKS scale denominator.
    pub area: f64,
    /// Bounding box `[x, y, w, h]`, used only when no GT keypoints are visible.
    pub bbox: [f64; 4],
}

/// COCO-decoupled port of pycocotools' `computeOks`, matrix-shaped so pose
/// tracking (and any future keypoint family) can reuse it without a `COCO`
/// object. The geometric IoU kinds are pure functions already; OKS was the one
/// kernel still buried inside the eval monolith, so it is lifted here and the
/// monolith delegates to it.
///
/// Each detection is a flat keypoint slice (`[x, y, v, ...]`, length `3*K`) —
/// OKS is asymmetric, so detections need only keypoints while ground-truths
/// carry area and bbox too (a [`GtPose`]). `sigmas` has length `K`. Detection
/// visibility is ignored (only GT visibility gates the average). An empty
/// keypoint slice (GT or DT) leaves that column/row zero.
///
/// Definition (per pycocotools): `vars = (2σ)²`; per keypoint
/// `e = (dx² + dy²) / vars / (area + ε) / 2`, `OKS = mean(exp(-e))`. If the GT
/// has no visible keypoints (`k1 == 0`), distance is measured to the GT bbox
/// (doubled) instead, over all keypoints.
///
/// Similarity is in `[0, 1]`, higher = better — the [module contract](self).
pub fn oks_matrix(dt_keypoints: &[&[f64]], gt: &[GtPose<'_>], sigmas: &[f64]) -> Vec<Vec<f64>> {
    let num_kpts = sigmas.len();
    // vars = (sigmas * 2)**2 = 4 * sigma^2  (matching pycocotools)
    let vars: Vec<f64> = sigmas.iter().map(|s| (2.0 * s).powi(2)).collect();

    let d = dt_keypoints.len();
    let g = gt.len();
    let mut result = vec![vec![0.0f64; g]; d];

    for (j, gt_pose) in gt.iter().enumerate() {
        let gt_kpts = gt_pose.keypoints;
        if gt_kpts.is_empty() {
            continue;
        }
        let gt_area = gt_pose.area + f64::EPSILON;
        let bb = gt_pose.bbox;

        // Count visible GT keypoints.
        let k1: usize = (0..num_kpts)
            .filter(|&ki| gt_kpts.get(ki * 3 + 2).copied().unwrap_or(0.0) > 0.0)
            .count();

        // Ignore-region bounds (double the GT bbox), used only when k1 == 0.
        let x0 = bb[0] - bb[2];
        let x1 = bb[0] + bb[2] * 2.0;
        let y0 = bb[1] - bb[3];
        let y1 = bb[1] + bb[3] * 2.0;

        for (i, &dt_kpts) in dt_keypoints.iter().enumerate() {
            if dt_kpts.is_empty() {
                continue;
            }

            let mut oks_sum = 0.0_f64;
            let mut oks_count = 0_usize;

            for (ki, &var_k) in vars.iter().enumerate() {
                // When k1 > 0, only include visible GT keypoints.
                let visible = gt_kpts.get(ki * 3 + 2).copied().unwrap_or(0.0) > 0.0;
                if k1 > 0 && !visible {
                    continue;
                }

                let gx = gt_kpts.get(ki * 3).copied().unwrap_or(0.0);
                let gy = gt_kpts.get(ki * 3 + 1).copied().unwrap_or(0.0);
                let xd = dt_kpts.get(ki * 3).copied().unwrap_or(0.0);
                let yd = dt_kpts.get(ki * 3 + 1).copied().unwrap_or(0.0);

                let (dx, dy) = if k1 > 0 {
                    (xd - gx, yd - gy)
                } else {
                    // No visible GT keypoints: measure distance to bbox boundary.
                    let dx = 0.0_f64.max(x0 - xd) + 0.0_f64.max(xd - x1);
                    let dy = 0.0_f64.max(y0 - yd) + 0.0_f64.max(yd - y1);
                    (dx, dy)
                };

                let e = (dx * dx + dy * dy) / var_k / gt_area / 2.0;
                oks_sum += (-e).exp();
                oks_count += 1;
            }

            if oks_count > 0 {
                result[i][j] = oks_sum / oks_count as f64;
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_roundtrips_and_aliases() {
        assert_eq!("bbox".parse(), Ok(SimKind::Bbox));
        assert_eq!("mask".parse(), Ok(SimKind::Mask));
        assert_eq!("obb".parse(), Ok(SimKind::Obb));
        assert_eq!("oks".parse(), Ok(SimKind::Oks));
        // aliases from the eval-config axis
        assert_eq!("segm".parse(), Ok(SimKind::Mask));
        assert_eq!("keypoints".parse(), Ok(SimKind::Oks));
        // canonical names round-trip
        for k in [SimKind::Bbox, SimKind::Mask, SimKind::Obb, SimKind::Oks] {
            assert_eq!(k.as_str().parse(), Ok(k));
        }
    }

    #[test]
    fn parse_rejects_unknown() {
        assert!("polygon".parse::<SimKind>().is_err());
    }

    // --- OKS ------------------------------------------------------------

    // One keypoint, sigma s, area A. e = (dx^2+dy^2)/(4 s^2)/(A+eps)/2.
    fn oks_1kpt(dx: f64, dy: f64, s: f64, area: f64) -> f64 {
        let e = (dx * dx + dy * dy) / (4.0 * s * s) / (area + f64::EPSILON) / 2.0;
        (-e).exp()
    }

    // A GT pose with a default bbox (only relevant to the k1 == 0 branch).
    fn gt(keypoints: &[f64], area: f64) -> GtPose<'_> {
        GtPose {
            keypoints,
            area,
            bbox: [0.0, 0.0, 30.0, 30.0],
        }
    }

    #[test]
    fn oks_identical_keypoints_is_one() {
        let sigmas = [0.05, 0.07];
        let kpts = [10.0, 10.0, 2.0, 20.0, 20.0, 2.0]; // both visible
        let dt = kpts; // identical
        let m = oks_matrix(&[&dt], &[gt(&kpts, 1000.0)], &sigmas);
        assert!((m[0][0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn oks_single_displaced_keypoint_matches_formula() {
        let sigmas = [0.05];
        let area = 1000.0;
        let kpts = [10.0, 10.0, 2.0]; // visible
        let dt = [13.0, 14.0, 2.0]; // displaced by (3, 4)
        let m = oks_matrix(&[&dt], &[gt(&kpts, area)], &sigmas);
        assert!((m[0][0] - oks_1kpt(3.0, 4.0, 0.05, area)).abs() < 1e-12);
    }

    #[test]
    fn oks_averages_only_visible_gt_keypoints() {
        // kpt 0 visible & identical (contributes 1.0); kpt 1 not visible (v=0)
        // and far away — must be excluded, so OKS == 1.0.
        let sigmas = [0.05, 0.05];
        let kpts = [10.0, 10.0, 2.0, 0.0, 0.0, 0.0];
        let dt = [10.0, 10.0, 2.0, 999.0, 999.0, 2.0];
        let m = oks_matrix(&[&dt], &[gt(&kpts, 1000.0)], &sigmas);
        assert!((m[0][0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn oks_no_visible_gt_uses_bbox_distance_branch() {
        // k1 == 0 (all GT keypoints invisible). A DT keypoint inside the doubled
        // GT bbox has zero boundary distance => e = 0 => OKS = 1.0.
        let sigmas = [0.05];
        let kpts = [0.0, 0.0, 0.0]; // invisible
        let pose = GtPose {
            keypoints: &kpts,
            area: 1000.0,
            bbox: [0.0, 0.0, 20.0, 20.0],
        };
        let inside = [10.0, 10.0, 2.0]; // within [x0,x1]x[y0,y1] = [-20,40]
        let m = oks_matrix(&[&inside], &[pose], &sigmas);
        assert!((m[0][0] - 1.0).abs() < 1e-12);
        // A DT keypoint far outside the bbox scores strictly less than 1.
        let outside = [1000.0, 1000.0, 2.0];
        let m2 = oks_matrix(&[&outside], &[pose], &sigmas);
        assert!(m2[0][0] < 1.0);
    }

    #[test]
    fn oks_empty_keypoints_leave_zero() {
        let sigmas = [0.05];
        let kpts = [10.0, 10.0, 2.0];
        let empty: &[f64] = &[];
        // empty DT row, present GT
        let m = oks_matrix(&[empty], &[gt(&kpts, 1000.0)], &sigmas);
        assert_eq!(m[0][0], 0.0);
        // present DT, empty GT column
        let m2 = oks_matrix(&[&kpts[..]], &[gt(empty, 1000.0)], &sigmas);
        assert_eq!(m2[0][0], 0.0);
    }

    #[test]
    fn bbox_kernel_matches_and_obeys_contract() {
        // Re-exported kernel is the same math as crate::mask::bbox_iou, and the
        // similarity contract holds: [0,1], self-overlap == 1.
        let dt = [[0.0, 0.0, 10.0, 10.0], [100.0, 100.0, 10.0, 10.0]];
        let gt = [[0.0, 0.0, 10.0, 10.0]];
        let iscrowd = [false];
        let m = bbox_iou(&dt, &gt, &iscrowd);
        assert_eq!(m, crate::mask::bbox_iou(&dt, &gt, &iscrowd));
        assert!((m[0][0] - 1.0).abs() < 1e-12, "identical box => IoU 1");
        assert_eq!(m[1][0], 0.0, "disjoint box => IoU 0");
        for row in &m {
            for &v in row {
                assert!((0.0..=1.0).contains(&v), "similarity in [0,1]");
            }
        }
    }
}
