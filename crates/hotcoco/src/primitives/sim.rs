//! Similarity kernels and the `SimKind` geometry axis.
//!
//! # Similarity contract
//!
//! Every kernel returns a dense `[D][G]` matrix (detections × ground-truths) of
//! similarities that are:
//! - in `[0, 1]`, higher = better,
//! - directly threshold-comparable (a match is `sim >= threshold`),
//! - crowd/ignore-aware: the per-GT `iscrowd` flag changes the formula for that
//!   column (e.g. IoU uses detection-area-only union against a crowd GT), exactly
//!   as pycocotools does.
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

use std::fmt;
use std::str::FromStr;

// The built-in matrix kernels already live on the geometry/mask modules and
// already satisfy the contract above. Re-export them here so `primitives::sim`
// is the one canonical similarity surface (no duplicated math).
pub use crate::geometry::obb_iou;
pub use crate::mask::bbox_iou;
pub use crate::mask::iou as mask_iou;

/// The geometry axis for similarity: which built-in kernel computes the matrix.
///
/// `Oks` is a declared kind whose matrix-shaped kernel is not yet extracted from
/// `eval/iou.rs` (that extraction is the next primitives slice — the current OKS
/// path is COCO-coupled via `compute_oks_static`). The other three kinds map to
/// the re-exported kernels in this module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SimKind {
    /// Axis-aligned bounding boxes — [`bbox_iou`].
    Bbox,
    /// Segmentation masks (RLE) — [`mask_iou`].
    Mask,
    /// Oriented bounding boxes — [`obb_iou`].
    Obb,
    /// Object keypoint similarity (pose). Kernel extraction pending.
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
