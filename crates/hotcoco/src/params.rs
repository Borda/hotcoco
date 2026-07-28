use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// A single area-range filter: a human-readable label paired with its `[min, max]` bounds.
///
/// Used in [`Params::area_ranges`] to keep labels and ranges in sync.
/// Standard COCO labels are `"all"`, `"small"`, `"medium"`, `"large"`.
#[derive(Debug, Clone)]
pub struct AreaRange {
    pub label: String,
    pub range: [f64; 2],
}

/// The type of IoU (intersection over union) computation to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum IouType {
    /// Bounding box IoU.
    Bbox,
    /// Segmentation mask IoU (RLE-based).
    Segm,
    /// Keypoint OKS (object keypoint similarity).
    Keypoints,
    /// Oriented bounding box IoU (rotated rectangle intersection).
    Obb,
}

impl fmt::Display for IouType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IouType::Bbox => write!(f, "bbox"),
            IouType::Segm => write!(f, "segm"),
            IouType::Keypoints => write!(f, "keypoints"),
            IouType::Obb => write!(f, "obb"),
        }
    }
}

impl From<IouType> for crate::primitives::sim::SimKind {
    /// Project this eval-config axis onto the geometry axis.
    ///
    /// The two are deliberately separate types: an `IouType` says what the user
    /// asked to evaluate (Tier-1 config surface, serialized, fixed at four
    /// variants), while a [`SimKind`](crate::primitives::sim::SimKind) says which
    /// kernel computes it (`#[non_exhaustive]`, expected to grow). The mapping is
    /// total *today*, which is why this is `From` and not `TryFrom`, and why it
    /// only goes this direction.
    ///
    /// It lives here rather than beside `SimKind` so that `primitives` — the
    /// bottom layer every family builds on — does not depend on the eval-config
    /// module above it.
    fn from(iou_type: IouType) -> Self {
        use crate::primitives::sim::SimKind;
        match iou_type {
            IouType::Bbox => SimKind::Bbox,
            IouType::Segm => SimKind::Mask,
            IouType::Keypoints => SimKind::Oks,
            IouType::Obb => SimKind::Obb,
        }
    }
}

impl FromStr for IouType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "bbox" => Ok(IouType::Bbox),
            "segm" => Ok(IouType::Segm),
            "keypoints" => Ok(IouType::Keypoints),
            "obb" => Ok(IouType::Obb),
            _ => Err(format!(
                "Unknown iou_type: '{}'. Expected 'bbox', 'segm', 'keypoints', or 'obb'",
                s
            )),
        }
    }
}

/// Small-object area upper bound: 32² = 1024 px².
pub(crate) const AREA_SMALL: f64 = 32.0 * 32.0;

/// Medium/large-object area boundary: 96² = 9216 px².
pub(crate) const AREA_LARGE: f64 = 96.0 * 96.0;

/// Default OKS sigmas for the 17 COCO keypoints (nose, eyes, ears, shoulders, …, ankles).
pub(crate) const KPT_OKS_SIGMAS: [f64; 17] = [
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107,
    0.087, 0.087, 0.089, 0.089,
];

/// `numpy.linspace(start, stop, num, endpoint=True)`, bit-for-bit.
///
/// pycocotools builds both threshold grids with `np.linspace`, and the obvious
/// Rust spellings do not reproduce it. `0.5 + 0.05 * i` disagrees at 2 of the 10
/// IoU thresholds, and `i / 100.0` disagrees at 10 of the 101 recall thresholds —
/// each by one ulp, because numpy computes a single `step` once and multiplies,
/// where those forms round twice or divide exactly.
///
/// One ulp sounds harmless and is not, on the recall grid. `rc[d] = tp / num_gt`
/// is a ratio of small integers, so it lands *exactly* on a grid point often: at
/// `num_gt = 20, tp = 7` the recall equals our old `rec_thrs[35]` bit-for-bit
/// while sitting strictly below numpy's. The two-pointer scan in
/// [`metrics::counts`](crate::metrics::counts) then stops one detection earlier
/// and reports a slightly different precision there. Neither grid is more
/// *correct* — both approximate 0.35 — so matching the reference is free, and
/// not matching it put a permanent floor under parity.
///
/// numpy's algorithm: `y[i] = i * step + start` with `step = (stop - start) /
/// (num - 1)`, then `y[num - 1] = stop` assigned exactly rather than computed.
fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    if num == 0 {
        return Vec::new();
    }
    if num == 1 {
        return vec![start];
    }
    let step = (stop - start) / (num - 1) as f64;
    let mut out: Vec<f64> = (0..num).map(|i| i as f64 * step + start).collect();
    // numpy pins the endpoint instead of trusting the arithmetic to land on it.
    out[num - 1] = stop;
    out
}

/// Generate the default COCO IoU threshold range: 0.50, 0.55, …, 0.95.
pub(crate) fn default_iou_thrs() -> Vec<f64> {
    linspace(0.5, 0.95, 10)
}

/// COCO's 101-point recall grid: 0.00, 0.01, …, 1.00.
///
/// The x-axis every AP in the crate is interpolated onto. Public because the
/// metric functions in [`metrics`](crate::metrics) take it as a parameter, so a
/// caller reaching for them directly needs the same grid `Params` defaults to —
/// otherwise their AP is on a different axis than `COCOeval`'s and the two
/// silently disagree.
pub fn default_rec_thrs() -> Vec<f64> {
    linspace(0.0, 1.0, 101)
}

/// Evaluation parameters controlling IoU thresholds, area ranges, and detection limits.
///
/// Defaults match pycocotools: 10 IoU thresholds (0.50:0.05:0.95), 101 recall
/// thresholds, and standard COCO area ranges. Keypoint evaluation uses different
/// defaults (3 area ranges instead of 4, max 20 detections instead of 1/10/100).
#[derive(Debug, Clone)]
pub struct Params {
    /// IoU computation type (bbox, segm, or keypoints).
    pub iou_type: IouType,
    /// Image IDs to evaluate (empty = all images).
    pub img_ids: Vec<u64>,
    /// Category IDs to evaluate (empty = all categories).
    pub cat_ids: Vec<u64>,
    /// IoU thresholds for matching (default: 0.50, 0.55, ..., 0.95).
    pub iou_thrs: Vec<f64>,
    /// Recall thresholds for interpolated precision (default: 0.00, 0.01, ..., 1.00).
    pub rec_thrs: Vec<f64>,
    /// Maximum detections per image for each summary metric (default: [1, 10, 100]).
    pub max_dets: Vec<usize>,
    /// Area ranges for filtering, each with a label and `[min, max]` bounds.
    /// Default labels: `"all"`, `"small"`, `"medium"`, `"large"` (3 ranges for keypoints).
    pub area_ranges: Vec<AreaRange>,
    /// Whether to evaluate per-category (true) or pool all categories (false).
    pub use_cats: bool,
    /// Per-keypoint OKS sigmas (default: 17 COCO keypoint sigmas).
    pub kpt_oks_sigmas: Vec<f64>,
    /// Whether to expand detections up the category hierarchy (OID mode).
    /// Default: false (only GT is expanded).
    pub expand_dt: bool,
}

impl Params {
    /// Index of the area range with the given label, or `None` if not found.
    pub fn area_range_idx(&self, label: &str) -> Option<usize> {
        self.area_ranges.iter().position(|ar| ar.label == label)
    }

    /// Index of the `"all"` area range, falling back to the first.
    ///
    /// Every whole-dataset metric is reported at `area="all"`, so this lookup runs
    /// in the summarize, report, calibration, diagnostics, and TIDE paths. The
    /// fallback matters: a caller with custom area labels and no `"all"` still gets
    /// a defined index rather than a panic, and index 0 is the widest range by
    /// convention.
    pub fn all_area_idx(&self) -> usize {
        self.area_range_idx("all").unwrap_or(0)
    }

    /// Create default parameters for the given evaluation type.
    ///
    /// Keypoint evaluation uses 3 area ranges (all/medium/large) and a single
    /// max-detections value of 20. All other types use 4 area ranges
    /// (all/small/medium/large) and max-detections of [1, 10, 100].
    pub fn new(iou_type: IouType) -> Self {
        let (max_dets, area_ranges) = match iou_type {
            IouType::Keypoints => (
                vec![20],
                vec![
                    AreaRange {
                        label: "all".into(),
                        range: [0.0, 1e10],
                    },
                    AreaRange {
                        label: "medium".into(),
                        range: [AREA_SMALL, AREA_LARGE],
                    },
                    AreaRange {
                        label: "large".into(),
                        range: [AREA_LARGE, 1e10],
                    },
                ],
            ),
            _ => (
                vec![1, 10, 100],
                vec![
                    AreaRange {
                        label: "all".into(),
                        range: [0.0, 1e10],
                    },
                    AreaRange {
                        label: "small".into(),
                        range: [0.0, AREA_SMALL],
                    },
                    AreaRange {
                        label: "medium".into(),
                        range: [AREA_SMALL, AREA_LARGE],
                    },
                    AreaRange {
                        label: "large".into(),
                        range: [AREA_LARGE, 1e10],
                    },
                ],
            ),
        };

        let kpt_oks_sigmas = KPT_OKS_SIGMAS.to_vec();
        let iou_thrs = default_iou_thrs();
        let rec_thrs = default_rec_thrs();

        Params {
            iou_type,
            img_ids: Vec::new(),
            cat_ids: Vec::new(),
            iou_thrs,
            rec_thrs,
            max_dets,
            area_ranges,
            use_cats: true,
            kpt_oks_sigmas,
            expand_dt: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The default grids must equal `numpy.linspace` bit-for-bit.
    ///
    /// Raw bit patterns, captured from the exact `np.linspace` calls pycocotools
    /// makes, because the whole point is the last ulp — an approximate comparison
    /// would pass against the very constructions this replaced.
    ///
    /// Regenerate with:
    /// ```text
    /// np.linspace(.5, 0.95, int(np.round((0.95-.5)/.05))+1, endpoint=True)
    /// np.linspace(.0, 1.00, int(np.round((1.00-.0)/.01))+1, endpoint=True)
    /// ```
    #[test]
    fn default_grids_match_numpy_linspace_bitwise() {
        const IOU_BITS: [u64; 10] = [
            4602678819172646912,
            4603129179135383962,
            4603579539098121011,
            4604029899060858061,
            4604480259023595110,
            4604930618986332160,
            4605380978949069210,
            4605831338911806259,
            4606281698874543308,
            4606732058837280358,
        ];
        const REC_BITS: [u64; 101] = [
            0,
            4576918229304087675,
            4581421828931458171,
            4584304132692975288,
            4585925428558828667,
            4587366580439587226,
            4588807732320345784,
            4589708452245819884,
            4590429028186199163,
            4591149604126578442,
            4591870180066957722,
            4592590756007337001,
            4593311331947716280,
            4593851763903000740,
            4594212051873190380,
            4594572339843380019,
            4594932627813569659,
            4595292915783759299,
            4595653203753948938,
            4596013491724138578,
            4596373779694328218,
            4596734067664517857,
            4597094355634707497,
            4597454643604897137,
            4597814931575086776,
            4598175219545276416,
            4598355363530371236,
            4598535507515466056,
            4598715651500560876,
            4598895795485655695,
            4599075939470750515,
            4599256083455845335,
            4599436227440940155,
            4599616371426034975,
            4599796515411129795,
            4599976659396224615,
            4600156803381319434,
            4600336947366414254,
            4600517091351509074,
            4600697235336603894,
            4600877379321698714,
            4601057523306793534,
            4601237667291888353,
            4601417811276983173,
            4601597955262077993,
            4601778099247172813,
            4601958243232267633,
            4602138387217362453,
            4602318531202457272,
            4602498675187552092,
            4602678819172646912,
            4602768891165194322,
            4602858963157741732,
            4602949035150289142,
            4603039107142836552,
            4603129179135383962,
            4603219251127931372,
            4603309323120478782,
            4603399395113026191,
            4603489467105573601,
            4603579539098121011,
            4603669611090668421,
            4603759683083215831,
            4603849755075763241,
            4603939827068310651,
            4604029899060858061,
            4604119971053405471,
            4604210043045952881,
            4604300115038500291,
            4604390187031047701,
            4604480259023595111,
            4604570331016142520,
            4604660403008689930,
            4604750475001237340,
            4604840546993784750,
            4604930618986332160,
            4605020690978879570,
            4605110762971426980,
            4605200834963974390,
            4605290906956521800,
            4605380978949069210,
            4605471050941616620,
            4605561122934164030,
            4605651194926711440,
            4605741266919258849,
            4605831338911806259,
            4605921410904353669,
            4606011482896901079,
            4606101554889448489,
            4606191626881995899,
            4606281698874543309,
            4606371770867090719,
            4606461842859638129,
            4606551914852185539,
            4606641986844732949,
            4606732058837280359,
            4606822130829827768,
            4606912202822375178,
            4607002274814922588,
            4607092346807469998,
            4607182418800017408,
        ];

        let iou = default_iou_thrs();
        assert_eq!(iou.len(), IOU_BITS.len());
        for (i, (&got, &want)) in iou.iter().zip(IOU_BITS.iter()).enumerate() {
            assert_eq!(
                got.to_bits(),
                want,
                "iou_thrs[{i}] = {got:?}, numpy has {:?}",
                f64::from_bits(want)
            );
        }

        let rec = default_rec_thrs();
        assert_eq!(rec.len(), REC_BITS.len());
        for (i, (&got, &want)) in rec.iter().zip(REC_BITS.iter()).enumerate() {
            assert_eq!(
                got.to_bits(),
                want,
                "rec_thrs[{i}] = {got:?}, numpy has {:?}",
                f64::from_bits(want)
            );
        }

        // numpy pins the endpoint rather than computing it; so must we.
        assert_eq!(iou[9], 0.95);
        assert_eq!(rec[100], 1.0);
    }

    #[test]
    fn linspace_handles_degenerate_lengths() {
        assert!(linspace(0.0, 1.0, 0).is_empty());
        assert_eq!(linspace(0.25, 1.0, 1), vec![0.25]);
        assert_eq!(linspace(0.0, 1.0, 2), vec![0.0, 1.0]);
    }
}
