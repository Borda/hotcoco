pub mod coco;
pub mod convert;
pub mod detection;
pub mod error;
pub mod geometry;
pub mod mask;
pub mod params;
pub mod primitives;
pub mod quality;
pub mod types;

pub use coco::COCO;
pub use convert::{ConvertError, CvatStats, DotaStats, VocStats, YoloStats};
pub use detection::{
    AccumulatedEval, AnnotationIndex, BootstrapCI, COCOeval, CalibrationBin, CalibrationResult,
    CategoryDelta, CompareOpts, ComparisonResult, ConfusionMatrix, DtStatus, ErrorProfile, EvalImg,
    EvalMode, EvalParams, EvalResults, EvalShape, GtStatus, ImageDiagnostics, ImageSummary,
    LabelError, LabelErrorType, SliceResult, SlicedResults, TideErrors, compare,
};
pub use error::Error;

/// The detection family under its pre-1.0 name.
///
/// `eval` was the module's name while detection was the only metric family in the
/// crate. It is now [`detection`], one family beside the panoptic, tracking, and
/// concepts families that follow — but every `hotcoco::eval::*` path keeps
/// resolving through this alias so existing Rust code compiles unchanged.
///
/// This is Tier-2 compatibility surface: kept through the whole 1.x series and
/// removed at 2.0. The Python API is unaffected — `hotcoco.COCOeval` and the
/// `pycocotools` drop-in surface are Tier 1 and permanent.
#[deprecated(
    since = "1.0.0",
    note = "renamed to `hotcoco::detection`; the `eval` alias is kept for the 1.x series and removal is slated for 2.0"
)]
pub use detection as eval;

/// Open Images label hierarchy, under its pre-1.0 path.
///
/// [`Hierarchy`] is Open Images machinery — it is consumed only by the detection
/// family's GT/DT expansion — so it now lives at [`detection::hierarchy`] rather
/// than beside the cross-family primitives, where its old top-level placement
/// wrongly implied it was one.
///
/// Tier-2 compatibility: this path is kept for the 1.x series and removed at 2.0.
/// The crate-root [`Hierarchy`] re-export is **not** deprecated and is the
/// recommended path.
#[deprecated(
    since = "1.0.0",
    note = "moved to `hotcoco::detection::hierarchy`; this alias is kept for the 1.x series and removal is slated for 2.0"
)]
pub mod hierarchy {
    pub use crate::detection::hierarchy::*;
}
pub use detection::hierarchy::Hierarchy;
pub use params::{AreaRange, IouType, Params};
pub use primitives::report::{EvalReport, Provenance};
pub use quality::{
    CategoryStats, DatasetStats, DatasetSummary, Finding, HealthReport, Layer, SummaryStats,
};
pub use types::{Annotation, Category, Dataset, Image, Rle, Segmentation};

/// Dataset health checks, under their pre-1.0 path.
///
/// Health checking joined `COCO::stats` and the statistics DTOs in [`quality`] at
/// 1.0: they are one concern — inspecting a dataset — and are distinct from both
/// the schema ([`types`]) and the metrics engine ([`detection`]).
///
/// Tier-2 compatibility: kept for the 1.x series, removed at 2.0. The crate-root
/// re-exports ([`HealthReport`], [`Finding`], [`Layer`], [`DatasetSummary`]) are
/// **not** deprecated.
#[deprecated(
    since = "1.0.0",
    note = "moved to `hotcoco::quality::healthcheck`; this alias is kept for the 1.x series and removal is slated for 2.0"
)]
pub mod healthcheck {
    pub use crate::quality::healthcheck::*;
}
