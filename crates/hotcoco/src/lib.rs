//! A pure-Rust implementation of COCO-style evaluation.
//!
//! ```no_run
//! use hotcoco::{COCO, COCOeval, params::IouType};
//! # fn main() -> hotcoco::error::Result<()> {
//! let gt = COCO::new(std::path::Path::new("instances_val2017.json"))?;
//! let dt = gt.load_res(std::path::Path::new("detections.json"))?;
//!
//! let mut ev = COCOeval::new(gt, dt, IouType::Bbox);
//! ev.run();                       // evaluate -> accumulate -> summarize
//! let report = ev.report()?;      // metrics, per-class, curves, provenance
//! # Ok(())
//! # }
//! ```
//!
//! # How the crate is laid out
//!
//! | Module | What lives there |
//! |---|---|
//! | [`types`] | The COCO schema — `Dataset`, `Image`, `Annotation`, `Category`, `Rle`. |
//! | [`coco`] | The dataset object: load, index, query, filter, merge, split, sample. |
//! | [`mask`], [`geometry`] | RLE codec and rotated-rect mechanics. |
//! | [`primitives`] | The shared evaluation substrate — similarity kernels, matching, accumulation, [`EvalReport`]. |
//! | [`detection`] | The detection metric family: AP/AR, LVIS, Open Images, TIDE, calibration, confusion. |
//! | [`quality`] | Dataset introspection: health checks and statistics. |
//! | [`convert`] | YOLO, Pascal VOC, CVAT, and DOTA conversion. |
//!
//! [`primitives`] is where an auditor should look to answer "how is similarity
//! computed?", "how are detections matched?", "how is AP accumulated?" — there is
//! exactly one implementation of each, and `tests/architecture.rs` fails the build
//! if a second appears.
//!
//! # Module renames in 1.0, and what still compiles
//!
//! 1.0 renamed `eval` to [`detection`], because detection is now one metric family
//! among several rather than the only one. Two modules moved for the same reason:
//! the Open Images hierarchy is detection machinery, and health checks belong with
//! dataset statistics rather than beside the schema.
//!
//! | Pre-1.0 path | Now | Status |
//! |---|---|---|
//! | `hotcoco::eval` | [`detection`] | deprecated alias |
//! | `hotcoco::hierarchy` | [`detection::hierarchy`] | deprecated alias |
//! | `hotcoco::healthcheck` | [`quality::healthcheck`] | deprecated alias |
//! | `hotcoco::types::{SummaryStats, CategoryStats, DatasetStats}` | [`quality`] | deprecated re-export |
//!
//! **Nothing stops compiling.** Every path above still resolves; each emits a
//! deprecation warning pointing at its replacement. The aliases are kept for the
//! whole 1.x series and removed at 2.0.
//!
//! **The crate-root re-exports are not deprecated and are the recommended paths.**
//! [`COCOeval`], [`Hierarchy`], [`HealthReport`], [`SummaryStats`] and the rest are
//! unchanged — most code needs no edit at all.
//!
//! The Python API is entirely unaffected: `hotcoco.COCOeval`,
//! `init_as_pycocotools()`, and the `pycocotools`/LVIS drop-in surface are
//! permanent compatibility guarantees.

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
