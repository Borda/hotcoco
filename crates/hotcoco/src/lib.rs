pub mod coco;
pub mod convert;
pub mod detection;
pub mod error;
pub mod geometry;
pub mod healthcheck;
pub mod hierarchy;
pub mod mask;
pub mod params;
pub mod primitives;
pub mod types;

pub use coco::COCO;
pub use convert::{ConvertError, CvatStats, DotaStats, VocStats, YoloStats};
pub use detection::{
    AccumulatedEval, AnnotationIndex, BootstrapCI, COCOeval, CalibrationBin, CalibrationResult,
    CategoryDelta, CompareOpts, ComparisonResult, ConfusionMatrix, DtStatus, ErrorProfile, EvalImg,
    EvalMode, EvalResults, EvalShape, GtStatus, ImageDiagnostics, ImageSummary, LabelError,
    LabelErrorType, SliceResult, SlicedResults, TideErrors, compare,
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
pub use healthcheck::{DatasetSummary, Finding, HealthReport, Layer};
pub use hierarchy::Hierarchy;
pub use params::{AreaRange, IouType, Params};
pub use types::{
    Annotation, Category, CategoryStats, Dataset, DatasetStats, Image, Rle, Segmentation,
    SummaryStats,
};
