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
//! | [`primitives`] | Matching kernels — similarity, greedy assignment, LSAP. |
//! | [`metrics`] | Metric functions over flat arrays — AP, calibration, confusion, bootstrap. |
//! | [`report`] | [`EvalReport`] — the shape every metric family reports in. |
//! | [`detection`] | The detection metric family: AP/AR, LVIS, Open Images, TIDE. |
//! | [`quality`] | Dataset introspection: health checks and statistics. |
//! | [`convert`] | YOLO, Pascal VOC, CVAT, and DOTA conversion. |
//!
//! # The functional layer
//!
//! [`primitives`] and [`metrics`] are free functions over flat arrays — no
//! evaluator required, the way `sklearn.metrics` and `torchmetrics.functional`
//! work:
//!
//! ```
//! use hotcoco::metrics::counts::average_precision;
//!
//! let ap = average_precision(&[0.9, 0.8, 0.3], &[true, false, true], None, 3, &[0.0, 0.5, 1.0]);
//! ```
//!
//! The two split by what a function *produces*: [`primitives`] produces matches
//! (which detection pairs with which ground truth), [`metrics`] produces numbers
//! from matches. Nothing in `primitives` scores; nothing in `metrics` matches.
//!
//! [`COCOeval`] is the stateful driver on top — it owns the pycocotools-compatible
//! `evaluate`/`accumulate`/`summarize` lifecycle, and its analysis methods are
//! adapters that marshal `eval_imgs` into arrays and call the functions above.
//!
//! Between them, [`primitives`] and [`metrics`] are where an auditor should look
//! to answer "how is similarity computed?", "how are detections matched?", "how is
//! AP accumulated?" — there is exactly one implementation of each, and
//! `tests/architecture.rs` fails the build if a second appears.
//!
//! # Module renames in 1.0
//!
//! 1.0 renamed `eval` to [`detection`], because detection is now one metric family
//! among several rather than the only one. Three modules moved for the same reason:
//! the Open Images hierarchy is detection machinery, health checks belong with
//! dataset statistics rather than beside the schema, and `counts` computes numbers
//! from matches so it belongs in [`metrics`], not [`primitives`].
//!
//! | Pre-1.0 module path | Now |
//! |---|---|
//! | `hotcoco::eval` | [`detection`] |
//! | `hotcoco::hierarchy` | [`detection::hierarchy`] |
//! | `hotcoco::healthcheck` | [`quality::healthcheck`](mod@quality::healthcheck) |
//! | `hotcoco::types::{SummaryStats, CategoryStats, DatasetStats}` | [`quality`] |
//! | `hotcoco::primitives::counts` | [`metrics::counts`] |
//!
//! **The crate-root re-exports absorbed every one of these**, so most code needs no
//! edit at all: [`COCOeval`], [`EvalImg`], [`Hierarchy`], [`HealthReport`],
//! [`SummaryStats`] and the rest resolve exactly as before.
//!
//! There are no compatibility aliases for the old *module* paths. 0.x is
//! pre-release under SemVer — "anything MAY change at any time" — so those paths
//! carried no stability promise, and keeping them would have meant a multi-year
//! obligation to a surface nothing depended on. 1.0 is where the API is fixed;
//! from here breaking changes wait for 2.0.
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
pub mod metrics;
pub mod params;
pub mod primitives;
pub mod quality;
pub mod report;
pub mod types;

pub use coco::COCO;
pub use convert::{ConvertError, CvatStats, DotaStats, VocStats, YoloStats};
pub use detection::{
    AccumulatedEval, AnnotationIndex, COCOeval, CalibrationResult, CategoryDelta, CompareOpts,
    ComparisonResult, ConfusionMatrix, DtStatus, ErrorProfile, EvalImg, EvalMode, EvalParams,
    EvalResults, EvalShape, GtStatus, ImageDiagnostics, ImageSummary, LabelError, LabelErrorType,
    SliceResult, SlicedResults, TideErrors, compare,
};
pub use error::Error;

pub use detection::hierarchy::Hierarchy;
// Re-exported from where they are defined, not through `detection`. Both are
// family-agnostic — any family that resamples gets a `BootstrapCI`, any family
// that bins confidences gets a `CalibrationBin` — so routing the crate-root path
// through the detection driver would make the next family import a detection path
// for a type detection does not own.
pub use metrics::bootstrap::BootstrapCI;
pub use metrics::calibration::CalibrationBin;
pub use params::{AreaRange, IouType, Params};
pub use quality::{
    CategoryStats, DatasetStats, DatasetSummary, Finding, HealthReport, Layer, SummaryStats,
};
pub use report::{EvalReport, Provenance};
pub use types::{Annotation, Category, Dataset, Image, Rle, Segmentation};
