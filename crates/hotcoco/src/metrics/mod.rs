//! Metric functions — numbers computed from matches.
//!
//! Every item here is a free function over flat arrays, callable without
//! constructing an evaluator:
//!
//! ```
//! use hotcoco::metrics::counts::average_precision;
//!
//! let scores  = [0.9, 0.8, 0.3];
//! let matched = [true, false, true];
//! let ap = average_precision(&scores, &matched, None, 3, &[0.0, 0.5, 1.0]);
//! ```
//!
//! This is the same shape `sklearn.metrics` and `torchmetrics.functional` use,
//! and for the same reason: a metric is a pure function of its inputs, so tying
//! it to an evaluator object only makes it harder to reach.
//!
//! # `metrics` vs [`primitives`](crate::primitives)
//!
//! The split is by what a function *produces*, not by which family calls it:
//!
//! | | Produces | Contains |
//! |---|---|---|
//! | [`primitives`](crate::primitives) | matches and similarities | `sim`, `greedy`, `assign` |
//! | `metrics` | numbers from matches | `counts`, `calibration`, `confusion`, `bootstrap` |
//!
//! `primitives::greedy::greedy_match` decides *which detection pairs with which
//! ground truth*. `metrics::counts::average_precision` turns that decision into a
//! number. Nothing here matches; nothing there scores.
//!
//! `tests/architecture.rs` enforces the direction of the dependency: `metrics`
//! may not import from a family driver such as [`detection`](crate::detection),
//! and `primitives` may not import from `metrics`.
//!
//! # Why these functions take flat arrays
//!
//! Taking `(scores, matched)` rather than a family-specific struct is what makes
//! them reusable. Detection produces those arrays from `eval_imgs`; tracking and
//! panoptic will produce them from their own match records. The function does not
//! know or care which family called it — that adapter is a dozen lines in the
//! driver, and it is the only detection-shaped code involved.
//!
//! # Stability
//!
//! Provisional through 1.x, like [`primitives`](crate::primitives): public so the
//! family drivers and Python can share them, but not frozen until 1.4. Expect
//! additive change — new functions, and the tracking count vocabulary in
//! [`counts`] — rather than reshaping of what is here.

pub mod bootstrap;
pub mod calibration;
pub mod confusion;
pub mod counts;
