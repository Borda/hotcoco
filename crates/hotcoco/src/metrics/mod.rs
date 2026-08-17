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
//! # Degenerate-input convention
//!
//! These free functions share one policy, the same as `primitives`:
//!
//! - **Mismatched parallel-array lengths are a programmer error** and panic via
//!   `assert!` with a message naming both lengths. Nothing silently truncates,
//!   no-ops, or degrades ([`primitives::assign::lsap`](crate::primitives::assign::lsap)
//!   set the pattern). Each function's `# Panics` section states its checks.
//! - **Empty input is not an error** — it produces the documented empty-set
//!   value (`0.0`, an empty `Vec`, an all-zero matrix), because "no detections"
//!   is a legitimate evaluation state, not a bug.
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

/// Whether a metric value was actually computed, as opposed to carrying the
/// crate's `-1.0` "not computed for this configuration" sentinel.
///
/// The sentinel is public contract: evaluation output uses `-1.0` for an area
/// range with no ground truth or a category absent from the split — never for a
/// genuinely low score. Callers filtering evaluation arrays should use this
/// predicate rather than re-deriving `v >= 0.0` by hand.
///
/// The predicate half of the sentinel convention whose *producer* is
/// [`detection::summarize::mean_or_missing`](crate::detection). It lives here
/// rather than beside the producer because the lower layer has to read it too:
/// [`counts::max_f_beta`] skips sentinel precisions, and `metrics` may not import
/// a family driver (`tests/architecture.rs` enforces that). Detection reaches
/// down; the dependency runs one way.
///
/// Spelled `v >= 0.0` rather than `!(v < 0.0)` so it is the exact test the five
/// hand-written sites used — the two differ only on `NaN`, which none of the
/// accumulated arrays can hold, and matching the old spelling keeps the fold
/// bit-identical without anyone having to prove that.
#[inline]
pub fn is_computed(v: f64) -> bool {
    v >= 0.0
}

/// The complement of [`is_computed`]: `v` is the "not computed" sentinel.
#[inline]
pub fn is_missing(v: f64) -> bool {
    !is_computed(v)
}
