//! Matching kernels — the shared substrate that decides what pairs with what.
//!
//! Three kernels, and nothing else: [`sim`] computes similarity between two sets,
//! [`greedy`] resolves it into COCO's rank-ordered assignment, [`assign`] resolves
//! it optimally via rectangular LSAP. Family drivers (detection at 1.0;
//! panoptic/tracking/concepts later) compose them. The one non-kernel export is
//! [`greedy::ThreshMatrix`], the container shape of the greedy kernel's
//! per-threshold output.
//!
//! Kernels here produce matches and similarities; the functions in
//! [`metrics`](crate::metrics) turn those into numbers. Nothing here scores;
//! nothing there matches. See [`metrics`](crate::metrics) for the full split.
//!
//! This module is where an auditor should look to answer "how is similarity
//! computed?" and "how are detections matched?" — there is exactly one
//! implementation of each, and `tests/architecture.rs` fails the build if a second
//! appears.
//!
//! # Stability
//!
//! These APIs are **provisional**. They are public so the family drivers can
//! share them, but they are not frozen until 1.4, after real-world soak — expect
//! additive change (new [`sim::SimKind`] variants) in the 1.x minors.
//!
//! The similarity kernels re-exported onto Tier-1 paths are the exception —
//! already frozen. See [`sim`][sim#where-the-math-lives].
//!
//! # Kernels are stateless
//!
//! No primitive here caches, memoizes, or otherwise retains input across calls,
//! and **no contract in this module may come to require whole-sequence retention**.
//! That is what lets HOTA's second pass recompute similarity instead of holding a
//! whole sequence's matrices in memory (at MOT20 scale, hundreds of MB per
//! sequence per thread). Batched or per-sequence helpers added later must
//! iterate-and-consume — never return every timestep's matrix at once.

pub mod assign;
pub mod greedy;
pub mod sim;
