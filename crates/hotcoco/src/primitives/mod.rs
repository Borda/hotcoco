//! Reusable evaluation primitives — the shared substrate that family drivers
//! (detection at 1.0; panoptic/tracking/concepts later) compose into metrics.
//!
//! This module is built alongside the existing `eval/` monolith and does not
//! change its behavior. At 1.0 the detection driver is rebuilt on these
//! primitives and `eval/` is deleted; until then the primitives are validated
//! against the monolith's output rather than replacing it.
//!
//! The public surface mirrors the planned Python `hotcoco.primitives` package.
//! Contracts here are designed against detection's needs *and* the documented
//! needs of tracking/panoptic/concepts, so they don't have to be reshaped when
//! those families land.
//!
//! # Stability
//!
//! These APIs are **provisional**. They are public so the family drivers can
//! share them, but they are not frozen until 1.4, after real-world soak — expect
//! additive change (new [`sim::SimKind`] variants, the tracking count vocabulary
//! in [`counts`]) in the 1.x minors.
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
//!
//! Submodules are added as each primitive is built:
//! - [`sim`] — similarity kernels + the `SimKind` geometry axis.
//! - [`greedy`] — COCO greedy matching (pycocotools-exact).
//! - [`assign`] — rectangular LSAP, semantic port of scipy.
//! - [`counts`] — detection rank-based PR accumulator (broader count vocabulary
//!   lands with the family that needs it).
//! - `report` — `EvalReport` (metrics + curves + params + provenance). *(pending)*

pub mod assign;
pub mod counts;
pub mod greedy;
pub mod sim;
