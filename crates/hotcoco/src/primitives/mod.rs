//! Reusable evaluation primitives — the shared substrate that family drivers
//! (detection at 1.0; panoptic/tracking/concepts later) compose into metrics.
//!
//! This module is built alongside the existing `eval/` monolith and does not
//! change its behaviour. At 1.0 the detection driver is rebuilt on these
//! primitives and `eval/` is deleted; until then the primitives are validated
//! against the monolith's output rather than replacing it.
//!
//! The public surface mirrors the planned Python `hotcoco.primitives` package.
//! Contracts here are designed against detection's needs *and* the documented
//! needs of tracking/panoptic/concepts, so they don't have to be reshaped when
//! those families land.
//!
//! Submodules are added as each primitive is built:
//! - [`sim`] — similarity kernels + the `SimKind` geometry axis (this slice).
//! - `greedy` — COCO greedy matching (pycocotools-exact). *(pending)*
//! - `assign` — rectangular LSAP, semantic port of scipy. *(pending)*
//! - `counts` — count structs, `GroupKey`, metric formulas, PR accumulator. *(pending)*
//! - `report` — `EvalReport` (metrics + curves + params + provenance). *(pending)*

pub mod sim;
