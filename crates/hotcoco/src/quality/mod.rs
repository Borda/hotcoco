//! Dataset quality and introspection — a tier of its own.
//!
//! Distinct from both the schema ([`crate::types`], which says what a COCO file
//! *contains*) and the metrics engine ([`crate::detection`], which scores
//! predictions against it). This module answers "is this dataset sound, and what
//! is in it?" — the questions you ask before evaluating anything.

pub mod healthcheck;
pub mod stats;

pub use healthcheck::{
    DatasetSummary, Finding, HealthReport, Layer, healthcheck, healthcheck_compatibility,
};
pub use stats::{CategoryStats, DatasetStats, SummaryStats};
