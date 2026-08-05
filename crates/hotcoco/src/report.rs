//! `EvalReport` — the shape every family's results take.
//!
//! Detection produces one today; panoptic, tracking, and concepts produce the
//! same type as they land, and composed metrics built from these primitives
//! produce it too. That is the point: a dashboard, a CLI renderer, or a report
//! writer consumes `EvalReport` and does not care which family filled it in.
//!
//! # Provenance is enforced, not advisory
//!
//! [`Provenance`] records whether a number is a benchmark-standard result, a
//! labeled hotcoco extension, or something a user composed from primitives.
//! Family drivers set it; the Python constructor hardwires
//! [`Provenance::UserComposed`], so a report cannot claim parity it was never
//! checked for. It survives serialization, so a deserialized report renders as
//! whatever it actually is rather than as whatever the renderer assumes.
//!
//! Without this, composed and extension numbers look exactly like leaderboard
//! numbers once they reach a chart.
//!
//! # What belongs in `curves`
//!
//! Whatever a renderer needs to draw the result and cannot recompute: PR curves,
//! HOTA's alpha sweep, calibration reliability bins. It is an open string-keyed
//! map rather than typed fields so families can add their own without a breaking
//! change.
//!
//! It is **not** a dump of the full evaluation arrays. Detection's complete
//! precision tensor is `T×R×K×A×M` — on COCO that is ~1M floats, which no report
//! should carry and no JSON file should hold. Producers put the *aggregate* slice
//! here (see [`crate::detection`]'s report assembly) and leave the full arrays
//! reachable through the family's own types.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Where a set of numbers came from, and therefore how much they may be trusted
/// to match a published leaderboard.
///
/// `#[non_exhaustive]`: families may need to describe provenance we have not
/// anticipated, and adding a variant must not be a breaking change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum Provenance {
    /// Verified against the reference implementation for this metric family —
    /// pycocotools, panopticapi, TrackEval — within its documented tolerance.
    /// These are the numbers you may compare against a leaderboard.
    ParityVerified,
    /// A hotcoco extension: a real metric computed over a geometry or
    /// configuration the reference implementation does not support, such as
    /// oriented bounding boxes. Internally consistent and useful for comparing
    /// models against each other, but **not** benchmark-standard, because no
    /// reference exists to be standard against.
    Extension,
    /// Composed by a user from public primitives. hotcoco has no idea whether it
    /// matches anything; it is reported as-is.
    UserComposed,
}

impl Provenance {
    /// Whether these numbers may be presented as benchmark-standard.
    pub fn is_benchmark_standard(self) -> bool {
        matches!(self, Provenance::ParityVerified)
    }
}

/// A completed evaluation: headline metrics, per-class and per-group breakdowns,
/// renderable curves, and the parameters that produced them.
///
/// Built by a family driver (or by a user composing primitives) and consumed by
/// anything that renders or serializes results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalReport {
    /// Which family produced this — `"detection"`, `"panoptic"`, `"tracking"`, …
    /// Free-form so a composed metric can name itself.
    pub task: String,
    /// See [`Provenance`]. Set by the producer; never inferred by a renderer.
    pub provenance: Provenance,
    /// Headline metrics, e.g. `AP`, `AP50`, `ARl`. `BTreeMap` so serialization
    /// is key-ordered and therefore diffable.
    pub metrics: BTreeMap<String, f64>,
    /// Per-class breakdown: class name -> metric name -> value.
    ///
    /// Nested rather than flat `"AP/person"` because families report more than
    /// one metric per class, and splitting a flattened key back apart is
    /// ambiguous when class names contain the separator.
    pub per_class: BTreeMap<String, BTreeMap<String, f64>>,
    /// Per-group breakdown, for whatever grouping the family defines — LVIS
    /// frequency buckets, dataset slices, area ranges.
    pub per_group: BTreeMap<String, BTreeMap<String, f64>>,
    /// Renderable curves. See the module docs for what belongs here.
    pub curves: BTreeMap<String, Vec<f64>>,
    /// The evaluation parameters, as the producer chose to describe them.
    /// Opaque `Value` because each family's parameters differ.
    pub params: serde_json::Value,
}

impl EvalReport {
    /// A report for `task` with the given provenance.
    ///
    /// Drivers use this. Anything reaching users that has not been checked
    /// against a reference should use [`user_composed`](Self::user_composed).
    pub fn new(task: impl Into<String>, provenance: Provenance) -> Self {
        EvalReport {
            task: task.into(),
            provenance,
            metrics: BTreeMap::new(),
            per_class: BTreeMap::new(),
            per_group: BTreeMap::new(),
            curves: BTreeMap::new(),
            params: serde_json::Value::Null,
        }
    }

    /// A report that claims nothing — provenance is hardwired to
    /// [`Provenance::UserComposed`].
    ///
    /// This is the constructor the Python bindings expose, so a report built
    /// outside a family driver cannot assert parity it never had.
    pub fn user_composed(task: impl Into<String>) -> Self {
        Self::new(task, Provenance::UserComposed)
    }

    /// Set the headline metrics.
    #[must_use]
    pub fn with_metrics<K: Into<String>>(
        mut self,
        metrics: impl IntoIterator<Item = (K, f64)>,
    ) -> Self {
        self.metrics = metrics.into_iter().map(|(k, v)| (k.into(), v)).collect();
        self
    }

    /// Record one metric for one class.
    #[must_use]
    pub fn with_class_metric(
        mut self,
        class: impl Into<String>,
        metric: impl Into<String>,
        value: f64,
    ) -> Self {
        self.per_class
            .entry(class.into())
            .or_default()
            .insert(metric.into(), value);
        self
    }

    /// Record one metric for one group.
    #[must_use]
    pub fn with_group_metric(
        mut self,
        group: impl Into<String>,
        metric: impl Into<String>,
        value: f64,
    ) -> Self {
        self.per_group
            .entry(group.into())
            .or_default()
            .insert(metric.into(), value);
        self
    }

    /// Attach a named curve.
    #[must_use]
    pub fn with_curve(mut self, name: impl Into<String>, values: Vec<f64>) -> Self {
        self.curves.insert(name.into(), values);
        self
    }

    /// Attach the producing parameters.
    #[must_use]
    pub fn with_params(mut self, params: serde_json::Value) -> Self {
        self.params = params;
        self
    }

    /// Look up a headline metric.
    pub fn metric(&self, name: &str) -> Option<f64> {
        self.metrics.get(name).copied()
    }

    /// Serialize to pretty-printed JSON.
    pub fn to_json(&self) -> crate::error::Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_facing_constructor_cannot_claim_parity() {
        let r = EvalReport::user_composed("my_metric");
        assert_eq!(r.provenance, Provenance::UserComposed);
        assert!(!r.provenance.is_benchmark_standard());
    }

    #[test]
    fn only_parity_verified_is_benchmark_standard() {
        assert!(Provenance::ParityVerified.is_benchmark_standard());
        assert!(!Provenance::Extension.is_benchmark_standard());
        assert!(!Provenance::UserComposed.is_benchmark_standard());
    }

    /// Provenance must survive a round trip — a report that loses it on
    /// serialization would render as whatever the renderer assumed.
    #[test]
    fn provenance_survives_serialization() {
        for p in [
            Provenance::ParityVerified,
            Provenance::Extension,
            Provenance::UserComposed,
        ] {
            let report = EvalReport::new("detection", p)
                .with_metrics([("AP", 0.5)])
                .with_curve("pr@0.50", vec![1.0, 0.5, 0.0]);
            let json = report.to_json().expect("serialize");
            let back: EvalReport = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(back.provenance, p);
            assert_eq!(back.metric("AP"), Some(0.5));
            assert_eq!(back.curves["pr@0.50"], vec![1.0, 0.5, 0.0]);
        }
    }

    #[test]
    fn nested_breakdowns_hold_several_metrics_per_key() {
        let r = EvalReport::new("detection", Provenance::ParityVerified)
            .with_class_metric("person", "AP", 0.6)
            .with_class_metric("person", "AR", 0.7)
            .with_group_metric("rare", "AP", 0.2);
        assert_eq!(r.per_class["person"]["AP"], 0.6);
        assert_eq!(r.per_class["person"]["AR"], 0.7);
        assert_eq!(r.per_group["rare"]["AP"], 0.2);
    }
}
