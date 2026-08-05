//! Presentation of summary metrics: printed lines, result maps, and DTOs.
//!
//! Everything here formats or reshapes numbers computed elsewhere: the reduction
//! is [`super::summarize`]'s, the metric formulas are
//! [`metrics::counts`](crate::metrics::counts)'. What this module owns is the
//! presentation — which numbers appear, under which names, in which order — plus
//! assembling [`EvalReport`].

use std::collections::BTreeMap;

use crate::params::{IouType, Params};
use crate::report::{EvalReport, Provenance};

use super::accumulate::AccumulatedEval;
use super::catalog::{MetricDef, build_metric_defs};
use super::results::{EvalParams, EvalResults};
use super::summarize::{mean_of_valid, mean_or_missing, per_cat_ap_static, summarize_impl};
use super::{COCOeval, EvalMode};

impl COCOeval {
    /// Ways this run's parameters depart from the reference configuration.
    ///
    /// Public because [`Provenance`] is a single bit and the *reason* is what a
    /// caller can act on: "extension" alone leaves someone re-deriving which of
    /// half a dozen conditions fired. The Python bindings turn each entry into a
    /// `warnings.warn`, which is also how these reach a notebook — `summarize()`
    /// writes them to fd 2, and that bypasses `sys.stderr` entirely.
    ///
    /// Empty means the numbers are directly comparable to the reference
    /// implementation's published output. Non-empty means they are not, and it is
    /// the *same* fact that drives both the `summarize()` warnings and
    /// [`Provenance`]: a report claiming `ParityVerified` on custom `iou_thrs` is
    /// claiming a check nobody ran.
    ///
    /// This is the *whole* comparability predicate, not part of one. An earlier
    /// version checked only the parameters here and tested geometry and eval mode
    /// separately at the `report()` call site — so an OBB run with default params,
    /// and every Open Images run, were downgraded to `Extension` while `summarize()`
    /// printed no warning at all. Those are the two *largest* comparability breaks,
    /// and they were the silent ones. Anything that can make a run incomparable
    /// belongs in this list.
    pub fn reference_deviations(&self) -> Vec<String> {
        let mut out = Vec::new();

        // No reference implementation exists for these at all, at any parameters.
        if self.params.iou_type == IouType::Obb {
            out.push(
                "oriented-box evaluation has no reference implementation to check against; \
                 these numbers are a hotcoco extension, not leaderboard-comparable."
                    .to_string(),
            );
        }
        if self.eval_mode == EvalMode::OpenImages {
            // `scripts/parity_oid.py` matches the TensorFlow reference on group-of
            // handling, IoA containment and all-points AP. What remains unimplemented
            // is the Challenge's non-exhaustive image-level-label rule — detections
            // of an unverified class are ignored, and of a negatively-labelled class
            // are false positives — which needs per-image label data COCO JSON
            // cannot carry. So a real challenge submission would still differ.
            out.push(
                "Open Images evaluation matches the TensorFlow reference for group-of \
                 handling and AP, but does not implement the challenge's non-exhaustive \
                 image-level-label rule, so these numbers are not leaderboard-comparable."
                    .to_string(),
            );
        }

        // Parameter deviations only mean something where there is a reference to
        // deviate *from*: `scripts/parity.py` (pycocotools) and `parity_lvis.py`.
        //
        // Default-deny, not default-allow: a mode with no checked reference is a
        // reason to say so, not a reason to skip the parameter checks and stay
        // silent. `EvalMode` has three variants today, so the only way to reach
        // this is to add a fourth — and the failure mode of the old early return
        // was that such a mode inherited `parity_verified` for free.
        match self.eval_mode {
            EvalMode::Coco | EvalMode::Lvis => {}
            EvalMode::OpenImages => return out, // already flagged above
        }

        let defaults = Params::new(self.params.iou_type);

        if self.params.iou_thrs != defaults.iou_thrs {
            out.push(
                "iou_thrs differ from default (0.50:0.05:0.95). AP50/AP75 lines may show -1.000."
                    .to_string(),
            );
        }

        let expected_max_dets = if self.eval_mode == EvalMode::Lvis {
            vec![300usize]
        } else {
            defaults.max_dets.clone()
        };
        if self.params.max_dets != expected_max_dets {
            out.push(format!(
                "max_dets differ from expected ({:?}). AR lines may use unexpected max_dets values.",
                expected_max_dets
            ));
        }

        let default_labels: Vec<&str> = defaults
            .area_ranges
            .iter()
            .map(|ar| ar.label.as_str())
            .collect();
        if !self
            .params
            .area_ranges
            .iter()
            .map(|ar| ar.label.as_str())
            .eq(default_labels.iter().copied())
        {
            out.push(format!(
                "area range labels differ from default ({:?}). Per-size metrics may not find their area range.",
                default_labels
            ));
        }

        // Area-range *bounds*, not just labels. Comparing labels alone made this
        // check structurally blind to the one path a Python caller actually takes:
        // the `params.areaRng` setter deliberately preserves the existing labels,
        // so redefining "small" as [0, 100] kept the name, passed the label check,
        // and reported different APs/APm/APl as parity_verified.
        if !self
            .params
            .area_ranges
            .iter()
            .map(|ar| ar.range)
            .eq(defaults.area_ranges.iter().map(|ar| ar.range))
        {
            out.push(
                "area range bounds differ from default. APs/APm/APl measure different \
                 object-size buckets than the reference."
                    .to_string(),
            );
        }

        // The 101-point recall grid defines what AP *means*: it is the x-axis the
        // precision curve is averaged over. A different grid is a different metric
        // wearing the same name.
        if self.params.rec_thrs != defaults.rec_thrs {
            out.push(format!(
                "rec_thrs differ from the default {}-point grid. AP is averaged over a \
                 different recall axis than the reference.",
                defaults.rec_thrs.len()
            ));
        }

        // Class-agnostic pooling is a different question than the leaderboard asks.
        if !self.params.use_cats {
            out.push(
                "use_cats is false, so detections are pooled across categories. This is \
                 not the per-category AP any COCO or LVIS leaderboard reports."
                    .to_string(),
            );
        }

        // Custom sigmas redefine OKS, and OKS is the similarity keypoint AP is built on.
        if self.params.iou_type == IouType::Keypoints
            && self.params.kpt_oks_sigmas != defaults.kpt_oks_sigmas
        {
            out.push(
                "kpt_oks_sigmas differ from the COCO defaults, which redefines OKS. \
                 Keypoint AP is no longer COCO keypoint AP."
                    .to_string(),
            );
        }

        out
    }

    /// Whether this run's numbers may be presented as leaderboard-comparable.
    ///
    /// The single mapping from [`reference_deviations`](Self::reference_deviations)
    /// to a [`Provenance`] bit. It exists as a named method rather than an
    /// expression inside [`report`](Self::report) because renderers need the bit
    /// without paying for a whole report — and a renderer that recomputes it from
    /// `iou_type` or `eval_mode` is exactly the silent downgrade `Provenance`
    /// exists to prevent. Read it; never re-derive it.
    pub fn provenance(&self) -> Provenance {
        if self.reference_deviations().is_empty() {
            Provenance::ParityVerified
        } else {
            Provenance::Extension
        }
    }

    /// Return the summary metric lines as strings without printing.
    ///
    /// Computes stats (setting `self.stats`) and returns each formatted line.
    /// Warnings about non-default parameters are printed to stderr.
    pub fn summarize_lines(&mut self) -> Vec<String> {
        let eval = match &self.eval {
            Some(e) => e,
            None => {
                eprintln!("Please run evaluate() and accumulate() first.");
                return Vec::new();
            }
        };

        for w in &self.reference_deviations() {
            eprintln!("Warning: {}", w);
        }

        // Delegate the actual computation to the free function.
        let metrics = self.metric_defs();
        let stats = summarize_impl(
            eval,
            &self.params,
            self.eval_mode,
            self.freq_groups(),
            &metrics,
        );

        let mut lines = Vec::with_capacity(metrics.len() + 1);

        for (m, &val) in metrics.iter().zip(stats.iter()) {
            let val_str = Self::format_metric(val);

            if self.eval_mode == EvalMode::Lvis || self.eval_mode == EvalMode::OpenImages {
                lines.push(format!(" {:>10} = {}", m.name, val_str));
            } else {
                let metric_name = if m.ap {
                    "Average Precision"
                } else {
                    "Average Recall"
                };
                let metric_short = if m.ap { "AP" } else { "AR" };
                let iou_str = match m.iou_thr {
                    Some(thr) => format!("{:.2}", thr),
                    None => "0.50:0.95".to_string(),
                };
                lines.push(format!(
                    " {:<22} @[ IoU={:<9} | area={:>6} | maxDets={:>3} ] = {}",
                    format!("{} ({})", metric_name, metric_short),
                    iou_str,
                    m.area_lbl,
                    m.max_det,
                    val_str
                ));
            }
        }

        self.stats = Some(stats);
        lines
    }

    /// Print the standard COCO evaluation summary.
    ///
    /// Calls [`Self::summarize_lines`] and prints each line to stdout.
    pub fn summarize(&mut self) {
        for line in self.summarize_lines() {
            println!("{}", line);
        }
    }

    /// Format a metric value: -1.0 sentinel stays as "-1.000", positive values use 3 decimal places.
    pub(super) fn format_metric(val: f64) -> String {
        if val < 0.0 {
            format!("{:0.3}", -1.0f64)
        } else {
            format!("{:0.3}", val)
        }
    }

    /// The summary-metric catalog for the current evaluation mode.
    ///
    /// One [`MetricDef`] per headline number, in the order `summarize()`,
    /// [`metric_keys`](Self::metric_keys) and [`stats`](Self::stats) use — so
    /// `metric_defs()[i]` describes `stats()[i]`, and a renderer can label a
    /// value from its definition instead of parsing its name.
    ///
    /// The list depends on `params` (it resolves the max-detection axis) and on
    /// `eval_mode`, so it is a method rather than a constant: COCO bbox/segm has
    /// 12 entries, keypoints 10, LVIS 13, Open Images 1.
    pub fn metric_defs(&self) -> Vec<MetricDef> {
        build_metric_defs(&self.params, self.eval_mode)
    }

    /// Metric key names in canonical display order for the current evaluation mode.
    ///
    /// Returns the same ordered list that drives `summarize()` and `get_results()`.
    /// For standard COCO bbox/segm: `["AP", "AP50", ..., "ARl"]` (12 keys).
    /// For keypoints: 10 keys. For LVIS: 13 keys.
    ///
    /// The names projected out of [`metric_defs`](Self::metric_defs) — one list,
    /// so the keys and the definitions cannot fall out of order with each other.
    pub fn metric_keys(&self) -> Vec<&'static str> {
        self.metric_defs().into_iter().map(|m| m.name).collect()
    }

    /// Per-category mean AP (averaged over all IoU thresholds and recall thresholds,
    /// at area="all" and the last max_dets setting). Returns one value per `params.cat_ids`
    /// entry; -1.0 for categories with no valid precision data.
    pub(super) fn per_cat_ap(&self, eval: &AccumulatedEval) -> Vec<f64> {
        per_cat_ap_static(eval, &self.params, self.eval_mode)
    }

    /// Per-category AP keyed by category *name*, in `params.cat_ids` order.
    ///
    /// The one place the name→AP table is derived. Categories reporting the
    /// `-1.0` "not computed" sentinel, and ids with no category record to name
    /// them, are dropped — and both [`report`](Self::report) and
    /// [`get_results`](Self::get_results) drop exactly the same ones, which is
    /// the point: they were two independent filters over the same data, free to
    /// disagree about which classes exist.
    fn per_class_ap_named(&self, eval: &AccumulatedEval) -> Vec<(String, f64)> {
        self.per_cat_ap(eval)
            .iter()
            .zip(self.params.cat_ids.iter())
            .filter(|&(&ap, _)| ap >= 0.0)
            .filter_map(|(&ap, &cat_id)| self.coco_gt.get_cat(cat_id).map(|c| (c.name.clone(), ap)))
            .collect()
    }

    /// Return summary metrics as a `BTreeMap<metric_name, value>`.
    ///
    /// Must be called after [`summarize`](COCOeval::summarize). Returns an empty map
    /// if `summarize` has not been run.
    ///
    /// # Arguments
    ///
    /// * `prefix` — When `Some("val/bbox")`, keys become `"val/bbox/AP"` etc.
    ///   When `None`, keys are bare metric names (`"AP"`, `"AR100"`, …).
    /// * `per_class` — When `true` and [`accumulate`](COCOeval::accumulate) has been
    ///   run, adds per-category AP entries keyed as `"AP/{cat_name}"` (or
    ///   `"{prefix}/AP/{cat_name}"` with a prefix). Categories where all precision
    ///   values are −1 are skipped.
    ///
    /// # Metric keys
    ///
    /// For LVIS mode: `AP`, `AP50`, `AP75`, `APs`, `APm`, `APl`, `APr`, `APc`, `APf`,
    /// `AR@300`, `ARs@300`, `ARm@300`, `ARl@300`.
    ///
    /// For standard COCO bbox/segm: `AP`, `AP50`, `AP75`, `APs`, `APm`, `APl`,
    /// `AR1`, `AR10`, `AR100`, `ARs`, `ARm`, `ARl`.
    ///
    /// For keypoints: `AP`, `AP50`, `AP75`, `APm`, `APl`,
    /// `AR`, `AR50`, `AR75`, `ARm`, `ARl`.
    pub fn get_results(&self, prefix: Option<&str>, per_class: bool) -> BTreeMap<String, f64> {
        let stats = match &self.stats {
            Some(s) => s,
            None => return BTreeMap::new(),
        };

        let keys = self.metric_keys();

        let make_key = |metric: &str| -> String {
            match prefix {
                Some(p) => format!("{p}/{metric}"),
                None => metric.to_string(),
            }
        };

        let mut results: BTreeMap<String, f64> = keys
            .iter()
            .zip(stats.iter())
            .map(|(&k, &v)| (make_key(k), v))
            .collect();

        if per_class {
            if let Some(eval) = &self.eval {
                for (name, ap) in self.per_class_ap_named(eval) {
                    results.insert(make_key(&format!("AP/{name}")), ap);
                }
            }
        }

        results
    }

    /// Compute F-beta scores after `accumulate()`.
    ///
    /// Returns three metrics analogous to AP/AP50/AP75, but using max F-beta instead of
    /// mean precision. For each (IoU threshold, category), finds the recall operating point
    /// that maximizes F-beta, then averages across categories.
    ///
    /// `beta` controls the precision/recall trade-off:
    /// - `beta = 1.0`  → F1 (harmonic mean, equal weight)
    /// - `beta < 1.0`  → weights precision more heavily
    /// - `beta > 1.0`  → weights recall more heavily
    ///
    /// Returns an empty map if `accumulate()` has not been run.
    ///
    /// Keys are `F1`, `F1_50`, `F1_75` (or `F0.5`, `F0.5_50`, … for other betas).
    /// The separator is not decorative: `F1` + `50` reads as an unrelated metric
    /// named `F150`, which is what these keys used to be.
    pub fn f_scores(&self, beta: f64) -> BTreeMap<String, f64> {
        let eval = match &self.eval {
            Some(e) => e,
            None => return BTreeMap::new(),
        };

        let a_idx = self.params.all_area_idx();
        // `max_det_idx`, not `shape.m - 1`: the F-scores are the AP/AP50/AP75
        // rows in another metric, so they must read the same M slot those do.
        let m_idx = self.params.max_det_idx();

        // `Params::iou_thr_idx` owns this lookup, tolerance included: `F1_50`
        // names the 0.50 slice or nothing, exactly as `AP50` does.
        let t50 = self.params.iou_thr_idx(0.5);
        let t75 = self.params.iou_thr_idx(0.75);

        // Single pass: compute max-F-beta per (t_idx, k_idx), accumulate into
        // three (sum, count) buckets — overall, then the two single-threshold ones.
        let mut buckets = [(0.0_f64, 0_usize); 3];

        for t_idx in 0..eval.shape.t {
            for k_idx in 0..eval.shape.k {
                let precisions: Vec<f64> = (0..eval.shape.r)
                    .map(|r_idx| {
                        eval.precision[eval.precision_idx(t_idx, r_idx, k_idx, a_idx, m_idx)]
                    })
                    .collect();

                if let Some(max_f) =
                    crate::metrics::counts::max_f_beta(&precisions, &self.params.rec_thrs, beta)
                {
                    let in_bucket = [true, Some(t_idx) == t50, Some(t_idx) == t75];
                    for (bucket, hit) in buckets.iter_mut().zip(in_bucket) {
                        if hit {
                            bucket.0 += max_f;
                            bucket.1 += 1;
                        }
                    }
                }
            }
        }

        let prefix = if (beta - 1.0).abs() < 1e-9 {
            "F1".to_string()
        } else {
            format!("F{:.1}", beta)
        };

        let names = [
            prefix.clone(),
            format!("{prefix}_50"),
            format!("{prefix}_75"),
        ];
        names
            .into_iter()
            .zip(buckets)
            .map(|(name, (sum, count))| (name, mean_or_missing(sum, count)))
            .collect()
    }

    /// Print results to stdout in a compact key=value format.
    ///
    /// Must be called after [`summarize`](COCOeval::summarize). Prints nothing if
    /// `summarize` has not been run (emits a warning to stderr instead).
    pub fn print_results(&self) {
        // Zipped straight against `stats`, which is what `metric_keys()` is
        // parallel to. Building a `BTreeMap` and then looking every key back out
        // of it needed an `unwrap_or(-1.0)` for a miss that cannot happen, and
        // that unreachable default is indistinguishable from a real `-1.000`.
        let keys = self.metric_keys();
        let stats = self.stats.as_deref().unwrap_or(&[]);

        if keys.is_empty() || stats.is_empty() {
            eprintln!("No results to print. Run evaluate(), accumulate(), and summarize() first.");
            return;
        }

        for (&key, &val) in keys.iter().zip(stats) {
            println!(" {:>10} = {}", key, Self::format_metric(val));
        }
    }

    /// Build a serializable [`EvalResults`] from the current evaluation state.
    ///
    /// Must be called after [`summarize`](COCOeval::summarize). Returns an error
    /// if `summarize` has not been run.
    ///
    /// # Arguments
    ///
    /// * `per_class` — When `true`, includes per-category AP values in the result.
    ///   Categories where all precision values are −1 are excluded.
    pub fn results(&self, per_class: bool) -> crate::error::Result<EvalResults> {
        // A projection of the report, so the two can never disagree about a
        // metric. `params` is taken from the typed `Params` rather than from the
        // report's opaque `serde_json::Value`, which keeps the serialized shape
        // of `EvalResults` byte-stable for anyone parsing saved result files.
        let report = self.report()?;

        // `None` (not an empty map) when `accumulate()` has not run, matching the
        // pre-1.0 behavior — an empty map would claim "no classes scored" where
        // the truth is "per-class data was never computed".
        let per_class_map = if per_class {
            self.eval.as_ref().map(|_| {
                report
                    .per_class
                    .iter()
                    .filter_map(|(name, m)| m.get("AP").map(|&ap| (name.clone(), ap)))
                    .collect()
            })
        } else {
            None
        };

        Ok(EvalResults {
            hotcoco_version: env!("CARGO_PKG_VERSION").to_string(),
            provenance: report.provenance,
            params: EvalParams::from_params(&self.params, self.eval_mode),
            metrics: report.metrics.into_iter().collect(),
            per_class: per_class_map,
        })
    }

    /// Assemble a [`EvalReport`] from this evaluation.
    ///
    /// Requires [`summarize`](COCOeval::summarize) to have been called. This is
    /// the shape every metric family reports in, so a renderer that can draw a
    /// detection report can draw a panoptic or tracking one unchanged.
    ///
    /// # Provenance
    ///
    /// [`Provenance::ParityVerified`] only when this run is actually comparable to
    /// a reference implementation's published numbers — COCO bbox/segm/keypoints
    /// against pycocotools, or LVIS against `lvis-api`, **with reference
    /// parameters**. Anything else is [`Provenance::Extension`]:
    ///
    /// - oriented boxes, which are a real metric with no reference to be standard against
    /// - Open Images, whose protocol hotcoco implements but has no checked reference for
    /// - any run with custom `iou_thrs`, `rec_thrs`, `max_dets`, area-range labels or
    ///   bounds, `use_cats = false`, or `kpt_oks_sigmas`
    ///
    /// That last case is the one worth stating plainly: parity is a property of a
    /// *configuration*, not of an `iou_type`. Deriving it from the type alone
    /// would stamp `parity_verified` on numbers pycocotools was never run against,
    /// which is exactly what [`Provenance`] exists to prevent.
    ///
    /// # Curves
    ///
    /// The aggregate precision-recall curve per IoU threshold (`pr@0.50` …),
    /// averaged over categories at `area="all"` and the largest `max_dets`, plus
    /// the shared `rec_thrs` x-axis. That is the slice a chart actually draws;
    /// the full `T×R×K×A×M` tensor stays reachable through
    /// [`accumulated`](COCOeval::accumulated) rather than being copied in here
    /// (on COCO it is ~1M floats).
    pub fn report(&self) -> crate::error::Result<EvalReport> {
        let stats = self.stats.as_ref().ok_or_else(|| {
            "summarize() must be called before report(). \
             Run evaluate(), accumulate(), and summarize() first."
                .to_string()
        })?;

        let provenance = self.provenance();

        let keys = self.metric_keys();
        let mut report = EvalReport::new("detection", provenance)
            .with_metrics(keys.iter().copied().zip(stats.iter().copied()))
            .with_params(serde_json::to_value(EvalParams::from_params(
                &self.params,
                self.eval_mode,
            ))?);

        let Some(eval) = &self.eval else {
            return Ok(report);
        };

        // Per-class AP. Categories with no valid precision anywhere report -1.0
        // and are omitted rather than recorded as a real score.
        for (name, ap) in self.per_class_ap_named(eval) {
            report = report.with_class_metric(name, "AP", ap);
        }

        // LVIS frequency buckets as a structured group axis. Same values as the
        // APr/APc/APf headline metrics — this is a view of them, not a second
        // computation — but it saves renderers from string-matching metric names
        // to discover that a grouping exists.
        if self.eval_mode == EvalMode::Lvis {
            for (group, key) in [("rare", "APr"), ("common", "APc"), ("frequent", "APf")] {
                if let Some(v) = report.metrics.get(key).copied() {
                    report = report.with_group_metric(group, "AP", v);
                }
            }
        }

        // Aggregate PR curves: mean precision over categories at each recall
        // threshold, for each IoU threshold.
        let a_idx = self.params.all_area_idx();
        // The curve a chart draws must be the curve the headline AP was averaged
        // from, so it reads the same M slot — `max_det_idx`, not the last one.
        let m_idx = self.params.max_det_idx();
        for (t_idx, &thr) in self.params.iou_thrs.iter().enumerate() {
            // Folded in place rather than collected: this runs T×R times (10×101
            // on COCO), and collecting a throwaway Vec per recall threshold cost
            // ~1000 heap allocations per `report()` call to compute a mean.
            let curve: Vec<f64> = (0..eval.shape.r)
                .map(|r_idx| {
                    mean_of_valid((0..eval.shape.k).map(|k_idx| {
                        eval.precision[eval.precision_idx(t_idx, r_idx, k_idx, a_idx, m_idx)]
                    }))
                })
                .collect();
            report = report.with_curve(format!("pr@{thr:.2}"), curve);
        }
        report = report.with_curve("rec_thrs", self.params.rec_thrs.clone());

        Ok(report)
    }
}
