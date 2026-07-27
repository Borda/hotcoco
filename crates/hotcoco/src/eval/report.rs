//! Presentation of summary metrics: printed lines, result maps, and DTOs.
//!
//! Everything here formats or reshapes numbers that [`super::summarize`] already
//! computed — this module performs no evaluation math of its own. At 1.0 it also
//! becomes where the detection driver assembles `primitives::report::EvalReport`.

use std::collections::HashMap;

use crate::params::{Params, default_iou_thrs};

use super::accumulate::AccumulatedEval;
use super::metrics::build_metric_defs;
use super::results::{EvalParams, EvalResults};
use super::summarize::{per_cat_ap_static, summarize_impl};
use super::{COCOeval, EvalMode};

impl COCOeval {
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

        // Warn if parameters differ from what the hardcoded summary display expects.
        // OID has its own defaults — skip these COCO-specific warnings.
        if self.eval_mode == EvalMode::Coco || self.eval_mode == EvalMode::Lvis {
            let defaults = Params::new(self.params.iou_type);
            let mut warnings = Vec::new();

            let default_iou = default_iou_thrs();
            if self.params.iou_thrs != default_iou {
                warnings.push(
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
                warnings.push(format!(
                    "max_dets differ from expected ({:?}). AR lines may use unexpected max_dets values.",
                    expected_max_dets
                ));
            }
            if !self
                .params
                .area_ranges
                .iter()
                .map(|ar| ar.label.as_str())
                .eq(defaults.area_ranges.iter().map(|ar| ar.label.as_str()))
            {
                let default_labels: Vec<&str> = defaults
                    .area_ranges
                    .iter()
                    .map(|ar| ar.label.as_str())
                    .collect();
                warnings.push(format!(
                    "area range labels differ from default ({:?}). Per-size metrics may not find their area range.",
                    default_labels
                ));
            }

            for w in &warnings {
                eprintln!("Warning: {}", w);
            }
        }

        // Delegate the actual computation to the free function.
        let metrics = build_metric_defs(&self.params, self.eval_mode);
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

    /// Index of the "all" area range, or 0 if not found.
    fn area_all_idx(&self) -> usize {
        self.params.area_range_idx("all").unwrap_or(0)
    }

    /// Metric key names in canonical display order for the current evaluation mode.
    ///
    /// Returns the same ordered list that drives `summarize()` and `get_results()`.
    /// For standard COCO bbox/segm: `["AP", "AP50", ..., "ARl"]` (12 keys).
    /// For keypoints: 10 keys. For LVIS: 13 keys.
    pub fn metric_keys(&self) -> Vec<&'static str> {
        build_metric_defs(&self.params, self.eval_mode)
            .into_iter()
            .map(|m| m.name)
            .collect()
    }

    /// Per-category mean AP (averaged over all IoU thresholds and recall thresholds,
    /// at area="all" and the last max_dets setting). Returns one value per `params.cat_ids`
    /// entry; -1.0 for categories with no valid precision data.
    pub(super) fn per_cat_ap(&self, eval: &AccumulatedEval) -> Vec<f64> {
        per_cat_ap_static(eval, &self.params)
    }

    /// Return summary metrics as a `HashMap<metric_name, value>`.
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
    pub fn get_results(&self, prefix: Option<&str>, per_class: bool) -> HashMap<String, f64> {
        let stats = match &self.stats {
            Some(s) => s,
            None => return HashMap::new(),
        };

        let keys = self.metric_keys();

        let make_key = |metric: &str| -> String {
            match prefix {
                Some(p) => format!("{p}/{metric}"),
                None => metric.to_string(),
            }
        };

        let mut results: HashMap<String, f64> = keys
            .iter()
            .zip(stats.iter())
            .map(|(&k, &v)| (make_key(k), v))
            .collect();

        if per_class {
            if let Some(eval) = &self.eval {
                let per_cat = self.per_cat_ap(eval);
                for (ap, cat_id) in per_cat.iter().zip(self.params.cat_ids.iter()) {
                    if *ap >= 0.0 {
                        if let Some(cat) = self.coco_gt.get_cat(*cat_id) {
                            results.insert(make_key(&format!("AP/{}", cat.name)), *ap);
                        }
                    }
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
    pub fn f_scores(&self, beta: f64) -> HashMap<String, f64> {
        let eval = match &self.eval {
            Some(e) => e,
            None => return HashMap::new(),
        };

        let beta2 = beta * beta;
        let a_idx = self.area_all_idx();
        let m_idx = eval.shape.m - 1;

        // Identify which IoU threshold indices correspond to 0.5 and 0.75.
        let mut is_t50 = vec![false; eval.shape.t];
        let mut is_t75 = vec![false; eval.shape.t];
        for (i, &thr) in self.params.iou_thrs.iter().enumerate() {
            if (thr - 0.5).abs() < 1e-9 {
                is_t50[i] = true;
            }
            if (thr - 0.75).abs() < 1e-9 {
                is_t75[i] = true;
            }
        }

        // Single pass: compute max-F-beta per (t_idx, k_idx), accumulate into three buckets.
        let mut sum_all = 0.0_f64;
        let mut count_all = 0_usize;
        let mut sum_50 = 0.0_f64;
        let mut count_50 = 0_usize;
        let mut sum_75 = 0.0_f64;
        let mut count_75 = 0_usize;

        for t_idx in 0..eval.shape.t {
            for k_idx in 0..eval.shape.k {
                let max_f = self
                    .params
                    .rec_thrs
                    .iter()
                    .enumerate()
                    .filter_map(|(r_idx, &r)| {
                        let p_idx = eval.precision_idx(t_idx, r_idx, k_idx, a_idx, m_idx);
                        let p = eval.precision[p_idx];
                        if p < 0.0 {
                            return None;
                        }
                        let denom = beta2 * p + r;
                        if denom < f64::EPSILON {
                            return Some(0.0);
                        }
                        Some((1.0 + beta2) * p * r / denom)
                    })
                    .fold(f64::NEG_INFINITY, f64::max);

                if max_f > f64::NEG_INFINITY {
                    sum_all += max_f;
                    count_all += 1;
                    if is_t50[t_idx] {
                        sum_50 += max_f;
                        count_50 += 1;
                    }
                    if is_t75[t_idx] {
                        sum_75 += max_f;
                        count_75 += 1;
                    }
                }
            }
        }

        let mean_or_neg1 =
            |sum: f64, count: usize| -> f64 { if count == 0 { -1.0 } else { sum / count as f64 } };

        let prefix = if (beta - 1.0).abs() < 1e-9 {
            "F1".to_string()
        } else {
            format!("F{:.1}", beta)
        };

        let mut out = HashMap::new();
        out.insert(prefix.clone(), mean_or_neg1(sum_all, count_all));
        out.insert(format!("{}50", prefix), mean_or_neg1(sum_50, count_50));
        out.insert(format!("{}75", prefix), mean_or_neg1(sum_75, count_75));
        out
    }

    /// Print results to stdout in a compact key=value format.
    ///
    /// Must be called after [`summarize`](COCOeval::summarize). Prints nothing if
    /// `summarize` has not been run (emits a warning to stderr instead).
    pub fn print_results(&self) {
        let results = self.get_results(None, false);
        if results.is_empty() {
            eprintln!("No results to print. Run evaluate(), accumulate(), and summarize() first.");
            return;
        }

        let keys = self.metric_keys();

        for key in keys {
            let val = results.get(key).copied().unwrap_or(-1.0);
            let val_str = Self::format_metric(val);
            println!(" {:>10} = {}", key, val_str);
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
        let stats = self.stats.as_ref().ok_or_else(|| {
            "summarize() must be called before results(). \
             Run evaluate(), accumulate(), and summarize() first."
                .to_string()
        })?;

        let keys = self.metric_keys();

        let metrics: HashMap<String, f64> = keys
            .iter()
            .zip(stats.iter())
            .map(|(&k, &v)| (k.to_string(), v))
            .collect();

        let per_class_map = if per_class {
            self.eval.as_ref().map(|eval| {
                let per_cat = self.per_cat_ap(eval);
                per_cat
                    .iter()
                    .zip(self.params.cat_ids.iter())
                    .filter(|&(&ap, _)| ap >= 0.0)
                    .filter_map(|(&ap, &cat_id)| {
                        self.coco_gt
                            .get_cat(cat_id)
                            .map(|cat| (cat.name.clone(), ap))
                    })
                    .collect()
            })
        } else {
            None
        };

        Ok(EvalResults {
            hotcoco_version: env!("CARGO_PKG_VERSION").to_string(),
            params: EvalParams::from_params(&self.params, self.eval_mode),
            metrics,
            per_class: per_class_map,
        })
    }
}
