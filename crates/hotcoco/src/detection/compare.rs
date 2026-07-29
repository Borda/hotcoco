use std::collections::{HashMap, HashSet};

use serde::Serialize;

use crate::metrics::bootstrap::bootstrap_ci;

use super::COCOeval;
use super::accumulate::accumulate_impl;
use super::catalog::{MetricDef, build_metric_defs};
use super::summarize::{per_cat_ap_static, summarize_impl};

/// Options for pairwise model comparison.
#[derive(Debug, Clone)]
pub struct CompareOpts {
    /// Number of bootstrap samples for confidence intervals. 0 = no bootstrap.
    pub n_bootstrap: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// Confidence level for bootstrap intervals (e.g. 0.95 for 95% CI).
    pub confidence: f64,
}

impl Default for CompareOpts {
    fn default() -> Self {
        Self {
            n_bootstrap: 0,
            seed: 42,
            confidence: 0.95,
        }
    }
}

// Private: `BootstrapCI` belongs to `metrics::bootstrap` and reaches the crate
// root from there. Re-exporting it publicly here would create
// `hotcoco::detection::BootstrapCI` — a detection path for a type detection does
// not own, which 1.0 would then freeze.
use crate::metrics::bootstrap::BootstrapCI;

/// Per-category AP comparison entry.
#[derive(Debug, Clone, Serialize)]
pub struct CategoryDelta {
    /// COCO category ID.
    pub cat_id: u64,
    /// Human-readable category name.
    pub cat_name: String,
    /// AP for model A.
    pub ap_a: f64,
    /// AP for model B.
    pub ap_b: f64,
    /// Delta (B - A).
    pub delta: f64,
}

/// Full pairwise comparison result.
#[derive(Debug, Clone, Serialize)]
pub struct ComparisonResult {
    /// Metric names in canonical display order (from the evaluation mode's MetricDef vec).
    pub metric_keys: Vec<String>,
    /// All summary metrics for model A.
    pub metrics_a: HashMap<String, f64>,
    /// All summary metrics for model B.
    pub metrics_b: HashMap<String, f64>,
    /// Per-metric delta (B - A).
    pub deltas: HashMap<String, f64>,
    /// Bootstrap CIs on summary metric deltas. `None` if bootstrap disabled.
    pub ci: Option<HashMap<String, BootstrapCI>>,
    /// Per-category AP comparison, sorted by delta ascending (worst regressions first).
    pub per_category: Vec<CategoryDelta>,
    /// Number of bootstrap samples used (0 if disabled).
    pub n_bootstrap: usize,
    /// Number of shared images in the comparison.
    pub num_images: usize,
}

/// Compare two evaluations on the same dataset.
///
/// Both evaluators must have had [`evaluate()`](COCOeval::evaluate) called and must
/// use the same `eval_mode` and `iou_type`. Accumulation and summarization are
/// performed internally on the shared image set — callers do not need to call
/// `accumulate()` or `summarize()` first.
///
/// When `opts.n_bootstrap > 0`, bootstrap confidence intervals are computed on
/// the summary metric deltas by resampling images with replacement and
/// re-accumulating for each sample. This is parallelized with rayon.
pub fn compare(
    eval_a: &COCOeval,
    eval_b: &COCOeval,
    opts: &CompareOpts,
) -> crate::error::Result<ComparisonResult> {
    // --- Validation ---
    if eval_a.eval_imgs.is_empty() {
        return Err("evaluate() must be called on eval_a before compare()".into());
    }
    if eval_b.eval_imgs.is_empty() {
        return Err("evaluate() must be called on eval_b before compare()".into());
    }
    if eval_a.eval_mode != eval_b.eval_mode {
        return Err(format!(
            "eval_mode mismatch: {:?} vs {:?}",
            eval_a.eval_mode, eval_b.eval_mode
        )
        .into());
    }
    if eval_a.params.iou_type != eval_b.params.iou_type {
        return Err(format!(
            "iou_type mismatch: {:?} vs {:?}",
            eval_a.params.iou_type, eval_b.params.iou_type
        )
        .into());
    }
    if opts.confidence <= 0.0 || opts.confidence >= 1.0 {
        return Err(format!("confidence must be in (0, 1), got {}", opts.confidence).into());
    }

    // --- Shared image set ---
    let imgs_a: HashSet<u64> = eval_a.params.img_ids.iter().copied().collect();
    let imgs_b: HashSet<u64> = eval_b.params.img_ids.iter().copied().collect();
    let shared_set: HashSet<u64> = imgs_a.intersection(&imgs_b).copied().collect();
    if shared_set.is_empty() {
        return Err("no shared images between eval_a and eval_b".into());
    }
    let num_images = shared_set.len();
    let shared_sorted: Vec<u64> = {
        let mut v: Vec<u64> = shared_set.iter().copied().collect();
        v.sort_unstable();
        v
    };

    // --- Accumulate + summarize on shared images ---
    let metrics = build_metric_defs(&eval_a.params, eval_a.eval_mode);
    let metric_keys: Vec<&str> = metrics.iter().map(|m| m.name).collect();

    let acc_a = accumulate_impl(
        &eval_a.eval_imgs,
        &eval_a.params,
        Some(&shared_set),
        eval_a.eval_mode,
    );
    let stats_a = summarize_impl(
        &acc_a,
        &eval_a.params,
        eval_a.eval_mode,
        eval_a.freq_groups(),
        &metrics,
    );

    let acc_b = accumulate_impl(
        &eval_b.eval_imgs,
        &eval_b.params,
        Some(&shared_set),
        eval_b.eval_mode,
    );
    let stats_b = summarize_impl(
        &acc_b,
        &eval_b.params,
        eval_b.eval_mode,
        eval_b.freq_groups(),
        &metrics,
    );

    // --- Metric maps ---
    let metrics_a = stats_to_map(&metric_keys, &stats_a);
    let metrics_b = stats_to_map(&metric_keys, &stats_b);

    let deltas: HashMap<String, f64> = metric_keys
        .iter()
        .zip(stats_a.iter().zip(stats_b.iter()))
        .map(|(&k, (&a, &b))| (k.to_string(), metric_delta(a, b)))
        .collect();

    // --- Per-category AP ---
    // `per_cat_ap_static` returns one entry per `params.cat_ids` slot, so both
    // sides must be looked up by category id rather than by position. The two
    // evaluators may carry different category lists — different GT files, or the
    // same file filtered differently — and a positional read would pair A's
    // category with whatever B happened to evaluate in that slot, under A's name.
    let by_cat = |ev: &COCOeval, acc: &_| -> HashMap<u64, f64> {
        ev.params
            .cat_ids
            .iter()
            .copied()
            .zip(per_cat_ap_static(acc, &ev.params, ev.eval_mode))
            .collect()
    };
    let per_cat_a = by_cat(eval_a, &acc_a);
    let per_cat_b = by_cat(eval_b, &acc_b);

    // Union, so a category only one side evaluated is reported rather than dropped.
    let mut all_cat_ids: Vec<u64> = eval_a
        .params
        .cat_ids
        .iter()
        .chain(eval_b.params.cat_ids.iter())
        .copied()
        .collect();
    all_cat_ids.sort_unstable();
    all_cat_ids.dedup();

    let mut per_category: Vec<CategoryDelta> = all_cat_ids
        .into_iter()
        .filter_map(|cat_id| {
            let ap_a = per_cat_a.get(&cat_id).copied().unwrap_or(-1.0);
            let ap_b = per_cat_b.get(&cat_id).copied().unwrap_or(-1.0);
            // Skip categories with no data in either model
            if ap_a < 0.0 && ap_b < 0.0 {
                return None;
            }
            let cat_name = eval_a
                .coco_gt
                .get_cat(cat_id)
                .or_else(|| eval_b.coco_gt.get_cat(cat_id))
                .map_or_else(|| cat_id.to_string(), |c| c.name.clone());
            Some(CategoryDelta {
                cat_id,
                cat_name,
                ap_a,
                ap_b,
                // `metric_delta`, not a raw subtraction: a category missing from
                // one side is no evidence, not a swing of up to 1.0. Since the
                // table sorts ascending as "worst regressions first", the raw form
                // put every B-missing category at the top.
                delta: metric_delta(ap_a, ap_b),
            })
        })
        .collect();

    // Sort by delta ascending (worst regressions first)
    per_category.sort_by(|a, b| {
        a.delta
            .partial_cmp(&b.delta)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // --- Bootstrap ---
    let ci = if opts.n_bootstrap > 0 {
        Some(bootstrap_compare(
            eval_a,
            eval_b,
            &shared_sorted,
            opts,
            &metrics,
            &metric_keys,
        ))
    } else {
        None
    };

    Ok(ComparisonResult {
        metric_keys: metric_keys.iter().map(|&k| k.to_string()).collect(),
        metrics_a,
        metrics_b,
        deltas,
        ci,
        per_category,
        n_bootstrap: opts.n_bootstrap,
        num_images,
    })
}

/// Run bootstrap resampling to compute confidence intervals on metric deltas.
///
/// The resampling, the percentile bounds, and `prob_positive` all belong to
/// [`metrics::bootstrap::bootstrap_ci`](crate::metrics::bootstrap::bootstrap_ci).
/// What is detection-specific — and all this function supplies — is the statistic:
/// re-accumulate both evaluators over the sampled images and take the metric
/// deltas. Missing metrics (`-1.0`) contribute a zero delta rather than a spurious
/// swing, since a metric undefined for a subset is not evidence either way.
fn bootstrap_compare(
    eval_a: &COCOeval,
    eval_b: &COCOeval,
    shared_img_ids: &[u64],
    opts: &CompareOpts,
    metrics: &[MetricDef],
    metric_keys: &[&str],
) -> HashMap<String, BootstrapCI> {
    let cis = bootstrap_ci(
        shared_img_ids,
        opts.n_bootstrap,
        opts.seed,
        opts.confidence,
        |sample| {
            let acc_a = accumulate_impl(
                &eval_a.eval_imgs,
                &eval_a.params,
                Some(sample),
                eval_a.eval_mode,
            );
            let stats_a = summarize_impl(
                &acc_a,
                &eval_a.params,
                eval_a.eval_mode,
                eval_a.freq_groups(),
                metrics,
            );

            let acc_b = accumulate_impl(
                &eval_b.eval_imgs,
                &eval_b.params,
                Some(sample),
                eval_b.eval_mode,
            );
            let stats_b = summarize_impl(
                &acc_b,
                &eval_b.params,
                eval_b.eval_mode,
                eval_b.freq_groups(),
                metrics,
            );

            stats_a
                .iter()
                .zip(stats_b.iter())
                .map(|(&a, &b)| metric_delta(a, b))
                .collect()
        },
    );

    metric_keys
        .iter()
        .zip(cis)
        .map(|(&name, ci)| (name.to_string(), ci))
        .collect()
}

/// B minus A, treating a metric missing from either side as no evidence.
///
/// `-1.0` is the "not computed for this configuration" sentinel — APs for an area
/// range with no ground truth, say. Subtracting it would manufacture a swing of up
/// to 1.0 out of missing data, so the delta is zero instead. The point estimate and
/// the bootstrap CIs must agree on this, which is why it is one function.
#[inline]
fn metric_delta(a: f64, b: f64) -> f64 {
    if a >= 0.0 && b >= 0.0 { b - a } else { 0.0 }
}

fn stats_to_map(metric_keys: &[&str], stats: &[f64]) -> HashMap<String, f64> {
    metric_keys
        .iter()
        .zip(stats.iter())
        .map(|(&k, &v)| (k.to_string(), v))
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::COCO;
    use crate::params::IouType;
    use std::path::PathBuf;

    fn fixtures_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
    }

    fn make_eval() -> COCOeval {
        let gt = COCO::new(&fixtures_dir().join("gt.json")).unwrap();
        let dt = gt.load_res(&fixtures_dir().join("dt.json")).unwrap();
        let mut ev = COCOeval::new(gt, dt, IouType::Bbox);
        ev.evaluate();
        ev
    }

    #[test]
    fn test_compare_identical_models() {
        let ev_a = make_eval();
        let ev_b = make_eval();
        let result = compare(&ev_a, &ev_b, &CompareOpts::default()).unwrap();

        assert_eq!(result.num_images, ev_a.params.img_ids.len());
        for (key, delta) in &result.deltas {
            assert!(
                delta.abs() < 1e-10,
                "expected zero delta for {key}, got {delta}"
            );
        }
        assert!(result.ci.is_none());
    }

    /// `per_cat_ap_static` returns one entry per `params.cat_ids` slot, so pairing
    /// the two sides by position reports B's category-2 AP under A's category-1
    /// name whenever the evaluators carry different category lists.
    #[test]
    fn per_category_pairs_by_cat_id_not_position() {
        let ev_a = make_eval();
        let cat_ids = ev_a.params.cat_ids.clone();
        assert!(cat_ids.len() >= 2, "fixture must have >= 2 categories");
        let only = cat_ids[1];

        // Same data, but B evaluates only the *second* category — so its
        // per-category vector has one entry, at the slot A uses for its first.
        let gt = COCO::new(&fixtures_dir().join("gt.json")).unwrap();
        let dt = gt.load_res(&fixtures_dir().join("dt.json")).unwrap();
        let mut ev_b = COCOeval::new(gt, dt, IouType::Bbox);
        ev_b.params.cat_ids = vec![only];
        ev_b.evaluate();

        let result = compare(&ev_a, &ev_b, &CompareOpts::default()).unwrap();

        for cat in &result.per_category {
            if cat.cat_id == only {
                assert!(
                    cat.ap_b >= 0.0,
                    "category {only} was evaluated by B and should carry its own AP"
                );
            } else {
                assert_eq!(
                    cat.ap_b, -1.0,
                    "category {} was not evaluated by B and must not borrow \
                     another category's AP",
                    cat.cat_id
                );
                assert_eq!(
                    cat.delta, 0.0,
                    "a category missing from B is no evidence, not a regression"
                );
            }
        }
    }

    #[test]
    fn test_compare_with_bootstrap() {
        let ev_a = make_eval();
        let ev_b = make_eval();
        let opts = CompareOpts {
            n_bootstrap: 50,
            seed: 42,
            confidence: 0.95,
        };
        let result = compare(&ev_a, &ev_b, &opts).unwrap();

        assert!(result.ci.is_some());
        let ci = result.ci.as_ref().unwrap();
        assert!(!ci.is_empty());

        // For identical models, CIs should be tight around zero
        for (key, boot_ci) in ci {
            assert!(
                boot_ci.lower.abs() < 0.1 && boot_ci.upper.abs() < 0.1,
                "expected tight CI for {key}, got [{}, {}]",
                boot_ci.lower,
                boot_ci.upper
            );
            assert_eq!(boot_ci.confidence, 0.95);
        }
    }

    #[test]
    fn test_bootstrap_seed_reproducibility() {
        let ev_a = make_eval();
        let ev_b = make_eval();
        let opts = CompareOpts {
            n_bootstrap: 20,
            seed: 123,
            confidence: 0.95,
        };

        let r1 = compare(&ev_a, &ev_b, &opts).unwrap();
        let r2 = compare(&ev_a, &ev_b, &opts).unwrap();

        let ci1 = r1.ci.as_ref().unwrap();
        let ci2 = r2.ci.as_ref().unwrap();
        for key in ci1.keys() {
            assert_eq!(
                ci1[key].lower, ci2[key].lower,
                "CI lower mismatch for {key}"
            );
            assert_eq!(
                ci1[key].upper, ci2[key].upper,
                "CI upper mismatch for {key}"
            );
        }
    }

    #[test]
    fn test_compare_before_evaluate_errors() {
        let gt = COCO::new(&fixtures_dir().join("gt.json")).unwrap();
        let dt = gt.load_res(&fixtures_dir().join("dt.json")).unwrap();
        let ev_a = COCOeval::new(gt, dt, IouType::Bbox);
        // ev_a has not called evaluate(), so compare should fail
        let gt2 = COCO::new(&fixtures_dir().join("gt.json")).unwrap();
        let dt2 = gt2.load_res(&fixtures_dir().join("dt.json")).unwrap();
        let ev_b = COCOeval::new(gt2, dt2, IouType::Bbox);

        let err = compare(&ev_a, &ev_b, &CompareOpts::default()).unwrap_err();
        assert!(err.to_string().contains("evaluate()"));
    }

    #[test]
    fn test_compare_per_category() {
        let ev_a = make_eval();
        let ev_b = make_eval();
        let result = compare(&ev_a, &ev_b, &CompareOpts::default()).unwrap();

        // Per-category deltas should all be zero for identical models
        for cat in &result.per_category {
            assert!(
                cat.delta.abs() < 1e-10,
                "expected zero delta for {}, got {}",
                cat.cat_name,
                cat.delta
            );
            assert!(cat.ap_a >= 0.0 || cat.ap_b >= 0.0);
        }
    }

    #[test]
    fn test_compare_invalid_confidence() {
        let ev_a = make_eval();
        let ev_b = make_eval();
        let opts = CompareOpts {
            confidence: 1.5,
            ..Default::default()
        };
        let err = compare(&ev_a, &ev_b, &opts).unwrap_err();
        assert!(err.to_string().contains("confidence"));
    }
}
