//! Bootstrap confidence intervals over any resampled statistic.
//!
//! "Model B scores 1.2 AP higher" is not a result until you know whether it
//! survives resampling the evaluation set. [`bootstrap_ci`] answers that for any
//! statistic you can recompute on a subset of units — images for detection,
//! sequences for tracking — by taking the statistic itself as a closure.
//!
//! ```
//! use hotcoco::metrics::bootstrap::bootstrap_ci;
//!
//! let units: Vec<u32> = (0..100).collect();
//! // A toy statistic: the fraction of sampled units below 50.
//! let cis = bootstrap_ci(&units, 200, 0xC0C0, 0.95, |sample| {
//!     let hits = sample.iter().filter(|&&u| u < 50).count();
//!     vec![hits as f64 / sample.len() as f64]
//! });
//!
//! assert_eq!(cis.len(), 1);
//! assert!(cis[0].lower < 0.5 && cis[0].upper > 0.5);
//! ```
//!
//! Nothing here knows what a detection is. Detection's adapter is
//! [`compare`](crate::detection::compare), whose closure re-accumulates two
//! evaluators on the sampled images and returns their metric deltas.
//!
//! # Sampling convention
//!
//! Each sample draws `n_units` indices **with replacement**, then deduplicates
//! them into a set — so a sample holds roughly 63% of the units, not all of them.
//! That is deliberate and standard for detection: the accumulators treat a unit as
//! present or absent rather than weighted, so a repeated draw can't count twice.
//! Sampling is seeded per-sample (`seed + i`), which makes results reproducible and
//! order-independent under parallel execution.

use std::collections::HashSet;

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use serde::Serialize;

/// Bootstrap confidence interval for a single statistic.
#[derive(Debug, Clone, Serialize)]
pub struct BootstrapCI {
    /// Lower percentile bound.
    pub lower: f64,
    /// Upper percentile bound.
    pub upper: f64,
    /// Confidence level these bounds were computed at (e.g. 0.95).
    pub confidence: f64,
    /// Fraction of bootstrap samples in which the statistic was positive.
    ///
    /// For a delta between two models this reads as "how often B beat A" — often
    /// more useful than the interval, because it stays interpretable when the
    /// interval straddles zero.
    pub prob_positive: f64,
    /// Standard deviation across bootstrap samples.
    pub std_err: f64,
}

/// Percentile confidence intervals for a vector-valued statistic.
///
/// Draws `n_samples` bootstrap samples from `units` and calls `statistic` on each.
/// Every call must return a vector of the same length — one entry per quantity
/// being measured — and the result holds one [`BootstrapCI`] per entry, in the
/// same order.
///
/// `statistic` receives the sampled units themselves, deduplicated. Generic over
/// the unit type rather than handing back indices: an index set would force every
/// caller to build a *second* set to map indices onto its own units, which for
/// detection meant ~0.63·n extra hash inserts per sample on top of the ones this
/// function already paid.
///
/// Bounds are raw percentiles: `floor(α/2 · n)` and `ceil((1-α/2) · n)` into the
/// sorted samples, clamped to the last index. No BCa correction — the intervals
/// are readable as "the middle 95% of what resampling produced", not as a
/// bias-corrected estimator.
///
/// `statistic` is called from multiple threads, hence the `Sync` bound. Returns an
/// empty vector when `n_samples` is 0 or `units` is empty.
pub fn bootstrap_ci<T, F>(
    units: &[T],
    n_samples: usize,
    seed: u64,
    confidence: f64,
    statistic: F,
) -> Vec<BootstrapCI>
where
    T: Copy + Eq + std::hash::Hash + Sync,
    F: Fn(&HashSet<T>) -> Vec<f64> + Sync,
{
    let n_units = units.len();
    if n_samples == 0 || n_units == 0 {
        return Vec::new();
    }

    let all_samples: Vec<Vec<f64>> = (0..n_samples)
        .into_par_iter()
        .map(|i| {
            // Seeded per sample, so the draw does not depend on thread scheduling.
            let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(i as u64));
            let sample: HashSet<T> = (0..n_units)
                .map(|_| units[rng.random_range(0..n_units)])
                .collect();
            statistic(&sample)
        })
        .collect();

    let num_stats = all_samples.first().map_or(0, Vec::len);
    let alpha = 1.0 - confidence;
    let nb = n_samples;

    (0..num_stats)
        .map(|m| {
            let mut samples: Vec<f64> = all_samples.iter().map(|s| s[m]).collect();
            samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let lo_idx = ((alpha / 2.0) * nb as f64).floor() as usize;
            let hi_idx = ((1.0 - alpha / 2.0) * nb as f64).ceil() as usize;

            let lower = samples[lo_idx.min(nb - 1)];
            let upper = samples[hi_idx.min(nb - 1)];

            let pos_count = samples.iter().filter(|&&x| x > 0.0).count();
            let prob_positive = pos_count as f64 / nb as f64;

            let mean: f64 = samples.iter().sum::<f64>() / nb as f64;
            let variance = if nb > 1 {
                samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (nb - 1) as f64
            } else {
                0.0
            };

            BootstrapCI {
                lower,
                upper,
                confidence,
                prob_positive,
                std_err: variance.sqrt(),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Indices standing in for units, when the test only cares about the sampling.
    fn units(n: usize) -> Vec<usize> {
        (0..n).collect()
    }

    /// Sample mean of a fixed population — the interval must bracket the truth.
    #[test]
    fn interval_brackets_the_population_mean() {
        let values: Vec<f64> = (0..200).map(|i| i as f64).collect();
        let truth = values.iter().sum::<f64>() / values.len() as f64;

        let cis = bootstrap_ci(&units(values.len()), 500, 42, 0.95, |sample| {
            let sum: f64 = sample.iter().map(|&i| values[i]).sum();
            vec![sum / sample.len() as f64]
        });

        assert_eq!(cis.len(), 1);
        assert!(
            cis[0].lower <= truth && truth <= cis[0].upper,
            "truth {truth} outside [{}, {}]",
            cis[0].lower,
            cis[0].upper
        );
    }

    /// The statistic sees the caller's own units, not positions into them.
    #[test]
    fn statistic_receives_the_units_themselves() {
        let image_ids: Vec<u64> = vec![101, 202, 303, 404];
        let cis = bootstrap_ci(&image_ids, 20, 1, 0.95, |sample| {
            assert!(sample.iter().all(|id| image_ids.contains(id)));
            vec![sample.len() as f64]
        });
        assert_eq!(cis.len(), 1);
    }

    #[test]
    fn same_seed_reproduces_the_same_interval() {
        let stat = |sample: &HashSet<usize>| vec![sample.len() as f64];
        let a = bootstrap_ci(&units(100), 50, 7, 0.9, stat);
        let b = bootstrap_ci(&units(100), 50, 7, 0.9, stat);
        assert_eq!(a[0].lower, b[0].lower);
        assert_eq!(a[0].upper, b[0].upper);
        assert_eq!(a[0].std_err, b[0].std_err);
    }

    #[test]
    fn different_seed_gives_a_different_draw() {
        let stat = |sample: &HashSet<usize>| vec![sample.len() as f64];
        let a = bootstrap_ci(&units(1000), 50, 1, 0.9, stat);
        let b = bootstrap_ci(&units(1000), 50, 2, 0.9, stat);
        // Asserting only that variance exists would pass with identical seeds.
        assert_ne!(a[0].std_err, b[0].std_err);
    }

    /// With-replacement draws deduplicated: ~1 - 1/e of units per sample.
    #[test]
    fn sample_holds_roughly_63_percent_of_units() {
        let cis = bootstrap_ci(&units(10_000), 20, 99, 0.95, |sample| {
            vec![sample.len() as f64 / 10_000.0]
        });
        assert!(
            cis[0].lower > 0.60 && cis[0].upper < 0.66,
            "expected ~0.632, got [{}, {}]",
            cis[0].lower,
            cis[0].upper
        );
    }

    #[test]
    fn always_positive_statistic_reports_probability_one() {
        let cis = bootstrap_ci(&units(50), 100, 3, 0.95, |_| vec![1.0]);
        assert_eq!(cis[0].prob_positive, 1.0);
        assert_eq!(cis[0].std_err, 0.0);
        assert_eq!(cis[0].lower, 1.0);
    }

    #[test]
    fn vector_statistic_yields_one_interval_per_entry() {
        let cis = bootstrap_ci(&units(50), 30, 5, 0.95, |s| {
            vec![s.len() as f64, -(s.len() as f64), 0.0]
        });
        assert_eq!(cis.len(), 3);
        assert_eq!(cis[1].prob_positive, 0.0);
        assert_eq!(cis[2].prob_positive, 0.0);
    }

    #[test]
    fn degenerate_inputs_return_empty_not_a_panic() {
        assert!(bootstrap_ci(&units(0), 10, 1, 0.95, |_| vec![1.0]).is_empty());
        assert!(bootstrap_ci(&units(10), 0, 1, 0.95, |_| vec![1.0]).is_empty());
    }
}
