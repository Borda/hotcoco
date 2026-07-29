//! Confidence calibration — reliability bins, ECE, and MCE.
//!
//! A well-calibrated model is right about 80% of the time when it says 0.8. These
//! functions measure how far from that a model is, given nothing but per-prediction
//! confidences and whether each prediction turned out correct.
//!
//! ```
//! use hotcoco::metrics::calibration::{calibration_curve, calibration_error};
//!
//! // Every prediction claims 0.9 confidence, but only half are right.
//! let scores: Vec<f64> = vec![0.9; 100];
//! let matched: Vec<bool> = (0..100).map(|i| i < 50).collect();
//!
//! let bins = calibration_curve(&scores, &matched, 10);
//! let (ece, mce) = calibration_error(&bins);
//! assert!((ece - 0.4).abs() < 1e-9); // |0.5 accuracy - 0.9 confidence|
//! ```
//!
//! `scores` and `matched` are the same parallel arrays
//! [`counts::average_precision`](crate::metrics::counts::average_precision) takes,
//! and for the same reason: nothing about calibration is detection-specific. A
//! tracking or classification driver produces those two arrays from its own match
//! records and calls the identical function.
//!
//! Detection's adapter is [`COCOeval::calibration`](crate::COCOeval::calibration),
//! which flattens `eval_imgs` into these arrays, adds per-category breakdowns, and
//! records which IoU threshold defined "correct".

use serde::Serialize;

/// Single bin in a calibration analysis.
///
/// Each bin covers an equal-width interval of the `[0, 1]` confidence range.
/// After bucketing predictions by confidence, `avg_confidence` and `avg_accuracy`
/// are the means within the bin. A perfectly calibrated model has them equal in
/// every bin — that diagonal is what a reliability diagram plots.
#[derive(Debug, Clone, Serialize)]
pub struct CalibrationBin {
    /// Lower bound of the confidence interval (inclusive).
    pub bin_lower: f64,
    /// Upper bound of the confidence interval (exclusive, except last bin).
    pub bin_upper: f64,
    /// Mean predicted confidence of predictions in this bin.
    pub avg_confidence: f64,
    /// Fraction of predictions in this bin that are correct.
    pub avg_accuracy: f64,
    /// Number of predictions in this bin.
    pub count: usize,
}

/// Bucket predictions into `n_bins` equal-width confidence bins.
///
/// `scores` and `matched` are parallel arrays over predictions in any order —
/// `scores[i]` is prediction `i`'s confidence in `[0, 1]`, `matched[i]` whether it
/// was correct. Bin membership is `floor(score * n_bins)`, clamped so `score == 1.0`
/// lands in the last bin rather than off the end.
///
/// Callers filter out ignored predictions before calling; there is no ignore mask
/// here because a prediction excluded from calibration should not influence the
/// bin means either.
///
/// Returns an empty vector when `n_bins` is 0. Predictions beyond the shorter of
/// the two arrays are dropped, so mismatched lengths can't panic.
pub fn calibration_curve(scores: &[f64], matched: &[bool], n_bins: usize) -> Vec<CalibrationBin> {
    if n_bins == 0 {
        return Vec::new();
    }

    let mut bins: Vec<CalibrationBin> = (0..n_bins)
        .map(|i| CalibrationBin {
            bin_lower: i as f64 / n_bins as f64,
            bin_upper: (i + 1) as f64 / n_bins as f64,
            avg_confidence: 0.0,
            avg_accuracy: 0.0,
            count: 0,
        })
        .collect();

    let n = scores.len().min(matched.len());
    for i in 0..n {
        let idx = ((scores[i] * n_bins as f64) as usize).min(n_bins - 1);
        bins[idx].avg_confidence += scores[i];
        bins[idx].avg_accuracy += if matched[i] { 1.0 } else { 0.0 };
        bins[idx].count += 1;
    }

    for bin in &mut bins {
        if bin.count > 0 {
            let n = bin.count as f64;
            bin.avg_confidence /= n;
            bin.avg_accuracy /= n;
        }
    }

    bins
}

/// Expected and Maximum Calibration Error over binned predictions.
///
/// Returns `(ece, mce)`:
/// - **ECE** — mean of `|accuracy - confidence|` across bins, weighted by how many
///   predictions each bin holds. The headline number.
/// - **MCE** — the worst single bin's gap, unweighted. Catches a badly calibrated
///   region that ECE averages away.
///
/// The weighting denominator is the total across `bins`, so pass the bins
/// [`calibration_curve`] returned rather than a filtered subset. Empty bins
/// contribute nothing; all-empty input returns `(0.0, 0.0)`.
pub fn calibration_error(bins: &[CalibrationBin]) -> (f64, f64) {
    let total: usize = bins.iter().map(|b| b.count).sum();
    if total == 0 {
        return (0.0, 0.0);
    }

    let mut ece = 0.0;
    let mut mce = 0.0f64;
    for bin in bins {
        if bin.count > 0 {
            let gap = (bin.avg_accuracy - bin.avg_confidence).abs();
            ece += (bin.count as f64 / total as f64) * gap;
            mce = mce.max(gap);
        }
    }
    (ece, mce)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// ECE must equal `netcal.metrics.ECE`, the field's reference implementation.
    ///
    /// Every quantitative test in this module is hand-derived, which checks the
    /// arithmetic but not that hotcoco bins the way everyone else does. An ECE
    /// computed over subtly different bin edges would satisfy all of them and
    /// still not be comparable to a number in a paper.
    ///
    /// The fixture's scores are deliberately lopsided (beta-distributed, bimodal),
    /// because a uniform draw fills every bin about equally and makes the
    /// occupancy weighting unobservable.
    ///
    /// Regenerate with
    /// `uv run --with scikit-learn --with netcal python scripts/gen_metrics_fixtures.py`.
    #[test]
    fn ece_matches_netcal() {
        #[derive(serde::Deserialize)]
        struct Case {
            n_bins: usize,
            style: String,
            scores: Vec<f64>,
            matched: Vec<bool>,
            ece: f64,
        }

        let data = include_str!("testdata/calibration_netcal.json");
        let cases: Vec<Case> = serde_json::from_str(data).expect("parse fixture");
        assert!(cases.len() > 150, "fixture looks truncated");

        let mut worst = 0.0f64;
        for (i, c) in cases.iter().enumerate() {
            let bins = calibration_curve(&c.scores, &c.matched, c.n_bins);
            let (ece, _) = calibration_error(&bins);
            let diff = (ece - c.ece).abs();
            worst = worst.max(diff);
            assert!(
                diff < 1e-12,
                "case {i} ({}, n_bins={}, n={}): ECE {ece} vs netcal {} (diff {diff:.3e})",
                c.style,
                c.n_bins,
                c.scores.len(),
                c.ece
            );
        }
        println!("ECE worst deviation from netcal: {worst:.3e}");
    }

    /// The binning contract, over scores that honour the documented `[0, 1]`
    /// precondition.
    ///
    /// Every existing quantitative test in this module puts all its mass in a
    /// single bin, so the occupancy weighting in [`calibration_error`] is only
    /// exercised here — a bug that ignored `count / total` would reproduce every
    /// hand-computed fixture above and fail only at uneven occupancy.
    #[test]
    fn calibration_binning_contract() {
        let mut rng = StdRng::seed_from_u64(0xCA11B);

        for case in 0..5000 {
            let n_bins = rng.random_range(1..=20);
            let n = rng.random_range(0..=60);

            // Skew the draw so occupancy is lopsided rather than uniform.
            let heavy_low = rng.random_bool(0.5);
            let scores: Vec<f64> = (0..n)
                .map(|_| {
                    if heavy_low && rng.random_bool(0.9) {
                        rng.random_range(0.0..=0.1)
                    } else {
                        rng.random_range(0.0..=1.0)
                    }
                })
                .collect();
            let matched: Vec<bool> = (0..n).map(|_| rng.random_bool(0.5)).collect();

            let bins = calibration_curve(&scores, &matched, n_bins);
            let ctx = format!("case {case}: n_bins={n_bins} n={n}");

            assert_eq!(bins.len(), n_bins, "{ctx}");
            assert_eq!(
                bins.iter().map(|b| b.count).sum::<usize>(),
                n,
                "{ctx}: bin counts do not partition the predictions"
            );

            for (i, b) in bins.iter().enumerate() {
                if b.count == 0 {
                    continue;
                }
                // A bin's mean confidence lies inside the bin. This is what fails
                // if a score is bucketed into the wrong interval.
                assert!(
                    b.avg_confidence >= b.bin_lower - 1e-12
                        && b.avg_confidence <= b.bin_upper + 1e-12,
                    "{ctx}: bin {i} mean confidence {} outside [{}, {}]",
                    b.avg_confidence,
                    b.bin_lower,
                    b.bin_upper
                );
                assert!(
                    (0.0..=1.0).contains(&b.avg_accuracy),
                    "{ctx}: bin {i} accuracy {} outside [0,1]",
                    b.avg_accuracy
                );
            }

            let (ece, mce) = calibration_error(&bins);
            assert!((0.0..=1.0).contains(&ece), "{ctx}: ECE {ece} outside [0,1]");
            assert!((0.0..=1.0).contains(&mce), "{ctx}: MCE {mce} outside [0,1]");
            // ECE is a weighted mean of the per-bin gaps; MCE is their maximum.
            assert!(mce >= ece - 1e-12, "{ctx}: MCE {mce} below ECE {ece}");
        }
    }

    /// Scores outside `[0, 1]` are bucketed into the end bins and carry their raw
    /// value into the bin mean, so `avg_confidence` escapes its own interval and
    /// the resulting ECE can exceed 1.
    ///
    /// `calibration_curve` clamps the bin *index* but never the *score*
    /// (`(score * n_bins) as usize` saturates at 0 for negatives and is capped at
    /// `n_bins - 1` above). `[0, 1]` is a documented precondition, and detection's
    /// adapter passes `score.unwrap_or(0.0)` straight from user JSON — so a model
    /// exporting logits gets a silently meaningless number rather than an error.
    ///
    /// Pinned as known behaviour, not endorsed: see the input-validation item in
    /// `docs/plans/AUDIT-2026-07.md`. Whether to clamp, reject, or keep documenting
    /// it is a live decision; this test exists so the choice is a choice.
    #[test]
    fn out_of_range_scores_escape_their_bin() {
        let bins = calibration_curve(&[5.0, -3.0], &[true, false], 10);

        let last = bins.last().expect("10 bins");
        assert_eq!(last.count, 1, "a score of 5.0 saturates into the last bin");
        assert!(
            last.avg_confidence > last.bin_upper,
            "expected {} to escape the bin upper bound {}",
            last.avg_confidence,
            last.bin_upper
        );

        assert_eq!(bins[0].count, 1, "a negative score saturates into bin 0");
        assert!(
            bins[0].avg_confidence < bins[0].bin_lower,
            "expected {} to fall below the bin lower bound {}",
            bins[0].avg_confidence,
            bins[0].bin_lower
        );

        let (ece, _) = calibration_error(&bins);
        assert!(ece > 1.0, "expected a meaningless ECE above 1.0, got {ece}");
    }

    #[test]
    fn bins_partition_every_prediction() {
        let scores: Vec<f64> = (0..100).map(|i| (i as f64 + 0.5) / 100.0).collect();
        let matched: Vec<bool> = (0..100).map(|i| i % 2 == 0).collect();

        let bins = calibration_curve(&scores, &matched, 10);
        assert_eq!(bins.len(), 10);
        for bin in &bins {
            assert_eq!(bin.count, 10);
        }
        // Every prediction landed in exactly one bin.
        assert_eq!(bins.iter().map(|b| b.count).sum::<usize>(), 100);
    }

    #[test]
    fn confidence_one_lands_in_last_bin_not_off_the_end() {
        let bins = calibration_curve(&[1.0], &[true], 10);
        assert_eq!(bins[9].count, 1);
        assert_eq!(bins.iter().map(|b| b.count).sum::<usize>(), 1);
    }

    #[test]
    fn underconfident_model_has_gap_equal_to_the_shortfall() {
        // Always right, but only claims 0.95.
        let bins = calibration_curve(&[0.95; 100], &[true; 100], 10);
        let (ece, mce) = calibration_error(&bins);
        assert!((ece - 0.05).abs() < 1e-9);
        assert!((mce - 0.05).abs() < 1e-9);
    }

    #[test]
    fn overconfident_model_gap_is_weighted_by_bin_occupancy() {
        // All at 0.9 confidence, half correct => single occupied bin, gap 0.4.
        let matched: Vec<bool> = (0..100).map(|i| i < 50).collect();
        let bins = calibration_curve(&[0.9; 100], &matched, 10);
        let (ece, mce) = calibration_error(&bins);
        assert!((ece - 0.4).abs() < 1e-9);
        assert!((mce - 0.4).abs() < 1e-9);
    }

    #[test]
    fn mce_exceeds_ece_when_a_small_bin_is_badly_off() {
        // 99 well-calibrated predictions at 0.05, plus one wildly overconfident.
        let mut scores = vec![0.05; 99];
        let mut matched = vec![false; 99];
        scores.push(0.95);
        matched.push(false);

        let bins = calibration_curve(&scores, &matched, 10);
        let (ece, mce) = calibration_error(&bins);
        // The lone bad bin dominates MCE but is averaged down in ECE.
        assert!((mce - 0.95).abs() < 1e-9);
        assert!(
            ece < 0.06,
            "ece={ece} should be diluted by the 99 good bins"
        );
    }

    #[test]
    fn empty_input_is_zero_not_a_panic() {
        assert_eq!(
            calibration_error(&calibration_curve(&[], &[], 10)),
            (0.0, 0.0)
        );
        // n_bins = 0 is degenerate but reachable from the public API.
        assert!(calibration_curve(&[0.5], &[true], 0).is_empty());
        assert_eq!(calibration_error(&[]), (0.0, 0.0));
    }

    #[test]
    fn mismatched_array_lengths_truncate_to_the_shorter() {
        let bins = calibration_curve(&[0.9, 0.9, 0.9], &[true], 10);
        assert_eq!(bins.iter().map(|b| b.count).sum::<usize>(), 1);
    }
}
