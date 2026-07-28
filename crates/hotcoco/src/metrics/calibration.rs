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
