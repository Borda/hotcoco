//! Confusion counts over matched prediction/ground-truth pairs.
//!
//! `sklearn.metrics.confusion_matrix(y_true, y_pred)` assumes every sample has
//! both a true and a predicted label. Detection, tracking, and panoptic don't:
//! a prediction can match nothing (a false positive) and a ground truth can go
//! unpredicted (a false negative). So these functions take **`Option` labels**,
//! and reserve one extra row and column — index `num_classes` — for background.
//!
//! ```
//! use hotcoco::metrics::confusion::confusion_matrix;
//!
//! // Three classes. One correct `0`, one `1` predicted as `2`, one missed `1`,
//! // and one spurious `0`.
//! let gt  = [Some(0), Some(1), Some(1), None];
//! let dt  = [Some(0), Some(2), None,    Some(0)];
//! let m = confusion_matrix(&gt, &dt, 3);
//!
//! let at = |gt_class: usize, dt_class: usize| m[gt_class * 4 + dt_class]; // side = 3 + 1
//! assert_eq!(at(0, 0), 1);   // class 0 predicted correctly
//! assert_eq!(at(1, 2), 1);   // class 1 confused for class 2
//! assert_eq!(at(1, 3), 1);   // class 1 missed    -> background column
//! assert_eq!(at(3, 0), 1);   // spurious class 0  -> background row
//! ```
//!
//! Producing the pairs is the caller's job, and it is the only family-specific
//! step: detection matches boxes by IoU, tracking matches tracks by identity.
//! Once matched, the counting is the same operation.

/// Accumulate a confusion matrix from aligned label pairs.
///
/// `gt_labels` and `dt_labels` are parallel arrays over **match records**, not
/// over predictions. Each entry `i` says what a single matching decision produced:
///
/// | `gt_labels[i]` | `dt_labels[i]` | Meaning | Lands at |
/// |---|---|---|---|
/// | `Some(g)` | `Some(d)` | matched pair (correct when `g == d`) | `[g][d]` |
/// | `Some(g)` | `None` | ground truth with no prediction | `[g][num_classes]` |
/// | `None` | `Some(d)` | prediction matching no ground truth | `[num_classes][d]` |
/// | `None` | `None` | nothing happened | ignored |
///
/// Returns a flat row-major matrix of side `num_classes + 1`, rows indexed by
/// ground truth and columns by prediction. Labels at or beyond `num_classes` are
/// skipped rather than panicking, so an out-of-range class can't take down an
/// evaluation run.
///
/// Counts are `u64` addition, so accumulating per-image and summing gives the same
/// answer as one whole-dataset call — which is what lets callers parallelize over
/// images and reduce afterwards.
pub fn confusion_matrix(
    gt_labels: &[Option<usize>],
    dt_labels: &[Option<usize>],
    num_classes: usize,
) -> Vec<u64> {
    let mut matrix = vec![0u64; (num_classes + 1) * (num_classes + 1)];
    accumulate_confusion(&mut matrix, gt_labels, dt_labels, num_classes);
    matrix
}

/// Add label pairs into an existing confusion matrix.
///
/// The accumulating form of [`confusion_matrix`], for callers folding many batches
/// into one result. Allocating a fresh `k²` matrix per batch and adding it in costs
/// a memset plus a read-modify-write of the whole matrix to record a handful of
/// records — on Objects365 (80k images, 365 classes) that is ~86 GB of memory
/// traffic for ~1.1M records, and it scales with `num_classes²`, not with the data.
///
/// `matrix` must be `(num_classes + 1)²` elements; shorter input is left untouched.
pub fn accumulate_confusion(
    matrix: &mut [u64],
    gt_labels: &[Option<usize>],
    dt_labels: &[Option<usize>],
    num_classes: usize,
) {
    let k = num_classes + 1;
    if matrix.len() < k * k {
        return;
    }

    let n = gt_labels.len().min(dt_labels.len());
    for i in 0..n {
        // Background index is `num_classes`; an out-of-range label is dropped.
        let row = match gt_labels[i] {
            Some(g) if g < num_classes => g,
            Some(_) => continue,
            None => num_classes,
        };
        let col = match dt_labels[i] {
            Some(d) if d < num_classes => d,
            Some(_) => continue,
            None => num_classes,
        };
        if row == num_classes && col == num_classes {
            continue; // neither a prediction nor a ground truth
        }
        matrix[row * k + col] += 1;
    }
}

/// Divide each row of a confusion matrix by its own sum.
///
/// Turns raw counts into "of the ground truths of this class, what fraction were
/// predicted as each class" — the form a confusion heatmap usually plots, because
/// it stays readable when class frequencies differ by orders of magnitude.
///
/// Takes `num_classes`, the same argument [`confusion_matrix`] takes, and adds the
/// background lane itself. Taking the matrix *side* instead would put two
/// conventions for one dimension forty lines apart in the same module, and passing
/// the wrong one would not panic — it would normalize the top-left sub-block and
/// return a short vector of plausible-looking garbage.
///
/// Rows summing to zero stay all-zero rather than producing `NaN`.
pub fn row_normalize(matrix: &[u64], num_classes: usize) -> Vec<f64> {
    let k = num_classes + 1;
    let mut norm = vec![0.0f64; k * k];
    for row in 0..k {
        let row_sum: u64 = (0..k).map(|col| matrix[row * k + col]).sum();
        if row_sum > 0 {
            let denom = row_sum as f64;
            for col in 0..k {
                norm[row * k + col] = matrix[row * k + col] as f64 / denom;
            }
        }
    }
    norm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagonal_counts_correct_predictions() {
        let gt = [Some(0), Some(1), Some(2)];
        let dt = [Some(0), Some(1), Some(2)];
        let m = confusion_matrix(&gt, &dt, 3);
        let k = 4;
        for c in 0..3 {
            assert_eq!(m[c * k + c], 1);
        }
        assert_eq!(m.iter().sum::<u64>(), 3);
    }

    #[test]
    fn unmatched_gt_and_dt_land_in_the_background_lane() {
        let gt = [Some(1), None];
        let dt = [None, Some(2)];
        let m = confusion_matrix(&gt, &dt, 3);
        let at = |gt_class: usize, dt_class: usize| m[gt_class * 4 + dt_class];
        assert_eq!(at(1, 3), 1, "missed GT -> background column");
        assert_eq!(at(3, 2), 1, "spurious DT -> background row");
    }

    #[test]
    fn background_to_background_is_not_counted() {
        let m = confusion_matrix(&[None, None], &[None, None], 3);
        assert_eq!(m.iter().sum::<u64>(), 0);
    }

    #[test]
    fn out_of_range_labels_are_dropped_not_panicking() {
        let m = confusion_matrix(&[Some(99), Some(0)], &[Some(0), Some(99)], 3);
        assert_eq!(m.iter().sum::<u64>(), 0);
    }

    #[test]
    fn per_batch_sums_equal_one_whole_call() {
        let gt = [Some(0), Some(1), None, Some(2)];
        let dt = [Some(1), None, Some(0), Some(2)];

        let whole = confusion_matrix(&gt, &dt, 3);
        let a = confusion_matrix(&gt[..2], &dt[..2], 3);
        let b = confusion_matrix(&gt[2..], &dt[2..], 3);
        let summed: Vec<u64> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();

        assert_eq!(whole, summed, "batching must not change the counts");
    }

    #[test]
    fn row_normalize_makes_rows_sum_to_one() {
        // One real class plus background => side 2. Row 0: 3 and 1 => 0.75 / 0.25.
        // Row 1 empty => stays zero.
        let m = vec![3, 1, 0, 0];
        let n = row_normalize(&m, 1);
        assert!((n[0] - 0.75).abs() < 1e-12);
        assert!((n[1] - 0.25).abs() < 1e-12);
        assert_eq!(n[2], 0.0);
        assert_eq!(n[3], 0.0);
    }

    #[test]
    fn mismatched_lengths_truncate_to_the_shorter() {
        let m = confusion_matrix(&[Some(0), Some(1), Some(2)], &[Some(0)], 3);
        assert_eq!(m.iter().sum::<u64>(), 1);
    }
}
