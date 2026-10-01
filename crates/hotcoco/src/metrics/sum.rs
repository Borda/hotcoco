//! Floating-point summation in numpy's order.
//!
//! Float addition is not associative, so two sums of the same values agree to the
//! last bit only when they add in the same order. pycocotools forms every summary
//! statistic with `np.mean`, which sums by pairwise reduction rather than left to
//! right; [`pairwise_sum`] reproduces that order so hotcoco's means are
//! bit-identical to the reference, not merely within rounding of it.

/// Below this length numpy adds serially.
const PW_UNROLL: usize = 8;

/// Up to this length numpy adds with [`PW_UNROLL`] interleaved accumulators;
/// above it, it splits the slice in two and recurses (numpy's `PW_BLOCKSIZE`).
const PW_BLOCKSIZE: usize = 128;

/// Sum `values` in the order numpy's `np.add.reduce` uses for a contiguous `f64`
/// array, and therefore the order `np.sum` and `np.mean` use.
///
/// A port of `pairwise_sum` in numpy's `loops_utils.h.src`:
///
/// - Fewer than 8 values: a plain left-to-right loop.
/// - Up to 128 values: eight accumulators over stride-8 lanes, combined as
///   `((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))`, then the tail of fewer
///   than 8 values added serially.
/// - More than 128 values: split at half the length rounded down to a multiple
///   of 8, and sum each half the same way.
///
/// To match `np.mean(x)` bit for bit, pass the elements in the order numpy
/// flattens `x` — C order — and divide by the length.
///
/// An empty slice sums to `0.0`.
pub(crate) fn pairwise_sum(values: &[f64]) -> f64 {
    let n = values.len();
    if n < PW_UNROLL {
        values.iter().fold(0.0, |s, &v| s + v)
    } else if n <= PW_BLOCKSIZE {
        let mut r = [0.0f64; PW_UNROLL];
        r.copy_from_slice(&values[..PW_UNROLL]);
        let blocks_end = n - n % PW_UNROLL;
        for block in values[PW_UNROLL..blocks_end].chunks_exact(PW_UNROLL) {
            for (acc, &v) in r.iter_mut().zip(block) {
                *acc += v;
            }
        }
        let head = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        values[blocks_end..].iter().fold(head, |s, &v| s + v)
    } else {
        let half = n / 2;
        let split = half - half % PW_UNROLL;
        pairwise_sum(&values[..split]) + pairwise_sum(&values[split..])
    }
}

#[cfg(test)]
mod tests {
    use super::pairwise_sum;

    /// A sequence whose serial and pairwise sums differ: a large value followed
    /// by many values too small to register against it one at a time.
    fn lopsided(n: usize) -> Vec<f64> {
        let mut v = vec![1e-16; n];
        v[0] = 1.0;
        v
    }

    fn serial(v: &[f64]) -> f64 {
        v.iter().fold(0.0, |s, &x| s + x)
    }

    #[test]
    fn empty_is_zero() {
        assert_eq!(pairwise_sum(&[]), 0.0);
    }

    #[test]
    fn below_unroll_is_serial() {
        // n = 7: numpy's plain loop, so pairwise and serial agree exactly.
        let v = lopsided(7);
        assert_eq!(pairwise_sum(&v), serial(&v));
        assert_eq!(pairwise_sum(&v), 1.0);
    }

    #[test]
    fn exactly_one_block_of_eight() {
        // Lanes hold one value each: 1, then seven 1e-16. The tree adds the six
        // small lanes pairwise first, (1e-16 + 1e-16) twice and so on, so they
        // reach 1.0 as 6e-16 plus the 1e-16 paired with it — enough to round up.
        let v = lopsided(8);
        let expected = ((1.0 + 1e-16) + (1e-16 + 1e-16)) + ((1e-16 + 1e-16) + (1e-16 + 1e-16));
        assert_eq!(pairwise_sum(&v), expected);
        assert_eq!(serial(&v), 1.0);
        assert_ne!(pairwise_sum(&v), serial(&v));
    }

    #[test]
    fn nine_adds_the_tail_serially() {
        let v = lopsided(9);
        let head = ((1.0 + 1e-16) + (1e-16 + 1e-16)) + ((1e-16 + 1e-16) + (1e-16 + 1e-16));
        assert_eq!(pairwise_sum(&v), head + 1e-16);
        assert_ne!(pairwise_sum(&v), serial(&v));
    }

    #[test]
    fn full_block_of_128_uses_strided_lanes() {
        // Lane j holds values j, j + 8, ..., j + 120: 16 values each.
        let v = lopsided(128);
        let small_lane = (0..16).fold(0.0, |s, _| s + 1e-16);
        let lane0 = (0..15).fold(1.0, |s, _| s + 1e-16);
        let expected = ((lane0 + small_lane) + (small_lane + small_lane))
            + ((small_lane + small_lane) + (small_lane + small_lane));
        assert_eq!(pairwise_sum(&v), expected);
        assert_ne!(pairwise_sum(&v), serial(&v));
    }

    #[test]
    fn above_blocksize_splits_at_a_multiple_of_eight() {
        // n = 129: half is 64, already a multiple of 8, so 64 + 65.
        let v = lopsided(129);
        assert_eq!(
            pairwise_sum(&v),
            pairwise_sum(&v[..64]) + pairwise_sum(&v[64..])
        );
        assert_ne!(pairwise_sum(&v), serial(&v));
        // n = 1000: half is 500, rounded down to 496; each half recurses again.
        let v = lopsided(1000);
        let left = pairwise_sum(&v[..248]) + pairwise_sum(&v[248..496]);
        let right = pairwise_sum(&v[496..748]) + pairwise_sum(&v[748..]);
        assert_eq!(pairwise_sum(&v), left + right);
        assert_ne!(pairwise_sum(&v), serial(&v));
    }

    /// Bit patterns of `np.sum(np.linspace(1, n, n) / 7.0)`, recorded with
    /// numpy 2.4.3 on arm64. Each term is a correctly rounded division, so Rust
    /// builds the same inputs; for n = 8, 9, and 128 the serial sum lands on a
    /// different bit pattern. To regenerate:
    ///
    /// ```text
    /// uv run python -c "import numpy as np
    /// for n in (7, 8, 9, 128, 129, 1000):
    ///     print(n, hex(np.sum(np.linspace(1, n, n) / 7.0).view(np.uint64)))"
    /// ```
    #[test]
    fn matches_numpy_on_sevenths() {
        for &(n, bits) in NUMPY_SEVENTHS {
            let v: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) / 7.0).collect();
            assert_eq!(pairwise_sum(&v).to_bits(), bits, "n = {n}");
        }
    }

    const NUMPY_SEVENTHS: &[(usize, u64)] = &[
        (7, 0x4010000000000000),
        (8, 0x4014924924924925),
        (9, 0x4019b6db6db6db6e),
        (128, 0x40926db6db6db6dc),
        (129, 0x4092b76db6db6db6),
        (1000, 0x40f174c000000000),
    ];
}
