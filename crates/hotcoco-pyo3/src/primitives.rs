//! Python bindings for `hotcoco::primitives` — the matching kernels.
//!
//! The similarity kernels (`bbox_iou`, `iou`) already ship under `hotcoco.mask`
//! for pycocotools compatibility; `python/hotcoco/primitives.py` re-exports them
//! here under their kernel names. This module adds the assignment solver, which
//! had no Python surface before.

use numpy::PyArray1;
use pyo3::prelude::*;

use hotcoco_core::primitives::assign;

use crate::convert::f64_matrix_arg;

/// Reject a cost matrix holding a NaN — an unsolvable matrix is an error
/// rather than an arbitrary assignment. `lsap`-specific: whether NaN is
/// meaningful is a property of the assignment problem, not of extracting a
/// 2-D array, so this stays local rather than living in `f64_matrix_arg`.
fn check_no_nan(flat: &[f64]) -> PyResult<()> {
    if flat.iter().any(|v| v.is_nan()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "cost contains NaN; the assignment is undefined",
        ));
    }
    Ok(())
}

#[pyfunction]
#[pyo3(
    signature = (cost, maximize=false),
    text_signature = "(cost, maximize=False)"
)]
#[doc = "Optimal one-to-one assignment on a rectangular cost matrix.

Solves the linear sum assignment problem — the same thing
``scipy.optimize.linear_sum_assignment`` solves, and a semantic port of it, so
results agree including on ties. Use it when greedy matching is not good enough:
tracking associates detections to tracks this way, and it is what HOTA and MOTA
are defined against.

Args:
    cost: 2-D ``numpy.ndarray`` of costs, or a nested sequence (list of
        lists), ``cost[i][j]`` for row ``i`` and column ``j``. Rows need not
        equal columns; the smaller side bounds the assignment. A ``float64``
        array is the fast path — it is read in one pass, any strides; other
        dtypes and nested sequences are converted element by element.
    maximize: Maximize total value instead of minimizing total cost. Pass
        ``True`` when the matrix holds similarities (IoU) rather than costs.

Returns:
    tuple[numpy.ndarray, numpy.ndarray]: ``(row_ind, col_ind)``, the matched
    pairs. ``cost[row_ind[k]][col_ind[k]]`` is the k-th matched entry, and
    ``len(row_ind) == min(n_rows, n_cols)``.

Raises:
    ValueError: If ``cost`` is ragged or holds a NaN — an unsolvable matrix is
        an error rather than an arbitrary assignment.

Example:
    >>> from hotcoco import primitives
    >>> rows, cols = primitives.lsap([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    >>> list(zip(rows.tolist(), cols.tolist()))   # total cost 1 + 2 + 2 = 5
    [(0, 1), (1, 0), (2, 2)]
"]
fn lsap(py: Python<'_>, cost: &Bound<'_, PyAny>, maximize: bool) -> PyResult<Py<PyAny>> {
    let (flat, nr, nc) = f64_matrix_arg(cost, "cost")?;
    check_no_nan(&flat)?;

    let (rows, cols) = assign::lsap(&flat, nr, nc, maximize);
    // `usize` implements `numpy::Element`, so these go straight out — remapping to
    // u64 first would allocate two more Vecs per call, and this is the primitive a
    // tracking loop calls once per frame.
    let rows = PyArray1::from_vec(py, rows);
    let cols = PyArray1::from_vec(py, cols);
    Ok((rows, cols).into_pyobject(py)?.into_any().unbind())
}

/// Build the `hotcoco.primitives` submodule.
pub fn register(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "primitives")?;
    m.add_function(wrap_pyfunction!(lsap, &m)?)?;
    Ok(m)
}
