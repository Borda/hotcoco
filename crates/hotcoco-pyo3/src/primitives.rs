//! Python bindings for `hotcoco::primitives` — the matching kernels.
//!
//! The similarity kernels (`bbox_iou`, `iou`) already ship under `hotcoco.mask`
//! for pycocotools compatibility; `python/hotcoco/primitives.py` re-exports them
//! here under their kernel names. This module adds the assignment solver, which
//! had no Python surface before.

use numpy::PyArray1;
use pyo3::prelude::*;

use hotcoco_core::primitives::assign;

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
    cost: 2D sequence of costs, ``cost[i][j]`` for row ``i`` and column ``j``.
        Rows need not equal columns; the smaller side bounds the assignment.
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
fn lsap(py: Python<'_>, cost: Vec<Vec<f64>>, maximize: bool) -> PyResult<Py<PyAny>> {
    let nr = cost.len();
    let nc = cost.first().map_or(0, Vec::len);

    let mut flat = Vec::with_capacity(nr * nc);
    for (i, row) in cost.iter().enumerate() {
        if row.len() != nc {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "cost must be rectangular; row 0 has {nc} columns but row {i} has {}",
                row.len()
            )));
        }
        for &v in row {
            if v.is_nan() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "cost contains NaN; the assignment is undefined",
                ));
            }
        }
        flat.extend_from_slice(row);
    }

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
