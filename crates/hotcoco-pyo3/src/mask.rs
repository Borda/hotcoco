use hotcoco_core::mask as rmask;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::convert::{
    bool_vec, extract_flag, f64_array, f64_matrix, numpy_dtype_name, py_to_rle, rle_to_coco_py,
    type_name,
};
use crate::to_pyerr;

/// Transpose between row-major (numpy) and column-major (hotcoco) mask layouts.
///
/// With `(h, w)`: reads row-major `src[y * w + x]` → writes column-major `dst[y + h * x]`.
/// For the reverse direction (column-major → row-major), swap the arguments: call with `(w, h)`.
pub(crate) fn transpose_mask(src: &[u8], h: usize, w: usize) -> Vec<u8> {
    debug_assert_eq!(src.len(), h * w);
    let mut dst = vec![0u8; h * w];
    for y in 0..h {
        for x in 0..w {
            dst[y + h * x] = src[y * w + x];
        }
    }
    dst
}

/// Extract a single RLE dict from a Python object (dict with "size"+"counts"
/// or "h"+"w"+"counts").
fn extract_coco_rle(obj: &Bound<'_, PyAny>) -> PyResult<hotcoco_core::Rle> {
    let dict = obj.cast::<PyDict>()?;
    py_to_rle(dict)
}

/// Extract a list of RLE dicts from a Python object. Accepts either a single
/// dict or a list of dicts.
fn extract_rle_list(obj: &Bound<'_, PyAny>) -> PyResult<Vec<hotcoco_core::Rle>> {
    if let Ok(dict) = obj.cast::<PyDict>() {
        Ok(vec![py_to_rle(dict)?])
    } else {
        let list: Vec<Bound<'_, PyAny>> = obj.extract()?;
        list.iter().map(|item| extract_coco_rle(item)).collect()
    }
}

// ---------------------------------------------------------------------------
// encode
// ---------------------------------------------------------------------------

/// Encode a binary mask to RLE in pycocotools format.
///
/// Parameters
/// ----------
/// mask : numpy.ndarray
///     2-D ``(H, W)`` → returns a single RLE dict.
///     3-D ``(H, W, N)`` → returns a list of *N* RLE dicts.
///     Accepts any memory layout, and the ``uint8``, ``bool``, and ``int8``
///     dtypes.
///
/// Returns
/// -------
/// dict or list[dict]
///     ``{"size": [H, W], "counts": b"..."}`` matching pycocotools.
#[pyfunction]
#[pyo3(text_signature = "(mask)")]
pub fn encode(py: Python<'_>, mask: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let mask = &as_uint8(mask)?;
    let ndim: usize = mask.getattr("ndim")?.extract()?;
    match ndim {
        2 => encode_2d(py, mask),
        3 => encode_3d(py, mask),
        _ => Err(pyo3::exceptions::PyValueError::new_err(
            "mask must be 2-D (H, W) or 3-D (H, W, N)",
        )),
    }
}

/// The `TypeError` for a mask that is not a one-byte numpy array.
///
/// numpy's own extraction failure reads `'ndarray' object is not an instance
/// of 'ndarray'`, which names the same type twice and never mentions the
/// dtype — the one thing the caller has to change. A list, or a torch tensor
/// passed by mistake, has no numpy `dtype` and is named by its type instead.
fn mask_dtype_error(mask: &Bound<'_, PyAny>) -> PyErr {
    let got = numpy_dtype_name(mask).map_or_else(
        || type_name(mask),
        |dtype| format!("{dtype}; cast it with mask.astype(numpy.uint8)"),
    );
    pyo3::exceptions::PyTypeError::new_err(format!(
        "encode(): mask must be a numpy array with dtype uint8, bool, or int8, got {got}"
    ))
}

/// The mask as a `uint8` array object.
///
/// `uint8` passes through. `bool` and `int8` are one byte wide too, so a
/// `view("uint8")` relabels them without a copy — shape and strides carry
/// over, a sliced view stays a view — and the core encoder's own rule, any
/// nonzero byte is foreground, does the rest (`-1` is `255`). The one-byte
/// limit is deliberate: a wider dtype is an error naming the dtype and the
/// cast, so nothing is silently truncated.
fn as_uint8<'py>(mask: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    match numpy_dtype_name(mask).as_deref() {
        Some("uint8") => Ok(mask.clone()),
        Some("bool" | "int8") => mask.call_method1("view", ("uint8",)),
        _ => Err(mask_dtype_error(mask)),
    }
}

/// Column-major (Fortran-order) bytes of one `(H, W)` view, whatever its
/// memory layout.
///
/// The transpose of a Fortran-order view is already standard layout, so
/// `as_standard_layout` borrows and the result is one copy. A C-order or
/// sliced view goes through ndarray's layout conversion first: its strided
/// copy is about 3× faster than iterating the transposed view element by
/// element (350 µs against 1.2 ms for a 480×640 mask), which is why this is
/// not `t.iter().copied().collect()`.
fn col_major(view: numpy::ndarray::ArrayView2<'_, u8>) -> Vec<u8> {
    let t = view.t();
    let std = t.as_standard_layout();
    std.as_slice()
        .map_or_else(|| std.iter().copied().collect(), <[u8]>::to_vec)
}

fn encode_2d(py: Python<'_>, mask: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let arr: PyReadonlyArray2<u8> = mask.extract()?;
    let [h, w] = [arr.shape()[0], arr.shape()[1]];
    let col_major = col_major(arr.as_array());
    // Owned buffer from here on, so the encode itself runs without the GIL —
    // same convention as the COCOeval driver paths.
    let rle = py
        .detach(|| rmask::encode(&col_major, h as u32, w as u32))
        .map_err(to_pyerr)?;
    rle_to_coco_py(py, &rle)
}

fn encode_3d(py: Python<'_>, mask: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let arr: PyReadonlyArray3<u8> = mask.extract()?;
    let [h, w, n] = [arr.shape()[0], arr.shape()[1], arr.shape()[2]];
    // Each slice is converted from the borrowed view; the stack is never
    // copied whole.
    let view = arr.as_array();
    let slices: Vec<Vec<u8>> = (0..n)
        .map(|i| col_major(view.index_axis(numpy::ndarray::Axis(2), i)))
        .collect();
    let rles = py
        .detach(|| {
            slices
                .iter()
                .map(|s| rmask::encode(s, h as u32, w as u32))
                .collect::<Result<Vec<_>, _>>()
        })
        .map_err(to_pyerr)?;
    let list = PyList::empty(py);
    for rle in &rles {
        list.append(rle_to_coco_py(py, rle)?)?;
    }
    Ok(list.into_any().unbind())
}

// ---------------------------------------------------------------------------
// decode
// ---------------------------------------------------------------------------

/// Decode RLE to a binary mask.
///
/// Parameters
/// ----------
/// rle : dict or list[dict]
///     Single RLE dict → ``(H, W)`` uint8 Fortran-order array.
///     List of *N* RLE dicts → ``(H, W, N)`` uint8 Fortran-order array.
///
/// Returns
/// -------
/// numpy.ndarray
#[pyfunction]
#[pyo3(text_signature = "(rle)")]
pub fn decode(py: Python<'_>, rle: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    if let Ok(dict) = rle.cast::<PyDict>() {
        // Single RLE → (H, W) Fortran-order
        let r = py_to_rle(dict)?;
        let col_major = py.detach(|| rmask::decode(&r));
        let h = r.h as usize;
        let w = r.w as usize;
        // col_major is already in Fortran order — create array and set flag
        let flat = PyArray1::from_vec(py, col_major);
        let arr2d = flat.reshape_with_order([h, w], numpy::npyffi::NPY_ORDER::NPY_FORTRANORDER)?;
        Ok(arr2d.into_any().unbind())
    } else {
        // List of RLEs → (H, W, N) Fortran-order
        let list: Vec<Bound<'_, PyAny>> = rle.extract()?;
        if list.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("empty RLE list"));
        }
        let rles: Vec<hotcoco_core::Rle> = list
            .iter()
            .map(|item| extract_coco_rle(item))
            .collect::<PyResult<_>>()?;

        let h = rles[0].h as usize;
        let w = rles[0].w as usize;
        let n = rles.len();

        // Build (H, W, N) Fortran-order: for each slice, decode gives
        // column-major data. In Fortran order for 3D, axis 0 varies fastest,
        // so memory layout is: all (h*w) of slice 0, then slice 1, etc.
        let data = py.detach(|| {
            let mut data = Vec::with_capacity(h * w * n);
            for r in &rles {
                data.extend_from_slice(&rmask::decode(r));
            }
            data
        });
        let flat = PyArray1::from_vec(py, data);
        let arr3d =
            flat.reshape_with_order([h, w, n], numpy::npyffi::NPY_ORDER::NPY_FORTRANORDER)?;
        Ok(arr3d.into_any().unbind())
    }
}

// ---------------------------------------------------------------------------
// area
// ---------------------------------------------------------------------------

/// Compute the area (number of foreground pixels) of RLE mask(s).
///
/// Parameters
/// ----------
/// rle : dict or list[dict]
///     Single RLE dict → scalar uint64.
///     List of RLE dicts → numpy uint32 array, matching pycocotools.
#[pyfunction]
#[pyo3(text_signature = "(rle)")]
pub fn area(py: Python<'_>, rle: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    if let Ok(dict) = rle.cast::<PyDict>() {
        let r = py_to_rle(dict)?;
        let a = rmask::area(&r);
        Ok(a.into_pyobject(py)?.into_any().unbind())
    } else {
        let rles = extract_rle_list(rle)?;
        // `uint32`, matching pycocotools' array dtype exactly — the parity
        // suite checks dtypes, not just values. (The scalar path above hands
        // back a Python int, which has no dtype to match.)
        let areas: Vec<u32> = rles.iter().map(|r| rmask::area(r) as u32).collect();
        let arr = PyArray1::from_vec(py, areas);
        Ok(arr.into_any().unbind())
    }
}

// ---------------------------------------------------------------------------
// to_bbox / toBbox
// ---------------------------------------------------------------------------

/// Compute bounding box(es) from RLE mask(s).
///
/// Parameters
/// ----------
/// rle : dict or list[dict]
///     Single RLE dict → numpy float64(4,).
///     List of RLE dicts → numpy float64(N, 4).
#[pyfunction]
#[pyo3(text_signature = "(rle)")]
pub fn to_bbox(py: Python<'_>, rle: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    if let Ok(dict) = rle.cast::<PyDict>() {
        let r = py_to_rle(dict)?;
        let bb = rmask::to_bbox(&r);
        let arr = PyArray1::from_vec(py, bb.to_vec());
        Ok(arr.into_any().unbind())
    } else {
        let rles = extract_rle_list(rle)?;
        let n = rles.len();
        let mut data = Vec::with_capacity(n * 4);
        for r in &rles {
            data.extend_from_slice(&rmask::to_bbox(r));
        }
        f64_array(py, data, [n, 4])
    }
}

/// Alias for `to_bbox` matching pycocotools naming.
#[pyfunction]
#[pyo3(name = "toBbox")]
pub fn to_bbox_camel(py: Python<'_>, rle: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    to_bbox(py, rle)
}

// ---------------------------------------------------------------------------
// merge
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (rles, intersect = false))]
pub fn merge(py: Python<'_>, rles: &Bound<'_, PyAny>, intersect: bool) -> PyResult<Py<PyAny>> {
    let rle_vec = extract_rle_list(rles)?;
    let result = rmask::merge(&rle_vec, intersect).map_err(to_pyerr)?;
    rle_to_coco_py(py, &result)
}

fn check_iscrowd_len(iscrowd_len: usize, gt_len: usize) -> PyResult<()> {
    crate::convert::check_parallel(iscrowd_len, gt_len, "iscrowd", "gt")
}

/// Extract `iscrowd` from bools, ints, or a numpy array of either.
///
/// COCO JSON stores `iscrowd` as `0`/`1` integers, and pycocotools takes them
/// straight through — `maskUtils.iou(dt, gt, [a["iscrowd"] for a in anns])` is the
/// idiomatic call, as is passing a numpy array. A `Vec<bool>` parameter rejects
/// both with a `TypeError`, which breaks the drop-in claim on a public API for a
/// reason the caller cannot guess from the error.
///
/// The crate already accepts either spelling when deserializing annotations (see
/// `types::deserialize_iscrowd`); this is the same convention at the Python edge.
fn extract_iscrowd(obj: &Bound<'_, PyAny>) -> PyResult<Vec<bool>> {
    // `bool_vec` covers bools (list or numpy bool array) with the same fast
    // path COCOeval's own bool arguments get. Anything else goes through the
    // one flag reader, element by element.
    if let Ok(v) = bool_vec(obj, "iscrowd") {
        return Ok(v);
    }
    obj.try_iter()?.map(|item| extract_flag(&item?)).collect()
}

// ---------------------------------------------------------------------------
// iou
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(text_signature = "(dt, gt, iscrowd)")]
pub fn iou(
    py: Python<'_>,
    dt: &Bound<'_, PyAny>,
    gt: &Bound<'_, PyAny>,
    iscrowd: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let dt_rles = extract_rle_list(dt)?;
    let gt_rles = extract_rle_list(gt)?;
    let iscrowd = extract_iscrowd(iscrowd)?;
    check_iscrowd_len(iscrowd.len(), gt_rles.len())?;
    // All inputs are owned by now; the O(D*G) kernel runs GIL-free like the
    // COCOeval paths.
    let result = py.detach(|| rmask::iou(&dt_rles, &gt_rles, &iscrowd));
    let d = dt_rles.len();
    let g = gt_rles.len();
    f64_matrix(py, &result, [d, g])
}

#[pyfunction]
#[pyo3(text_signature = "(dt, gt, iscrowd)")]
pub fn bbox_iou(
    py: Python<'_>,
    dt: Vec<[f64; 4]>,
    gt: Vec<[f64; 4]>,
    iscrowd: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let iscrowd = extract_iscrowd(iscrowd)?;
    check_iscrowd_len(iscrowd.len(), gt.len())?;
    let result = rmask::bbox_iou(&dt, &gt, &iscrowd);
    let d = dt.len();
    let g = gt.len();
    f64_matrix(py, &result, [d, g])
}

// ---------------------------------------------------------------------------
// fr_poly / frPoly
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(text_signature = "(xy, h, w)")]
pub fn fr_poly(py: Python<'_>, xy: Vec<f64>, h: u32, w: u32) -> PyResult<Py<PyAny>> {
    let rle = rmask::fr_poly(&xy, h, w).map_err(to_pyerr)?;
    rle_to_coco_py(py, &rle)
}

/// Alias for `fr_poly` matching pycocotools naming.
#[pyfunction]
#[pyo3(name = "frPoly")]
pub fn fr_poly_camel(py: Python<'_>, xy: Vec<f64>, h: u32, w: u32) -> PyResult<Py<PyAny>> {
    fr_poly(py, xy, h, w)
}

// ---------------------------------------------------------------------------
// fr_bbox / frBbox
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(text_signature = "(bb, h, w)")]
pub fn fr_bbox(py: Python<'_>, bb: [f64; 4], h: u32, w: u32) -> PyResult<Py<PyAny>> {
    let rle = rmask::fr_bbox(&bb, h, w).map_err(to_pyerr)?;
    rle_to_coco_py(py, &rle)
}

/// Alias for `fr_bbox` matching pycocotools naming.
#[pyfunction]
#[pyo3(name = "frBbox")]
pub fn fr_bbox_camel(py: Python<'_>, bb: [f64; 4], h: u32, w: u32) -> PyResult<Py<PyAny>> {
    fr_bbox(py, bb, h, w)
}

// ---------------------------------------------------------------------------
// rle_to_string / rle_from_string
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(text_signature = "(rle)")]
pub fn rle_to_string(rle: &Bound<'_, PyDict>) -> PyResult<String> {
    let rle = py_to_rle(rle)?;
    Ok(rmask::rle_to_string(&rle))
}

#[pyfunction]
#[pyo3(text_signature = "(s, h, w)")]
pub fn rle_from_string(py: Python<'_>, s: &str, h: u32, w: u32) -> PyResult<Py<PyAny>> {
    let rle = rmask::rle_from_string(s, h, w)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    rle_to_coco_py(py, &rle)
}

// ---------------------------------------------------------------------------
// frPyObjects / fr_py_objects
// ---------------------------------------------------------------------------

/// Encode segmentation objects to RLEs (pycocotools compatibility).
///
/// Parameters
/// ----------
/// seg : list[list[float]] | numpy.ndarray | dict | list[dict]
///     - List (or 2-D array) of boxes ``[x, y, w, h]`` → list of RLE dicts.
///     - List of flattened polygons ``[x1, y1, x2, y2, ...]`` → list of RLE dicts.
///       Entries of exactly 4 values are boxes, more than 4 are polygons —
///       the same length-based dispatch pycocotools uses.
///     - Single RLE dict (compressed or uncompressed) → one RLE dict.
///     - List of uncompressed RLE dicts → list of RLE dicts.
/// h : int
///     Image height.
/// w : int
///     Image width.
///
/// Returns
/// -------
/// dict or list[dict]
///     A single RLE dict when ``seg`` is a dict, a list of RLE dicts when it
///     is a list — dict in, dict out, exactly as pycocotools does it.
#[pyfunction]
#[pyo3(name = "frPyObjects", text_signature = "(seg, h, w)")]
pub fn fr_py_objects(
    py: Python<'_>,
    seg: &Bound<'_, PyAny>,
    h: u32,
    w: u32,
) -> PyResult<Py<PyAny>> {
    // Case 1: single dict (uncompressed or compressed RLE) → single dict
    if let Ok(dict) = seg.cast::<PyDict>() {
        let rle = py_to_rle(dict)?;
        return rle_to_coco_py(py, &rle);
    }

    // Case 2: list — could be list of polygons or list of dicts
    let items: Vec<Bound<'_, PyAny>> = seg.extract()?;
    if items.is_empty() {
        return Ok(PyList::empty(py).into_any().unbind());
    }

    // Check first element to determine type
    let first = &items[0];
    if first.cast::<PyDict>().is_ok() {
        // List of RLE dicts
        let list = PyList::empty(py);
        for item in &items {
            let rle = extract_coco_rle(item)?;
            list.append(rle_to_coco_py(py, &rle)?)?;
        }
        Ok(list.into_any().unbind())
    } else {
        // List (or ndarray) of coordinate sequences, dispatched on entry length like
        // pycocotools: exactly 4 is a `[x, y, w, h]` box, more than 4 a flattened
        // polygon. `fr_poly` returns an empty RLE below three points, so a box must
        // not reach it.
        //
        // Deliberate deviation: pycocotools' box path requires a numpy array and
        // raises `TypeError` on a list of lists. Accepting both is strictly more
        // permissive.
        let list = PyList::empty(py);
        for item in &items {
            let coords: Vec<f64> = item.extract()?;
            let rle = match coords.len() {
                4 => rmask::fr_bbox(&[coords[0], coords[1], coords[2], coords[3]], h, w)
                    .map_err(to_pyerr)?,
                n if n > 4 => rmask::fr_poly(&coords, h, w).map_err(to_pyerr)?,
                n => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "frPyObjects: each entry must be a box [x, y, w, h] (4 values) \
                         or a flattened polygon [x1, y1, x2, y2, ...] (more than 4, \
                         even count); got {n}"
                    )));
                }
            };
            list.append(rle_to_coco_py(py, &rle)?)?;
        }
        Ok(list.into_any().unbind())
    }
}

/// Snake-case alias for `frPyObjects`.
///
/// The explicit `name` is load-bearing: PyO3 falls back to the Rust identifier,
/// so without it the snake_case spelling this alias exists to provide is the one
/// spelling not reachable from Python.
#[pyfunction]
#[pyo3(name = "fr_py_objects", text_signature = "(seg, h, w)")]
pub fn fr_py_objects_snake(
    py: Python<'_>,
    seg: &Bound<'_, PyAny>,
    h: u32,
    w: u32,
) -> PyResult<Py<PyAny>> {
    fr_py_objects(py, seg, h, w)
}
