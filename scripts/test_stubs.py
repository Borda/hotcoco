"""Verify that __init__.pyi covers every public symbol in the hotcoco module.

Run with: uv run pytest scripts/test_stubs.py -v
"""

from __future__ import annotations

import ast
from pathlib import Path

import hotcoco

STUB_PATH = Path(__file__).resolve().parent.parent / "python" / "hotcoco" / "__init__.pyi"


def _parse_stub_names() -> dict[str, set[str]]:
    """Parse the .pyi file and return {class_name: {method_names}} and top-level names."""
    source = STUB_PATH.read_text()
    tree = ast.parse(source)

    top_level: set[str] = set()
    classes: dict[str, set[str]] = {}

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            top_level.add(node.name)
        elif isinstance(node, ast.ClassDef):
            top_level.add(node.name)
            members: set[str] = set()
            for item in ast.iter_child_nodes(node):
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    members.add(item.name)
                elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    members.add(item.target.id)
            classes[node.name] = members
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    top_level.add(target.id)

    return {"__top__": top_level, **classes}


def _public_names(obj: object) -> set[str]:
    """Get public attribute names (no underscore prefix)."""
    return {name for name in dir(obj) if not name.startswith("_")}


def test_stub_file_exists():
    assert STUB_PATH.exists(), f"Missing stub file: {STUB_PATH}"


def test_py_typed_exists():
    py_typed = STUB_PATH.parent / "py.typed"
    assert py_typed.exists(), "Missing py.typed marker"


def test_top_level_exports_covered():
    """Every public name in hotcoco should appear in the stub."""
    stub_names = _parse_stub_names()["__top__"]
    runtime_names = _public_names(hotcoco)

    # These are re-exports or internal names we don't need to stub.
    # "annotations" is the _Feature object bound by `from __future__ import
    # annotations` in __init__.py — a language directive, not public API.
    skip = {
        "LVIS",
        "LVISeval",
        "LVISEval",
        "LVISResults",
        "CocoDetection",
        "CocoEvaluator",
        "hotcoco",
        "integrations",
        "annotations",
        # Family namespace: it re-exports names already stubbed at the top level
        # and has its own detection.pyi. Covered by the tests below instead.
        "detection",
        # The functional layer. Unlike `detection`, these are *not* re-exported
        # at the top level — `average_precision` and `lsap` exist only under
        # their namespace — so their signatures live in metrics.pyi and
        # primitives.pyi, checked by the tests below.
        "metrics",
        "primitives",
    }
    runtime_names -= skip

    missing = runtime_names - stub_names
    assert not missing, f"Public names missing from stubs: {sorted(missing)}"


def test_coco_methods_covered():
    """Every public method on COCO should appear in the stub."""
    stub_members = _parse_stub_names().get("COCO", set())
    runtime_members = _public_names(hotcoco.COCO)

    missing = runtime_members - stub_members
    assert not missing, f"COCO methods missing from stubs: {sorted(missing)}"


def test_cocoeval_methods_covered():
    """Every public method on COCOeval should appear in the stub."""
    stub_members = _parse_stub_names().get("COCOeval", set())
    runtime_members = _public_names(hotcoco.COCOeval)

    missing = runtime_members - stub_members
    assert not missing, f"COCOeval methods missing from stubs: {sorted(missing)}"


def test_params_attrs_covered():
    """Every public attr on Params should appear in the stub."""
    stub_members = _parse_stub_names().get("Params", set())
    runtime_members = _public_names(hotcoco.Params)

    missing = runtime_members - stub_members
    assert not missing, f"Params attrs missing from stubs: {sorted(missing)}"


def test_mask_methods_covered():
    """Every public function in mask should appear in the stub."""
    stub_members = _parse_stub_names().get("mask", set())
    runtime_members = _public_names(hotcoco.mask)

    missing = runtime_members - stub_members
    assert not missing, f"mask methods missing from stubs: {sorted(missing)}"


def test_hierarchy_methods_covered():
    """Every public method on Hierarchy should appear in the stub."""
    stub_members = _parse_stub_names().get("Hierarchy", set())
    runtime_members = _public_names(hotcoco.Hierarchy)

    missing = runtime_members - stub_members
    assert not missing, f"Hierarchy methods missing from stubs: {sorted(missing)}"


# ---------------------------------------------------------------------------
# Family namespaces
# ---------------------------------------------------------------------------


def test_detection_namespace_importable_both_ways():
    """`import hotcoco.detection` and `from hotcoco import detection` must agree.

    PyO3's `add_submodule` makes a submodule reachable as an *attribute* without
    registering it in `sys.modules`, so the `import x.y` form fails while
    `from x import y` works. That bit `hotcoco.mask`; this test keeps it from
    biting the family namespaces as panoptic, tracking, and concepts land.
    """
    import sys

    import hotcoco.detection
    from hotcoco import detection

    assert "hotcoco.detection" in sys.modules
    assert detection is hotcoco.detection


def test_mask_importable_both_ways():
    """The regression that motivated the check above."""
    import hotcoco.mask
    from hotcoco import mask

    assert mask is hotcoco.mask
    assert hasattr(hotcoco.mask, "iou")


def test_detection_namespace_reexports_are_the_same_objects():
    """The namespace is additive sugar, not a parallel implementation."""
    import hotcoco
    from hotcoco import detection

    for name in detection.__all__:
        assert getattr(detection, name) is getattr(hotcoco, name), f"hotcoco.detection.{name} is not hotcoco.{name}"


def test_detection_stub_matches_runtime():
    stub = ast.parse((STUB_PATH.parent / "detection.pyi").read_text())
    stubbed: set[str] = set()
    for node in ast.iter_child_nodes(stub):
        if isinstance(node, ast.ImportFrom):
            stubbed.update(alias.asname or alias.name for alias in node.names)

    from hotcoco import detection

    missing = set(detection.__all__) - stubbed
    assert not missing, f"Names missing from detection.pyi: {sorted(missing)}"


# ---------------------------------------------------------------------------
# The functional layer
# ---------------------------------------------------------------------------


def _stub_function_names(filename: str) -> set[str]:
    """Top-level `def` names declared in a sibling stub file."""
    tree = ast.parse((STUB_PATH.parent / filename).read_text())
    return {n.name for n in ast.iter_child_nodes(tree) if isinstance(n, ast.FunctionDef)}


def test_functional_layer_importable_both_ways():
    """`import hotcoco.metrics` must work, not only `from hotcoco import metrics`."""
    import sys

    import hotcoco.metrics
    import hotcoco.primitives
    from hotcoco import metrics, primitives

    assert "hotcoco.metrics" in sys.modules
    assert "hotcoco.primitives" in sys.modules
    assert metrics is hotcoco.metrics
    assert primitives is hotcoco.primitives


def test_metrics_stub_matches_runtime():
    from hotcoco import metrics

    missing = set(metrics.__all__) - _stub_function_names("metrics.pyi")
    assert not missing, f"Names missing from metrics.pyi: {sorted(missing)}"


def test_metrics_facade_reexports_every_extension_function():
    """The facade is hand-maintained, so check the *other* direction too.

    `test_metrics_stub_matches_runtime` asserts stub >= __all__. Without this,
    adding a #[pyfunction] to crates/hotcoco-pyo3/src/metrics.rs and forgetting
    the two-line edit in metrics.py leaves it unreachable from Python with the
    whole suite green.
    """
    from hotcoco import metrics
    from hotcoco.hotcoco import metrics as _ext

    exported = {n for n in dir(_ext) if not n.startswith("_")}
    missing = exported - set(metrics.__all__)
    assert not missing, f"In the extension but not re-exported by metrics.py: {sorted(missing)}"


def test_primitives_stub_matches_runtime():
    from hotcoco import primitives

    missing = set(primitives.__all__) - _stub_function_names("primitives.pyi")
    assert not missing, f"Names missing from primitives.pyi: {sorted(missing)}"


def test_functional_layer_needs_no_evaluator():
    """The whole point: metric functions callable on bare arrays.

    If these ever start requiring a COCOeval, the functional layer has collapsed
    back into the god object 1.0 pulled them out of.
    """
    from hotcoco import metrics, primitives

    assert metrics.average_precision([0.9, 0.1], [True, False], num_gt=2) > 0.0
    ece, mce = metrics.calibration_error([0.9] * 10, [True] * 5 + [False] * 5)
    assert ece == mce  # single occupied bin
    assert metrics.confusion_matrix([0, None], [0, 1], num_classes=2).shape == (3, 3)
    assert len(metrics.calibration_curve([0.5], [True], n_bins=4)) == 4
    rows, _ = primitives.lsap([[1.0, 2.0], [3.0, 4.0]])
    assert len(rows) == 2


def test_metric_functions_reject_mismatched_arrays():
    """Parallel arrays of different lengths are a caller bug, not a silent truncation."""
    import pytest
    from hotcoco import metrics

    with pytest.raises(ValueError):
        metrics.average_precision([0.9, 0.8], [True], num_gt=1)
    with pytest.raises(ValueError):
        metrics.calibration_error([0.9, 0.8], [True])
    with pytest.raises(ValueError):
        metrics.confusion_matrix([0, 1], [0], num_classes=2)


def test_lsap_rejects_ragged_and_nan():
    import math

    import pytest
    from hotcoco import primitives

    with pytest.raises(ValueError):
        primitives.lsap([[1.0, 2.0], [3.0]])
    with pytest.raises(ValueError):
        primitives.lsap([[1.0, math.nan], [3.0, 4.0]])


def test_lvis_dropin_matches_lvis_api_spelling():
    """`from lvis import LVISEval` must work — capital E, as lvis-api spells it.

    hotcoco names the class `LVISeval` after pycocotools' `COCOeval`, but
    lvis-api exports `LVISEval`, and that is what Detectron2 and MMDetection
    import. Without the alias, `init_as_lvis()` registered a `lvis` module that
    the canonical import could not use.
    """
    import hotcoco

    hotcoco.init_as_lvis()
    from lvis import LVIS, LVISEval, LVISResults

    assert LVISEval is hotcoco.LVISeval
    assert LVIS is hotcoco.COCO
    assert LVISResults is hotcoco.LVISResults
