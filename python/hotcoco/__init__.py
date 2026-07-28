from __future__ import annotations

from .hotcoco import COCO as _RustCOCO
from .hotcoco import COCOeval, Hierarchy, Params, compare, init_as_lvis, init_as_pycocotools, mask  # noqa: F401


class COCO(_RustCOCO):
    """COCO dataset — extends the Rust core with Python-only methods."""

    def __init__(self, annotation_file=None, *, image_dir=None):  # noqa: ARG002
        # Rust __new__ handles construction and stores image_dir.
        # This __init__ exists only to accept the same kwargs without complaint.
        pass

    def browse(
        self,
        image_dir: str | None = None,
        dt=None,
        iou_type: str = "bbox",
        iou_thr: float = 0.5,
        eval=None,
        slices: dict[str, list[int]] | str | None = None,
        batch_size: int = 12,
        port: int = 7860,
    ):
        """Launch an interactive dataset browser.

        Parameters
        ----------
        image_dir : str, optional
            Root directory for image files. Overrides ``self.image_dir``.
        dt : COCO or str, optional
            Detection results to overlay. Pass a COCO object (from
            ``self.load_res()``) or a path string (auto-loaded).
        iou_type : str
            Evaluation type: ``"bbox"``, ``"segm"``, or ``"keypoints"``
            (default ``"bbox"``). Only used when ``dt`` is provided.
        iou_thr : float
            IoU threshold for TP/FP classification (default 0.5).
        eval : COCOeval, optional
            Pre-computed COCOeval (must have ``evaluate()`` called).
            When provided, ``iou_type`` is ignored.
        slices : dict or str, optional
            Image subsets for sliced browsing. Pass a dict mapping slice
            names to image ID lists, or a path to a JSON file.
        batch_size : int
            Number of images loaded per batch (default 12).
        port : int
            Local server port (default 7860).

        Raises
        ------
        ValueError
            If ``image_dir`` is ``None`` and ``self.image_dir`` is also ``None``.
        ImportError
            If browse dependencies are not installed (``pip install hotcoco[browse]``).
        """
        from . import browse as _browse
        from .server import create_app, run_server, start_server_background

        _browse._require_browse_deps()

        dt_coco = self.load_res(dt) if isinstance(dt, str) else dt

        # Build coco_eval when detections are provided
        coco_eval = None
        if dt_coco is not None:
            if eval is not None:
                coco_eval = eval
            else:
                ev = COCOeval(self, dt_coco, iou_type)
                ev.evaluate()
                coco_eval = ev

        # Load slices from JSON if path given
        resolved_slices = slices
        if isinstance(slices, str):
            import json

            with open(slices) as f:
                resolved_slices = json.load(f)

        app = create_app(
            self,
            image_dir=image_dir,
            batch_size=batch_size,
            dt_coco=dt_coco,
            coco_eval=coco_eval,
            slices=resolved_slices,
        )

        if _browse._is_jupyter():
            actual_port = start_server_background(app, port=port)
            from IPython.display import IFrame, display

            display(IFrame(f"http://127.0.0.1:{actual_port}", width="100%", height=700))
        else:
            run_server(app, port=port, open_browser=True)


class LVISeval:
    """Drop-in replacement for lvis-api LVISEval.

    Returns a ``COCOeval`` instance configured for LVIS federated evaluation
    (``lvis_style=True``). Supports ``run()`` / ``print_results()`` /
    ``get_results()`` as required by Detectron2 and MMDetection.

    Parameters
    ----------
    gt : COCO
        Ground-truth COCO object loaded from an LVIS annotation file.
    dt : COCO
        Detection results COCO object (e.g. from ``gt.load_res(...)``).
    iou_type : str
        One of ``"bbox"``, ``"segm"``, or ``"keypoints"``.
    """

    def __new__(cls, gt, dt, iou_type="segm"):
        return COCOeval(gt, dt, iou_type, lvis_style=True)


# lvis-api spells it `LVISEval`, with a capital E — `from lvis import LVISEval`
# is the canonical import, and it is what Detectron2 and MMDetection write. The
# `LVISeval` spelling above follows pycocotools' `COCOeval`, so both exist: one
# for consistency with the rest of hotcoco, one so `init_as_lvis()` actually
# satisfies the import it promises to.
LVISEval = LVISeval

# lvis-api uses LVIS as the dataset class name, not COCO.
LVIS = COCO


class LVISResults:
    """Drop-in replacement for lvis-api LVISResults.

    ``LVISResults(lvis_gt, predictions, max_dets=300)`` returns a ``COCO``
    object. ``max_dets`` is accepted for API compatibility; detection
    truncation is handled by ``LVISeval`` params (``max_dets=300``).
    """

    def __new__(cls, lvis_gt, results, max_dets=300):  # noqa: ARG003
        return lvis_gt.load_res(results)


import sys as _sys  # noqa: E402

from . import detection, metrics, primitives  # noqa: E402, F401

# `mask` is a PyO3 submodule object, which `from hotcoco import mask` finds as an
# attribute but `import hotcoco.mask` does not — the import system looks in
# sys.modules, and PyO3's add_submodule does not register there. Anyone migrating
# from `import pycocotools.mask` writes the second form, so register it.
# (`init_as_pycocotools()` already does the equivalent for `pycocotools.mask`,
# which is why that path worked while the hotcoco one did not.)
_sys.modules.setdefault("hotcoco.mask", mask)
from .integrations import CocoDetection, CocoEvaluator  # noqa: E402, F401

__all__ = [
    "COCO",
    "COCOeval",
    "CocoDetection",
    "CocoEvaluator",
    "Hierarchy",
    "LVIS",
    "LVISEval",
    "LVISResults",
    "LVISeval",
    "Params",
    "compare",
    "detection",
    "init_as_lvis",
    "init_as_pycocotools",
    "mask",
    "metrics",
    "primitives",
]
