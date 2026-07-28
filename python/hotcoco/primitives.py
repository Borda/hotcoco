"""Matching kernels — deciding what pairs with what.

The layer underneath :mod:`hotcoco.metrics`: similarity between two sets, and the
assignment that turns similarity into pairs::

    from hotcoco import primitives

    sim = primitives.bbox_iou(detections, ground_truths, iscrowd)
    rows, cols = primitives.lsap(sim, maximize=True)

Nothing here scores. Feed the pairs to :mod:`hotcoco.metrics` for that.

The IoU kernels are the same functions ``hotcoco.mask`` exposes — re-exported
under their kernel names. ``hotcoco.mask`` mirrors ``pycocotools.mask`` and is a
permanent compatibility surface; this module is where they live as primitives.
Both call one implementation, so they cannot disagree.

Greedy matching (COCO's rank-ordered assignment) has no binding yet. Its Rust
signature carries pycocotools' crowd and ignore semantics, and exposing that
faithfully needs a Python-facing shape designed on purpose rather than
transliterated — it lands in a 1.x minor. ``COCOeval.evaluate()`` uses it today.

Stability
---------

Provisional through the 1.x series, like :mod:`hotcoco.metrics`. The IoU kernels
are the exception: those are frozen, since ``pycocotools`` parity depends on them.
"""

from __future__ import annotations

from .hotcoco import mask as _mask
from .hotcoco import primitives as _primitives

lsap = _primitives.lsap

#: Pairwise IoU between two sets of boxes, shape ``(len(dt), len(gt))``.
bbox_iou = _mask.bbox_iou
#: Pairwise IoU between two sets of RLE masks.
mask_iou = _mask.iou

__all__ = ["bbox_iou", "lsap", "mask_iou"]
