"""Type stubs for the matching-kernel namespace.

``lsap`` is new surface, so its signature lives here. ``bbox_iou`` and
``mask_iou`` are the ``hotcoco.mask`` kernels under their primitive names; their
signatures are restated rather than aliased, since ``mask`` is stubbed as a class
in ``__init__.pyi`` and its members are not importable names.
"""

from typing import Any, Sequence

import numpy as np
import numpy.typing as npt

def lsap(
    cost: npt.NDArray[np.float64] | Sequence[Sequence[float]], maximize: bool = False
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64]]: ...
def bbox_iou(
    dt: Sequence[Sequence[float]], gt: Sequence[Sequence[float]], iscrowd: Sequence[bool]
) -> npt.NDArray[np.float64]: ...
def mask_iou(dt: Any, gt: Any, iscrowd: Sequence[bool]) -> npt.NDArray[np.float64]: ...

__all__ = ["bbox_iou", "lsap", "mask_iou"]
