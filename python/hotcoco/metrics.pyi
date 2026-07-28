"""Type stubs for the metric-function namespace.

Unlike ``detection.pyi``, these names are *not* re-exported from the top level —
they exist only under ``hotcoco.metrics`` — so the real signatures live here.
"""

from typing import Any, Optional, Sequence

import numpy as np
import numpy.typing as npt

def average_precision(
    scores: Sequence[float],
    matched: Sequence[bool],
    num_gt: int,
    ignored: Optional[Sequence[bool]] = None,
    rec_thrs: Optional[Sequence[float]] = None,
) -> float: ...
def precision_recall_curve(
    tp_cum: Sequence[float], fp_cum: Sequence[float], num_gt: int, rec_thrs: Optional[Sequence[float]] = None
) -> tuple[float, list[tuple[int, float, int]]]: ...
def calibration_curve(scores: Sequence[float], matched: Sequence[bool], n_bins: int = 10) -> list[dict[str, Any]]: ...
def calibration_error(scores: Sequence[float], matched: Sequence[bool], n_bins: int = 10) -> tuple[float, float]: ...
def confusion_matrix(
    gt: Sequence[Optional[int]], dt: Sequence[Optional[int]], num_classes: int
) -> npt.NDArray[np.uint64]: ...

__all__ = ["average_precision", "calibration_curve", "calibration_error", "confusion_matrix", "precision_recall_curve"]
