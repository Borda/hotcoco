"""PlotData: validated data extraction from a COCOeval object."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import _mask_invalid_prec


@dataclass
class PlotData:
    """Extracted, validated data from a COCOeval object.

    Built via :meth:`from_coco_eval`. All plot functions consume this
    instead of reaching into COCOeval internals directly.
    """

    eval_mode: str  # "coco" | "lvis" | "openimages"
    iou_type: str  # "bbox" | "segm" | "keypoints"
    provenance: str  # "parity_verified" | "extension" — read from Rust, never re-derived
    is_benchmark_standard: bool  # Rust's default-deny predicate, never re-derived here
    deviations: list[str]  # why, when provenance is not "parity_verified"
    iou_thresholds: list[float]  # T values — matches precision axis 0
    area_labels: list[str]  # ordered area range labels — matches precision axis 3
    area_ranges: dict[str, tuple[float, float]]
    max_dets: list[int]  # matches precision axis 4
    metrics: dict[str, float]
    per_class: dict[str, float] | None
    precision: np.ndarray  # shape (T, R, K, A, M)
    recall_pts: np.ndarray  # params.rec_thrs — the evaluator's own grid, length R
    cat_ids: list[int]  # ordered — matches precision axis 2
    cat_names: dict[int, str]
    # The metric catalog from Rust, in canonical display order: one dict per
    # metric with name/ap/iou_thr/area/max_det/freq_group. Every renderer reads
    # a metric's axes from here rather than parsing them back out of its name.
    metric_defs: list[dict]
    version: str

    # ------------------------------------------------------------------
    # Index helpers — used by plot functions to resolve axis positions
    # ------------------------------------------------------------------

    def area_idx(self, label: str) -> int:
        """Return the area-range axis index for *label*, defaulting to 0."""
        return self.area_labels.index(label) if label in self.area_labels else 0

    def max_det_idx(self, max_det: int | None) -> int:
        """Return the max-det axis index, defaulting to the per-image cap.

        The default mirrors ``Params::max_det()`` in Rust — the *maximum* entry,
        not the last one. The two agree on the sorted default ``[1, 10, 100]``
        and diverge on unsorted params, where taking the last would plot the
        AR1 slice under the AP label.
        """
        if max_det is not None and max_det in self.max_dets:
            return self.max_dets.index(max_det)
        return self.max_dets.index(max(self.max_dets))

    def nearest_iou_idx(self, target: float) -> int:
        """Return the IoU-threshold index closest to *target*."""
        return min(range(len(self.iou_thresholds)), key=lambda i: abs(self.iou_thresholds[i] - target))

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def mean_precision(self, t_idx: int, a_idx: int, m_idx: int) -> np.ndarray:
        """Mean precision over categories for one (IoU, area, maxDets) slice.

        Returns one value per recall threshold, length R. The ``-1`` sentinel
        ("not computed for this configuration") is excluded from the mean rather
        than averaged in as a low score, and a recall threshold no category
        reached comes back as NaN.

        The single owner of this reduction on the Python side: `report()`'s
        ``curves`` is the same aggregate computed in Rust, so a chart that is
        parameterized over area or maxDets calls this, and a chart that draws
        the standard slice reads ``curves``. Anything spelling the nanmean
        itself is a third answer to the same question.
        """
        return np.nanmean(_mask_invalid_prec(self.precision[t_idx, :, :, a_idx, m_idx]), axis=1)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_coco_eval(cls, coco_eval, *, per_class: bool = False) -> "PlotData":
        """Extract and validate plot data from a COCOeval object.

        Parameters
        ----------
        coco_eval : COCOeval
            Must have ``run()`` called first.
        per_class : bool
            Whether to include per-category AP values. Default False.

        Raises
        ------
        ValueError
            If ``run()`` has not been called, the eval mode is unrecognized,
            or the precision array does not have the expected 5D shape.
        """
        if coco_eval.eval is None:
            raise ValueError("Call coco_eval.run() before plotting.")

        r = coco_eval.results(per_class=per_class)
        params_dict = r["params"]

        valid_modes = {"coco", "lvis", "openimages"}
        if params_dict["eval_mode"] not in valid_modes:
            raise ValueError(f"Unknown eval_mode: {params_dict['eval_mode']!r}")

        precision = np.asarray(coco_eval.eval["precision"])
        if precision.ndim != 5:
            raise ValueError(f"Expected 5-dimensional precision array, got shape {precision.shape}")

        area_labels = list(coco_eval.params.area_rng_lbl)
        cat_ids = list(coco_eval.params.cat_ids)

        # The recall axis is the evaluator's own grid, never a fresh linspace:
        # `rec_thrs` is configurable, and fabricating 0..1 silently mislabels
        # every x coordinate on a run that customized it.
        recall_pts = np.asarray(coco_eval.params.rec_thrs, dtype=float)
        if recall_pts.shape[0] != precision.shape[1]:
            raise ValueError(
                f"params.rec_thrs has {recall_pts.shape[0]} points but precision axis 1 has "
                f"{precision.shape[1]} — accumulate() ran against a different recall grid."
            )

        try:
            cats = coco_eval.coco_gt.load_cats(cat_ids)
            cat_names = {c["id"]: c["name"] for c in cats}
        except Exception:
            cat_names = {cid: str(cid) for cid in cat_ids}

        return cls(
            eval_mode=params_dict["eval_mode"],
            iou_type=params_dict["iou_type"],
            # Read from the results dict Rust already produced, not re-derived.
            # Provenance depends on the whole configuration, so `eval_mode ==
            # "coco"` says nothing about it — a bbox run with custom iou_thrs is
            # an extension too.
            provenance=r["provenance"],
            is_benchmark_standard=coco_eval.is_benchmark_standard(),
            deviations=list(coco_eval.reference_deviations()),
            iou_thresholds=params_dict["iou_thresholds"],
            area_labels=area_labels,
            area_ranges={k: tuple(v) for k, v in params_dict["area_ranges"].items()},
            max_dets=params_dict["max_dets"],
            metrics=r["metrics"],
            per_class=r.get("per_class"),
            precision=precision,
            recall_pts=recall_pts,
            cat_ids=cat_ids,
            cat_names=cat_names,
            metric_defs=list(coco_eval.metric_defs()),
            version=r["hotcoco_version"],
        )
