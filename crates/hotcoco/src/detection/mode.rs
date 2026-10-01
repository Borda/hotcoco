//! Evaluation modes and the per-mode state they carry.
//!
//! [`EvalMode`] selects matching semantics, the metric set, and output format.
//! The LVIS frequency buckets live here too: they are populated during
//! `evaluate()` only in LVIS mode and are meaningless in the others, so they
//! belong beside the mode that owns them rather than in a shared type bag.

/// Evaluation mode: determines matching semantics, metric sets, and output formatting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalMode {
    /// Standard COCO evaluation (12 bbox/segm metrics or 10 keypoint metrics).
    Coco,
    /// LVIS federated evaluation (13 metrics including frequency-group AP).
    Lvis,
    /// Open Images detection evaluation (hierarchy-aware, group-of matching).
    OpenImages,
}

impl EvalMode {
    /// The parameters each mode's protocol starts from — what the `COCOeval`
    /// constructors and `StreamingEval` bindings hand to `Params`. COCO is
    /// [`Params::new`](crate::params::Params::new); LVIS caps detections at 300; Open Images scores one
    /// IoU threshold (0.5), one area range, and 100 detections.
    pub fn default_params(self, iou_type: crate::params::IouType) -> crate::params::Params {
        let mut params = crate::params::Params::new(iou_type);
        match self {
            EvalMode::Coco => {}
            EvalMode::Lvis => params.max_dets = vec![300],
            EvalMode::OpenImages => {
                params.iou_thrs = vec![0.5];
                params.area_ranges = vec![crate::AreaRange {
                    label: "all".to_string(),
                    range: [0.0, 1e10],
                }];
                params.max_dets = vec![100];
            }
        }
        params
    }
}

/// LVIS category frequency bucket, as stored in `Category.frequency`.
///
/// Public because it is a field of [`MetricDef`](super::MetricDef): a renderer
/// walking the metric catalog has to be able to tell `APr` from `APc` without
/// string-matching the display name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FreqGroup {
    Rare,
    Common,
    Frequent,
}

/// LVIS category-index buckets grouped by frequency.
///
/// Each field holds the `k_idx` (position in `params.cat_ids`) of all categories
/// in that frequency bucket. Populated during `evaluate()` when `eval_mode == Lvis`.
#[derive(Debug, Clone, Default)]
pub(super) struct FreqGroups {
    pub rare: Vec<usize>,
    pub common: Vec<usize>,
    pub frequent: Vec<usize>,
}

impl FreqGroups {
    /// Bucket `categories` by their LVIS `frequency` tag, as positions in
    /// `cat_ids` (the K axis). A category outside `cat_ids`, or without a tag,
    /// lands in no bucket — so the buckets are empty for any non-LVIS dataset.
    pub fn from_categories(categories: &[crate::types::Category], cat_ids: &[u64]) -> Self {
        let cat_id_to_k_idx: std::collections::HashMap<u64, usize> =
            cat_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        let mut groups = FreqGroups::default();
        for cat in categories {
            if let Some(&k_idx) = cat_id_to_k_idx.get(&cat.id) {
                match cat.frequency.as_deref() {
                    Some("r") => groups.rare.push(k_idx),
                    Some("c") => groups.common.push(k_idx),
                    Some("f") => groups.frequent.push(k_idx),
                    _ => {}
                }
            }
        }
        groups
    }

    pub fn get(&self, group: FreqGroup) -> &[usize] {
        match group {
            FreqGroup::Rare => &self.rare,
            FreqGroup::Common => &self.common,
            FreqGroup::Frequent => &self.frequent,
        }
    }
}
