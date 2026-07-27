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

/// LVIS category frequency bucket, as stored in `Category.frequency`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FreqGroup {
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
    pub fn get(&self, group: FreqGroup) -> &[usize] {
        match group {
            FreqGroup::Rare => &self.rare,
            FreqGroup::Common => &self.common,
            FreqGroup::Frequent => &self.frequent,
        }
    }
}
