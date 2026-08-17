//! Dataset statistics: the DTOs and the `COCO::stats` computation.
//!
//! These are *outputs* of inspecting a dataset, not part of its schema:
//! [`crate::types`] describes what a COCO file contains; this describes what
//! was found in one.

use std::collections::HashMap;

use crate::coco::COCO;

/// Summary statistics (min/max/mean/median) for a numeric field.
#[derive(Debug, Clone)]
pub struct SummaryStats {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub median: f64,
}

/// Per-category dataset statistics.
#[derive(Debug, Clone)]
pub struct CategoryStats {
    pub id: u64,
    pub name: String,
    pub ann_count: usize,
    pub img_count: usize,
    pub crowd_count: usize,
}

/// Dataset health-check statistics returned by [`crate::COCO::stats`].
#[derive(Debug, Clone)]
pub struct DatasetStats {
    pub image_count: usize,
    pub annotation_count: usize,
    pub category_count: usize,
    pub crowd_count: usize,
    /// Per-category breakdown, sorted by `ann_count` descending.
    pub per_category: Vec<CategoryStats>,
    pub image_width: SummaryStats,
    pub image_height: SummaryStats,
    /// Summary over annotations that have an `area` value.
    pub annotation_area: SummaryStats,
}

impl COCO {
    /// Compute dataset health-check statistics.
    pub fn stats(&self) -> DatasetStats {
        let mut cat_ann_counts: HashMap<u64, usize> = HashMap::new();
        let mut cat_crowd_counts: HashMap<u64, usize> = HashMap::new();
        let mut areas: Vec<f64> = Vec::new();
        let mut crowd_count = 0usize;

        for ann in &self.dataset.annotations {
            *cat_ann_counts.entry(ann.category_id).or_default() += 1;
            if ann.iscrowd {
                crowd_count += 1;
                *cat_crowd_counts.entry(ann.category_id).or_default() += 1;
            }
            if let Some(area) = ann.area {
                areas.push(area);
            }
        }

        let (widths, heights): (Vec<f64>, Vec<f64>) = self
            .dataset
            .images
            .iter()
            .map(|img| (img.width as f64, img.height as f64))
            .unzip();

        let mut per_category: Vec<CategoryStats> = self
            .dataset
            .categories
            .iter()
            .map(|cat| CategoryStats {
                id: cat.id,
                name: cat.name.clone(),
                ann_count: cat_ann_counts.get(&cat.id).copied().unwrap_or(0),
                img_count: self.cat_to_imgs.get(&cat.id).map_or(0, std::vec::Vec::len),
                crowd_count: cat_crowd_counts.get(&cat.id).copied().unwrap_or(0),
            })
            .collect();
        per_category.sort_by_key(|b| std::cmp::Reverse(b.ann_count));

        DatasetStats {
            image_count: self.dataset.images.len(),
            annotation_count: self.dataset.annotations.len(),
            category_count: self.dataset.categories.len(),
            crowd_count,
            per_category,
            image_width: summary_stats(widths),
            image_height: summary_stats(heights),
            annotation_area: summary_stats(areas),
        }
    }
}

fn summary_stats(mut values: Vec<f64>) -> SummaryStats {
    if values.is_empty() {
        return SummaryStats {
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            median: 0.0,
        };
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let sorted = values;
    let min = sorted[0];
    let max = *sorted.last().expect("non-empty after early return");
    let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
    let n = sorted.len();
    let median = if n % 2 == 1 {
        sorted[n / 2]
    } else {
        f64::midpoint(sorted[n / 2 - 1], sorted[n / 2])
    };
    SummaryStats {
        min,
        max,
        mean,
        median,
    }
}
