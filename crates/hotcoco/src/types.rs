use std::collections::HashMap;

use serde::{Deserialize, Deserializer, Serialize};

/// Top-level COCO dataset structure.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Dataset {
    #[serde(default)]
    pub info: Option<Info>,
    #[serde(default)]
    pub images: Vec<Image>,
    #[serde(default)]
    pub annotations: Vec<Annotation>,
    #[serde(default)]
    pub categories: Vec<Category>,
    #[serde(default)]
    pub licenses: Vec<License>,
}

/// Dataset metadata: version, description, date, and the rest of the `info` block.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Info {
    #[serde(default)]
    pub year: Option<u32>,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub contributor: Option<String>,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub date_created: Option<String>,
}

/// A single image in the dataset.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Image {
    pub id: u64,
    #[serde(default)]
    pub file_name: String,
    pub height: u32,
    pub width: u32,
    #[serde(default)]
    pub license: Option<u64>,
    #[serde(default)]
    pub coco_url: Option<String>,
    #[serde(default)]
    pub flickr_url: Option<String>,
    #[serde(default)]
    pub date_captured: Option<String>,
    /// LVIS: categories confirmed absent in this image (unmatched DTs are FP).
    #[serde(default)]
    pub neg_category_ids: Vec<u64>,
    /// LVIS: categories not exhaustively checked in this image (unmatched DTs are ignored).
    #[serde(default)]
    pub not_exhaustive_category_ids: Vec<u64>,
    /// Keys not in the COCO schema, preserved verbatim so
    /// load → filter/split/merge → save round-trips user metadata
    /// (pycocotools keeps unknown keys because it stores raw dicts).
    #[serde(flatten, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// A single object annotation (ground truth or detection result).
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Annotation {
    #[serde(default)]
    pub id: u64,
    pub image_id: u64,
    pub category_id: u64,
    #[serde(default)]
    pub bbox: Option<[f64; 4]>,
    #[serde(default)]
    pub area: Option<f64>,
    #[serde(default)]
    pub segmentation: Option<Segmentation>,
    #[serde(default, deserialize_with = "deserialize_iscrowd")]
    pub iscrowd: bool,
    #[serde(default)]
    pub keypoints: Option<Vec<f64>>,
    #[serde(default)]
    pub num_keypoints: Option<u32>,
    /// Oriented bounding box as `[cx, cy, w, h, angle]` where angle is in radians.
    /// Used for rotated detection evaluation (aerial imagery, document analysis, scene text).
    #[serde(default)]
    pub obb: Option<[f64; 5]>,
    /// Detection score (present only in result annotations).
    #[serde(default)]
    pub score: Option<f64>,
    /// Open Images group-of flag. When true, the annotation represents a group of objects
    /// rather than a single instance. Distinct from `iscrowd` — different matching semantics.
    #[serde(default)]
    pub is_group_of: Option<bool>,
    /// Keys not in the COCO schema, preserved verbatim so
    /// load → filter/split/merge → save round-trips user metadata
    /// (pycocotools keeps unknown keys because it stores raw dicts).
    #[serde(flatten, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// Deserialize `iscrowd` from either a boolean or an integer (0/1).
///
/// COCO JSON files use both representations, so both are accepted.
fn deserialize_iscrowd<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum IsCrowd {
        Bool(bool),
        Int(u8),
    }
    match IsCrowd::deserialize(deserializer)? {
        IsCrowd::Bool(b) => Ok(b),
        IsCrowd::Int(i) => Ok(i != 0),
    }
}

/// Segmentation mask in one of three COCO formats.
///
/// `#[serde(untagged)]` auto-detects the format when *serializing* (it just
/// writes the variant's content, which is the COCO wire shape). Deserialization
/// is hand-written below instead of untagged: untagged buffers the entire value
/// into serde's internal `Content` tree and then tries each variant against it,
/// which materializes every polygon coordinate twice — measured as the dominant
/// cost of loading a polygon-heavy GT file, on serde_json and simd-json alike.
/// The visitor streams instead: a JSON array is a polygon list, a JSON object
/// is an RLE whose variant is decided by the type of its `counts` value.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum Segmentation {
    /// Polygon format: list of polygons, each a flat list of [x, y, x, y, ...] coordinates.
    Polygon(Vec<Vec<f64>>),
    /// Compressed RLE format (as stored in COCO JSON results).
    CompressedRle { size: [u32; 2], counts: String },
    /// Uncompressed RLE format.
    UncompressedRle { size: [u32; 2], counts: Vec<u32> },
}

impl<'de> Deserialize<'de> for Segmentation {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        /// `counts` value: a compressed-RLE string or an uncompressed run list,
        /// decided by the token serde hands the visitor — no buffering.
        enum Counts {
            Str(String),
            Ints(Vec<u32>),
        }

        impl<'de> Deserialize<'de> for Counts {
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                struct CountsVisitor;
                impl<'de> serde::de::Visitor<'de> for CountsVisitor {
                    type Value = Counts;

                    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                        f.write_str("an RLE counts string or an array of run lengths")
                    }

                    fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Counts, E> {
                        Ok(Counts::Str(v.to_owned()))
                    }

                    fn visit_string<E: serde::de::Error>(self, v: String) -> Result<Counts, E> {
                        Ok(Counts::Str(v))
                    }

                    fn visit_seq<A: serde::de::SeqAccess<'de>>(
                        self,
                        mut seq: A,
                    ) -> Result<Counts, A::Error> {
                        let mut v = Vec::with_capacity(seq.size_hint().unwrap_or(0));
                        while let Some(c) = seq.next_element()? {
                            v.push(c);
                        }
                        Ok(Counts::Ints(v))
                    }
                }
                deserializer.deserialize_any(CountsVisitor)
            }
        }

        struct SegVisitor;
        impl<'de> serde::de::Visitor<'de> for SegVisitor {
            type Value = Segmentation;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a list of polygons or an RLE object with `size` and `counts`")
            }

            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<Segmentation, A::Error> {
                let mut polys = Vec::with_capacity(seq.size_hint().unwrap_or(0));
                while let Some(p) = seq.next_element()? {
                    polys.push(p);
                }
                Ok(Segmentation::Polygon(polys))
            }

            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> Result<Segmentation, A::Error> {
                let mut size: Option<[u32; 2]> = None;
                let mut counts: Option<Counts> = None;
                while let Some(key) = map.next_key::<std::borrow::Cow<'_, str>>()? {
                    match key.as_ref() {
                        "size" => size = Some(map.next_value()?),
                        "counts" => counts = Some(map.next_value()?),
                        _ => {
                            map.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                let size = size.ok_or_else(|| serde::de::Error::missing_field("size"))?;
                match counts.ok_or_else(|| serde::de::Error::missing_field("counts"))? {
                    Counts::Str(counts) => Ok(Segmentation::CompressedRle { size, counts }),
                    Counts::Ints(counts) => Ok(Segmentation::UncompressedRle { size, counts }),
                }
            }
        }

        deserializer.deserialize_any(SegVisitor)
    }
}

/// An object category, such as "person" or "car".
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Category {
    pub id: u64,
    pub name: String,
    #[serde(default)]
    pub supercategory: Option<String>,
    #[serde(default)]
    pub skeleton: Option<Vec<[u32; 2]>>,
    #[serde(default)]
    pub keypoints: Option<Vec<String>>,
    /// LVIS frequency bucket: "r" (rare), "c" (common), "f" (frequent).
    #[serde(default)]
    pub frequency: Option<String>,
    /// Keys not in the COCO schema, preserved verbatim so
    /// load → filter/split/merge → save round-trips user metadata
    /// (pycocotools keeps unknown keys because it stores raw dicts).
    #[serde(flatten, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// Map `category_id -> name`.
///
/// The single owner of this reduction — `convert`'s exporters that write
/// categories by name, `detection::hierarchy`, and `quality::healthcheck` all
/// call this rather than re-collecting `dataset.categories` themselves.
pub(crate) fn cat_id_to_name(dataset: &Dataset) -> HashMap<u64, &str> {
    dataset
        .categories
        .iter()
        .map(|c| (c.id, c.name.as_str()))
        .collect()
}

/// Map `name -> category_id`.
///
/// Takes a category slice rather than a [`Dataset`] — some callers resolve
/// names against categories they are still assembling, before a `Dataset`
/// exists to hold them.
pub(crate) fn cat_name_to_id(categories: &[Category]) -> HashMap<&str, u64> {
    categories.iter().map(|c| (c.name.as_str(), c.id)).collect()
}

/// Image license information.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct License {
    #[serde(default)]
    pub id: u64,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub url: Option<String>,
}

/// Run-length encoding for masks.
#[derive(Debug, Clone, PartialEq)]
pub struct Rle {
    pub h: u32,
    pub w: u32,
    /// Run counts: alternating runs of 0s and 1s, starting with 0s.
    pub counts: Vec<u32>,
}

impl Rle {
    /// Validated constructor: errors unless `counts` sums to exactly `h * w`.
    ///
    /// Validation happens in release builds too — use this at untrusted
    /// boundaries. Internal code that produces RLEs it already knows to be
    /// well-formed (the codecs in [`crate::mask`]) constructs the struct
    /// directly instead; the fields stay public for that reason.
    pub fn new(h: u32, w: u32, counts: Vec<u32>) -> crate::error::Result<Self> {
        let sum: u64 = counts.iter().map(|&c| c as u64).sum();
        let expected = h as u64 * w as u64;
        if sum != expected {
            return Err(
                format!("RLE counts must sum to h*w ({h} * {w} = {expected}), got {sum}").into(),
            );
        }
        Ok(Self { h, w, counts })
    }
}
