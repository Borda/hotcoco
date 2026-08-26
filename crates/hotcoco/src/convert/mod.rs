//! Format converters: COCO ↔ YOLO, Pascal VOC, CVAT, DOTA, and Open Images.
//!
//! Five formats, one design. Every converter follows the same policies, so
//! learning one is learning all of them.
//!
//! # Malformed input is an error; inexpressible input is a counted skip
//!
//! A record that is *wrong* — an unparsable number, a missing required
//! attribute or column, the wrong field count — fails the conversion with a
//! [`ConvertError`] naming the file and, where one exists, the line or byte
//! position. A record the target format *cannot express* — an
//! unsupported CVAT shape kind, a polygon with fewer than three points, a crowd
//! annotation in YOLO — is skipped and counted in the returned stats struct.
//! Nothing is ever dropped without either a counter or an error. An annotation
//! whose `category_id` is not in the dataset is a data-integrity error
//! ([`ConvertError::UnknownCategory`]), not a skip.
//!
//! # Missing image dimensions are an error where geometry depends on them
//!
//! YOLO and Open Images store coordinates normalized by the image size, so
//! converting them in either direction must scale by the true width/height: a
//! zero or unknown dimension is [`ConvertError::MissingImageDimensions`]. The
//! one documented exception is [`oid_to_coco`], which keeps normalized
//! coordinates against a 1×1 image when no dimensions are supplied — IoU and
//! IoA (and therefore Open Images AP) are unaffected, only absolute areas lose
//! meaning. Where dimensions are pure metadata (DOTA's corners are absolute
//! pixels), an unknown size is recorded as `0×0` and never multiplied into
//! geometry.
//!
//! # `file_name` is never invented
//!
//! CVAT and VOC record the image file name, and it round-trips verbatim. YOLO,
//! DOTA, and Open Images record only a file stem or image ID, so their
//! importers use that stem as `file_name` without fabricating an extension
//! (dimension lookups accept both the bare stem and stem + common image
//! extensions). Exporters key per-image output by the stem of `file_name`; two
//! images whose stems collide (`train/img.jpg` vs `val/img.jpg`) would
//! silently overwrite each other's output, so that is
//! [`ConvertError::StemCollision`].
//!
//! # One shape for signatures and stats
//!
//! Importers are `*_to_coco(source_path, <format-specific options>,
//! image_dims)` — source first, `image_dims` last, and only present when the
//! format does not record dimensions itself. Exporters are
//! `coco_to_*(dataset, output_path)` and return a `*Stats` struct with shared
//! vocabulary: `images` and `annotations` count what was written, `skipped_*`
//! fields count records the format cannot express. Every field is documented
//! on its struct.

mod cvat;
mod dota;
mod oid;
mod voc;
mod yolo;

pub use cvat::{CvatImportStats, CvatStats, coco_to_cvat, cvat_to_coco};
pub use dota::{DotaStats, coco_to_dota, dota_to_coco};
pub use oid::{OidStats, coco_to_oid, oid_results_to_anns, oid_to_coco, read_class_descriptions};
pub use voc::{VocStats, coco_to_voc, voc_to_coco};
pub use yolo::{YoloStats, coco_to_yolo, yolo_to_coco};

use std::collections::HashMap;
use std::io;
use std::path::Path;

use crate::types::{Annotation, Dataset};

/// Errors that can occur during format conversion.
#[derive(Debug, thiserror::Error)]
pub enum ConvertError {
    /// An I/O error occurred while reading or writing files.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    /// An image has zero or unknown dimensions where the conversion must scale
    /// coordinates by the image size. The payload identifies the image.
    #[error(
        "image `{0}` has zero or unknown width/height, which this conversion needs to scale coordinates"
    )]
    MissingImageDimensions(String),
    /// No `data.yaml` found in the YOLO directory.
    #[error("data.yaml not found in YOLO directory")]
    MissingDataYaml,
    /// A label file, CSV, or `data.yaml` could not be parsed. The message names
    /// the file and, where one exists, the line or position.
    #[error("parse error: {0}")]
    ParseError(String),
    /// An XML parsing or writing error occurred.
    #[error("XML error: {0}")]
    XmlError(String),
    /// Two images share a file stem, so per-image outputs keyed by stem would
    /// silently overwrite each other.
    #[error(
        "images `{first}` and `{second}` share the file stem `{stem}`; per-image outputs are keyed by stem and would overwrite each other — rename one image"
    )]
    StemCollision {
        /// The colliding stem.
        stem: String,
        /// `file_name` of the image seen first.
        first: String,
        /// `file_name` of the image seen second.
        second: String,
    },
    /// An annotation references a `category_id` that is not in the dataset's
    /// category list.
    #[error(
        "annotation {ann_id} references category id {category_id}, which is not in the dataset's categories"
    )]
    UnknownCategory {
        /// The annotation's id.
        ann_id: u64,
        /// The dangling category id it references.
        category_id: u64,
    },
}

impl From<quick_xml::Error> for ConvertError {
    fn from(e: quick_xml::Error) -> Self {
        ConvertError::XmlError(e.to_string())
    }
}

impl ConvertError {
    /// Prefix a parse/XML error's message with the file it came from.
    ///
    /// This is the single place file context is attached, so parsers that take
    /// a reader (and cannot know the path) stay path-agnostic while every error
    /// that leaves the module still names its file.
    pub(crate) fn with_path(self, path: &Path) -> Self {
        match self {
            ConvertError::ParseError(msg) => {
                ConvertError::ParseError(format!("{}: {msg}", path.display()))
            }
            ConvertError::XmlError(msg) => {
                ConvertError::XmlError(format!("{}: {msg}", path.display()))
            }
            other => other,
        }
    }
}

/// A parse error carrying its file.
pub(crate) fn parse_err(path: &Path, detail: impl std::fmt::Display) -> ConvertError {
    ConvertError::ParseError(format!("{}: {detail}", path.display()))
}

/// A parse error carrying its file and 1-based line number.
pub(crate) fn line_err(
    path: &Path,
    line_no: usize,
    detail: impl std::fmt::Display,
) -> ConvertError {
    ConvertError::ParseError(format!("{}: line {line_no}: {detail}", path.display()))
}

/// Group annotations by `image_id`.
pub(crate) fn anns_by_image(dataset: &Dataset) -> HashMap<u64, Vec<&Annotation>> {
    let mut map: HashMap<u64, Vec<&Annotation>> = HashMap::new();
    for ann in &dataset.annotations {
        map.entry(ann.image_id).or_default().push(ann);
    }
    map
}

/// Extract the filename stem (without extension) from a file path string.
pub(crate) fn file_stem(file_name: &str) -> &str {
    Path::new(file_name)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(file_name)
}

/// The UTF-8 stem of an on-disk label file, or an error naming the file.
///
/// A non-UTF-8 file name cannot become a COCO `file_name`; silently mapping it
/// to an empty stem would fabricate colliding image records.
pub(crate) fn utf8_stem(path: &Path) -> Result<&str, ConvertError> {
    path.file_stem().and_then(|s| s.to_str()).ok_or_else(|| {
        ConvertError::ParseError(format!(
            "label file `{}` has a non-UTF-8 name; cannot derive an image file name from it",
            path.display()
        ))
    })
}

/// Verify every image's file stem is unique before writing per-image outputs.
///
/// Exporters that write one file (or CSV image id) per image key it by the stem
/// of `file_name`; `train/img.jpg` and `val/img.jpg` share the stem `img` and
/// the second would silently overwrite the first.
pub(crate) fn check_unique_stems(dataset: &Dataset) -> Result<(), ConvertError> {
    let mut seen: HashMap<&str, &str> = HashMap::new();
    for img in &dataset.images {
        let stem = file_stem(&img.file_name);
        if let Some(first) = seen.insert(stem, &img.file_name) {
            return Err(ConvertError::StemCollision {
                stem: stem.to_string(),
                first: first.to_string(),
                second: img.file_name.clone(),
            });
        }
    }
    Ok(())
}

/// Look up image dimensions by stem; try common extensions as fallback.
///
/// Returns `None` when the stem is absent or maps to a zero dimension — both
/// mean "unknown" to callers, which apply the missing-dimensions policy
/// documented in the module docs.
pub(crate) fn lookup_image_dims(
    image_dims: &HashMap<String, (u32, u32)>,
    stem: &str,
) -> Option<(u32, u32)> {
    let found = image_dims.get(stem).or_else(|| {
        ["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
            .iter()
            .find_map(|ext| image_dims.get(&format!("{stem}.{ext}")))
    });
    match found {
        Some(&(w, h)) if w > 0 && h > 0 => Some((w, h)),
        _ => None,
    }
}

/// Write a simple `<tag>text</tag>` XML element.
pub(crate) fn write_text_element<W: std::io::Write>(
    writer: &mut quick_xml::writer::Writer<W>,
    tag: &str,
    text: &str,
) -> Result<(), quick_xml::Error> {
    use quick_xml::events::{BytesEnd, BytesStart, BytesText, Event};
    writer.write_event(Event::Start(BytesStart::new(tag)))?;
    writer.write_event(Event::Text(BytesText::new(text)))?;
    writer.write_event(Event::End(BytesEnd::new(tag)))?;
    Ok(())
}
