//! Reading COCO JSON into [`Dataset`] and [`Annotation`] records.
//!
//! One rule shapes this module: **peak memory is the output, not a parse
//! tree.** A tape-building parser turns every number and bracket into a node
//! before serde sees any of it, and on COCO files — which are almost entirely
//! numbers — that tape dwarfs the records it produces. serde_json streams
//! straight into the structs, so the peak is the file plus the records.
//!
//! The annotations array is the bulk of any COCO file, so it is parsed in
//! parallel, in place: a dataset object is walked by hand until the array,
//! the bytes from there are cut into chunks at guessed record boundaries (a
//! `{` that follows `},`), each chunk is parsed as a run of records, and the
//! run that reaches the array's `]` says where the walk resumes. A guess can
//! land inside a string — a caption containing `},{` — so every run must end
//! exactly where the next begins. When that fails, or the file has a shape
//! the walk does not expect, one serial serde pass takes over and is also
//! what reports the error for a malformed file.
//!
//! Floats are read with serde_json's `float_roundtrip` feature. Its default
//! parser is best-effort and can land one ULP off, and an `area` one ULP from
//! 32² changes an annotation's size bucket.

use std::borrow::Cow;
use std::path::Path;

use rayon::prelude::*;
use serde::Deserialize;
use serde::de::IgnoredAny;

use crate::types::{Annotation, Dataset};

/// Read and parse a dataset file. The second value is the number of
/// non-finite float tokens rewritten to `null` on the way in.
pub(crate) fn read_dataset(path: &Path) -> crate::error::Result<(Dataset, usize)> {
    read(path, dataset_from_slice)
}

/// Read and parse a results file: a bare array of annotations, or a dataset
/// object whose `annotations` are the results.
pub(crate) fn read_results(path: &Path) -> crate::error::Result<(Vec<Annotation>, usize)> {
    read(path, results_from_slice)
}

/// Parse first and sanitize only on failure: a clean file, which is nearly
/// every file, never pays the sanitizer's scans, and a dirty one parses twice.
fn read<T>(
    path: &Path,
    parse: fn(&[u8]) -> serde_json::Result<T>,
) -> crate::error::Result<(T, usize)> {
    let raw = std::fs::read(path)?;
    match parse(&raw) {
        Ok(value) => Ok((value, 0)),
        Err(err) => {
            let (fixed, n_fixed) = sanitize_non_finite(&raw);
            if n_fixed == 0 {
                return Err(err.into());
            }
            Ok((parse(&fixed)?, n_fixed))
        }
    }
}

/// Parse a results file's bytes, whichever of its two shapes they take.
pub(crate) fn results_from_slice(bytes: &[u8]) -> serde_json::Result<Vec<Annotation>> {
    if bytes.get(skip_ws(bytes, 0)) == Some(&b'[') {
        annotations_from_slice(bytes)
    } else {
        dataset_from_slice(bytes).map(|dataset| dataset.annotations)
    }
}

/// Parse a COCO dataset object. Keys outside the schema are ignored, a
/// `null` or missing `annotations` reads as none, and a repeated key keeps
/// its last value, as Python's `json` does.
pub(crate) fn dataset_from_slice(bytes: &[u8]) -> serde_json::Result<Dataset> {
    parse_dataset(bytes).map_or_else(|| serde_json::from_slice(bytes), Ok)
}

/// The hand walk over a dataset object, or `None` for a shape it does not
/// expect, which the serde derive then judges. The struct literal at the end
/// is deliberate: a field added to [`Dataset`] fails to compile here instead
/// of being silently dropped.
fn parse_dataset(bytes: &[u8]) -> Option<Dataset> {
    let (mut info, mut images, mut annotations, mut categories, mut licenses) = Default::default();
    let mut pos = skip_ws(bytes, 0);
    if bytes.get(pos) != Some(&b'{') {
        return None;
    }
    pos = skip_ws(bytes, pos + 1);
    if bytes.get(pos) != Some(&b'}') {
        loop {
            let (key, next) = value_at::<Cow<str>>(bytes, pos)?;
            pos = skip_ws(bytes, next);
            if bytes.get(pos) != Some(&b':') {
                return None;
            }
            pos = skip_ws(bytes, pos + 1);
            pos = match &*key {
                "info" => take(bytes, pos, &mut info)?,
                "images" => take(bytes, pos, &mut images)?,
                "categories" => take(bytes, pos, &mut categories)?,
                "licenses" => take(bytes, pos, &mut licenses)?,
                "annotations" => take_annotations(bytes, pos, &mut annotations)?,
                _ => value_at::<IgnoredAny>(bytes, pos)?.1,
            };
            pos = skip_ws(bytes, pos);
            match bytes.get(pos) {
                Some(b',') => pos = skip_ws(bytes, pos + 1),
                Some(b'}') => break,
                _ => return None,
            }
        }
    }
    (skip_ws(bytes, pos + 1) == bytes.len()).then_some(Dataset {
        info,
        images,
        annotations,
        categories,
        licenses,
    })
}

/// Parse the value at `pos` into `slot`; the offset just past it.
fn take<'de, T: Deserialize<'de>>(bytes: &'de [u8], pos: usize, slot: &mut T) -> Option<usize> {
    let (value, next) = value_at(bytes, pos)?;
    *slot = value;
    Some(next)
}

/// [`take`] for the annotations array: chunked when it is worth it, serial
/// otherwise, and `null` reads as none.
fn take_annotations(bytes: &[u8], pos: usize, slot: &mut Vec<Annotation>) -> Option<usize> {
    let (anns, next) = parse_chunked(bytes, pos, chunk_count(bytes.len() - pos)).or_else(|| {
        value_at::<Option<Vec<Annotation>>>(bytes, pos)
            .map(|(anns, next)| (anns.unwrap_or_default(), next))
    })?;
    *slot = anns;
    Some(next)
}

/// One JSON value starting at `pos`, and the offset just past it.
fn value_at<'de, T: Deserialize<'de>>(bytes: &'de [u8], pos: usize) -> Option<(T, usize)> {
    let mut stream = serde_json::Deserializer::from_slice(bytes.get(pos..)?).into_iter::<T>();
    let value = stream.next()?.ok()?;
    Some((value, pos + stream.byte_offset()))
}

/// Parse a JSON array of annotation records, in parallel when the array is
/// large enough to be worth cutting up.
pub(crate) fn annotations_from_slice(bytes: &[u8]) -> serde_json::Result<Vec<Annotation>> {
    let open = skip_ws(bytes, 0);
    match parse_chunked(bytes, open, chunk_count(bytes.len())) {
        Some((anns, end)) if skip_ws(bytes, end) == bytes.len() => Ok(anns),
        _ => serde_json::from_slice(bytes),
    }
}

/// How many runs to cut `len` bytes of array into: none below one chunk's
/// worth, and no more than a few per thread.
fn chunk_count(len: usize) -> usize {
    (len / MIN_CHUNK_BYTES).min(RUNS_PER_THREAD * rayon::current_num_threads())
}

/// Below this many bytes per chunk, cutting the array up costs more than the
/// parallel parse saves.
const MIN_CHUNK_BYTES: usize = 64 * 1024;

/// Several runs per thread, so a thread that draws polygon-heavy records does
/// not hold the join.
const RUNS_PER_THREAD: usize = 4;

/// The chunked parallel parse of the array whose `[` is at `open`, and the
/// offset just past its `]`. `None` when there are fewer than two chunks,
/// when a chunk boundary guess was wrong, or when the bytes are not an array
/// of objects.
///
/// The array may end before the bytes do (a dataset's `categories` usually
/// follow it), so the chunk targets are spread over everything after `open`
/// and the runs that start past the array's end are discarded: the first run
/// to reach a `]` ends the array, and every run before it must hand over
/// exactly at the next run's start.
fn parse_chunked(bytes: &[u8], open: usize, chunks: usize) -> Option<(Vec<Annotation>, usize)> {
    if chunks < 2 || bytes.get(open) != Some(&b'[') {
        return None;
    }
    let first = skip_ws(bytes, open + 1);
    if bytes.get(first) != Some(&b'{') {
        return None;
    }
    let span = bytes.len() - open;
    let mut starts: Vec<usize> = std::iter::once(first)
        .chain(
            (1..chunks)
                .filter_map(|i| record_start(bytes, (open + span / chunks * i).max(first + 1))),
        )
        .collect();
    starts.dedup();
    let runs: Vec<Run> = (0..starts.len())
        .into_par_iter()
        .map(|i| parse_run(bytes, starts[i], starts.get(i + 1).copied()))
        .collect();
    // Sized exactly: the vector outlives the parse as `Dataset::annotations`,
    // and growing it by doubling would keep up to a run's worth of slack per
    // doubling. Runs past the one that ends the array are not part of it.
    let ends = runs
        .iter()
        .position(|run| !matches!(run, Run::Continues(_)))
        .map_or(runs.len(), |i| i + 1);
    let total: usize = runs[..ends]
        .iter()
        .map(|run| match run {
            Run::Continues(anns) | Run::Ends(anns, _) => anns.len(),
            Run::Failed => 0,
        })
        .sum();
    let mut out = Vec::with_capacity(total);
    for run in runs {
        match run {
            Run::Continues(anns) => out.extend(anns),
            Run::Ends(anns, end) => {
                out.extend(anns);
                return Some((out, end));
            }
            Run::Failed => return None,
        }
    }
    None
}

/// What one run of records turned out to be.
enum Run {
    /// Records up to exactly the next run's start.
    Continues(Vec<Annotation>),
    /// Records up to the array's `]`, with the offset just past it.
    Ends(Vec<Annotation>, usize),
    /// Not records, or records that overran the next run's start: a boundary
    /// guess inside a string, a run that began past the array, or a
    /// malformed file.
    Failed,
}

/// The first `{` at or after `from` that follows a `}` and a `,`, with JSON
/// whitespace allowed around the comma: where a record starts, unless the
/// bytes are inside a string.
fn record_start(bytes: &[u8], from: usize) -> Option<usize> {
    let mut pos = from;
    loop {
        pos += memchr::memchr(b'}', bytes.get(pos..)?)? + 1;
        let comma = skip_ws(bytes, pos);
        if bytes.get(comma) == Some(&b',') {
            let next = skip_ws(bytes, comma + 1);
            if bytes.get(next) == Some(&b'{') {
                return Some(next);
            }
        }
    }
}

/// The records from `start` to exactly `end` (the next run's start) or to
/// the array's `]`, whichever comes first.
fn parse_run(bytes: &[u8], start: usize, end: Option<usize>) -> Run {
    let mut out = Vec::new();
    let mut pos = start;
    loop {
        let Some((ann, next)) = value_at::<Annotation>(bytes, pos) else {
            return Run::Failed;
        };
        out.push(ann);
        pos = skip_ws(bytes, next);
        match bytes.get(pos) {
            Some(b',') => {
                pos = skip_ws(bytes, pos + 1);
                if end.is_some_and(|end| pos >= end) {
                    return if Some(pos) == end {
                        Run::Continues(out)
                    } else {
                        Run::Failed
                    };
                }
            }
            Some(b']') => return Run::Ends(out, pos + 1),
            _ => return Run::Failed,
        }
    }
}

fn skip_ws(bytes: &[u8], mut pos: usize) -> usize {
    while let Some(b' ' | b'\t' | b'\n' | b'\r') = bytes.get(pos) {
        pos += 1;
    }
    pos
}

/// Normalize non-finite JSON float tokens (`NaN`, `Infinity`, `-Infinity`) to
/// `null`, matching the leniency of Python's `json` module.
///
/// Python emits these bare tokens by default and reads them back, so files
/// produced by pycocotools / numpy pipelines frequently contain them, even
/// though they are not valid JSON. serde_json (correctly) rejects them. To load
/// such files, each non-finite token is rewritten to `null` — which serde also
/// uses when *serializing* a non-finite `f64` — but only when the token appears
/// outside a JSON string, so string values that merely contain the substring
/// `"NaN"`/`"Infinity"` — a file name, say — are left untouched. On `Option<f64>`
/// fields (`area`, `score`) the `null` deserializes to `None`.
///
/// Returns the input unchanged and borrowed (no allocation) when it contains no
/// such tokens, so the common case pays only a single linear scan. The second
/// element is the number of tokens rewritten.
fn sanitize_non_finite(input: &[u8]) -> (Cow<'_, [u8]>, usize) {
    // Prefilter: if the tokens never occur as substrings *anywhere* — even
    // inside strings, where they would not count — the scan below cannot
    // rewrite anything. Two SIMD substring searches cost ~1ms on a 19 MB
    // file; the byte-at-a-time state machine they skip cost ~24ms, paid on
    // every load of a clean file, which is nearly every load. ("-Infinity"
    // contains "Infinity", so two needles cover all three tokens.)
    if memchr::memmem::find(input, b"NaN").is_none()
        && memchr::memmem::find(input, b"Infinity").is_none()
    {
        return (Cow::Borrowed(input), 0);
    }

    let n = input.len();
    let mut out: Option<Vec<u8>> = None;
    let mut count = 0usize;
    let mut in_string = false;
    let mut i = 0;

    while i < n {
        let b = input[i];

        if in_string {
            if b == b'\\' {
                // Copy the backslash and the escaped byte verbatim so an
                // escaped quote (`\"`) does not toggle the string state.
                if let Some(o) = out.as_mut() {
                    o.push(b);
                    if i + 1 < n {
                        o.push(input[i + 1]);
                    }
                }
                i += 2;
                continue;
            }
            if b == b'"' {
                in_string = false;
            }
            if let Some(o) = out.as_mut() {
                o.push(b);
            }
            i += 1;
            continue;
        }

        if b == b'"' {
            in_string = true;
            if let Some(o) = out.as_mut() {
                o.push(b);
            }
            i += 1;
            continue;
        }

        // Outside a string, the only bare identifier-like tokens are
        // true/false/null and the non-finite floats we rewrite here. Gate the
        // substring comparisons on the first byte so the common case (digits,
        // punctuation, whitespace) skips them entirely.
        let token_len = match b {
            b'N' if input[i..].starts_with(b"NaN") => Some(3),
            b'I' if input[i..].starts_with(b"Infinity") => Some(8),
            b'-' if input[i..].starts_with(b"-Infinity") => Some(9),
            _ => None,
        };

        if let Some(len) = token_len {
            let o = out.get_or_insert_with(|| {
                let mut v = Vec::with_capacity(n);
                v.extend_from_slice(&input[..i]);
                v
            });
            o.extend_from_slice(b"null");
            count += 1;
            i += len;
            continue;
        }

        if let Some(o) = out.as_mut() {
            o.push(b);
        }
        i += 1;
    }

    match out {
        Some(v) => (Cow::Owned(v), count),
        None => (Cow::Borrowed(input), count),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod sanitize_tests {
    use super::sanitize_non_finite;

    fn run(s: &str) -> (String, usize) {
        let (bytes, n) = sanitize_non_finite(s.as_bytes());
        (String::from_utf8(bytes.into_owned()).unwrap(), n)
    }

    #[test]
    fn clean_input_is_borrowed_unchanged() {
        let input = br#"{"a": [1.0, -2.5], "b": null}"#;
        let (bytes, n) = sanitize_non_finite(input);
        assert_eq!(n, 0);
        assert!(matches!(bytes, std::borrow::Cow::Borrowed(_)));
    }

    #[test]
    fn rewrites_the_non_finite_family() {
        let (out, n) = run(r#"{"a": NaN, "b": Infinity, "c": -Infinity, "d": -3.5}"#);
        assert_eq!(n, 3);
        // -3.5 (a real negative number) must be preserved, not mangled.
        assert_eq!(out, r#"{"a": null, "b": null, "c": null, "d": -3.5}"#);
    }

    #[test]
    fn leaves_non_finite_substrings_inside_strings_alone() {
        // Strings containing the tokens — including an escaped quote — untouched.
        let (out, n) = run(r#"{"name": "NaN and \"Infinity\"", "v": NaN}"#);
        assert_eq!(n, 1);
        assert_eq!(out, r#"{"name": "NaN and \"Infinity\"", "v": null}"#);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(i: u64, tail: &str) -> String {
        format!(
            r#"{{"image_id": {i}, "category_id": {}, "bbox": [1.5, 2.25, 3.125, 4.0625], "score": 0.{i}{tail}}}"#,
            i % 7 + 1
        )
    }

    fn array(records: impl IntoIterator<Item = String>, sep: &str) -> Vec<u8> {
        let records: Vec<String> = records.into_iter().collect();
        format!("[\n{}\n]\n", records.join(sep)).into_bytes()
    }

    fn serial(bytes: &[u8]) -> Vec<Annotation> {
        serde_json::from_slice(bytes).expect("fixture parses serially")
    }

    /// Serialized form, so a comparison covers every field bit for bit (ryu
    /// output round-trips) and a failure prints the differing record.
    fn json(anns: &[Annotation]) -> String {
        serde_json::to_string(anns).expect("annotations serialize")
    }

    #[test]
    fn chunked_parse_matches_serial_for_every_chunk_count() {
        let bytes = array((0..200).map(|i| record(i, "")), ",\n  ");
        let expected = serial(&bytes);
        for chunks in 2..=50 {
            let (got, end) = parse_chunked(&bytes, 0, chunks).expect("clean array parses chunked");
            assert_eq!(json(&got), json(&expected), "chunks={chunks}");
            assert_eq!(got.capacity(), got.len(), "chunks={chunks}");
            assert_eq!(skip_ws(&bytes, end), bytes.len(), "chunks={chunks}");
        }
    }

    #[test]
    fn boundary_guess_inside_a_string_is_rejected_not_misparsed() {
        // Every third record carries a string that looks like a record boundary.
        let bytes = array(
            (0..90).map(|i| {
                if i % 3 == 0 {
                    record(
                        i,
                        r#", "caption": "a}, {\"image_id\": 1, \"category_id\": 1}, {""#,
                    )
                } else {
                    record(i, "")
                }
            }),
            ", ",
        );
        let expected = serial(&bytes);
        assert_eq!(expected.len(), 90);
        for chunks in 2..=40 {
            if let Some((got, _)) = parse_chunked(&bytes, 0, chunks) {
                assert_eq!(json(&got), json(&expected), "chunks={chunks}");
            }
        }
        let got =
            annotations_from_slice(&bytes).expect("the public entry point falls back to serial");
        assert_eq!(json(&got), json(&expected));
    }

    #[test]
    fn a_run_that_overruns_its_end_is_rejected() {
        let bytes = array((0..3).map(|i| record(i, "")), ",");
        let second = record_start(&bytes, 2).expect("three records have a second start");
        // Cutting the first run one byte before the real boundary must fail,
        // not hand back a record that belongs to the next run.
        assert!(matches!(
            parse_run(&bytes, skip_ws(&bytes, 1), Some(second - 1)),
            Run::Failed
        ));
        let Run::Continues(run) = parse_run(&bytes, skip_ws(&bytes, 1), Some(second)) else {
            panic!("exact boundary parses");
        };
        assert_eq!(run.len(), 1);
        // The last run reports where the array ends, whatever follows it.
        let Run::Ends(rest, end) = parse_run(&bytes, second, None) else {
            panic!("last run reaches the closing bracket");
        };
        assert_eq!((rest.len(), &bytes[end - 1..end]), (2, &b"]"[..]));
    }

    #[test]
    fn results_shape_is_decided_by_the_first_byte() {
        let array = array((0..2).map(|i| record(i, "")), ",");
        let object = format!(
            r#" {{"images": [], "annotations": {}}}"#,
            String::from_utf8_lossy(&array)
        );
        assert_eq!(
            json(&results_from_slice(&array).expect("array form")),
            json(&results_from_slice(object.as_bytes()).expect("object form"))
        );
    }

    #[test]
    fn a_dirty_file_is_sanitized_only_after_a_clean_parse_fails() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("res.json");
        std::fs::write(
            &path,
            br#"[{"image_id": 1, "category_id": 1, "score": NaN}]"#,
        )
        .expect("write");
        let (anns, n_fixed) = read_results(&path).expect("NaN reads as null");
        assert_eq!((anns.len(), n_fixed, anns[0].score), (1, 1, None));
        std::fs::write(&path, br#"[{"image_id": 1,"#).expect("write");
        let err = read_results(&path).expect_err("truncated file fails");
        assert!(matches!(err, crate::error::Error::Json(_)), "{err}");
    }

    #[test]
    fn non_object_arrays_and_malformed_input_go_serial() {
        assert!(parse_chunked(b"[]", 0, 4).is_none());
        assert!(parse_chunked(b"[1, 2]", 0, 4).is_none());
        assert!(parse_chunked(b"{\"annotations\": []}", 0, 4).is_none());
        let truncated = &array((0..5).map(|i| record(i, "")), ",")[..80];
        assert!(parse_chunked(truncated, 0, 4).is_none());
        let err = annotations_from_slice(truncated).expect_err("truncated input fails");
        assert!(err.to_string().contains("line"), "{err}");
    }

    #[test]
    fn dataset_object_ignores_unknown_keys_and_defaults_missing_ones() {
        let json = br#"{"custom": {"nested": [1, 2]}, "images": [{"id": 3, "width": 4, "height": 5}],
                       "annotations": [{"id": 9, "image_id": 3, "category_id": 1, "bbox": [0, 0, 1, 1]}],
                       "categories": [{"id": 1, "name": "x"}]}"#;
        let ds = dataset_from_slice(json).expect("dataset parses");
        assert_eq!(ds.images.len(), 1);
        assert_eq!(ds.annotations[0].id, 9);
        assert_eq!(ds.categories[0].name, "x");
        assert!(ds.info.is_none());
        let ds = dataset_from_slice(br#"{"images": []}"#).expect("dataset parses");
        assert!(ds.annotations.is_empty());
        let ds =
            dataset_from_slice(br#"{"annotations": null}"#).expect("null annotations read as none");
        assert!(ds.annotations.is_empty());
        assert!(dataset_from_slice(br"[1]").is_err());
        assert!(dataset_from_slice(br#"{"images": []} trailing"#).is_err());
    }

    #[test]
    fn dataset_walk_parses_annotations_in_place_whatever_follows_them() {
        // Categories after the annotations, unknown keys before and after,
        // and enough bytes that chunk targets land inside the categories.
        let anns: Vec<String> = (0..300).map(|i| record(i, "")).collect();
        let cats: Vec<String> = (0..2000)
            .map(|i| format!(r#"{{"id": {i}, "name": "cat{i}", "supercategory": "s"}}"#))
            .collect();
        let doc = format!(
            "{{\n \"info\": {{\"year\": 2026}}, \"extra\": [{{\"a\": 1}}, {{\"b\": 2}}],\n \"images\": [{{\"id\": 1, \"width\": 2, \"height\": 3}}],\n \"annotations\": [\n{}\n],\n \"categories\": [{}],\n \"licenses\": [], \"more\": {{\"x\": [1, 2]}}\n}}\n",
            anns.join(",\n"),
            cats.join(", ")
        );
        let fast = parse_dataset(doc.as_bytes()).expect("the walk handles this shape");
        let slow = dataset_from_slice(doc.as_bytes()).expect("the derive parses it too");
        assert_eq!(json(&fast.annotations), json(&slow.annotations));
        assert_eq!(fast.annotations.len(), 300);
        assert_eq!((fast.categories.len(), fast.images.len()), (2000, 1));
        assert_eq!(fast.info.map(|i| i.year), Some(Some(2026)));
        // The forced chunking exercises runs that start past the array's end.
        let open = doc.find("\"annotations\": [").unwrap_or(0) + "\"annotations\": ".len();
        let (anns, end) = parse_chunked(doc.as_bytes(), open, 64).expect("chunked in place");
        assert_eq!(anns.len(), 300);
        assert_eq!(&doc.as_bytes()[end - 1..end], b"]");
    }

    #[test]
    fn dataset_walk_defers_unusual_shapes_to_the_derive() {
        assert!(parse_dataset(br#"{"images": [] trailing"#).is_none());
        assert!(parse_dataset(br#"["not", "an", "object"]"#).is_none());
        assert!(parse_dataset(br#"{"images": [}"#).is_none());
        let ds = parse_dataset(br"{}").expect("empty object");
        assert!(ds.annotations.is_empty());
        // The derive reports the error the walk declined to judge.
        assert!(dataset_from_slice(br#"{"images": [}"#).is_err());
        // Repeated keys keep the last value, as Python's `json` does.
        let ds = parse_dataset(br#"{"images": [{"id": 1}], "annotations": null, "images": []}"#)
            .expect("repeated key");
        assert!(ds.images.is_empty() && ds.annotations.is_empty());
    }
}
