//! Architecture conformance — the crate's "one obvious home" rule, as CI.
//!
//! hotcoco's product is auditability: someone checking our numbers should find
//! **one** place answering "how is similarity computed?", "how are detections
//! matched?", "how is AP accumulated?". Before 0.5 there were four bbox-IoU
//! implementations, three copies of the crowd/union formula, two greedy matchers,
//! and two disagreeing `MIN_PARALLEL_WORK` constants (1024 vs 1000, under a
//! comment claiming they matched).
//!
//! Those are consolidated into `primitives/`. These tests exist so the
//! duplication cannot silently creep back between 0.5 and the 1.0 rewrite. They
//! are deliberately crude — grep over source text, no parsing. A determined
//! author can evade them; the point is to catch the *accidental* re-introduction
//! that code review misses, and to make the rule explicit and enforced rather
//! than folklore in a planning doc.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

/// The workspace root, derived from this crate's location.
fn workspace_root() -> &'static Path {
    static ROOT: LazyLock<PathBuf> = LazyLock::new(|| {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("crate lives at <workspace>/crates/<name>")
            .to_path_buf()
    });
    &ROOT
}

/// Every `.rs` file under `crates/*/src/` (the shipped source; tests excluded),
/// paired with its workspace-relative path and its code lines.
///
/// Read once and shared: all three tests scan the same tree.
fn sources() -> &'static [SourceFile] {
    static SOURCES: LazyLock<Vec<SourceFile>> = LazyLock::new(|| {
        let mut paths = Vec::new();
        let crates = fs::read_dir(workspace_root().join("crates")).expect("crates/ dir");
        for entry in crates.flatten() {
            collect_rs(&entry.path().join("src"), &mut paths);
        }
        assert!(!paths.is_empty(), "found no source files to scan");
        paths.sort();
        paths.iter().map(|p| SourceFile::read(p)).collect()
    });
    &SOURCES
}

struct SourceFile {
    /// Workspace-relative path with forward slashes, e.g. `crates/hotcoco/src/mask.rs`.
    path: String,
    /// 1-based line numbers paired with their text, with whole-line `//` comments
    /// dropped — prose mentioning a banned name is not a violation.
    lines: Vec<(usize, String)>,
}

impl SourceFile {
    fn read(path: &Path) -> Self {
        // `expect`, not a silent default: a conformance test that cannot read a
        // file must fail loudly, never pass by scanning nothing.
        let text = fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let rel = path
            .strip_prefix(workspace_root())
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        SourceFile {
            path: rel,
            lines: text
                .lines()
                .enumerate()
                .map(|(i, l)| (i + 1, l.to_string()))
                .filter(|(_, l)| !l.trim_start().starts_with("//"))
                .collect(),
        }
    }
}

fn collect_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rs(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// Every code line matching `hit`, in files not excluded by `skip`, formatted as
/// `path:line  text`. The shared body of all three checks below.
fn scan(skip: impl Fn(&str) -> bool, hit: impl Fn(&str) -> bool) -> Vec<String> {
    sources()
        .iter()
        .filter(|f| !skip(&f.path))
        .flat_map(|f| {
            f.lines
                .iter()
                .filter(|(_, l)| hit(l))
                .map(move |(n, l)| format!("{}:{n}  {}", f.path, l.trim()))
        })
        .collect()
}

/// The IoU formula itself lives in `primitives::sim` — nowhere else.
///
/// Matching on the *arithmetic* rather than on function names is deliberate:
/// plenty of legitimate code is named `…iou…` (config accessors, PyO3 bindings,
/// the marshaling helpers that reshape annotations before calling a kernel).
/// What must not recur is a second site computing `union = a + b - intersection`
/// — that was copy-pasted three times (mask, bbox, OBB) before 0.5.
///
/// # Adding a sanctioned home
///
/// [`FORMULA_HOMES`] is a list, not a constant, because a *genuinely different*
/// formula may legitimately need its own home. The known case is panoptic
/// quality (1.1): panopticapi computes
/// `union = pred_area + gt_area - intersection - void_intersection`, whose
/// void-overlap term has no analogue here, over a label-map co-occurrence
/// histogram rather than a dense `[D][G]` kernel. When that lands, add
/// `primitives/pq.rs` (or wherever it lives) to the list with a comment — do not
/// weaken or `#[ignore]` this test.
#[test]
fn iou_formula_only_in_primitives_sim() {
    /// Files permitted to define a similarity denominator. One entry per
    /// *distinct* formula — never a place that should have called `sim`.
    const FORMULA_HOMES: &[&str] = &["crates/hotcoco/src/primitives/sim.rs"];

    // `union = <area> + <area> - <intersection>`: the IoU denominator. RLE
    // set-union (`merge(.., false)`) has no subtraction and is not caught.
    let violations = scan(
        |path| FORMULA_HOMES.contains(&path),
        |line| line.contains("union") && line.contains('+') && line.contains(" - "),
    );

    assert!(
        violations.is_empty(),
        "a similarity denominator may only be defined in: {}.\n\
         Found what looks like another copy at:\n  {}\n\n\
         Call `primitives::sim` instead — it exposes matrix kernels (`bbox_iou`, \
         `mask_iou`, `obb_iou`) and the scalar `bbox_iou_pair`. If this really is \
         a different formula (see the panoptic note on this test), add its file to \
         FORMULA_HOMES with a comment.",
        FORMULA_HOMES.join(", "),
        violations.join("\n  ")
    );
}

/// One parallelism threshold for every kernel.
///
/// Two of these drifted apart (1024 vs 1000) while a comment insisted they
/// matched — the exact silent divergence this test exists to prevent.
#[test]
fn exactly_one_min_parallel_work_constant() {
    let found = scan(|_| false, |line| line.contains("const MIN_PARALLEL_WORK"));

    assert_eq!(
        found.len(),
        1,
        "expected exactly one MIN_PARALLEL_WORK declaration, found {}:\n  {}\n\n\
         Per-kernel copies drift. Kernels share the one in `primitives::sim`.",
        found.len(),
        found.join("\n  ")
    );
}

/// Matching lives in `primitives/` — no hand-rolled greedy loops elsewhere.
///
/// `best_iou`/`best_gi` is the signature of the pycocotools greedy scan; a second
/// copy is how match semantics (tie-breaking especially, which is public contract
/// via `evalImgs`) drift apart.
///
/// This is a *naming* heuristic, so it can catch an innocent bystander — panoptic
/// (1.1) has a unique-match scan where `best_iou` would be a natural variable
/// name, even though PQ needs no solver and is not re-implementing this matcher.
/// If that happens, rename the variable (`pq_overlap`, say) rather than deleting
/// the check: the tokens are cheap to avoid, and their absence is what makes the
/// guard trustworthy.
#[test]
fn greedy_matching_only_in_primitives() {
    let violations = scan(
        |path| path.contains("/primitives/"),
        |line| line.contains("best_iou") || line.contains("best_gi"),
    );

    assert!(
        violations.is_empty(),
        "greedy matching belongs to `primitives::greedy::greedy_match`.\n\
         Hand-rolled matcher found at:\n  {}",
        violations.join("\n  ")
    );
}

/// The whole-dataset similarity cache stays inside the detection driver.
///
/// `COCOeval::ious` maps every evaluated `(image, category)` cell to its
/// similarity matrix. That is fine for detection — but the 0.5 primitives
/// contract review identified it as the single most likely route by which
/// *retention* leaks into a shared contract, and retention is the one thing the
/// primitives layer forbids: `primitives/mod.rs` states that no contract there may
/// require whole-sequence similarity retention, so HOTA's second pass (1.2) stays
/// free to recompute rather than hold hundreds of megabytes per sequence per
/// thread at MOT20 scale.
///
/// The analysis layer only ever needs one cell, so it goes through
/// `COCOeval::cell_ious(img_id, cat_id)`. The accessor's *shape* is the
/// enforcement — it cannot hand out the map — and this test keeps callers from
/// bypassing it by touching the field directly.
///
/// # Adding a sanctioned home
///
/// [`CACHE_OWNERS`] is the driver, and only the driver: the module that owns the
/// field, the one that populates it, and the one that consumes it per cell. When
/// `eval/` becomes `detection/` at 1.0 these paths are renamed — update them.
/// Do not add an analysis or `primitives` file to this list; needing to is the
/// signal that the code should call `cell_ious` instead.
#[test]
fn similarity_cache_stays_driver_private() {
    /// Files permitted to touch the `ious` field directly.
    const CACHE_OWNERS: &[&str] = &[
        // owns the field and the `cell_ious` accessor
        "crates/hotcoco/src/detection/mod.rs",
        // builds the cache during `evaluate()`
        "crates/hotcoco/src/detection/evaluate.rs",
        // declares the read-only context that carries it into the fan-out, and
        // consumes exactly one cell per call inside that fan-out
        "crates/hotcoco/src/detection/matching.rs",
    ];

    let violations = scan(
        |path| CACHE_OWNERS.contains(&path),
        |line| line.contains(".ious") || line.contains("ious:"),
    );

    assert!(
        violations.is_empty(),
        "the whole-dataset similarity cache is driver-private; only {} may touch \
         it.\nFound a direct field access at:\n  {}\n\n\
         Call `COCOeval::cell_ious(img_id, cat_id)` instead — it hands out one \
         cell and cannot leak the map. Exposing the map as a shared \"similarity \
         cache\" type would foreclose the 1.2 recompute-instead-of-retain lever.",
        CACHE_OWNERS.join(", "),
        violations.join("\n  ")
    );
}
