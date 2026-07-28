#!/usr/bin/env python3
"""Differential test of `hotcoco.mask` against `pycocotools.mask`.

The RLE codec is a documented pycocotools-compatible surface, and
`pycocotools.mask` is an exact oracle for it — but nothing compared them. Coverage
was transitive through segmentation AP parity, which exercises decode and IoU and
never touches `merge`, `frPyObjects` polygon rasterization, or the string codec.

That gap matters twice over. It is the only unchecked path in a public API, and it
is where segmentation's residual parity difference (AP ~1e-5 on val2017, against
bbox's ~1e-14) has to originate, since the metric arithmetic above it is now exact.

Every operation is compared bit-for-bit. RLE is integer run lengths and masks are
`uint8` — there is no floating point to be tolerant about, so any difference is a
real difference. `iou` returns doubles and is compared exactly for the same reason:
both sides compute intersection and union as integer areas, so the quotients should
be identical.

Run: uv run python scripts/parity_mask.py [--cases N] [--seed S]
"""

from __future__ import annotations

import argparse
import sys

import hotcoco
import numpy as np
import pycocotools.mask as pm

hm = hotcoco.mask

# Every failure seen, as (operation, detail). Collected rather than raised so one
# run reports the whole picture instead of the first thing to break.
FAILURES: list[tuple[str, str]] = []


def fail(op: str, detail: str) -> None:
    FAILURES.append((op, detail))
    if len(FAILURES) <= 12:
        print(f"  FAIL [{op}] {detail}")


def norm_rle(r: dict) -> tuple:
    """RLE dicts as comparable tuples; `counts` may be bytes or str."""
    counts = r["counts"]
    if isinstance(counts, str):
        counts = counts.encode()
    return (tuple(r["size"]), counts)


def random_masks(rng: np.random.Generator, h: int, w: int, n: int) -> np.ndarray:
    """Fortran-order uint8 masks, the layout pycocotools requires.

    Mixes blobs with salt-and-pepper: run-length coding is most likely to diverge
    where runs are long (large blobs) or maximally short (alternating pixels), and
    uniform random noise only ever produces the latter.
    """
    out = np.zeros((h, w, n), dtype=np.uint8, order="F")
    for k in range(n):
        style = rng.integers(0, 4)
        if style == 0:  # solid rectangle
            y0, y1 = sorted(rng.integers(0, h + 1, size=2))
            x0, x1 = sorted(rng.integers(0, w + 1, size=2))
            out[y0:y1, x0:x1, k] = 1
        elif style == 1:  # salt and pepper — shortest possible runs
            out[:, :, k] = rng.integers(0, 2, size=(h, w), dtype=np.uint8)
        elif style == 2:  # a few overlapping blobs
            for _ in range(rng.integers(1, 4)):
                y0, y1 = sorted(rng.integers(0, h + 1, size=2))
                x0, x1 = sorted(rng.integers(0, w + 1, size=2))
                out[y0:y1, x0:x1, k] = 1
        else:  # degenerate: all-empty or all-full
            out[:, :, k] = rng.integers(0, 2, dtype=np.uint8)
    return np.asfortranarray(out)


def check_encode_decode(masks: np.ndarray, ctx: str) -> tuple[list, list]:
    """encode + decode + round-trip. Returns both sides' RLEs for reuse."""
    py_rles = pm.encode(masks)
    hc_rles = hm.encode(masks)

    if len(py_rles) != len(hc_rles):
        fail("encode", f"{ctx}: {len(hc_rles)} RLEs vs pycocotools {len(py_rles)}")
        return py_rles, hc_rles

    for i, (p, h) in enumerate(zip(py_rles, hc_rles)):
        if norm_rle(p) != norm_rle(h):
            fail("encode", f"{ctx}[{i}]: {norm_rle(h)} != {norm_rle(p)}")

    py_dec = pm.decode(py_rles)
    hc_dec = hm.decode(hc_rles)
    if py_dec.shape != hc_dec.shape:
        fail("decode", f"{ctx}: shape {hc_dec.shape} != {py_dec.shape}")
    elif not np.array_equal(py_dec, hc_dec):
        n = int((py_dec != hc_dec).sum())
        fail("decode", f"{ctx}: {n} pixels differ")

    # Round-trip must be the identity — this is what catches a row/column-major
    # transposition that encode and decode both make, and so cancel out on their
    # own output while disagreeing with pycocotools.
    if not np.array_equal(hc_dec, masks):
        n = int((hc_dec != masks).sum())
        fail("roundtrip", f"{ctx}: decode(encode(m)) differs from m in {n} pixels")

    return py_rles, hc_rles


def check_area_bbox(py_rles, hc_rles, ctx: str) -> None:
    py_a, hc_a = pm.area(py_rles), hm.area(hc_rles)
    if not np.array_equal(np.asarray(py_a), np.asarray(hc_a)):
        fail("area", f"{ctx}: {list(hc_a)} != {list(py_a)}")

    py_b, hc_b = pm.toBbox(py_rles), hm.toBbox(hc_rles)
    if not np.array_equal(np.asarray(py_b), np.asarray(hc_b)):
        fail("toBbox", f"{ctx}: {np.asarray(hc_b).tolist()} != {np.asarray(py_b).tolist()}")


def check_iou(py_rles, hc_rles, rng: np.random.Generator, ctx: str) -> None:
    n = len(py_rles)
    if n < 2:
        return
    split = max(1, n // 2)
    dt_p, gt_p = py_rles[:split], py_rles[split:]
    dt_h, gt_h = hc_rles[:split], hc_rles[split:]

    for crowd_mode in ("none", "all", "mixed"):
        if crowd_mode == "none":
            iscrowd = [0] * len(gt_p)
        elif crowd_mode == "all":
            iscrowd = [1] * len(gt_p)
        else:
            iscrowd = rng.integers(0, 2, size=len(gt_p)).tolist()

        py_i = np.asarray(pm.iou(dt_p, gt_p, iscrowd))
        hc_i = np.asarray(hm.iou(dt_h, gt_h, iscrowd))
        if py_i.shape != hc_i.shape:
            fail("iou", f"{ctx}/{crowd_mode}: shape {hc_i.shape} != {py_i.shape}")
            continue
        if not np.array_equal(py_i, hc_i):
            d = np.abs(py_i - hc_i)
            idx = np.unravel_index(int(np.argmax(d)), d.shape)
            fail(
                "iou",
                f"{ctx}/{crowd_mode}: max diff {d.max():.3e} at {idx} "
                f"(hotcoco {hc_i[idx]!r} vs pycocotools {py_i[idx]!r})",
            )


def check_merge(py_rles, hc_rles, ctx: str) -> None:
    for intersect in (False, True):
        p = pm.merge(py_rles, intersect=int(intersect))
        h = hm.merge(hc_rles, intersect=intersect)
        if norm_rle(p) != norm_rle(h):
            fail("merge", f"{ctx}/intersect={intersect}: {norm_rle(h)} != {norm_rle(p)}")


def check_string_codec(hc_rles, ctx: str) -> None:
    """The LEB128-ish `counts` string codec, round-tripped through hotcoco."""
    for i, r in enumerate(hc_rles):
        s = hm.rle_to_string(r)
        back = hm.rle_from_string(s, r["size"][0], r["size"][1])
        if norm_rle(back) != norm_rle(r):
            fail("rle_string", f"{ctx}[{i}]: round-trip changed the RLE")


def check_frpyobjects(rng: np.random.Generator, h: int, w: int, ctx: str) -> None:
    """Polygon and bbox rasterization — reachable only through this entry point."""
    # Polygon: a random simple-ish quad, coordinates as flat [x0,y0,x1,y1,...].
    poly = [
        float(rng.integers(0, w)),
        float(rng.integers(0, h)),
        float(rng.integers(0, w)),
        float(rng.integers(0, h)),
        float(rng.integers(0, w)),
        float(rng.integers(0, h)),
        float(rng.integers(0, w)),
        float(rng.integers(0, h)),
    ]
    p = pm.frPyObjects([poly], h, w)
    hc = hm.frPyObjects([poly], h, w)
    if len(p) != len(hc):
        fail("frPyObjects/poly", f"{ctx}: {len(hc)} RLEs vs {len(p)}")
    else:
        for i, (a, b) in enumerate(zip(p, hc)):
            if norm_rle(a) != norm_rle(b):
                fail("frPyObjects/poly", f"{ctx}[{i}]: poly={poly} -> {norm_rle(b)} != {norm_rle(a)}")

    # Bbox form: [x, y, w, h].
    bbox = [float(rng.integers(0, w)), float(rng.integers(0, h)), float(rng.integers(0, w)), float(rng.integers(0, h))]
    # pycocotools routes a length-4 entry to its bbox path, which requires an
    # ndarray; hotcoco accepts a plain list too. Feed each what it takes.
    pb = pm.frPyObjects(np.array([bbox], dtype=np.float64), h, w)
    hb = hm.frPyObjects([bbox], h, w)
    if len(pb) == len(hb):
        for i, (a, b) in enumerate(zip(pb, hb)):
            if norm_rle(a) != norm_rle(b):
                fail("frPyObjects/bbox", f"{ctx}[{i}]: bbox={bbox} -> {norm_rle(b)} != {norm_rle(a)}")
    else:
        fail("frPyObjects/bbox", f"{ctx}: {len(hb)} RLEs vs {len(pb)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    print("=" * 68)
    print(f"  hotcoco.mask vs pycocotools.mask — {args.cases} cases, seed {args.seed}")
    print("=" * 68)

    for case in range(args.cases):
        # Include 1-pixel and single-row/column shapes: RLE boundary handling is
        # exactly where an off-by-one in the run encoding would hide.
        h = int(rng.choice([1, 1, 2, 3, 7, 16, 32, 64]))
        w = int(rng.choice([1, 2, 3, 5, 16, 32, 64]))
        n = int(rng.integers(1, 6))
        ctx = f"case{case}({h}x{w}x{n})"

        masks = random_masks(rng, h, w, n)
        py_rles, hc_rles = check_encode_decode(masks, ctx)
        if len(py_rles) != len(hc_rles):
            continue

        check_area_bbox(py_rles, hc_rles, ctx)
        check_iou(py_rles, hc_rles, rng, ctx)
        check_merge(py_rles, hc_rles, ctx)
        check_string_codec(hc_rles, ctx)
        check_frpyobjects(rng, h, w, ctx)

    print()
    if not FAILURES:
        print(f"  ALL MASK OPERATIONS MATCH  ({args.cases} cases)")
        return 0

    by_op: dict[str, int] = {}
    for op, _ in FAILURES:
        by_op[op] = by_op.get(op, 0) + 1
    print(f"  {len(FAILURES)} DIVERGENCES across {len(by_op)} operations:")
    for op, n in sorted(by_op.items(), key=lambda kv: -kv[1]):
        print(f"    {op:<22} {n}")
    if len(FAILURES) > 12:
        print("  (first 12 shown above)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
