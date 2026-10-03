---
name: bench
description: Run hotcoco speed benchmarks against pycocotools and faster-coco-eval, then update the README and docs benchmark tables. Use when the user says "benchmark", "run bench", "how fast is it", or after changes that could affect evaluation performance.
---

# Bench

Run speed benchmarks and update README tables.

## Steps

1. Run the full benchmark suite from the repo root:
   ```bash
   just bench
   ```
   This builds the extension if needed, then runs `scripts/bench.py`. Benchmarks
   bbox, segm, and keypoints against pycocotools, faster-coco-eval,
   ultrafast-pycocotools, and vernier on COCO val2017 — the last two are dev
   extras and a missing one just leaves its column blank. Each cell is the
   median of 3 fresh-process runs (`--reps`). Use **wall clock time**, not CPU
   time.

   **Machine state matters more than library versions.** The March 2026 capture
   ran ~1.8× slow across *all three* libraries (throttling or memory pressure on
   the fanless 8 GB M1 Air — numpy and script changes were ruled out by
   experiment). Bench plugged in, on a quiet machine, and run the suite 3× taking
   per-cell medians. Cross-capture absolute times are not comparable; only
   within-capture ratios are.

   For extra flags, run the script directly:
   ```bash
   just build
   uv run python scripts/bench.py --scale 10
   uv run python scripts/bench.py --types bbox segm
   ```

2. Compare new numbers against the current tables.
   `docs/benchmarks.md` is the **canonical owner** of every benchmark and parity
   number. If any hotcoco time changed by more than ~5%:
   - Update `docs/benchmarks.md` first, including the **version label** on each
     table — a table labeled with an old version while reporting new timings is
     the exact drift this ownership rule exists to prevent.
   - Then sync `README.md`'s headline table and any speedup multiplier quoted in
     its prose. Every figure in the README must match `docs/benchmarks.md`
     digit for digit.
   - `docs/index.md` carries only a headline claim and a link — check it still
     reads true, but do not add a table to it.
   - Re-run `just docs-links` if you added or renamed any heading.
   - the "Where the time goes" phase table in `docs/benchmarks.md` — refresh it
     with `uv run python scripts/bench.py --phases` (single run is fine; it is
     labeled as such)

3. Table format:
   - Columns: `Eval Type | pycocotools | faster-coco-eval | ultrafast-pycocotools | vernier | hotcoco`
   - Times in seconds, 2 decimal places
   - Speedups in parentheses vs pycocotools, e.g. `0.74s (15.9x)`
   - The script also prints a peak-RSS table from the same runs; carry it into
     `docs/benchmarks.md` beside the timing table it came from. Every cell runs in
     its own process, so the memory column is per library, not cumulative.

4. Only scale detections (never ground truth) for any synthetic load tests.

5. If numbers changed, run `just parity` to confirm metrics still match before
   committing updated tables.
