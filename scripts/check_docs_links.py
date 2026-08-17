"""Resolve every internal Markdown link and anchor under docs/ (plus README.md).

Reports three failure kinds:
  MISSING FILE    — relative link to a path that does not exist
  MISSING ANCHOR  — link to a heading that no target page defines
  ORPHAN/NAV      — page not in zensical nav, or nav entry with no file
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import tomllib

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
while not (ROOT / "zensical.toml").exists():
    if ROOT == ROOT.parent:
        sys.exit("zensical.toml not found — pass the repo root as argv[1]")
    ROOT = ROOT.parent

DOCS = ROOT / "docs"
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*$", re.M)
EXPLICIT_ID = re.compile(r"\{#([\w-]+)\}\s*$")


def slugify(text: str) -> str:
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = text.strip().lower()
    # python-markdown's toc slugify: strip non-word chars, collapse space/hyphen
    # runs to one hyphen. Underscores are \w and survive — `f_scores` stays
    # `f_scores`, it does not become `f-scores`.
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"[-\s]+", "-", text).strip("-")


def anchors_of(path: Path) -> set[str]:
    out: set[str] = set()
    for raw in HEADING.findall(path.read_text(encoding="utf-8")):
        explicit = EXPLICIT_ID.search(raw)
        if explicit:
            out.add(explicit.group(1))
            raw = EXPLICIT_ID.sub("", raw)
        out.add(slugify(raw))
    return out


pages = sorted(DOCS.rglob("*.md")) + [ROOT / "README.md", ROOT / "CONTRIBUTING.md"]
anchor_cache = {p: anchors_of(p) for p in pages if p.exists()}
problems: list[str] = []

for page in pages:
    if not page.exists():
        continue
    for target in LINK.findall(page.read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        rel, _, anchor = target.partition("#")
        here = page.relative_to(ROOT)
        if not rel:  # same-page anchor
            if anchor and anchor not in anchor_cache[page]:
                problems.append(f"MISSING ANCHOR  {here} -> #{anchor} (same page)")
            continue
        dest = (page.parent / rel).resolve()
        if not dest.exists():
            problems.append(f"MISSING FILE    {here} -> {rel}")
            continue
        if anchor and dest.suffix == ".md":
            if dest not in anchor_cache:
                anchor_cache[dest] = anchors_of(dest)
            if anchor not in anchor_cache[dest]:
                problems.append(f"MISSING ANCHOR  {here} -> {rel}#{anchor}")

# Nav coverage
nav_files: set[str] = set()


def walk(node) -> None:
    if isinstance(node, str):
        nav_files.add(node)
    elif isinstance(node, list):
        for item in node:
            walk(item)
    elif isinstance(node, dict):
        for value in node.values():
            walk(value)


cfg = tomllib.loads((ROOT / "zensical.toml").read_text(encoding="utf-8"))
walk(cfg["project"]["nav"])

for entry in sorted(nav_files):
    if not (DOCS / entry).exists():
        problems.append(f"NAV -> MISSING  zensical.toml lists {entry}")
for page in sorted(DOCS.rglob("*.md")):
    rel = page.relative_to(DOCS).as_posix()
    if rel not in nav_files:
        problems.append(f"ORPHAN PAGE     {rel} not in nav (would still be built)")

print(f"checked {len(pages)} pages, {len(nav_files)} nav entries")
if problems:
    print(f"\n{len(problems)} problem(s):\n")
    print("\n".join(problems))
    sys.exit(1)
print("all internal links, anchors, and nav entries resolve")
