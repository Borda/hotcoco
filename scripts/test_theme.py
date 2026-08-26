"""The Cyanotype design system, asserted rather than asserted-in-prose.

Six surfaces each hold their own copy of the theme values, and only three of
them can import Python. Before this file the spec was a markdown convention
with nothing stopping a copy from drifting — which is exactly how the previous
system ended up with the same colour in five places at four values.

These tests are cheap and they fail loudly. To prove one still works, change a
hex in `style.css` and watch it go red.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
STYLE_CSS = ROOT / "python" / "hotcoco" / "static" / "style.css"
BASE_HTML = ROOT / "python" / "hotcoco" / "templates" / "base.html"
FONTS_DIR = ROOT / "python" / "hotcoco" / "_fonts"

pytest.importorskip("matplotlib", reason="theme constants live behind the plot extra")

from hotcoco.plot.theme import CHROME_DARK, EVAL_COLORS_DARK, SERIES_COLORS_DARK  # noqa: E402


def _root_tokens() -> dict[str, str]:
    """Parse the `:root { … }` custom properties out of the browse stylesheet."""
    css = STYLE_CSS.read_text()
    block = re.search(r":root\s*\{(.*?)\n\}", css, re.S)
    assert block, "style.css has no :root block"
    return {m.group(1): m.group(2).strip() for m in re.finditer(r"(--[\w-]+)\s*:\s*([^;]+);", block.group(1))}


TOKENS = _root_tokens()


def _same(a: str, b: str) -> bool:
    return a.strip().lstrip("#").lower() == b.strip().lstrip("#").lower()


# ---------------------------------------------------------------------------
# style.css cannot import Python, so it is the copy most able to drift.
# ---------------------------------------------------------------------------


def test_browse_accent_is_the_dark_signature():
    assert _same(TOKENS["--accent"], SERIES_COLORS_DARK[0])


def test_browse_caveat_is_palette_slot_five():
    """The non-comparable flag. plot/report.py and dashboard.py derive it."""
    assert _same(TOKENS["--caveat"], SERIES_COLORS_DARK[4])


@pytest.mark.parametrize("state", ["tp", "fp", "fn"])
def test_browse_eval_semantics_match_the_palette(state):
    assert _same(TOKENS[f"--{state}"], EVAL_COLORS_DARK[state])


@pytest.mark.parametrize(
    ("token", "chrome_key"),
    [
        ("--bg-base", "background"),
        ("--bg-surface", "plot_bg"),
        ("--text-primary", "text"),
        ("--text-secondary", "label"),
        ("--text-tertiary", "tick"),
        ("--border-subtle", "grid"),
        ("--border-strong", "spine"),
    ],
)
def test_browse_chrome_matches_the_dark_theme(token, chrome_key):
    assert _same(TOKENS[token], CHROME_DARK[chrome_key])


def test_dashboard_constants_are_derived_not_copied():
    """The dashboard imports from theme.py; catch a regression to literals."""
    from hotcoco import dashboard

    assert dashboard._ACCENT == SERIES_COLORS_DARK[0]
    assert dashboard._CAVEAT == SERIES_COLORS_DARK[4]
    assert dashboard._TEXT_PRIMARY == CHROME_DARK["text"]
    assert dashboard._BORDER_SUBTLE == CHROME_DARK["grid"]


# ---------------------------------------------------------------------------
# Fonts: the browse UI must render with no CDN, so every face it asks for by
# URL has to exist on disk.
# ---------------------------------------------------------------------------


def test_every_served_font_file_exists():
    """A @font-face pointing at a missing TTF is a 404 on every page load."""
    requested = re.findall(r"url\('/fonts/([^']+)'\)", BASE_HTML.read_text())
    assert requested, "base.html declares no @font-face rules"
    missing = [f for f in requested if not (FONTS_DIR / f).is_file()]
    assert not missing, f"base.html requests fonts not vendored in _fonts/: {missing}"


def test_matplotlib_resolves_only_installed_families():
    """Naming a missing family costs a findfont warning per text object."""
    from hotcoco.plot.core import _resolve_font_family
    from matplotlib import font_manager

    # Resolve first: it registers the vendored TTFs, which is what puts them
    # in ttflist. Snapshotting before the call reads a font manager that has
    # not seen _fonts/ yet.
    resolved = _resolve_font_family()
    available = {f.name for f in font_manager.fontManager.ttflist}
    assert set(resolved) <= available


# ---------------------------------------------------------------------------
# The retired palette must not creep back in.
# ---------------------------------------------------------------------------


def test_no_cold_brew_theme_names_survive():
    from hotcoco.plot.theme import _THEMES

    assert set(_THEMES) == {"cyanotype", "cyanotype-dark"}


def test_red_belongs_to_false_positives():
    """Cyanotype's second rule: the signature gave up red so FP could own it."""
    assert not _same(TOKENS["--accent"], EVAL_COLORS_DARK["fp"])
    assert not _same(TOKENS["--caveat"], EVAL_COLORS_DARK["fp"])


def test_matplotlib_actually_uses_the_body_face():
    """Cyanotype's body face must win, not merely appear in the stack."""
    from hotcoco.plot.core import _resolve_font_family

    assert _resolve_font_family()[0] == "IBM Plex Sans"


def test_the_vendored_weights_cover_what_the_code_asks_for():
    """style.css and plot/core.py request 400/500/600; all three must ship."""
    from matplotlib import font_manager

    weights = {
        font_manager.ttfFontProperty(font_manager.get_font(str(ttf))).weight
        for ttf in sorted(FONTS_DIR.glob("IBMPlexSans-*.ttf"))
    }
    assert {400, 500, 600} <= weights, f"vendored weights: {sorted(weights)}"
