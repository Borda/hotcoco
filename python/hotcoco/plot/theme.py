"""Theme definitions, rcParams builder, and style() context manager."""

from __future__ import annotations

from contextlib import contextmanager

from .core import _import_mpl, _resolve_font_family

_THEMES: dict[str, dict] = {
    # Cyanotype. Series 1 leads with the Prussian signature; series 9 sits a
    # step brighter than the TP semantic so no series is exactly a state color.
    "cyanotype": {
        "series": [
            "#23467E",
            "#9E3B39",
            "#6F7B3A",
            "#47807B",
            "#7E5580",
            "#B08A22",
            "#3E9AA8",
            "#CE6B33",
            "#2E7A56",
            "#6E6E72",
        ],
        "chrome": {"text": "#18181A", "label": "#52525A", "tick": "#66666A", "grid": "#E4E4E2", "spine": "#DBDBD8"},
        "background": "#F5F5F4",
        "plot_bg": "#FFFFFF",
        "sequential": ["#F5F5F4", "#23467E", "#0A1730"],
        "cmap": "hotcoco_cyanotype",
    },
    # The same ten hues lifted until they hold against a dark ground: a chart
    # embedded in a dark notebook, dark slide, or the dark docs site should not
    # be the brightest thing there.
    "cyanotype-dark": {
        "series": [
            "#8FB3E2",
            "#D9736E",
            "#A3AF6C",
            "#79B4AE",
            "#B189B3",
            "#DDB855",
            "#74C4D2",
            "#F0A050",
            "#8FBC96",
            "#B4B4B8",
        ],
        "chrome": {"text": "#EAEAEA", "label": "#B0B0B3", "tick": "#8C8C90", "grid": "#2A2A2D", "spine": "#3E3E43"},
        "background": "#141415",
        "plot_bg": "#1C1C1E",
        # Inverted against the light theme: low values sink toward the ground,
        # high values rise toward the light.
        "sequential": ["#141415", "#2E5A96", "#A8C8F0"],
        "cmap": "hotcoco_cyanotype_dark",
        "dark": True,
    },
}


_CMAPS_REGISTERED: set[str] = set()


def _get_theme(name: str) -> dict:
    if name not in _THEMES:
        raise ValueError(f"Unknown theme {name!r}. Choose from: {list(_THEMES)}")
    return _THEMES[name]


def _ensure_cmap(theme: dict) -> None:
    import matplotlib
    from matplotlib.colors import LinearSegmentedColormap

    cmap_name = theme["cmap"]
    if cmap_name not in _CMAPS_REGISTERED:
        cmap = LinearSegmentedColormap.from_list(cmap_name, theme["sequential"])
        try:
            matplotlib.colormaps.register(cmap)
        except ValueError:
            pass
        _CMAPS_REGISTERED.add(cmap_name)


def _build_rc(theme_name: str = "cyanotype", paper_mode: bool = False) -> dict:
    from cycler import cycler

    t = _get_theme(theme_name)
    _ensure_cmap(t)
    bg = "#ffffff" if paper_mode else t["background"]
    plot_bg = t["background"] if paper_mode else t["plot_bg"]
    c = t["chrome"]
    return {
        "figure.facecolor": bg,
        "axes.facecolor": plot_bg,
        "axes.edgecolor": c["spine"],
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": c["grid"],
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.prop_cycle": cycler(color=t["series"]),
        "xtick.color": c["tick"],
        "ytick.color": c["tick"],
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 0.75,
        "ytick.major.width": 0.75,
        "axes.labelcolor": c["label"],
        "axes.labelsize": 11,
        "axes.labelpad": 8,
        "text.color": c["text"],
        "font.family": _resolve_font_family(),
        "legend.frameon": False,
        "image.cmap": t["cmap"],
    }


# Public palette constants — the Cyanotype defaults, for callers styling a
# surface matplotlib does not own.
_DEFAULT_THEME = _THEMES["cyanotype"]

SERIES_COLORS: list[str] = _DEFAULT_THEME["series"]
CHROME: dict[str, str] = {
    **_DEFAULT_THEME["chrome"],
    "background": _DEFAULT_THEME["background"],
    "plot_bg": _DEFAULT_THEME["plot_bg"],
}
SEQUENTIAL: list[str] = _DEFAULT_THEME["sequential"]

# The dark-lifted counterparts, for dark surfaces matplotlib does not own
# (the Plotly dashboard). Same ten hues, raised to hold against #141415.
_DARK_THEME = _THEMES["cyanotype-dark"]

SERIES_COLORS_DARK: list[str] = _DARK_THEME["series"]
SEQUENTIAL_DARK: list[str] = _DARK_THEME["sequential"]
CHROME_DARK: dict[str, str] = {
    **_DARK_THEME["chrome"],
    "background": _DARK_THEME["background"],
    "plot_bg": _DARK_THEME["plot_bg"],
}

# Eval semantics. Deliberately outside the chart palette: a series that happens
# to be green does not mean "true positive". Cyanotype spends red exclusively on
# false positives, which is why the signature is blue and the caveat is plum.
EVAL_COLORS: dict[str, str] = {"tp": "#47714E", "fp": "#B24A2E", "fn": "#5A5FB0"}
EVAL_COLORS_DARK: dict[str, str] = {"tp": "#7FBC98", "fp": "#F0A050", "fn": "#9296EE"}


@contextmanager
def style(theme: str = "cyanotype", paper_mode: bool = False):
    """Context manager that applies a hotcoco matplotlib theme.

    Parameters
    ----------
    theme : str
        ``"cyanotype"`` (default) or ``"cyanotype-dark"``.
    paper_mode : bool
        White figure background with the theme tint on axes. Useful for
        LaTeX inclusion or PowerPoint embedding.

    Usage::

        with hotcoco.plot.style():
            fig, ax = pr_curve(ev)

        with hotcoco.plot.style(paper_mode=True):
            fig, ax = pr_curve(ev)

    All plot functions also accept ``theme`` and ``paper_mode`` directly,
    which is equivalent and more concise for single calls.
    """
    mpl, _, _ = _import_mpl()
    with mpl.rc_context(_build_rc(theme, paper_mode)):
        yield
