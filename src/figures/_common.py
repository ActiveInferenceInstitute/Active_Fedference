"""Shared style + path helpers for the Active Fedference figure generators.

Headless (``Agg``) matplotlib only; no ``infrastructure.*`` imports (layer
contract). Every figure generator imports :func:`apply_style`,
:func:`figures_dir` and :func:`save_figure` from here so the colony / free-energy
/ robustness plots share one visual language.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TypeAlias

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless backend before any pyplot import
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb, to_rgba  # noqa: E402
from matplotlib.text import Text  # noqa: E402

#: Project root (two levels above ``src/figures/_common.py``).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

# Publication contracts shared by all generators. Schematic labels may be
# slightly smaller than quantitative labels because they must fit node layouts,
# but neither floor is allowed to drift silently.
MIN_QUANTITATIVE_FONT_SIZE: float = 9.5
MIN_SCHEMATIC_FONT_SIZE: float = 8.5
FIGURE_EXPORT_DPI: int = 220

#: Shared palette: naive (Friston) vs robust (FedGVI) consensus. The pair is
#: chosen for a large luminance gap (deep brick red vs sky blue, ΔL ~ 0.30) so the
#: two headline series stay distinguishable in a greyscale print, not only in
#: colour (gated by tests/figures/test_palette.py).
COLOR_NAIVE: str = "#922B21"
COLOR_ROBUST: str = "#5DADE2"
COLOR_ACCENT: str = "#2C3E50"
#: Muted grey for secondary / de-emphasized series (e.g. a negated EFE term).
COLOR_MUTED: str = "#626D6E"
#: Near-black for reference lines / threshold rules drawn over the data.
COLOR_AXIS: str = "#222222"
#: Mid-grey for annotation box borders and zero lines.
COLOR_GRID: str = "#555555"

#: Federated-network semantic colours (used by graphical_abstract and system_overview).
COLOR_ADVERSARY: str = "#E84855"  # adversarial agent fill / highlight
COLOR_SERVER: str = "#F4A261"  # robust-server / aggregation node fill
COLOR_SERVER_EDGE: str = "#D4833A"  # border / outline of server node
COLOR_HONEST_EDGE: str = "#1A5F80"  # border of honest-agent circles
COLOR_ADVERSARY_EDGE: str = "#A01020"  # border of adversarial-agent circles
COLOR_CORRECT: str = "#27AE60"  # correct-state highlight / success green
COLOR_VARIATE: str = "#2A9D8F"  # variational-method teal
COLOR_NAIVE_LIGHT: str = "#ADB5BD"  # light-grey naive-method fill in comparison bars
COLOR_PANEL_BG: str = "#F8F9FA"  # panel / axes background (near-white)
COLOR_PANEL_GRID: str = "#DEE2E6"  # panel border / grid lines (light grey)
COLOR_PANEL_FAIL: str = "#FFF0F0"  # panel background for failure / naive case
COLOR_PANEL_GOOD: str = "#F0FFF4"  # panel background for success / robust case
COLOR_PANEL_NOTE: str = "#E8F8F0"  # annotation box fill for correct-state callout
COLOR_ARROW: str = "#626D6E"  # neutral annotation arrow / secondary text
COLOR_EDGE_LIGHT: str = "#AAAAAA"  # light-grey outline for consensus boxes
COLOR_EDGE_PANEL: str = "#CED4DA"  # legend / panel border (slightly darker than PANEL_GRID)
COLOR_DARK: str = "#343A40"  # axis labels / tick text dark-grey
COLOR_DEEP: str = "#1A1A2E"  # near-black for panel titles / callouts
COLOR_MULTI_1: str = "#1A5F80"  # multi-curve accent 1 (dark teal-blue)
COLOR_MULTI_2: str = "#1B7066"  # multi-curve accent 2 (dark teal-green)
COLOR_BLACK: str = "#000000"  # maximum-contrast dark text on arbitrary fills
COLOR_WHITE: str = "#FFFFFF"  # accessible light text / open canvas
COLOR_GOLD: str = "#A65A00"  # additional operating point
COLOR_PURPLE: str = "#6F58A8"  # additional operating point
COLOR_SIGNED_NEGATIVE: str = "#3B4CC0"  # signed-map negative endpoint
COLOR_SIGNED_NEUTRAL: str = "#F7F7F7"  # signed-map zero midpoint
COLOR_SIGNED_POSITIVE: str = "#B35806"  # signed-map positive endpoint

LineStyle: TypeAlias = str | tuple[int, tuple[int, ...]]
ColorSpec: TypeAlias = str | tuple[float, float, float] | tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class SemanticStyle:
    """Non-colour-complete drawing style for one visual meaning.

    ``keyline`` is the contrasting outline used for filled marks.  The tuple is
    deliberately complete enough for line, point, and bar variants so figure
    modules do not invent a second semantic vocabulary.
    """

    color: str
    marker: str
    dash: LineStyle
    hatch: str
    linewidth: float
    keyline: str


# One immutable registry is the source of visual meaning across the figure
# collection.  Colour is never the sole cue: every simultaneously plotted role
# differs by marker, dash, hatch/open fill, or direct role label as appropriate.
SEMANTIC_STYLES = MappingProxyType(
    {
        "naive": SemanticStyle(
            color=COLOR_NAIVE,
            marker="o",
            dash="-",
            hatch="",
            linewidth=2.4,
            keyline="#541813",
        ),
        "reference": SemanticStyle(
            color=COLOR_NAIVE,
            marker="o",
            dash="-",
            hatch="",
            linewidth=2.4,
            keyline="#541813",
        ),
        "heuristic_robust": SemanticStyle(
            color=COLOR_ROBUST,
            marker="s",
            dash="--",
            hatch="///",
            linewidth=2.0,
            keyline=COLOR_HONEST_EDGE,
        ),
        "variational": SemanticStyle(
            color=COLOR_VARIATE,
            marker="^",
            dash="-.",
            hatch="xx",
            linewidth=2.0,
            keyline=COLOR_MULTI_2,
        ),
        "operating_point_1": SemanticStyle(
            color=COLOR_GOLD,
            marker="D",
            dash=":",
            hatch="..",
            linewidth=1.8,
            keyline="#613400",
        ),
        "operating_point_2": SemanticStyle(
            color=COLOR_PURPLE,
            marker="p",
            dash=(0, (7, 2)),
            hatch="\\\\",
            linewidth=1.8,
            keyline="#3C2C68",
        ),
        "operating_point_3": SemanticStyle(
            color=COLOR_MULTI_1,
            marker="v",
            dash=(0, (5, 1, 1, 1)),
            hatch="++",
            linewidth=1.8,
            keyline="#0E3549",
        ),
        "operating_point_4": SemanticStyle(
            color=COLOR_MULTI_2,
            marker="h",
            dash=(0, (1, 1, 5, 1)),
            hatch="oo",
            linewidth=1.8,
            keyline="#0E3B36",
        ),
        "reference_rule": SemanticStyle(
            color=COLOR_AXIS,
            marker="",
            dash=":",
            hatch="",
            linewidth=1.3,
            keyline=COLOR_AXIS,
        ),
        "adversarial": SemanticStyle(
            color=COLOR_ADVERSARY,
            marker="X",
            dash="--",
            hatch="xx",
            linewidth=1.8,
            keyline=COLOR_ADVERSARY_EDGE,
        ),
        "honest": SemanticStyle(
            color=COLOR_ROBUST,
            marker="o",
            dash="-",
            hatch="",
            linewidth=1.8,
            keyline=COLOR_HONEST_EDGE,
        ),
        "display_band": SemanticStyle(
            color=COLOR_GRID,
            marker="",
            dash=":",
            hatch="////",
            linewidth=1.0,
            keyline=COLOR_GRID,
        ),
    }
)

#: Qualitative palette for the *robust* divergence curves (everything that is not
#: the naive ``KLD`` baseline). Color-blind-safe, ordered so adjacent methods
#: stay distinguishable. ``COLOR_NAIVE`` is reserved for the naive baseline.
ROBUST_CYCLE: tuple[str, ...] = (
    "#1F77B4",  # blue
    "#2CA02C",  # green
    "#9467BD",  # purple
    "#FF7F0E",  # orange
    "#17BECF",  # cyan
    "#8C564B",  # brown
)


def robust_color(index: int) -> str:
    """Return a stable color from :data:`ROBUST_CYCLE` for a robust-method index."""
    return ROBUST_CYCLE[index % len(ROBUST_CYCLE)]


def semantic_style(role: str) -> SemanticStyle:
    """Return the immutable style for *role*, failing closed on unknown roles."""
    try:
        return SEMANTIC_STYLES[role]
    except KeyError as exc:
        known = ", ".join(sorted(SEMANTIC_STYLES))
        raise ValueError(f"unknown semantic style {role!r}; expected one of: {known}") from exc


def relative_luminance(color: ColorSpec) -> float:
    """Return WCAG relative luminance for a Matplotlib colour specification."""
    red, green, blue = to_rgb(color)

    def _linearize(component: float) -> float:
        return component / 12.92 if component <= 0.04045 else ((component + 0.055) / 1.055) ** 2.4

    return (
        0.2126 * _linearize(float(red)) + 0.7152 * _linearize(float(green)) + 0.0722 * _linearize(float(blue))
    )


def contrast_ratio(foreground: ColorSpec, background: ColorSpec) -> float:
    """Return the WCAG contrast ratio between two opaque colours."""
    light, dark = sorted(
        (relative_luminance(foreground), relative_luminance(background)),
        reverse=True,
    )
    return (light + 0.05) / (dark + 0.05)


def require_contrast(
    foreground: str,
    background: str,
    *,
    large_text: bool = False,
) -> float:
    """Validate text contrast and return the measured ratio.

    Normal text requires 4.5:1.  The 3:1 exception is available only when the
    caller has already established genuinely large text.
    """
    ratio = contrast_ratio(foreground, background)
    minimum = 3.0 if large_text else 4.5
    if ratio + 1e-12 < minimum:
        raise ValueError(
            f"text contrast {ratio:.2f}:1 is below the required {minimum:.1f}:1 "
            f"for foreground {foreground} on background {background}"
        )
    return ratio


def contrasting_text_color(background: str, *, large_text: bool = False) -> str:
    """Choose accessible dark or light text for *background*.

    The higher-contrast candidate wins; the selected pair is then validated at
    4.5:1 for normal text or 3:1 for explicitly large text.
    """
    # Pure black and white provide the mathematical contrast envelope for an
    # arbitrary opaque sRGB background.  A branded near-black can leave a
    # narrow mid-luminance band below 4.5:1 even when white also fails.
    candidates = (COLOR_BLACK, COLOR_WHITE)
    selected = max(candidates, key=lambda color: contrast_ratio(color, background))
    require_contrast(selected, background, large_text=large_text)
    return selected


def signed_difference_colormap() -> LinearSegmentedColormap:
    """Return the project zero-centred, colour-vision-deficiency-safe map."""
    return LinearSegmentedColormap.from_list(
        "fedference_signed_difference",
        (COLOR_SIGNED_NEGATIVE, COLOR_SIGNED_NEUTRAL, COLOR_SIGNED_POSITIVE),
    )


def _composite_rgba(
    foreground: Sequence[float],
    background: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Alpha-composite one RGBA colour over an opaque RGB background."""
    red, green, blue, alpha = (float(component) for component in foreground)
    return (
        alpha * red + (1.0 - alpha) * background[0],
        alpha * green + (1.0 - alpha) * background[1],
        alpha * blue + (1.0 - alpha) * background[2],
    )


def _effective_text_background(
    artist: Text,
    fig: "plt.Figure",
) -> tuple[float, float, float]:
    """Resolve the painted background at a visible text artist's anchor.

    The resolver composites the figure and axes faces, then any lower-z-order
    axes patches or raster images containing the text anchor, and finally the
    artist's own bounding box. This covers ordinary labels, text boxes, bar
    annotations, and heatmap-cell labels without assuming an opaque canvas.
    """
    figure_rgba = to_rgba(fig.get_facecolor())
    background = _composite_rgba(figure_rgba, (1.0, 1.0, 1.0))
    axis = artist.axes
    if axis is not None:
        background = _composite_rgba(to_rgba(axis.get_facecolor()), background)
        transformed_anchor = artist.get_transform().transform(artist.get_position())
        anchor = (float(transformed_anchor[0]), float(transformed_anchor[1]))
        artist_zorder = float(artist.get_zorder())
        layers: list[tuple[float, tuple[float, float, float, float]]] = []
        for patch in axis.patches:
            if (
                patch.get_visible()
                and float(patch.get_zorder()) <= artist_zorder
                and patch.contains_point(anchor)
            ):
                layers.append((float(patch.get_zorder()), to_rgba(patch.get_facecolor())))
        if axis.images:
            data_x, data_y = axis.transData.inverted().transform(anchor)
            for image in axis.images:
                if not image.get_visible() or float(image.get_zorder()) > artist_zorder:
                    continue
                left, right, bottom, top = (float(value) for value in image.get_extent())
                if not (min(left, right) <= data_x <= max(left, right)):
                    continue
                if not (min(bottom, top) <= data_y <= max(bottom, top)):
                    continue
                values = np.asanyarray(image.get_array())
                if values.ndim < 2 or values.shape[0] == 0 or values.shape[1] == 0:
                    continue
                col_fraction = (data_x - left) / (right - left) if right != left else 0.0
                row_fraction = (data_y - bottom) / (top - bottom) if top != bottom else 0.0
                col = int(np.clip(np.floor(col_fraction * values.shape[1]), 0, values.shape[1] - 1))
                row = int(np.clip(np.floor(row_fraction * values.shape[0]), 0, values.shape[0] - 1))
                sample = values[row, col]
                if np.ma.is_masked(sample):
                    continue
                if values.ndim == 2:
                    image_rgba = image.cmap(image.norm(float(sample)))
                else:
                    components = [float(value) for value in np.ravel(sample)]
                    if len(components) == 3:
                        image_rgba = to_rgba((components[0], components[1], components[2]))
                    elif len(components) == 4:
                        image_rgba = to_rgba(
                            (components[0], components[1], components[2], components[3])
                        )
                    else:
                        continue
                layers.append((float(image.get_zorder()), to_rgba(image_rgba)))
        for _, rgba in sorted(layers, key=lambda layer: layer[0]):
            background = _composite_rgba(rgba, background)

    bbox_patch = artist.get_bbox_patch()
    if bbox_patch is not None and bbox_patch.get_visible():
        background = _composite_rgba(to_rgba(bbox_patch.get_facecolor()), background)
    return background


def _is_genuinely_large_text(artist: Text) -> bool:
    """Return whether the text qualifies for the 3:1 large-text exception."""
    size = float(artist.get_fontsize())
    weight = artist.get_fontweight()
    bold = (isinstance(weight, str) and weight.lower() in {"bold", "heavy", "semibold", "demibold"}) or (
        isinstance(weight, (int, float)) and not isinstance(weight, bool) and float(weight) >= 700.0
    )
    return size >= 18.0 or (bold and size >= 14.0)


def validate_figure_text(
    fig: "plt.Figure",
    *,
    minimum_font_size: float,
) -> None:
    """Validate every visible text artist's size and effective contrast."""
    if minimum_font_size <= 0:
        raise ValueError("minimum_font_size must be positive")
    # Materialise tick labels and legend text before traversing artists.
    fig.canvas.draw()
    undersized: list[str] = []
    low_contrast: list[str] = []
    for artist in fig.findobj(match=Text):
        text = artist.get_text().strip()
        if not artist.get_visible() or not text:
            continue
        size = float(artist.get_fontsize())
        if size + 1e-12 < minimum_font_size:
            compact = " ".join(text.split())
            undersized.append(f"{compact[:72]!r} ({size:g} pt)")
        background = _effective_text_background(artist, fig)
        foreground_rgba = list(to_rgba(artist.get_color()))
        artist_alpha = artist.get_alpha()
        if artist_alpha is not None:
            foreground_rgba[3] *= float(artist_alpha)
        painted_foreground = _composite_rgba(foreground_rgba, background)
        ratio = contrast_ratio(painted_foreground, background)
        minimum_contrast = 3.0 if _is_genuinely_large_text(artist) else 4.5
        if ratio + 1e-12 < minimum_contrast:
            compact = " ".join(text.split())
            low_contrast.append(f"{compact[:72]!r} ({ratio:.2f}:1; required {minimum_contrast:.1f}:1)")
    if undersized:
        details = ", ".join(undersized[:8])
        if len(undersized) > 8:
            details += f", and {len(undersized) - 8} more"
        raise ValueError(f"visible figure text below {minimum_font_size:g} pt: {details}")
    if low_contrast:
        details = ", ".join(low_contrast[:8])
        if len(low_contrast) > 8:
            details += f", and {len(low_contrast) - 8} more"
        raise ValueError(f"visible figure text has insufficient effective contrast: {details}")


def apply_style() -> None:
    """Apply the shared publication style to matplotlib's global rcParams."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": FIGURE_EXPORT_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
            "figure.autolayout": True,  # tight_layout for every figure
            "font.family": "DejaVu Sans",
            "font.size": 12.5,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 13,
            "axes.grid": True,
            "axes.facecolor": COLOR_PANEL_BG,
            "grid.alpha": 0.34,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.frameon": True,
            "legend.framealpha": 0.85,
            "legend.fontsize": 10.5,
        }
    )


def figures_dir(project_root: Path | None = None) -> Path:
    """Return (creating if needed) the ``output/figures`` directory."""
    root = project_root or PROJECT_ROOT
    out = root / "output" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_figure(
    fig: "plt.Figure",
    path: Path,
    *,
    minimum_font_size: float = MIN_QUANTITATIVE_FONT_SIZE,
) -> Path:
    """Validate and save deterministic quantitative PNG/PDF companions."""
    validate_figure_text(fig, minimum_font_size=minimum_font_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(
        path.with_suffix(".pdf"),
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)
    return path


def save_figure_pair(
    fig: "plt.Figure",
    path: Path,
    *,
    minimum_font_size: float = MIN_SCHEMATIC_FONT_SIZE,
) -> Path:
    """Save a publication figure as deterministic PNG and PDF companions.

    The manuscript embeds the PNG for predictable HTML and Beamer rendering;
    the sibling PDF preserves a vector-friendly archival artifact.  Creation
    metadata is suppressed so repeated schematic renders are byte-stable in
    tests and do not acquire run-specific timestamps.
    """
    validate_figure_text(fig, minimum_font_size=minimum_font_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(
        path.with_suffix(".pdf"),
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)
    return path


def annotate_stats_box(
    ax: "plt.Axes",
    text: str,
    *,
    loc: str = "upper right",
    fontsize: float = 10.0,
    alpha: float = 0.85,
) -> None:
    """Draw a rounded white bbox text annotation on *ax*.

    Args:
        ax: The matplotlib :class:`~matplotlib.axes.Axes` to annotate.
        text: The annotation string (may contain newlines).
        loc: One of ``"upper right"``, ``"upper left"``, ``"lower right"``,
            ``"lower left"``.  Maps to ``(x, y)`` axes-fraction coordinates.
        fontsize: Font size in points.
        alpha: Background box opacity.
    """
    _loc_map: dict[str, tuple[float, float, str, str]] = {
        "upper right": (0.97, 0.97, "right", "top"),
        "upper left": (0.03, 0.97, "left", "top"),
        "lower right": (0.97, 0.03, "right", "bottom"),
        "lower left": (0.03, 0.03, "left", "bottom"),
    }
    x, y, ha, va = _loc_map.get(loc, _loc_map["upper right"])
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=fontsize,
        ha=ha,
        va=va,
        bbox={
            "boxstyle": "round,pad=0.35",
            "fc": "white",
            "ec": COLOR_GRID,
            "alpha": alpha,
        },
    )


def shade_ci(
    ax: "plt.Axes",
    xs: "Sequence[float] | np.ndarray",
    lo: "Sequence[float] | np.ndarray",
    hi: "Sequence[float] | np.ndarray",
    color: str,
    *,
    alpha: float = 0.18,
) -> None:
    """Shade a confidence interval band on *ax* using :func:`fill_between`.

    Args:
        ax: The matplotlib :class:`~matplotlib.axes.Axes` to draw on.
        xs: x-axis values.
        lo: Lower CI bound (same length as *xs*).
        hi: Upper CI bound (same length as *xs*).
        color: Fill colour (any matplotlib colour spec).
        alpha: Opacity of the shaded band.
    """
    ax.fill_between(xs, lo, hi, color=color, alpha=alpha)


__all__ = [
    "COLOR_ACCENT",
    "COLOR_ADVERSARY",
    "COLOR_ADVERSARY_EDGE",
    "COLOR_ARROW",
    "COLOR_AXIS",
    "COLOR_BLACK",
    "COLOR_CORRECT",
    "COLOR_DARK",
    "COLOR_DEEP",
    "COLOR_EDGE_LIGHT",
    "COLOR_EDGE_PANEL",
    "COLOR_GRID",
    "COLOR_HONEST_EDGE",
    "COLOR_MULTI_1",
    "COLOR_MULTI_2",
    "COLOR_MUTED",
    "COLOR_NAIVE",
    "COLOR_NAIVE_LIGHT",
    "COLOR_PANEL_BG",
    "COLOR_PANEL_FAIL",
    "COLOR_PANEL_GOOD",
    "COLOR_PANEL_GRID",
    "COLOR_PANEL_NOTE",
    "COLOR_ROBUST",
    "COLOR_SERVER",
    "COLOR_SERVER_EDGE",
    "COLOR_SIGNED_NEGATIVE",
    "COLOR_SIGNED_NEUTRAL",
    "COLOR_SIGNED_POSITIVE",
    "COLOR_VARIATE",
    "COLOR_WHITE",
    "COLOR_GOLD",
    "COLOR_PURPLE",
    "FIGURE_EXPORT_DPI",
    "MIN_QUANTITATIVE_FONT_SIZE",
    "MIN_SCHEMATIC_FONT_SIZE",
    "PROJECT_ROOT",
    "ROBUST_CYCLE",
    "SEMANTIC_STYLES",
    "SemanticStyle",
    "annotate_stats_box",
    "apply_style",
    "contrast_ratio",
    "contrasting_text_color",
    "figures_dir",
    "plt",
    "relative_luminance",
    "require_contrast",
    "robust_color",
    "semantic_style",
    "signed_difference_colormap",
    "save_figure",
    "save_figure_pair",
    "shade_ci",
    "validate_figure_text",
]
