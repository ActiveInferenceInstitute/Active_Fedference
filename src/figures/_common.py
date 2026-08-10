"""Shared style + path helpers for the Active Fedference figure generators.

Headless (``Agg``) matplotlib only; no ``infrastructure.*`` imports (layer
contract). Every figure generator imports :func:`apply_style`,
:func:`figures_dir` and :func:`save_figure` from here so the colony / free-energy
/ robustness plots share one visual language.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless backend before any pyplot import
import matplotlib.pyplot as plt  # noqa: E402

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
COLOR_MUTED: str = "#7F8C8D"
#: Near-black for reference lines / threshold rules drawn over the data.
COLOR_AXIS: str = "#222222"
#: Mid-grey for annotation box borders and zero lines.
COLOR_GRID: str = "#555555"

#: Federated-network semantic colours (used by graphical_abstract and system_overview).
COLOR_ADVERSARY: str = "#E84855"   # adversarial agent fill / highlight
COLOR_SERVER: str = "#F4A261"      # robust-server / aggregation node fill
COLOR_SERVER_EDGE: str = "#D4833A" # border / outline of server node
COLOR_HONEST_EDGE: str = "#1A5F80" # border of honest-agent circles
COLOR_ADVERSARY_EDGE: str = "#A01020" # border of adversarial-agent circles
COLOR_CORRECT: str = "#27AE60"     # correct-state highlight / success green
COLOR_VARIATE: str = "#2A9D8F"     # variational-method teal
COLOR_NAIVE_LIGHT: str = "#ADB5BD" # light-grey naive-method fill in comparison bars
COLOR_PANEL_BG: str = "#F8F9FA"    # panel / axes background (near-white)
COLOR_PANEL_GRID: str = "#DEE2E6"  # panel border / grid lines (light grey)
COLOR_PANEL_FAIL: str = "#FFF0F0"  # panel background for failure / naive case
COLOR_PANEL_GOOD: str = "#F0FFF4"  # panel background for success / robust case
COLOR_PANEL_NOTE: str = "#E8F8F0"  # annotation box fill for correct-state callout
COLOR_ARROW: str = "#6C757D"       # neutral annotation arrow / secondary text
COLOR_EDGE_LIGHT: str = "#AAAAAA"  # light-grey outline for consensus boxes
COLOR_EDGE_PANEL: str = "#CED4DA"  # legend / panel border (slightly darker than PANEL_GRID)
COLOR_DARK: str = "#343A40"        # axis labels / tick text dark-grey
COLOR_DEEP: str = "#1A1A2E"        # near-black for panel titles / callouts
COLOR_MULTI_1: str = "#1A5F80"     # multi-curve accent 1 (dark teal-blue)
COLOR_MULTI_2: str = "#1B7066"     # multi-curve accent 2 (dark teal-green)

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


def save_figure(fig: "plt.Figure", path: Path) -> Path:
    """Save deterministic PNG and vector PDF companions, then close *fig*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(
        path.with_suffix(".pdf"),
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)
    return path


def save_figure_pair(fig: "plt.Figure", path: Path) -> Path:
    """Save a publication figure as deterministic PNG and PDF companions.

    The manuscript embeds the PNG for predictable HTML and Beamer rendering;
    the sibling PDF preserves a vector-friendly archival artifact.  Creation
    metadata is suppressed so repeated schematic renders are byte-stable in
    tests and do not acquire run-specific timestamps.
    """
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
    "COLOR_VARIATE",
    "FIGURE_EXPORT_DPI",
    "MIN_QUANTITATIVE_FONT_SIZE",
    "MIN_SCHEMATIC_FONT_SIZE",
    "PROJECT_ROOT",
    "ROBUST_CYCLE",
    "annotate_stats_box",
    "apply_style",
    "figures_dir",
    "plt",
    "robust_color",
    "save_figure",
    "save_figure_pair",
    "shade_ci",
]
