"""Shared drawing and ordered frontmatter contract; no inference mathematics.

Scientific anchors are Friston et al. (2024), Figs. 1/4 and Eq. 7, and
Mildner et al. (2025), with the distinct claim owners in manuscript/07.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from ._common import (
    COLOR_ARROW,
    COLOR_DEEP,
    COLOR_EDGE_PANEL,
    COLOR_PANEL_BG,
    COLOR_WHITE,
    apply_style,
    plt,
)
from ._presentation import save_presentation_panels
from ._presentation_flows import _text_panels

# This is explicit placement intent. The sibling producer must implement the
# contract described in the handoff before complete-paper placement is accepted.
FRONTMATTER: tuple[tuple[str, str], ...] = (
    ("frontmatter_model", "00a_graphical_model.md"),
    ("frontmatter_robustness", "00b_graphical_robustness.md"),
    ("frontmatter_execution", "00c_graphical_execution.md"),
)


def canvas(title: str, subtitle: str) -> tuple[Figure, Axes]:
    """Create the shared portrait composition at manuscript text width."""
    apply_style()
    fig = plt.figure(figsize=(6.5, 8.0), facecolor=COLOR_WHITE)
    fig.set_layout_engine("none")
    ax = fig.add_axes((0.035, 0.025, 0.93, 0.95))
    ax.set(xlim=(0, 100), ylim=(0, 100))
    ax.axis("off")
    label(ax, 0, 99, title, size=18, bold=True)
    label(ax, 0, 93, subtitle, size=10.5)
    return fig, ax


def label(
    ax: Axes,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 11,
    bold: bool = False,
    color: str = COLOR_DEEP,
) -> None:
    """Paint selectable text; positions are in the shared hundred-unit frame."""
    ax.text(
        x,
        y,
        text,
        ha="left",
        va="top",
        fontsize=size,
        color=color,
        weight="bold" if bold else "normal",
        linespacing=1.45,
        zorder=5,
    )


def node(ax: Axes, x: float, y: float, w: float, h: float, text: str) -> None:
    """Draw a neutral, square node with a direct semantic label."""
    ax.add_patch(
        Rectangle((x, y), w, h, facecolor=COLOR_PANEL_BG, edgecolor=COLOR_EDGE_PANEL, linewidth=0.8, zorder=1)
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=11,
        linespacing=1.45,
        color=COLOR_DEEP,
        zorder=5,
    )


def arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    dashed: bool = False,
) -> None:
    """A solid causal/data arrow or a visibly dashed optional/sharing edge."""
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "mutation_scale": 13,
            "linewidth": 1.4,
            "color": COLOR_ARROW,
            "linestyle": "--" if dashed else "-",
        },
        zorder=2,
    )


def emit_readings(path: Path, panels: tuple[str, ...]) -> Path:
    """Save a complete ordered projection reading set for a canonical pair."""
    reading = [panel for i, text in enumerate(panels) for panel in _text_panels(f"reading-{i + 1}", text)]
    save_presentation_panels(
        reading,
        canonical_path=path,
        expected_identifiers=tuple(panel.identifier for panel in reading),
    )
    return path


def validate_frontmatter(project_root: Path, *, required: bool = True) -> None:
    """Reject missing/reordered placement, embeds, captions or alternatives.

    This validates source intent only. It does not certify a sibling producer,
    physical complete-paper placement, HTML accessibility, or publication.
    """
    from ._metadata import FIGURE_METADATA

    path = project_root / "manuscript/config.yaml"
    if not required and not path.exists():
        return
    config = yaml.safe_load(path.read_text())
    if not required and "frontmatter" not in config.get("paper", {}):
        return
    expected = [filename for _, filename in FRONTMATTER]
    if config["paper"].get("frontmatter") != expected:
        raise ValueError("frontmatter must follow model, robustness, execution order")
    for stem, filename in FRONTMATTER:
        text = (project_root / "manuscript" / filename).read_text()
        label_id = stem.replace("_", "-")
        if f"../output/figures/{stem}.png){{#fig:{label_id} " not in text:
            raise ValueError(f"missing frontmatter embed: {stem}")
        if "Source relation:" not in text or "no measured estimand" not in text:
            raise ValueError(f"missing conceptual caption: {stem}")
        metadata = FIGURE_METADATA[stem]
        if not metadata["alt_text"] or not metadata["long_description"]:
            raise ValueError(f"missing frontmatter alternative: {stem}")
