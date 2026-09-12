"""Sentinel-world, active-inference, and federation-loop schematic.

The figure makes the Friston et al. (2024) setting concrete: several agents
observe the same hidden nine-cell location through private categorical sensors,
infer local posteriors, and exchange those posteriors. A separate lower panel
shows the optional transition and action pathway.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from figures._common import (
    COLOR_ACCENT,
    COLOR_ARROW,
    COLOR_DARK,
    COLOR_DEEP,
    COLOR_EDGE_PANEL,
    COLOR_MULTI_1,
    COLOR_PANEL_BG,
    COLOR_PANEL_NOTE,
    COLOR_ROBUST,
    COLOR_VARIATE,
    apply_style,
    contrasting_text_color,
    figures_dir,
    save_figure_pair,
)
from figures._presentation_schematics import pomdp_presentation


def _panel(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str,
) -> None:
    """Draw a panel in normalized figure coordinates."""
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012",
            linewidth=0.9,
            edgecolor=COLOR_EDGE_PANEL,
            facecolor="white",
        )
    )
    ax.text(
        x + 0.018,
        y + h - 0.035,
        title,
        ha="left",
        va="top",
        fontsize=10.2,
        fontweight="bold",
        color=COLOR_DEEP,
    )
    ax.text(
        x + 0.018, y + h - 0.068,
        textwrap.fill(subtitle, width=95, break_long_words=False, break_on_hyphens=False),
        ha="left", va="top", fontsize=8.5, color=COLOR_ARROW,
    )


def _world_grid(ax: plt.Axes, x: float, y: float, size: float, highlighted: int = 4) -> None:
    """Draw the shared hidden nine-cell world and its den."""
    cell = size / 3.0
    for row in range(3):
        for col in range(3):
            index = row * 3 + col
            fill = COLOR_PANEL_NOTE if index == highlighted else COLOR_PANEL_BG
            ax.add_patch(
                mpatches.Rectangle(
                    (x + col * cell, y + (2 - row) * cell),
                    cell,
                    cell,
                    facecolor=fill,
                    edgecolor="white",
                    linewidth=1.0,
                )
            )
            if index == highlighted:
                ax.text(
                    x + (col + 0.5) * cell,
                    y + (2 - row + 0.5) * cell,
                    "den",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color=COLOR_DEEP,
                    fontweight="bold",
                )
    ax.add_patch(mpatches.Rectangle((x, y), size, size, fill=False, edgecolor=COLOR_ACCENT, linewidth=1.0))


def _agent(ax: plt.Axes, x: float, y: float, label: str, color: str) -> None:
    """Draw a sentinel agent and its visual field line."""
    ax.add_patch(mpatches.Circle((x, y), 0.027, facecolor=color, edgecolor=COLOR_DEEP, linewidth=0.9))
    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=8.5,
        color=contrasting_text_color(color),
        fontweight="bold",
        bbox={"facecolor": color, "edgecolor": "none", "pad": 0.0},
    )


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLOR_ARROW,
    linestyle: str = "-",
    connectionstyle: str = "arc3,rad=0.0",
) -> None:
    """Draw a directed edge."""
    ax.add_patch(
        mpatches.FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            linestyle=linestyle,
            color=color,
            connectionstyle=connectionstyle,
            shrinkA=5,
            shrinkB=5,
        )
    )


def _posterior_card(ax: plt.Axes, x: float, y: float, label: str, color: str, highlighted: int) -> None:
    """Draw a compact posterior card with a nine-cell mass glyph."""
    w, h = 0.12, 0.075
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.008",
            linewidth=0.8,
            edgecolor=color,
            facecolor=COLOR_PANEL_BG,
        )
    )
    ax.text(
        x + 0.016,
        y + h - 0.021,
        label,
        ha="left",
        va="top",
        fontsize=8.5,
        # The accent is a keyline; the painted card background is pale.
        color=COLOR_DEEP,
        fontweight="bold",
        bbox={"facecolor": COLOR_PANEL_BG, "edgecolor": "none", "pad": 0.2},
    )
    cell = 0.012
    gx, gy = x + 0.016, y + 0.015
    for row in range(3):
        for col in range(3):
            index = row * 3 + col
            fill = color if index == highlighted else "white"
            ax.add_patch(
                mpatches.Rectangle(
                    (gx + col * cell, gy + (2 - row) * cell),
                    cell,
                    cell,
                    facecolor=fill,
                    edgecolor="white",
                    linewidth=0.35,
                )
            )
    ax.text(
        x + 0.065,
        y + 0.020,
        "q(s)",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color=COLOR_DEEP,
        bbox={"facecolor": COLOR_PANEL_BG, "edgecolor": "none", "pad": 0.25},
    )


def _node(ax: plt.Axes, x: float, y: float, label: str, description: str, *, fill: str, edge: str) -> None:
    """Draw a node in the lower temporal loop."""
    ax.add_patch(mpatches.Circle((x, y), 0.043, facecolor=fill, edgecolor=edge, linewidth=1.1))
    ax.text(
        x,
        y + 0.004,
        label,
        ha="center",
        va="center",
        fontsize=9.2,
        color=contrasting_text_color(fill),
        bbox={"facecolor": fill, "edgecolor": "none", "pad": 0.0},
    )
    ax.text(
        x,
        y - 0.060,
        description,
        ha="center",
        va="top",
        fontsize=8.5,
        color=COLOR_DEEP,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.3},
    )


def generate_pomdp_loop(*, project_root: Path | None = None) -> Path:
    """Generate the sentinel-world and active-inference loop schematic."""
    apply_style()
    plt.rcParams["figure.autolayout"] = False
    # A compact portrait canvas preserves the full three-stage schematic while
    # keeping every 8.5-point label legible after the 95%-width page embed.
    fig, ax = plt.subplots(figsize=(7.4, 9.8), dpi=150, facecolor="white")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.set_facecolor(COLOR_PANEL_BG)

    fig.text(
        0.5,
        0.965,
        "Sentinel world, private observations, and federated active inference",
        ha="center",
        va="top",
        fontsize=16.2,
        fontweight="bold",
        color=COLOR_DEEP,
    )
    fig.text(
        0.5,
        0.928,
        "Agents share beliefs about one hidden location; observations and controls remain local",
        ha="center",
        va="top",
        fontsize=9.2,
        color=COLOR_ARROW,
    )

    _panel(
        ax,
        0.04,
        0.66,
        0.92,
        0.27,
        "A  Shared nine-cell world",
        "Three sentinels see noisy categorical reports of the same hidden location",
    )
    _world_grid(ax, 0.43, 0.685, 0.14)
    _agent(ax, 0.24, 0.775, "1", COLOR_MULTI_1)
    _agent(ax, 0.24, 0.705, "2", COLOR_ROBUST)
    _agent(ax, 0.75, 0.750, "n", COLOR_VARIATE)
    for start, end in [
        ((0.268, 0.770), (0.43, 0.770)),
        ((0.268, 0.710), (0.43, 0.720)),
        ((0.722, 0.745), (0.57, 0.750)),
    ]:
        _arrow(ax, start, end, color=COLOR_ARROW, linestyle="--")
    ax.text(
        0.50,
        0.675,
        r"shared hidden state $s_t$; each private outcome $o_n$ stays local",
        ha="center",
        va="top",
        fontsize=8.5,
        color=COLOR_DARK,
    )
    _panel(
        ax,
        0.04,
        0.38,
        0.92,
        0.24,
        "B  One belief-sharing round",
        "Local posteriors are the messages; the return is cavity-excluded",
    )
    _posterior_card(ax, 0.09, 0.435, "$q_1$", COLOR_MULTI_1, 0)
    _posterior_card(ax, 0.23, 0.435, "$q_2$", COLOR_ROBUST, 4)
    _posterior_card(ax, 0.37, 0.435, "$q_n$", COLOR_VARIATE, 8)
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.55, 0.420),
            0.15,
            0.10,
            boxstyle="round,pad=0.012",
            linewidth=1.0,
            edgecolor=COLOR_ACCENT,
            facecolor=COLOR_PANEL_BG,
        )
    )
    ax.text(
        0.625,
        0.485,
        "server",
        ha="center",
        va="center",
        fontsize=9.0,
        fontweight="bold",
        color=COLOR_DEEP,
    )
    ax.text(
        0.625,
        0.450,
        "qualified Eq. 7 pool\nor robust route",
        ha="center",
        va="center",
        fontsize=8.5,
        color=COLOR_DARK,
    )
    _posterior_card(ax, 0.77, 0.435, "$\\bar q_n$", COLOR_ACCENT, 4)
    # A shared transport bus keeps the private posterior glyphs unobscured.
    # Each card joins the bus; its single arrow enters the fusion server.
    for x in (0.15, 0.29, 0.43):
        ax.plot([x, x], [0.427, 0.415], color=COLOR_ARROW, linewidth=1.2)
    ax.plot([0.15, 0.49], [0.415, 0.415], color=COLOR_ARROW, linewidth=1.2)
    _arrow(ax, (0.49, 0.415), (0.54, 0.415), color=COLOR_ARROW)
    _arrow(ax, (0.70, 0.470), (0.76, 0.472), color=COLOR_ACCENT)
    _arrow(ax, (0.83, 0.435), (0.83, 0.405), color=COLOR_ACCENT, linestyle="--")
    ax.text(
        0.83,
        0.397,
        r"recipient n; $m\ne n$",
        ha="center",
        va="top",
        fontsize=8.5,
        color=COLOR_ACCENT,
    )
    ax.text(
        0.30,
        0.397,
        "no raw sensory data are pooled",
        ha="center",
        va="top",
        fontsize=8.5,
        color=COLOR_ARROW,
    )

    _panel(
        ax,
        0.04,
        0.020,
        0.92,
        0.315,
        "C  Active-inference temporal loop",
        "The flat federation uses inference and communication; the moving-world "
        "extension also executes B and EFE-guided control",
    )
    positions = {
        "state": (0.17, 0.155),
        "observation": (0.35, 0.195),
        "belief": (0.55, 0.195),
        "action": (0.75, 0.155),
        "next": (0.45, 0.075),
    }
    _node(ax, *positions["state"], r"$s_t$", "hidden location", fill=COLOR_ROBUST, edge=COLOR_ACCENT)
    _node(ax, *positions["observation"], r"$o_t$", "private report", fill=COLOR_MULTI_1, edge=COLOR_ACCENT)
    _node(ax, *positions["belief"], r"$q_t(s)$", "local posterior", fill="white", edge=COLOR_ROBUST)
    _node(ax, *positions["action"], r"$u_t$", "still / left / right", fill=COLOR_VARIATE, edge=COLOR_ACCENT)
    _node(ax, *positions["next"], r"$s_{t+1}$", "", fill=COLOR_ROBUST, edge=COLOR_ACCENT)
    _arrow(ax, (0.21, 0.222), (0.31, 0.230), color=COLOR_MULTI_1)
    _arrow(ax, (0.39, 0.238), (0.51, 0.238), color=COLOR_ROBUST)
    _arrow(ax, (0.59, 0.238), (0.71, 0.230), color=COLOR_VARIATE)
    _arrow(ax, (0.71, 0.135), (0.49, 0.080), color=COLOR_VARIATE)
    _arrow(ax, (0.41, 0.080), (0.21, 0.130), color=COLOR_ROBUST, connectionstyle="arc3,rad=-0.25")
    ax.text(
        0.26,
        0.205,
        "$A=P(o|s)$",
        fontsize=8.5,
        color=COLOR_DEEP,
        ha="center",
    )
    ax.text(
        0.65,
        0.205,
        "$C$ preferences\n/ EFE",
        va="center",
        fontsize=8.5,
        color=COLOR_DEEP,
        ha="center",
    )
    ax.text(
        0.61,
        0.050,
        "$B=P(s'|s,u)$",
        fontsize=8.5,
        color=COLOR_DEEP,
        ha="center",
    )
    fig.text(
        0.5,
        0.018,
        "Model schematic, not an empirical result: panels A and B explain the "
        "Friston-style agents and messages; panel C separates the optional "
        "active-control pathway from the belief-sharing transport.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=COLOR_ARROW,
        style="italic",
    )
    fig.subplots_adjust(left=0.035, right=0.965, top=0.90, bottom=0.055)
    path = save_figure_pair(fig, figures_dir(project_root) / "pomdp_loop.png")
    pomdp_presentation(path)
    return path


__all__ = ["generate_pomdp_loop"]
