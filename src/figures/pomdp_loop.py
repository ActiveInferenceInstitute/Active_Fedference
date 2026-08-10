"""Sentinel-world, active-inference, and federation-loop schematic.

The figure makes the Friston et al. (2024) setting concrete: several agents
observe the same hidden nine-cell location through private categorical sensors,
infer local posteriors, and exchange those posteriors. A separate lower panel
shows the optional transition and action pathway.
"""

from __future__ import annotations

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
    figures_dir,
    save_figure_pair,
)


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
        x + 0.018, y + h - 0.035, title, ha="left", va="top", fontsize=10.2,
        fontweight="bold", color=COLOR_DEEP,
    )
    ax.text(x + 0.018, y + h - 0.078, subtitle, ha="left", va="top", fontsize=8.5, color=COLOR_ARROW)


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
    ax.text(x, y, label, ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")


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
        x + 0.016, y + h - 0.021, label, ha="left", va="top", fontsize=8.5,
        color=color, fontweight="bold",
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
    ax.text(x + 0.065, y + 0.020, "q(s)", ha="left", va="bottom", fontsize=8.5, color=COLOR_DARK)


def _node(ax: plt.Axes, x: float, y: float, label: str, description: str, *, fill: str, edge: str) -> None:
    """Draw a node in the lower temporal loop."""
    ax.add_patch(mpatches.Circle((x, y), 0.043, facecolor=fill, edgecolor=edge, linewidth=1.1))
    ax.text(x, y + 0.004, label, ha="center", va="center", fontsize=9.2, color=COLOR_DEEP)
    ax.text(x, y - 0.060, description, ha="center", va="top", fontsize=8.5, color=COLOR_DARK)


def generate_pomdp_loop(*, project_root: Path | None = None) -> Path:
    """Generate the sentinel-world and active-inference loop schematic."""
    apply_style()
    plt.rcParams["figure.autolayout"] = False
    fig, ax = plt.subplots(figsize=(12.4, 8.2), dpi=150, facecolor="white")
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
        0.49,
        0.43,
        0.38,
        "A  Shared nine-cell world",
        "Three sentinels see noisy categorical reports of the same hidden location",
    )
    _world_grid(ax, 0.18, 0.55, 0.19)
    _agent(ax, 0.105, 0.72, "1", COLOR_MULTI_1)
    _agent(ax, 0.105, 0.58, "2", COLOR_ROBUST)
    _agent(ax, 0.39, 0.71, "n", COLOR_VARIATE)
    for start, end in [
        ((0.128, 0.705), (0.18, 0.70)),
        ((0.128, 0.595), (0.18, 0.62)),
        ((0.367, 0.692), (0.34, 0.70)),
    ]:
        _arrow(ax, start, end, color=COLOR_ARROW, linestyle="--")
    ax.text(0.105, 0.515, "private outcome $o_n$", ha="center", va="top", fontsize=8.5, color=COLOR_DARK)
    ax.text(0.29, 0.515, "hidden state $s_t$", ha="center", va="top", fontsize=8.5, color=COLOR_DEEP)
    _panel(
        ax,
        0.51,
        0.49,
        0.47,
        0.38,
        "B  One belief-sharing round",
        "Local posteriors are the messages; the return is cavity-excluded",
    )
    _posterior_card(ax, 0.55, 0.690, "$q_1$", COLOR_MULTI_1, 0)
    _posterior_card(ax, 0.55, 0.600, "$q_2$", COLOR_ROBUST, 4)
    _posterior_card(ax, 0.55, 0.510, "$q_n$", COLOR_VARIATE, 8)
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.72, 0.575),
            0.13,
            0.14,
            boxstyle="round,pad=0.012",
            linewidth=1.0,
            edgecolor=COLOR_ACCENT,
            facecolor=COLOR_PANEL_BG,
        )
    )
    ax.text(
        0.785, 0.665, "server", ha="center", va="center", fontsize=9.0,
        fontweight="bold", color=COLOR_DEEP,
    )
    ax.text(
        0.785, 0.625, "qualified Eq. 7 pool\nor robust route", ha="center", va="center",
        fontsize=8.5, color=COLOR_DARK,
    )
    _posterior_card(ax, 0.855, 0.610, "$\\bar q_n$", COLOR_ACCENT, 4)
    for y in (0.727, 0.637, 0.547):
        _arrow(ax, (0.675, y), (0.71, 0.645), color=COLOR_ARROW)
    _arrow(ax, (0.85, 0.645), (0.855, 0.645), color=COLOR_ACCENT)
    _arrow(ax, (0.88, 0.605), (0.88, 0.545), color=COLOR_ACCENT, linestyle="--")
    ax.text(0.88, 0.532, "heard by n\nwith m ≠ n", ha="center", va="top", fontsize=8.5, color=COLOR_ACCENT)
    ax.text(
        0.735, 0.515, "no raw sensory data are pooled", ha="center", va="top",
        fontsize=8.5, color=COLOR_ARROW,
    )

    _panel(
        ax,
        0.04,
        0.05,
        0.92,
        0.35,
        "C  Active-inference temporal loop",
        "The flat federation uses inference and communication; the moving-world "
        "extension also executes B and EFE-guided control",
    )
    positions = {
        "state": (0.17, 0.20),
        "observation": (0.35, 0.245),
        "belief": (0.55, 0.245),
        "action": (0.75, 0.20),
        "next": (0.55, 0.14),
    }
    _node(ax, *positions["state"], r"$s_t$", "hidden location", fill=COLOR_ROBUST, edge=COLOR_ACCENT)
    _node(ax, *positions["observation"], r"$o_t$", "private report", fill=COLOR_MULTI_1, edge=COLOR_ACCENT)
    _node(ax, *positions["belief"], r"$q_t(s)$", "local posterior", fill="white", edge=COLOR_ROBUST)
    _node(ax, *positions["action"], r"$u_t$", "still / left / right", fill=COLOR_VARIATE, edge=COLOR_ACCENT)
    _node(ax, *positions["next"], r"$s_{t+1}$", "", fill=COLOR_ROBUST, edge=COLOR_ACCENT)
    _arrow(ax, (0.21, 0.255), (0.31, 0.278), color=COLOR_MULTI_1)
    _arrow(ax, (0.39, 0.288), (0.51, 0.288), color=COLOR_ROBUST)
    _arrow(ax, (0.59, 0.275), (0.71, 0.255), color=COLOR_VARIATE)
    _arrow(ax, (0.73, 0.205), (0.59, 0.170), color=COLOR_VARIATE)
    _arrow(ax, (0.51, 0.165), (0.21, 0.215), color=COLOR_ROBUST, connectionstyle="arc3,rad=0.25")
    ax.text(0.26, 0.235, "$A=P(o|s)$", fontsize=8.5, color=COLOR_MULTI_1, ha="center")
    ax.text(0.65, 0.235, "$C$ preferences / EFE", fontsize=8.5, color=COLOR_VARIATE, ha="center")
    ax.text(0.43, 0.165, "$B=P(s'|s,u)$", fontsize=8.5, color=COLOR_ROBUST, ha="center")
    ax.text(
        0.50, 0.075,
        "Inference and federation are the executed bridge; action selection is "
        "an explicitly scoped extension.",
        ha="center", fontsize=8.5, color=COLOR_ARROW,
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
    return save_figure_pair(fig, figures_dir(project_root) / "pomdp_loop.png")


__all__ = ["generate_pomdp_loop"]
