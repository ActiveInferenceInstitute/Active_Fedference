"""Formal schematic of the categorical sentinel generative model.

The figure adapts the generative-model and depth conventions of Friston et al.
(2024) to the Active Fedference implementation. It makes the private sensory
view concrete while retaining the A/B/C/D_0 notation, temporal loop, and
optional hierarchical context. It is a deterministic model schematic.
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
    COLOR_MULTI_2,
    COLOR_PANEL_BG,
    COLOR_PANEL_GRID,
    COLOR_ROBUST,
    COLOR_VARIATE,
    apply_style,
    figures_dir,
    save_figure_pair,
)


def _panel(ax: plt.Axes, title: str, subtitle: str) -> None:
    """Draw a quiet publication panel."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.01, 0.02),
            0.98,
            0.95,
            boxstyle="round,pad=0.012",
            linewidth=0.9,
            edgecolor=COLOR_EDGE_PANEL,
            facecolor=COLOR_PANEL_BG,
        )
    )
    ax.text(
        0.05,
        0.92,
        title,
        ha="left",
        va="top",
        fontsize=11.2,
        fontweight="bold",
        color=COLOR_DEEP,
    )
    ax.text(
        0.05,
        0.855,
        subtitle,
        ha="left",
        va="top",
        fontsize=8.5,
        color=COLOR_ARROW,
    )


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLOR_ARROW,
    linestyle: str = "-",
) -> None:
    """Draw a directed dependency edge."""
    ax.add_patch(
        mpatches.FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.1,
            linestyle=linestyle,
            color=color,
            shrinkA=4,
            shrinkB=4,
        )
    )


def _node(
    ax: plt.Axes,
    x: float,
    y: float,
    label: str,
    *,
    fill: str = "white",
    edge: str = COLOR_ACCENT,
    radius: float = 0.043,
    fontsize: float = 8.8,
) -> None:
    """Draw a labelled latent or observation node."""
    ax.add_patch(
        mpatches.Circle(
            (x, y),
            radius,
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.0,
        )
    )
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, color=COLOR_DARK)


def _grid(
    ax: plt.Axes,
    x: float,
    y: float,
    size: float,
    *,
    highlighted: int,
    accent: str,
    likelihood: bool = False,
) -> None:
    """Draw a nine-cell categorical location grid."""
    cell = size / 3.0
    for row in range(3):
        for col in range(3):
            index = row * 3 + col
            if likelihood:
                fill = accent if index == highlighted else COLOR_PANEL_GRID
                alpha = 0.95 if index == highlighted else 0.45
            else:
                fill = accent if index == highlighted else "white"
                alpha = 0.9 if index == highlighted else 1.0
            ax.add_patch(
                mpatches.Rectangle(
                    (x + col * cell, y + (2 - row) * cell),
                    cell,
                    cell,
                    facecolor=fill,
                    alpha=alpha,
                    edgecolor="white",
                    linewidth=0.8,
                )
            )
    ax.add_patch(
        mpatches.Rectangle(
            (x, y),
            size,
            size,
            fill=False,
            edgecolor=accent,
            linewidth=0.9,
        )
    )


def _factor_card(
    ax: plt.Axes,
    x: float,
    y: float,
    label: str,
    description: str,
    *,
    color: str,
) -> None:
    """Draw one A/B/C/D_0 model-factor card."""
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x, y),
            0.19,
            0.16,
            boxstyle="round,pad=0.012",
            linewidth=0.9,
            edgecolor=color,
            facecolor="white",
        )
    )
    ax.text(x + 0.095, y + 0.115, label, ha="center", va="center", fontsize=12, color=color)
    ax.text(
        x + 0.095,
        y + 0.045,
        description,
        ha="center",
        va="center",
        fontsize=8.5,
        color=COLOR_DARK,
    )


def _draw_sensor(ax: plt.Axes) -> None:
    _panel(
        ax,
        "A  What each sentinel sees",
        "A private categorical report over the shared nine-cell location space",
    )
    _grid(ax, 0.12, 0.34, 0.25, highlighted=4, accent=COLOR_ROBUST)
    _grid(ax, 0.63, 0.34, 0.25, highlighted=4, accent=COLOR_MULTI_1, likelihood=True)
    ax.text(0.245, 0.30, "hidden location s", ha="center", va="top", fontsize=8.5, color=COLOR_DARK)
    ax.text(0.755, 0.30, "one outcome o", ha="center", va="top", fontsize=8.5, color=COLOR_DARK)
    _arrow(ax, (0.41, 0.47), (0.59, 0.47), color=COLOR_ROBUST)
    ax.text(0.50, 0.54, r"$A[o,s]=P(o\mid s)$", ha="center", va="center", fontsize=9.0, color=COLOR_ROBUST)
    ax.text(
        0.50,
        0.14,
        "diagonal mass = acuity; residual mass = categorical noise",
        ha="center",
        va="center",
        fontsize=8.5,
        color=COLOR_ARROW,
    )
    ax.text(
        0.50,
        0.08,
        "The raw outcome stays with the agent; only q(s) is broadcast.",
        ha="center",
        va="center",
        fontsize=8.5,
        color=COLOR_DEEP,
        style="italic",
    )


def _draw_factors(ax: plt.Axes) -> None:
    _panel(
        ax,
        "B  A/B/C/D_0 model factors",
        "Likelihood, dynamics, preferences, and prior shape local inference and action",
    )
    factors = (
        (0.055, r"$A$", "sensory likelihood", COLOR_ROBUST),
        (0.275, r"$B$", "state transition", COLOR_VARIATE),
        (0.495, r"$C$", "preferred outcomes", COLOR_MULTI_1),
        (0.715, r"$D_0$", "initial prior", COLOR_MULTI_2),
    )
    for x, label, description, color in factors:
        _factor_card(ax, x, 0.55, label, description, color=color)
        _arrow(ax, (x + 0.095, 0.54), (0.50, 0.34), color=color)
    _node(ax, 0.50, 0.28, r"$q(s)$", fill=COLOR_ROBUST, edge=COLOR_ACCENT, radius=0.055, fontsize=10.0)
    ax.text(
        0.50, 0.20, "local posterior over location", ha="center", va="center",
        fontsize=8.5, color=COLOR_DARK,
    )
    ax.text(
        0.50, 0.10, r"$q(s)=\mathrm{softmax}(\ln D_0+\ln A[o,\cdot])$",
        ha="center", va="center", fontsize=8.5, color=COLOR_DEEP,
    )


def _draw_temporal(ax: plt.Axes) -> None:
    _panel(ax, "C  Temporal depth", "Inference and control across successive times")
    xs = (0.16, 0.38, 0.60, 0.82)
    labels = (r"$s_t$", r"$o_t$", r"$q_t(s)$", r"$u_t$")
    fills = (COLOR_ROBUST, COLOR_MULTI_1, "white", COLOR_VARIATE)
    descriptions = ("hidden location", "private report", "posterior", "control")
    for x, label, fill, description in zip(xs, labels, fills, descriptions):
        _node(ax, x, 0.57, label, fill=fill, edge=COLOR_ACCENT)
        ax.text(x, 0.43, description, ha="center", va="top", fontsize=8.5, color=COLOR_DARK)
    for start, end in zip(xs[:-1], xs[1:]):
        _arrow(ax, (start + 0.045, 0.57), (end - 0.045, 0.57))
    _node(ax, 0.30, 0.25, r"$s_{t+1}$", fill=COLOR_ROBUST, edge=COLOR_ACCENT)
    _arrow(ax, (0.82, 0.52), (0.34, 0.30), color=COLOR_VARIATE)
    ax.text(0.58, 0.36, r"$B=P(s'\mid s,u)$", fontsize=8.5, color=COLOR_VARIATE, ha="center")
    ax.text(
        0.50, 0.12,
        "Flat studies stop after posterior sharing; moving-world studies execute "
        "B and EFE-guided u.",
        ha="center", fontsize=8.5, color=COLOR_ARROW,
    )


def _draw_hierarchy(ax: plt.Axes) -> None:
    _panel(ax, "D  Hierarchical context", "Optional top-down context conditions the location-level prior")
    levels = ((0.78, r"$s^L$", "meta-context"), (0.56, r"$s^2$", "context"), (0.34, r"$s^1$", "location"))
    for y, label, name in levels:
        _node(ax, 0.25, y, label, fill=COLOR_MULTI_2 if y > 0.4 else COLOR_ROBUST, edge=COLOR_ACCENT)
        ax.text(0.36, y, name, fontsize=8.5, color=COLOR_DARK, va="center", ha="left")
    for upper, lower in zip(levels[:-1], levels[1:]):
        _arrow(ax, (0.25, upper[0] - 0.05), (0.25, lower[0] + 0.05), color=COLOR_MULTI_2)
    _node(ax, 0.76, 0.56, r"$q_1(s)$", fill="white", edge=COLOR_ROBUST)
    _arrow(ax, (0.31, 0.34), (0.70, 0.53), color=COLOR_ROBUST)
    _arrow(ax, (0.31, 0.56), (0.70, 0.56), color=COLOR_MULTI_2)
    _arrow(ax, (0.31, 0.78), (0.70, 0.59), color=COLOR_MULTI_2, linestyle="--")
    ax.text(0.76, 0.41, r"$\bar q_1=\sum_k q_2[k]D_{1|k}$", fontsize=8.5, color=COLOR_DARK, ha="center")
    ax.text(
        0.76, 0.23,
        "The hierarchy is an extension, not a hidden assumption in every study.",
        ha="center", fontsize=8.5, color=COLOR_ARROW,
    )


def generate_generative_model_schema(*, project_root: Path | None = None) -> Path:
    """Generate the formal categorical generative-model schematic."""
    apply_style()
    plt.rcParams["figure.autolayout"] = False
    fig = plt.figure(figsize=(12.4, 8.5), dpi=150, facecolor="white")
    gs = fig.add_gridspec(
        2,
        2,
        left=0.035,
        right=0.965,
        top=0.85,
        bottom=0.10,
        wspace=0.045,
        hspace=0.12,
    )
    _draw_sensor(fig.add_subplot(gs[0, 0]))
    _draw_factors(fig.add_subplot(gs[0, 1]))
    _draw_temporal(fig.add_subplot(gs[1, 0]))
    _draw_hierarchy(fig.add_subplot(gs[1, 1]))

    fig.text(
        0.5,
        0.972,
        "Categorical sentinel generative model and federated belief paths",
        ha="center",
        va="top",
        fontsize=16.5,
        fontweight="bold",
        color=COLOR_DEEP,
    )
    fig.text(
        0.5,
        0.935,
        "Friston-style hidden location, private observation, local posterior, and optional control depth",
        ha="center",
        va="top",
        fontsize=9.5,
        color=COLOR_ARROW,
    )
    fig.text(
        0.5,
        0.885,
        r"$q(s)=\mathrm{softmax}\!\left(\ln D_0(s)+\ln A[o,\cdot]\right)$"
        r"   |   $\mathrm{robust\_aggregate}(0)=\mathrm{log\_linear\_pool}$",
        ha="center",
        va="center",
        fontsize=9.7,
        color=COLOR_DARK,
        bbox={
            "boxstyle": "round,pad=0.42",
            "facecolor": "white",
            "edgecolor": COLOR_ACCENT,
            "linewidth": 1.0,
        },
    )
    fig.text(
        0.5,
        0.043,
        "Schematic formalization: the agent's private sensory outcome becomes a posterior message; "
        "the server fuses messages over the shared categorical state.",
        ha="center",
        va="center",
        fontsize=9.5,
        color=COLOR_ARROW,
        style="italic",
    )
    return save_figure_pair(fig, figures_dir(project_root) / "generative_model_schema.png")


__all__ = ["generate_generative_model_schema"]
