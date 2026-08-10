"""Message-passing schematic for categorical sentinel belief sharing.

The figure follows Friston et al. (2024) while making the local observation,
posterior broadcast, server choice, and claim boundaries visible. It is a
deterministic protocol schematic, not a benchmark or uncertainty estimate.
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
    COLOR_NAIVE,
    COLOR_PANEL_BG,
    COLOR_ROBUST,
    COLOR_VARIATE,
    apply_style,
    figures_dir,
    save_figure_pair,
)


def _box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    *,
    accent: str,
    badge: str | None = None,
) -> None:
    """Draw a labelled protocol card."""
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012",
            linewidth=0.9,
            edgecolor=accent,
            facecolor="white",
        )
    )
    ax.text(
        x + 0.018,
        y + h - 0.032,
        title,
        ha="left",
        va="top",
        fontsize=9.0,
        fontweight="bold",
        color=accent,
    )
    ax.text(
        x + 0.018,
        y + h - 0.067,
        body,
        ha="left",
        va="top",
        fontsize=8.5,
        color=COLOR_DARK,
        linespacing=1.25,
    )
    if badge:
        ax.text(
            x + w - 0.018,
            y + 0.012,
            badge,
            ha="right",
            va="bottom",
            fontsize=8.5,
            color=accent,
            style="italic",
        )


def _agent_card(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    agent: str,
    observed_cell: int,
    accent: str,
) -> None:
    """Draw one sentinel with a private nine-cell categorical view."""
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.010",
            linewidth=0.9,
            edgecolor=accent,
            facecolor="white",
        )
    )
    gx, gy, size = x + 0.015, y + 0.040, 0.066
    cell = size / 3.0
    for row in range(3):
        for col in range(3):
            index = row * 3 + col
            fill = accent if index == observed_cell else COLOR_PANEL_BG
            ax.add_patch(
                mpatches.Rectangle(
                    (gx + col * cell, gy + (2 - row) * cell),
                    cell,
                    cell,
                    facecolor=fill,
                    edgecolor="white",
                    linewidth=0.5,
                )
            )
    ax.text(
        x + 0.092,
        y + h - 0.030,
        f"Sentinel {agent}",
        ha="left",
        va="top",
        fontsize=9.5,
        fontweight="bold",
        color=COLOR_DEEP,
    )
    ax.text(
        x + 0.092,
        y + h - 0.066,
        "private view",
        ha="left",
        va="top",
        fontsize=8.5,
        color=COLOR_DARK,
    )
    ax.text(
        x + 0.092,
        y + 0.025,
        f"outcome o_{agent}",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color=accent,
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
    """Draw a directed message edge."""
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


def _lane(ax: plt.Axes, y: float, label: str, color: str) -> None:
    """Draw a quiet lane rule and label."""
    ax.plot([0.035, 0.965], [y, y], color=COLOR_EDGE_PANEL, linewidth=0.7, zorder=0)
    ax.text(
        0.02,
        y + 0.065,
        label,
        ha="left",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=color,
    )


def generate_message_passing(*, project_root: Path | None = None) -> Path:
    """Generate the claim-bounded belief-sharing message-passing schematic."""
    apply_style()
    plt.rcParams["figure.autolayout"] = False
    fig, ax = plt.subplots(figsize=(18.5, 5.2), dpi=150, facecolor="white")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.set_facecolor(COLOR_PANEL_BG)

    fig.text(
        0.5,
        0.975,
        "Message passing from private sensory views to federated consensus",
        ha="center",
        va="top",
        fontsize=15.5,
        fontweight="bold",
        color=COLOR_DEEP,
    )
    fig.text(
        0.5,
        0.932,
        "Agents keep observations local; the server receives categorical posteriors",
        ha="center",
        va="top",
        fontsize=9.0,
        color=COLOR_ARROW,
    )

    _lane(ax, 0.82, "A  LOCAL INFERENCE AND BROADCAST", COLOR_ROBUST)
    _agent_card(ax, 0.055, 0.635, 0.155, 0.145, "1", 0, COLOR_MULTI_1)
    _agent_card(ax, 0.235, 0.635, 0.155, 0.145, "2", 4, COLOR_ROBUST)
    _agent_card(ax, 0.415, 0.635, 0.155, 0.145, "n", 8, COLOR_VARIATE)
    _box(
        ax,
        0.615,
        0.635,
        0.16,
        0.145,
        r"$q_n(s)$  local posterior",
        "generalized-Bayes update\nprivate outcome",
        accent=COLOR_ROBUST,
        badge="local route",
    )
    _box(
        ax,
        0.825,
        0.635,
        0.125,
        0.145,
        r"$q_n$  broadcast",
        "posterior only\nnot raw $o_n$",
        accent=COLOR_MULTI_1,
    )
    for x in (0.21, 0.39, 0.57):
        _arrow(ax, (x, 0.707), (0.605, 0.707), color=COLOR_ROBUST)
    _arrow(ax, (0.78, 0.707), (0.815, 0.707), color=COLOR_MULTI_1)

    _lane(ax, 0.51, "B  SERVER FUSION", COLOR_ACCENT)
    _box(
        ax,
        0.055,
        0.345,
        0.255,
        0.145,
        "Standard pool",
        r"$\mathrm{softmax}(\sum_n w_n\log q_n)$" "\n" r"$\mathrm{KLD/NLL/\beta=0}$",
        accent=COLOR_NAIVE,
        badge="categorical Eq. 7 bridge",
    )
    _box(
        ax,
        0.365,
        0.345,
        0.255,
        0.145,
        "Heuristic server",
        r"$\exp[-c\,\mathrm{KL}(q_n\|q)]$" "\n" "iterative reweighting",
        accent=COLOR_ROBUST,
        badge="recovery limit only",
    )
    _box(
        ax,
        0.675,
        0.345,
        0.255,
        0.145,
        "Variational server",
        r"$q,a$ block updates" "\n" r"objective $F(q,a)$ descends",
        accent=COLOR_VARIATE,
        badge="objective-backed",
    )
    for x, color in ((0.182, COLOR_NAIVE), (0.492, COLOR_ROBUST), (0.802, COLOR_VARIATE)):
        _arrow(ax, (0.887, 0.635), (x, 0.505), color=color, linestyle="--")

    ax.text(
        0.94,
        0.545,
        "return\n\ncavity:\n$m\\ne n$",
        ha="center",
        va="center",
        fontsize=8.5,
        color=COLOR_ACCENT,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": COLOR_ACCENT,
            "linewidth": 0.8,
        },
    )

    ax.text(
        0.02,
        0.255,
        "C  CLAIM OWNERSHIP",
        ha="left",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=COLOR_DEEP,
    )
    _box(
        ax,
        0.055,
        0.085,
        0.27,
        0.135,
        "Client-side FedGVI",
        "bounded-loss update\nunder source assumptions",
        accent=COLOR_ROBUST,
        badge="rigorous / conditional",
    )
    _box(
        ax,
        0.365,
        0.085,
        0.27,
        0.135,
        "robust_aggregate",
        "conditional accuracy\nunder declared attacks",
        accent=COLOR_NAIVE,
        badge="heuristic / empirical",
    )
    _box(
        ax,
        0.675,
        0.085,
        0.27,
        0.135,
        "variational_aggregate",
        "descent and raw-weight\ncontrol with conservatism",
        accent=COLOR_VARIATE,
        badge="objective-backed",
    )
    for x in (0.182, 0.492, 0.802):
        _arrow(ax, (x, 0.345), (x, 0.23), color=COLOR_ARROW, linestyle="--")

    ax.text(
        0.5,
        0.018,
        "Protocol schematic: each agent sees one private categorical outcome, "
        "forms a posterior, broadcasts that posterior, and receives a consensus "
        "with its own message excluded. No empirical data or CI is shown.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=COLOR_ARROW,
        style="italic",
    )
    return save_figure_pair(fig, figures_dir(project_root) / "message_passing.png")


__all__ = ["generate_message_passing"]
