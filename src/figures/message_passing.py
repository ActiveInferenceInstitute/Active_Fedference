"""Vertically ordered categorical belief-sharing message-path schematic.

The deterministic figure separates private client updating from server fusion,
assigns each mathematical or empirical statement to its owning route, and
retains posterior-only broadcast plus recipient-specific cavity exclusion. It
is a protocol schematic, not a benchmark or full source-protocol replication.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from figures._common import (
    COLOR_ARROW,
    COLOR_DARK,
    COLOR_DEEP,
    COLOR_EDGE_PANEL,
    COLOR_MULTI_1,
    COLOR_PANEL_BG,
    apply_style,
    figures_dir,
    save_figure_pair,
    semantic_style,
)
from figures._presentation_schematics import message_presentation


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
    owner: str | None = None,
) -> None:
    """Draw one labelled stage card with an optional claim-owner badge."""
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012",
            linewidth=1.1,
            edgecolor=accent,
            facecolor="white",
            zorder=2,
        )
    )
    ax.text(
        x + 0.018,
        y + h - 0.030,
        title,
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color=accent,
        zorder=4,
    )
    ax.text(
        x + 0.018,
        y + h - 0.060,
        body,
        ha="left",
        va="top",
        fontsize=9.5,
        color=COLOR_DARK,
        linespacing=1.28,
        zorder=4,
    )
    if owner:
        ax.text(
            x + w - 0.015,
            y + 0.014,
            owner,
            ha="right",
            va="bottom",
            fontsize=8.5,
            color=accent,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": accent,
                "linewidth": 0.8,
            },
            zorder=5,
        )


def _agent_card(
    ax: plt.Axes,
    x: float,
    y: float,
    agent: str,
    observed_cell: int,
    accent: str,
) -> None:
    """Draw one sentinel's local view and posterior-update label."""
    w, h = 0.265, 0.105
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.010",
            linewidth=1.0,
            edgecolor=accent,
            facecolor="white",
            zorder=2,
        )
    )
    gx, gy, size = x + 0.017, y + 0.025, 0.062
    cell = size / 3.0
    for row in range(3):
        for col in range(3):
            index = row * 3 + col
            ax.add_patch(
                mpatches.Rectangle(
                    (gx + col * cell, gy + (2 - row) * cell),
                    cell,
                    cell,
                    facecolor=accent if index == observed_cell else COLOR_PANEL_BG,
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=3,
                )
            )
    ax.text(
        x + 0.105,
        y + h - 0.024,
        f"Sentinel {agent}",
        ha="left",
        va="top",
        fontsize=9.5,
        fontweight="bold",
        color=COLOR_DEEP,
        zorder=4,
    )
    ax.text(
        x + 0.105,
        y + 0.022,
        rf"$o_{{{agent}}}\rightarrow q_{{{agent}}}(s)$",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color=accent,
        zorder=4,
    )


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLOR_ARROW,
    linestyle: str = "-",
) -> None:
    """Draw a directed stage edge."""
    ax.add_patch(
        mpatches.FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.3,
            linestyle=linestyle,
            color=color,
            shrinkA=5,
            shrinkB=5,
            zorder=1,
        )
    )


def _band(ax: plt.Axes, y: float, height: float, label: str, accent: str) -> None:
    """Draw a quiet horizontal stage band with explicit reading order."""
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.025, y),
            0.95,
            height,
            boxstyle="round,pad=0.010",
            linewidth=0.8,
            edgecolor=COLOR_EDGE_PANEL,
            facecolor=COLOR_PANEL_BG,
            zorder=0,
        )
    )
    ax.text(
        0.04,
        y + height - 0.020,
        label,
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color=accent,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": COLOR_PANEL_BG,
            "edgecolor": "none",
        },
        zorder=6,
    )


def _claim_badge(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    accent: str,
) -> None:
    """Draw a compact client-route ownership badge."""
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=8.5,
        color=accent,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": accent,
            "linewidth": 0.8,
        },
        zorder=5,
    )


def generate_message_passing(*, project_root: Path | None = None) -> Path:
    """Generate the vertically ordered, claim-bounded message-path figure."""
    apply_style()
    plt.rcParams["figure.autolayout"] = False
    # The vertical stage design is intentionally narrow enough that the
    # manuscript-scale 8.5-point labels remain at least 7 points effective.
    fig, ax = plt.subplots(figsize=(7.4, 10.2), dpi=150, facecolor="white")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.set_facecolor("white")

    naive = semantic_style("naive")
    robust = semantic_style("heuristic_robust")
    variational = semantic_style("variational")

    fig.text(
        0.5,
        0.975,
        "Categorical belief sharing from private updates to cavity-excluded return",
        ha="center",
        va="top",
        fontsize=16,
        fontweight="bold",
        color=COLOR_DEEP,
    )
    fig.text(
        0.5,
        0.942,
        "Vertical bands separate client-update guarantees from server-fusion properties",
        ha="center",
        va="top",
        fontsize=10,
        color=COLOR_ARROW,
    )

    # 1. Private observations and client-owned update routes.
    _band(ax, 0.690, 0.245, "1  PRIVATE OBSERVATION AND CLIENT UPDATE", robust.keyline)
    _agent_card(ax, 0.055, 0.775, "1", 0, COLOR_MULTI_1)
    _agent_card(ax, 0.365, 0.775, "2", 4, robust.keyline)
    _agent_card(ax, 0.675, 0.775, "n", 8, variational.keyline)
    _claim_badge(
        ax,
        0.285,
        0.720,
        "NLL/KLD/β = 0\nproject recovery identity",
        accent=naive.keyline,
    )
    _claim_badge(
        ax,
        0.715,
        0.720,
        "Robust client losses\nsource theorem under assumptions",
        accent=robust.keyline,
    )

    # 2. Posterior-only transport boundary.
    _band(ax, 0.515, 0.135, "2  POSTERIOR-ONLY BROADCAST", COLOR_MULTI_1)
    _box(
        ax,
        0.205,
        0.535,
        0.59,
        0.085,
        r"Broadcast $q_n(s)$; private $o_n$ stays local",
        "categorical posterior only",
        accent=COLOR_MULTI_1,
        owner="protocol-v1 payload",
    )
    for x in (0.187, 0.497, 0.807):
        _arrow(ax, (x, 0.775), (0.50, 0.622), color=COLOR_MULTI_1)

    # 3. Three server routes; properties stay in their owning boxes.
    _band(ax, 0.245, 0.225, "3  SERVER FUSION ROUTES", COLOR_DEEP)
    _box(
        ax,
        0.045,
        0.265,
        0.285,
        0.160,
        "Log-linear pool",
        r"$\mathrm{softmax}(\sum_n w_n\log q_n)$",
        accent=naive.color,
        owner="qualified Eq. 7\nspecialization",
    )
    _box(
        ax,
        0.3575,
        0.265,
        0.285,
        0.160,
        "robust_aggregate",
        "configured divergence\nreweighting",
        accent=robust.keyline,
        owner="conditional heuristic\nevidence",
    )
    _box(
        ax,
        0.670,
        0.265,
        0.285,
        0.160,
        "Variational server",
        "$q,a$ block updates\nof $F(q,a)$",
        accent=variational.keyline,
        owner="objective-backed\nweight property",
    )
    for x, color in (
        (0.192, naive.color),
        (0.502, robust.keyline),
        (0.812, variational.keyline),
    ):
        _arrow(ax, (0.50, 0.535), (x, 0.427), color=color, linestyle="--")

    # 4. Recipient-specific return, preserving cavity exclusion.
    _band(ax, 0.050, 0.150, "4  RECIPIENT-SPECIFIC RETURN", COLOR_MULTI_1)
    _box(
        ax,
        0.155,
        0.070,
        0.69,
        0.090,
        r"Cavity-excluded return $q_{-n}(s)$",
        r"senders $m\ne n$ only",
        accent=COLOR_MULTI_1,
        owner="self-exclusion preserved",
    )
    for x, color in (
        (0.192, naive.color),
        (0.502, robust.keyline),
        (0.812, variational.keyline),
    ):
        _arrow(ax, (x, 0.265), (0.50, 0.160), color=color, linestyle="--")

    fig.text(
        0.5,
        0.020,
        "Deterministic schematic; no empirical estimate or uncertainty interval. "
        "The Eq. 7 bridge is not a reconstruction of the complete source protocol.",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color=COLOR_ARROW,
    )
    path = save_figure_pair(fig, figures_dir(project_root) / "message_passing.png")
    message_presentation(path)
    return path


__all__ = ["generate_message_passing"]
