"""Graphical abstract for the Robust Federated Active Inference paper.

Two-panel publication-quality figure:
  LEFT  — Network diagram: metadata-backed honest and adversarial agents
           sharing beliefs with a central robust-aggregation server.
  RIGHT — Schematic outcome cards comparing token-backed naive and robust
           consensus states, with the variational objective kept separate.

Saved to output/figures/graphical_abstract.{png,pdf} and manuscript/cover_image.png.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from PIL import Image

from figures._common import (
    COLOR_ADVERSARY,
    COLOR_ADVERSARY_EDGE,
    COLOR_ARROW,
    COLOR_DARK,
    COLOR_EDGE_LIGHT,
    COLOR_EDGE_PANEL,
    COLOR_HONEST_EDGE,
    COLOR_NAIVE,
    COLOR_NAIVE_LIGHT,
    COLOR_PANEL_BG,
    COLOR_PANEL_GRID,
    COLOR_ROBUST,
    COLOR_SERVER,
    COLOR_SERVER_EDGE,
    COLOR_VARIATE,
    PROJECT_ROOT,
    apply_style,
)
from figures.system_overview import SYSTEM_OVERVIEW_METADATA, build_data

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
_OUT = PROJECT_ROOT / "output" / "figures"
_COVER = PROJECT_ROOT / "manuscript" / "cover_image.png"


# ---------------------------------------------------------------------------
# Style constants (semantic aliases for local readability)
# ---------------------------------------------------------------------------
HONEST_COLOR = COLOR_ROBUST
ADVERSARY_COLOR = COLOR_ADVERSARY
SERVER_COLOR = COLOR_SERVER
PANEL_BG = COLOR_PANEL_BG
GRID_COLOR = COLOR_PANEL_GRID
ARROW_COLOR = COLOR_ARROW
AXIS_COLOR = COLOR_DARK

NAIVE_COLOR = COLOR_NAIVE_LIGHT
ROBUST_COLOR = COLOR_ROBUST
VARIATE_COLOR = COLOR_VARIATE

FONT_FAMILY = "DejaVu Sans"
COVER_NETWORK_N_AGENTS = int(SYSTEM_OVERVIEW_METADATA["n_agents"])
COVER_NETWORK_N_ADVERSARIAL = int(SYSTEM_OVERVIEW_METADATA["n_adversarial"])
METHOD_LABELS = ("Naive log-linear pool", "Heuristic reweighting")
PDF_SAFE_RASTER_WIDTH = 4000


def _rewrite_as_rgb_png(path: Path) -> None:
    """Rewrite an opaque Matplotlib PNG as deterministic RGB for XeTeX.

    Matplotlib emits RGBA PNGs even when the rendered canvas is fully opaque.
    The XeTeX/dvipdfmx path can also mis-handle a wide raster above its
    approximately 4096-pixel image-tile boundary, producing a horizontal wrap
    in the lower strip. RGB plus a conservative 4000-pixel width cap preserves
    the visible pixels, keeps the HTML asset portable, and makes PDF raster
    embedding stable without relying on renderer-specific image limits.
    """
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        if rgb.width > PDF_SAFE_RASTER_WIDTH:
            height = round(rgb.height * PDF_SAFE_RASTER_WIDTH / rgb.width)
            rgb = rgb.resize(
                (PDF_SAFE_RASTER_WIDTH, height),
                resample=Image.Resampling.LANCZOS,
            )
        rgb.save(path, format="PNG", dpi=(300, 300), compress_level=9)


# ---------------------------------------------------------------------------
# Helper: tiny bar-chart glyph drawn inside an agent node
# ---------------------------------------------------------------------------
def _draw_mini_bars(ax: plt.Axes, cx: float, cy: float, r: float,
                    heights: list[float], color: str) -> None:
    """Draw a tiny 4-bar histogram inside a circle of radius r centred on (cx, cy)."""
    n = len(heights)
    bar_w = r * 0.28
    total_w = n * bar_w + (n - 1) * bar_w * 0.2
    x0 = cx - total_w / 2
    max_h = max(heights) if max(heights) > 0 else 1.0
    for i, h in enumerate(heights):
        bx = x0 + i * (bar_w + bar_w * 0.2)
        bh = (h / max_h) * r * 0.65
        rect = mpatches.FancyBboxPatch(
            (bx, cy - r * 0.42),
            bar_w,
            bh,
            boxstyle="square,pad=0",
            linewidth=0,
            facecolor=color,
            alpha=0.85,
            zorder=5,
        )
        ax.add_patch(rect)


# ---------------------------------------------------------------------------
# Helper: draw X mark for adversarial agents
# ---------------------------------------------------------------------------
def _draw_x_mark(ax: plt.Axes, cx: float, cy: float, r: float) -> None:
    d = r * 0.35
    for dx, dy in [((-d, d), (d, -d)), ((-d, -d), (d, d))]:
        ax.plot(
            [cx + dx[0], cx + dy[0]],
            [cy + dx[1], cy + dy[1]],
            color="white",
            linewidth=2.0,
            solid_capstyle="round",
            zorder=6,
        )


# ---------------------------------------------------------------------------
# LEFT PANEL — federated network diagram
# ---------------------------------------------------------------------------
def _draw_network_panel(ax: plt.Axes) -> None:
    ax.set_xlim(-2.8, 2.8)
    ax.set_ylim(-3.2, 3.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(PANEL_BG)

    # Panel background
    bg = mpatches.FancyBboxPatch(
        (-2.8, -3.2), 5.6, 6.2,
        boxstyle="round,pad=0.05",
        linewidth=1.2,
        edgecolor=COLOR_EDGE_PANEL,
        facecolor=PANEL_BG,
        zorder=0,
    )
    ax.add_patch(bg)

    # ---------- Server node (central gold hexagon) ----------
    server_r = 0.66
    hex_x = [server_r * math.cos(math.pi / 6 + k * math.pi / 3) for k in range(6)]
    hex_y = [server_r * math.sin(math.pi / 6 + k * math.pi / 3) for k in range(6)]
    hex_x.append(hex_x[0])
    hex_y.append(hex_y[0])
    ax.fill(hex_x, hex_y, color=SERVER_COLOR, zorder=4, alpha=0.92)
    ax.plot(hex_x, hex_y, color=COLOR_SERVER_EDGE, linewidth=1.5, zorder=4)
    ax.text(
        0, 0.17, "Robust", ha="center", va="center",
        fontsize=8.8, fontweight="bold", color=AXIS_COLOR,
        fontfamily=FONT_FAMILY, zorder=5,
    )
    ax.text(
        0, -0.03, "aggregation", ha="center", va="center",
        fontsize=8.5, fontweight="bold", color=AXIS_COLOR,
        fontfamily=FONT_FAMILY, zorder=5,
    )
    ax.text(
        0, -0.23, "server", ha="center", va="center",
        fontsize=8.5, fontweight="bold", color=AXIS_COLOR,
        fontfamily=FONT_FAMILY, zorder=5,
    )

    ring_r = 2.05
    n_agents = COVER_NETWORK_N_AGENTS
    adversary_idx = {
        int(round((idx + 0.5) * n_agents / COVER_NETWORK_N_ADVERSARIAL)) % n_agents
        for idx in range(COVER_NETWORK_N_ADVERSARIAL)
    }

    honest_beliefs = [0.1, 0.15, 0.65, 0.1]    # peaked at state 3

    node_r = 0.42
    agent_positions: list[tuple[float, float]] = []

    for i in range(n_agents):
        angle = math.pi / 2 + i * 2 * math.pi / n_agents   # start from top
        px = ring_r * math.cos(angle)
        py = ring_r * math.sin(angle)
        agent_positions.append((px, py))

        is_adv = i in adversary_idx
        color = ADVERSARY_COLOR if is_adv else HONEST_COLOR
        edge_color = COLOR_ADVERSARY_EDGE if is_adv else COLOR_HONEST_EDGE

        # Circle fill
        circle = mpatches.Circle(
            (px, py), node_r,
            facecolor=color, edgecolor=edge_color,
            linewidth=1.8, zorder=4, alpha=0.93,
        )
        ax.add_patch(circle)

        # Inner glyph
        if is_adv:
            _draw_x_mark(ax, px, py, node_r)
        else:
            _draw_mini_bars(ax, px, py, node_r, honest_beliefs, "white")

        # Label below node — honest agents numbered contiguously so labels
        # never skip an index when adversaries occupy ring slots.
        if is_adv:
            label = "Adversary"
        else:
            honest_seen = sum(
                1 for j in range(i + 1) if j not in adversary_idx
            )
            label = f"Agent {honest_seen}"
        ax.text(
            px, py - node_r - 0.18,
            label,
            ha="center", va="top",
            fontsize=8.5, color=AXIS_COLOR,
            fontfamily=FONT_FAMILY, zorder=5,
        )

    # ---------- Arrows: agents → server ----------
    for i, (px, py) in enumerate(agent_positions):
        # Arrow tip stops at edge of server hexagon; tail starts at edge of node
        dx = -px
        dy = -py
        dist = math.hypot(dx, dy)
        dx_n, dy_n = dx / dist, dy / dist

        tail_x = px + dx_n * (node_r + 0.04)
        tail_y = py + dy_n * (node_r + 0.04)
        head_x = dx_n * (server_r + 0.10)
        head_y = dy_n * (server_r + 0.10)

        is_adv = i in adversary_idx
        arrow_color = ADVERSARY_COLOR if is_adv else HONEST_COLOR
        alpha = 0.55 if is_adv else 0.45
        lw = 1.6 if is_adv else 1.3
        ls = "dashed" if is_adv else "solid"

        ax.annotate(
            "",
            xy=(head_x, head_y),
            xytext=(tail_x, tail_y),
            arrowprops=dict(
                arrowstyle="-|>",
                color=arrow_color,
                lw=lw,
                linestyle=ls,
                mutation_scale=9,
                alpha=alpha,
            ),
            zorder=3,
        )

    # ---------- Consensus belief bar chart (below server) ----------
    consensus_x = 0.0
    consensus_y = -1.88
    bar_panel_w = 1.10
    bar_panel_h = 0.65
    bar_bg = mpatches.FancyBboxPatch(
        (consensus_x - bar_panel_w / 2, consensus_y - 0.06),
        bar_panel_w,
        bar_panel_h,
        boxstyle="round,pad=0.05",
        linewidth=1.0,
        edgecolor=COLOR_EDGE_LIGHT,
        facecolor="white",
        zorder=4,
    )
    ax.add_patch(bar_bg)

    consensus_beliefs = [0.07, 0.10, 0.75, 0.08]
    bar_w_c = 0.16
    x0_c = consensus_x - 1.5 * (bar_w_c + 0.06)
    max_h_c = max(consensus_beliefs)
    for j, h in enumerate(consensus_beliefs):
        bx = x0_c + j * (bar_w_c + 0.06)
        bh = (h / max_h_c) * (bar_panel_h - 0.15)
        rect = mpatches.FancyBboxPatch(
            (bx, consensus_y - 0.02),
            bar_w_c,
            bh,
            boxstyle="square,pad=0",
            linewidth=0,
            facecolor=SERVER_COLOR,
            alpha=0.9,
            zorder=5,
        )
        ax.add_patch(rect)

    ax.text(
        consensus_x, consensus_y - 0.24,
        "Consensus Belief",
        ha="center", va="top",
        fontsize=8.5, fontweight="bold", color=AXIS_COLOR,
        fontfamily=FONT_FAMILY, zorder=5,
    )

    # Short arrow from server to consensus panel
    ax.annotate(
        "",
        xy=(0.0, consensus_y + bar_panel_h - 0.02),
        xytext=(0.0, -(server_r + 0.10)),
        arrowprops=dict(
            arrowstyle="-|>",
            color=SERVER_COLOR,
            lw=1.8,
            mutation_scale=10,
        ),
        zorder=3,
    )

    # ---------- Title ----------
    ax.text(
        0, 2.82,
        "Federated Belief Sharing Under Contamination",
        ha="center", va="top",
        fontsize=9.5, fontweight="bold", color=AXIS_COLOR,
        fontfamily=FONT_FAMILY, zorder=5,
    )

    # ---------- Legend ----------
    legend_items = [
        mpatches.Patch(facecolor=HONEST_COLOR, edgecolor=COLOR_HONEST_EDGE, label="Honest agent"),
        mpatches.Patch(facecolor=ADVERSARY_COLOR, edgecolor=COLOR_ADVERSARY_EDGE, label="Adversarial agent"),
        mpatches.Patch(facecolor=SERVER_COLOR, edgecolor=COLOR_SERVER_EDGE, label="Robust server"),
    ]
    leg = ax.legend(
        handles=legend_items,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        fontsize=8.5,
        frameon=True,
        framealpha=0.85,
        edgecolor=COLOR_EDGE_PANEL,
        handlelength=1.2,
        handleheight=0.9,
    )
    leg.get_frame().set_linewidth(0.8)


# ---------------------------------------------------------------------------
# RIGHT PANEL — schematic outcome cards
# ---------------------------------------------------------------------------
def _draw_distribution_card(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str,
    values: list[float],
    accent: str,
    edge: str,
    badge: str,
) -> None:
    card = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        transform=ax.transAxes,
        boxstyle="round,pad=0.018",
        linewidth=1.4,
        edgecolor=edge,
        facecolor="white",
        zorder=2,
    )
    ax.add_patch(card)

    ax.text(
        x + 0.045,
        y + h - 0.065,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.2,
        fontweight="bold",
        color=AXIS_COLOR,
        fontfamily=FONT_FAMILY,
        zorder=4,
    )
    ax.text(
        x + 0.045,
        y + h - 0.125,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color=COLOR_ARROW,
        fontfamily=FONT_FAMILY,
        zorder=4,
    )

    badge_box = mpatches.FancyBboxPatch(
        (x + w - 0.20, y + h - 0.15),
        0.14,
        0.09,
        transform=ax.transAxes,
        boxstyle="round,pad=0.015",
        linewidth=0,
        facecolor=accent,
        alpha=0.95,
        zorder=4,
    )
    ax.add_patch(badge_box)
    ax.text(
        x + w - 0.13,
        y + h - 0.105,
        badge,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color="white",
        fontfamily=FONT_FAMILY,
        zorder=5,
    )

    plot_x = x + 0.045
    plot_y = y + 0.025
    plot_w = w - 0.31
    plot_h = h * 0.18
    baseline = plot_y + 0.015
    max_v = max(values)
    n = len(values)
    gap = plot_w * 0.045
    bar_w = (plot_w - gap * (n - 1)) / n
    for idx, value in enumerate(values):
        bx = plot_x + idx * (bar_w + gap)
        bh = (value / max_v) * plot_h
        color = accent if idx == 2 else COLOR_EDGE_PANEL
        rect = mpatches.FancyBboxPatch(
            (bx, baseline),
            bar_w,
            bh,
            transform=ax.transAxes,
            boxstyle="round,pad=0.002",
            linewidth=0,
            facecolor=color,
            alpha=0.95 if idx == 2 else 0.78,
            zorder=3,
        )
        ax.add_patch(rect)

def _draw_performance_panel(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    bg = mpatches.FancyBboxPatch(
        (0.0, 0.0),
        1.0,
        1.0,
        transform=ax.transAxes,
        boxstyle="round,pad=0.018",
        linewidth=1.2,
        edgecolor=COLOR_EDGE_PANEL,
        facecolor=PANEL_BG,
        zorder=0,
    )
    ax.add_patch(bg)

    ax.text(
        0.05,
        0.93,
        "Consensus after contamination",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13.5,
        fontweight="bold",
        color=AXIS_COLOR,
        fontfamily=FONT_FAMILY,
        zorder=3,
    )
    ax.text(
        0.05,
        0.865,
        "Same broadcasts, different fusion rule",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color=COLOR_ARROW,
        fontfamily=FONT_FAMILY,
        zorder=3,
    )

    naive_acc = SYSTEM_OVERVIEW_METADATA["naive_acc_pct"]
    robust_acc = SYSTEM_OVERVIEW_METADATA["robust_acc_pct"]
    schematic = build_data()
    gain_pp = (
        SYSTEM_OVERVIEW_METADATA["robust_acc_pct"] - SYSTEM_OVERVIEW_METADATA["naive_acc_pct"]
    )
    _draw_distribution_card(
        ax,
        x=0.06,
        y=0.55,
        w=0.88,
        h=0.25,
        title=METHOD_LABELS[0],
        subtitle="confident wrong broadcasts pull the product",
        values=list(schematic["naive"]),
        accent=COLOR_NAIVE,
        edge=COLOR_ADVERSARY_EDGE,
        badge=f"{naive_acc}%",
    )
    _draw_distribution_card(
        ax,
        x=0.06,
        y=0.25,
        w=0.88,
        h=0.25,
        title=METHOD_LABELS[1],
        subtitle="server suppresses outlying beliefs before fusion",
        values=list(schematic["robust"]),
        accent=COLOR_ROBUST,
        edge=COLOR_HONEST_EDGE,
        badge=f"{robust_acc}%",
    )

    arrow = mpatches.FancyArrowPatch(
        (0.50, 0.53),
        (0.50, 0.51),
        transform=ax.transAxes,
        arrowstyle="simple",
        mutation_scale=22,
        color=COLOR_ARROW,
        alpha=0.45,
        zorder=4,
    )
    ax.add_patch(arrow)
    ax.text(
        0.76,
        0.515,
        f"+{gain_pp}pp true-state mass",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.7,
        fontweight="bold",
        color=AXIS_COLOR,
        fontfamily=FONT_FAMILY,
        bbox={
            "boxstyle": "round,pad=0.18",
            "fc": "white",
            "ec": COLOR_EDGE_PANEL,
            "alpha": 0.92,
            "lw": 0.8,
        },
        zorder=5,
    )

    n_agents = SYSTEM_OVERVIEW_METADATA["n_agents"]
    n_adv = SYSTEM_OVERVIEW_METADATA["n_adversarial"]
    ax.text(
        0.06,
        0.145,
        f"Deterministic schematic: {n_adv}/{n_agents} agents contaminated",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=8.5,
        color=COLOR_ARROW,
        fontfamily=FONT_FAMILY,
        zorder=4,
    )
    ax.text(
        0.06,
        0.075,
        "Variational server: objective-backed fusion is a separate conservative rule",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=8.5,
        color=AXIS_COLOR,
        fontfamily=FONT_FAMILY,
        bbox={
            "boxstyle": "round,pad=0.35",
            "fc": "white",
            "ec": COLOR_VARIATE,
            "alpha": 0.95,
            "lw": 1.1,
        },
        zorder=5,
    )


# ---------------------------------------------------------------------------
# Main figure assembly
# ---------------------------------------------------------------------------
def _draw_guarantee_strip(ax: plt.Axes) -> None:
    """Draw the claim-bounded three-axis strip beneath the abstract panels."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.text(
        0.0,
        0.96,
        "Three robustness axes — related routes, non-transferable guarantees",
        ha="left",
        va="top",
        fontsize=9.5,
        fontweight="bold",
        color=AXIS_COLOR,
    )
    cards = (
        (
            0.00,
            0.08,
            0.31,
            "CLIENT-SIDE FEDGVI",
            "source / β generalized-Bayes update\nsource-conditional bounded-influence result",
            HONEST_COLOR,
        ),
        (
            0.345,
            0.08,
            0.31,
            "SERVER HEURISTIC",
            "robust_aggregate reweighting\nconditional accuracy; recovery limit only",
            COLOR_NAIVE,
        ),
        (
            0.69,
            0.08,
            0.31,
            "VARIATIONAL SERVER",
            "variational_aggregate and F(q,a)\nobjective-backed raw-weight control; conservative",
            VARIATE_COLOR,
        ),
    )
    for x, y, w, title, body, color in cards:
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x, y),
                w,
                0.66,
                transform=ax.transAxes,
                boxstyle="round,pad=0.012",
                linewidth=1.0,
                edgecolor=color,
                facecolor="white",
            )
        )
        ax.text(
            x + 0.025,
            y + 0.49,
            title,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=color,
        )
        ax.text(
            x + 0.025,
            y + 0.25,
            body,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=8.5,
            color=AXIS_COLOR,
            linespacing=1.35,
        )


def generate_graphical_abstract(*, project_root: Path | None = None) -> Path:
    """Generate the refreshed graphical abstract and manuscript cover.

    The network and deterministic outcome cards retain the existing metadata-
    backed schematic, while the recovery ribbon and three-axis strip make the
    formal identity and guarantee boundaries explicit.  This is not a sampled
    result and carries no uncertainty interval.

    Args:
        project_root: Optional project root receiving the figure and cover.

    Returns:
        Path to the generated graphical-abstract PNG.
    """
    root = project_root or PROJECT_ROOT
    out_dir = root / "output" / "figures"
    cover_path = root / "manuscript" / "cover_image.png"
    apply_style()
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.facecolor": "white",
        "axes.facecolor": PANEL_BG,
        # Disable global autolayout for this figure — explicit gridspec
        # positioning is used so tight_layout would conflict.
        "figure.autolayout": False,
    })

    fig = plt.figure(figsize=(15.0, 8.5), dpi=220)

    # Upper row: network and deterministic outcome cards. Lower row: claim map.
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[4.45, 1.25],
        width_ratios=[5.8, 4.2],
        left=0.03, right=0.97,
        top=0.80, bottom=0.13,
        wspace=0.08,
        hspace=0.12,
    )

    ax_net = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_axes = fig.add_subplot(gs[1, :])

    _draw_network_panel(ax_net)
    _draw_performance_panel(ax_bar)
    _draw_guarantee_strip(ax_axes)

    fig.text(
        0.5, 0.965,
        "Federated belief sharing: from generative model to robust consensus",
        ha="center", va="top",
        fontsize=18,
        fontweight="bold",
        color=AXIS_COLOR,
        fontfamily=FONT_FAMILY,
    )
    fig.text(
        0.5, 0.915,
        "A categorical, recovery-tested bridge between active inference and generalized Bayes",
        ha="center", va="top",
        fontsize=10.5,
        color=COLOR_ARROW,
        fontfamily=FONT_FAMILY,
    )

    # Footer text bar
    fig.text(
        0.5, 0.875,
        "Recovery anchor: robust_aggregate(0) = log_linear_pool = standard belief sharing",
        ha="center", va="center",
        fontsize=9.0, color=AXIS_COLOR,
        fontfamily=FONT_FAMILY,
        bbox={
            "boxstyle": "round,pad=0.42",
            "fc": "white",
            "ec": COLOR_EDGE_PANEL,
            "lw": 1.0,
        },
    )
    fig.text(
        0.5, 0.025,
        "belief broadcasts → fusion rule → claim-bounded consensus schematic",
        ha="center", va="bottom",
        fontsize=9.5, color=COLOR_ARROW,
        fontfamily=FONT_FAMILY,
        style="italic",
    )

    # Panel labels A / B
    for letter, ax in zip(["A", "B"], [ax_net, ax_bar]):
        x_pos = ax.get_position().x0 + 0.005
        y_pos = ax.get_position().y1 + 0.015
        fig.text(
            x_pos, y_pos, letter,
            fontsize=14, fontweight="bold",
            color=AXIS_COLOR, fontfamily=FONT_FAMILY,
        )

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    cover_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "graphical_abstract.png"
    pdf_path = out_dir / "graphical_abstract.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(cover_path, dpi=300, bbox_inches="tight", facecolor="white")
    _rewrite_as_rgb_png(png_path)
    _rewrite_as_rgb_png(cover_path)
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)

    return png_path


def main() -> None:
    """Generate the graphical abstract and print its output paths."""
    png_path = generate_graphical_abstract()
    print(f"Saved PNG → {png_path}")
    print(f"Saved cover → {_COVER}")
    print(f"Saved PDF → {png_path.with_suffix('.pdf')}")


__all__ = ["generate_graphical_abstract", "main"]


if __name__ == "__main__":
    main()
