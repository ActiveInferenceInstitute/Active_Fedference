"""System overview figure for the Active Fedference introduction.

Three-panel figure illustrating the problem and solution:
  Panel A — The Setup: 5 agents (2 adversarial, 3 honest) with heterogeneous beliefs.
  Panel B — Naive equal-weight pooling: log-linear pool pulled off-target by adversarial agents.
  Panel C — Heuristic robust aggregation: adversarial agents down-weighted in this example.

Saves to output/figures/system_overview.{png,pdf} at 200 dpi.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from typing import NotRequired, TypedDict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from fedference.aggregation import log_linear_pool, robust_aggregate
from figures._common import (
    COLOR_ADVERSARY,
    COLOR_CORRECT,
    COLOR_DEEP,
    COLOR_MUTED,
    COLOR_PANEL_FAIL,
    COLOR_PANEL_GOOD,
    COLOR_PANEL_NOTE,
    COLOR_ROBUST,
    PROJECT_ROOT,
    apply_style,
    figures_dir,
)

# ---------------------------------------------------------------------------
# Palette (semantic aliases for local readability)
# ---------------------------------------------------------------------------
HONEST_BLUE = COLOR_ROBUST
ADVERSARIAL_RED = COLOR_ADVERSARY
CORRECT_GREEN = COLOR_CORRECT
PANEL_BG_FAIL = COLOR_PANEL_FAIL
PANEL_BG_GOOD = COLOR_PANEL_GOOD
DARK = COLOR_DEEP
GREY = COLOR_MUTED

N_STATES = 8
TRUE_STATE = 2  # 0-indexed → displays as state 3
ADV_STATE = 5  # 0-indexed → adversary peaks at state 6
# Schematic colony composition (the only authored choice; every percentage in
# SYSTEM_OVERVIEW_METADATA below is DERIVED from the pooled beliefs, never typed).
_N_HONEST = 3
_N_ADVERSARIAL = 2
_SCHEMATIC_SEED = 0
_HONEST_CONCENTRATION = 1.2
_ADVERSARIAL_CONCENTRATION = 1.8
_HONEST_SIDE_CONCENTRATION = 0.6
_ADVERSARIAL_SIDE_CONCENTRATION = 0.4
_SCHEMATIC_NOISE = 0.02
_SCHEMATIC_ROBUSTNESS = 1.5

RNG = np.random.default_rng(_SCHEMATIC_SEED)


class SystemOverviewMetadata(TypedDict):
    """Scalar provenance exported to manuscript tokens and the cover."""

    n_agents: int
    n_adversarial: int
    n_honest: int
    contamination_pct: int
    naive_acc_pct: int
    robust_acc_pct: int
    robustness: float


class SystemOverviewData(TypedDict):
    """Numerical arrays drawn by the system-overview and cover figures."""

    local_posteriors: list[np.ndarray]
    beliefs: NotRequired[list[np.ndarray]]
    naive: np.ndarray
    robust: np.ndarray
    normalized_effective_weights: np.ndarray
    weights: NotRequired[np.ndarray]
    naive_acc: float
    robust_acc: float


# ---------------------------------------------------------------------------
# Belief generators
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def honest_belief(
    true_state: int,
    concentration: float = _HONEST_CONCENTRATION,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Peaked categorical near true_state with mild uncertainty."""
    rng = RNG if rng is None else rng
    logits = np.zeros(N_STATES)
    logits[true_state] = concentration
    logits[(true_state - 1) % N_STATES] = _HONEST_SIDE_CONCENTRATION
    logits[(true_state + 1) % N_STATES] = _HONEST_SIDE_CONCENTRATION
    return _softmax(logits + rng.normal(0, _SCHEMATIC_NOISE, N_STATES))


def adversarial_belief(
    adv_state: int,
    concentration: float = _ADVERSARIAL_CONCENTRATION,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Peaked categorical at wrong state."""
    rng = RNG if rng is None else rng
    logits = np.zeros(N_STATES)
    logits[adv_state] = concentration
    logits[(adv_state - 1) % N_STATES] = _ADVERSARIAL_SIDE_CONCENTRATION
    logits[(adv_state + 1) % N_STATES] = _ADVERSARIAL_SIDE_CONCENTRATION
    return _softmax(logits + rng.normal(0, _SCHEMATIC_NOISE, N_STATES))


def naive_pool(
    local_posteriors: list[np.ndarray], base_weights: np.ndarray | None = None
) -> np.ndarray:
    """Log-linear pool: softmax of weighted sum of log-posteriors."""
    return log_linear_pool(
        local_posteriors=local_posteriors, base_weights=base_weights
    )


def robust_pool(
    local_posteriors: list[np.ndarray],
    base_weights: np.ndarray | None = None,
    concentration: float = _SCHEMATIC_ROBUSTNESS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the heuristic robust consensus and normalized influence weights."""
    result = robust_aggregate(
        local_posteriors,
        base_weights=base_weights,
        robustness=concentration,
        max_iter=128,
    )
    return result.consensus, result.normalized_effective_weights


# ---------------------------------------------------------------------------
# Build data
# ---------------------------------------------------------------------------

def _derive_schematic() -> tuple[SystemOverviewMetadata, SystemOverviewData]:
    """Build the schematic colony ONCE and derive every reported percentage.

    The only authored inputs are the colony composition and concentrations at
    the top of this module. The naive/robust "true-state mass" percentages and
    the contamination percentage are computed from the pooled beliefs, so the
    manuscript tokens exported from ``SYSTEM_OVERVIEW_METADATA`` can never
    drift from what the figure actually draws.
    """
    rng = np.random.default_rng(_SCHEMATIC_SEED)
    # agents: [adv0, adv1, honest0, honest1, honest2]
    adversarial_posteriors = [
        adversarial_belief(ADV_STATE, rng=rng) for _ in range(_N_ADVERSARIAL)
    ]
    honest_posteriors = [
        honest_belief(TRUE_STATE, rng=rng) for _ in range(_N_HONEST)
    ]
    local_posteriors = adversarial_posteriors + honest_posteriors
    n_agents = _N_ADVERSARIAL + _N_HONEST

    naive_consensus = naive_pool(local_posteriors)
    robust_consensus, normalized_effective_weights = robust_pool(
        local_posteriors,
        concentration=_SCHEMATIC_ROBUSTNESS,
    )

    metadata: SystemOverviewMetadata = {
        "n_agents": n_agents,
        "n_adversarial": _N_ADVERSARIAL,
        "n_honest": _N_HONEST,
        "contamination_pct": int(round(100.0 * _N_ADVERSARIAL / n_agents)),
        "naive_acc_pct": int(round(100.0 * float(naive_consensus[TRUE_STATE]))),
        "robust_acc_pct": int(round(100.0 * float(robust_consensus[TRUE_STATE]))),
        "robustness": _SCHEMATIC_ROBUSTNESS,
    }
    data: SystemOverviewData = {
        "local_posteriors": local_posteriors,
        # Compatibility key for older figure consumers; new code must use the
        # canonical local-posteriors field above.
        "beliefs": local_posteriors,
        "naive": naive_consensus,
        "robust": robust_consensus,
        "normalized_effective_weights": normalized_effective_weights,
        # Compatibility key for the pre-canonical figure-data mapping.
        "weights": normalized_effective_weights,
        "naive_acc": float(metadata["naive_acc_pct"]),
        "robust_acc": float(metadata["robust_acc_pct"]),
    }
    return metadata, data


SYSTEM_OVERVIEW_METADATA, _SCHEMATIC_DATA = _derive_schematic()


def build_data() -> SystemOverviewData:
    """Return the derived schematic colony (posteriors, consensuses, weights).

    Returns the same cached structure ``SYSTEM_OVERVIEW_METADATA`` was derived
    from, so figure pixels and exported metadata are one computation.
    """
    return _SCHEMATIC_DATA.copy()


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_agent(
    ax: plt.Axes,
    cx: float,
    cy: float,
    belief: np.ndarray,
    color: str,
    label: str,
    bar_width: float = 0.014,
    bar_height_scale: float = 0.35,
    true_state: int = TRUE_STATE,
) -> None:
    """Draw an agent circle with a mini bar chart above it."""
    radius = 0.062
    circle = plt.Circle((cx, cy), radius, color=color, zorder=3, linewidth=1.5, ec="white")
    ax.add_patch(circle)
    ax.text(cx, cy, label, ha="center", va="center", fontsize=10, color="white",
            fontweight="bold", zorder=4)

    # Mini bar chart above the circle
    bar_top = cy + radius + 0.04
    for s, p in enumerate(belief):
        bx = cx - (N_STATES / 2 - 0.5) * bar_width + s * bar_width
        bh = p * bar_height_scale
        bar_color = CORRECT_GREEN if s == true_state else GREY
        ax.bar(bx, bh, bottom=bar_top, width=bar_width * 0.85,
               color=bar_color, alpha=0.85, zorder=2)


def _draw_consensus_bar(
    ax: plt.Axes,
    cx: float,
    cy: float,
    belief: np.ndarray,
    highlight_color: str,
    true_state: int = TRUE_STATE,
    width: float = 0.55,
    height_scale: float = 1.4,
    argmax_color: str | None = None,
) -> None:
    """Draw a wide consensus bar chart centered at (cx, cy)."""
    bar_w = width / N_STATES
    bottom = cy - 0.05
    peak = int(np.argmax(belief))
    for s, p in enumerate(belief):
        bx = cx - width / 2 + (s + 0.5) * bar_w
        bh = p * height_scale
        if s == true_state:
            bar_color, alpha = highlight_color, 0.9
        elif argmax_color is not None and s == peak:
            bar_color, alpha = argmax_color, 0.9
        else:
            bar_color, alpha = GREY, 0.5
        ax.bar(bx, bh, bottom=bottom, width=bar_w * 0.85,
               color=bar_color, alpha=alpha, zorder=3, linewidth=0)
    # x-axis labels
    for s in range(N_STATES):
        bx = cx - width / 2 + (s + 0.5) * bar_w
        ax.text(bx, bottom - 0.04, str(s + 1), ha="center", va="top",
                fontsize=9.5, color=DARK)
    ax.text(cx, bottom - 0.12, "State", ha="center", va="top", fontsize=9.5, color=GREY)


def _draw_weight_dots(
    ax: plt.Axes,
    agent_xs: list[float],
    agent_y: float,
    weights: np.ndarray,
    is_adv: list[bool],
) -> None:
    """Draw weight magnitude bars below each agent."""
    dot_y = agent_y - 0.28
    ax.text(agent_xs[2], dot_y - 0.10, "Influence weights", ha="center", va="top",
            fontsize=9.0, color=DARK, style="italic")
    max_w = weights.max()
    for i, (x, w, adv) in enumerate(zip(agent_xs, weights, is_adv)):
        bar_h = (w / max_w) * 0.12
        color = ADVERSARIAL_RED if adv else HONEST_BLUE
        ax.bar(x, bar_h, bottom=dot_y - 0.13, width=0.06,
               color=color, alpha=0.8, zorder=3)
        ax.text(x, dot_y - 0.13 + bar_h + 0.01, f"{w:.2f}",
                ha="center", va="bottom", fontsize=9.0, color=DARK)


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def generate_system_overview() -> None:
    """Generate the three-panel system-overview figure (beliefs, weights, recovery).

    Panel 1: honest vs adversarial belief vectors.
    Panel 2: per-agent influence weights under naive, robust, and variational aggregation.
    Panel 3: recovery of the log-linear pool by robust_aggregate(robustness=0).
    Writes PNG + PDF to the configured ``figures`` output directory.
    """
    apply_style()
    data = build_data()

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.0), dpi=220)
    fig.patch.set_facecolor("white")

    agent_xs = [0.10, 0.26, 0.42, 0.58, 0.74]
    agent_y = 0.38
    is_adv = [True, True, False, False, False]
    adv_labels = ["A₁", "A₂"]
    hon_labels = ["H₁", "H₂", "H₃"]
    all_labels = adv_labels + hon_labels

    # ------------------------------------------------------------------
    # Panel A — The Setup
    # ------------------------------------------------------------------
    ax_a = axes[0]
    ax_a.set_xlim(0, 0.84)
    ax_a.set_ylim(-0.05, 1.0)
    ax_a.axis("off")
    ax_a.set_facecolor("white")

    ax_a.set_title("A   The Setup", fontsize=13, fontweight="bold",
                   loc="left", pad=6, color=DARK)

    # Legend — every color used in this panel gets a key (agents AND mini bars).
    ax_a.legend(handles=[
        mpatches.Patch(color=ADVERSARIAL_RED, label="Adversarial agent"),
        mpatches.Patch(color=HONEST_BLUE, label="Honest agent"),
        mpatches.Patch(color=CORRECT_GREEN, label="True-state mass"),
        mpatches.Patch(color=GREY, label="Other states"),
    ], loc="lower left", bbox_to_anchor=(0.0, -0.04), fontsize=9.5,
       framealpha=0.85, borderpad=0.35, handlelength=1.1, labelspacing=0.3)

    for i, (x, belief, adv, label) in enumerate(
        zip(agent_xs, data["local_posteriors"], is_adv, all_labels)
    ):
        color = ADVERSARIAL_RED if adv else HONEST_BLUE
        _draw_agent(ax_a, x, agent_y, belief, color, label)

    # "True state" annotation (value tracks the module constant)
    ax_a.text(0.60, 0.06, f"True hidden state = {TRUE_STATE + 1}",
              ha="center", va="center",
              fontsize=10.5, color=CORRECT_GREEN, fontweight="bold",
              bbox={"boxstyle": "round,pad=0.3", "fc": COLOR_PANEL_NOTE,
                    "ec": CORRECT_GREEN, "alpha": 0.9})

    # Dashed line below agents; contamination text derived from metadata
    ax_a.axhline(y=0.22, xmin=0.08, xmax=0.92, color=GREY, linewidth=0.6,
                 linestyle="--", alpha=0.5)
    ax_a.text(0.60, 0.17,
              f"{SYSTEM_OVERVIEW_METADATA['contamination_pct']}% contamination "
              f"({SYSTEM_OVERVIEW_METADATA['n_adversarial']}/"
              f"{SYSTEM_OVERVIEW_METADATA['n_agents']} adversarial)",
              ha="center", va="center", fontsize=9.5, color=GREY)

    # ------------------------------------------------------------------
    # Panel B — Naive Pooling Fails
    # ------------------------------------------------------------------
    ax_b = axes[1]
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(-0.05, 1.0)
    ax_b.set_facecolor(PANEL_BG_FAIL)
    ax_b.axis("off")

    ax_b.set_title("B   Naive Equal-Weight Pooling", fontsize=13, fontweight="bold",
                   loc="left", pad=6, color=ADVERSARIAL_RED)

    ax_b.text(0.5, 0.95, "Log-linear pool (equal weights)", ha="center", va="bottom",
              fontsize=10, color=DARK)

    # Shared mass→height scale across Panels B and C so bar heights compare.
    height_scale = 0.30 / max(float(data["naive"].max()), float(data["robust"].max()))
    _draw_consensus_bar(ax_b, 0.5, 0.56, data["naive"],
                        highlight_color=CORRECT_GREEN,
                        height_scale=height_scale,
                        argmax_color=ADVERSARIAL_RED)

    # argmax arrow pointing to the (wrong) peak state
    adv_peak = int(np.argmax(data["naive"]))
    bar_w_b = 0.55 / N_STATES
    peak_x = 0.5 - 0.55 / 2 + (adv_peak + 0.5) * bar_w_b
    ax_b.annotate("", xy=(peak_x, 0.83), xytext=(peak_x, 0.89),
                  arrowprops={"arrowstyle": "->", "color": ADVERSARIAL_RED, "lw": 1.4})
    ax_b.text(peak_x, 0.90, "argmax", ha="center", va="bottom",
              fontsize=9, color=ADVERSARIAL_RED)

    naive_peak_state = adv_peak + 1  # 1-indexed
    ax_b.text(0.5, 0.14,
              f"True-state mass: {int(data['naive_acc'])}%\n"
              f"argmax → state {naive_peak_state}  (true: {TRUE_STATE + 1})",
              ha="center", va="center", fontsize=10.5, color=ADVERSARIAL_RED,
              fontweight="bold", linespacing=1.4,
              bbox={"boxstyle": "round,pad=0.4", "fc": "white",
                    "ec": ADVERSARIAL_RED, "alpha": 0.9})
    ax_b.text(0.5, 0.01, "Adversarial agents pull the consensus off-target",
              ha="center", va="center", fontsize=9.5, color=GREY)

    # ------------------------------------------------------------------
    # Panel C — Robust Aggregation Succeeds
    # ------------------------------------------------------------------
    ax_c = axes[2]
    ax_c.set_xlim(0, 1)
    ax_c.set_ylim(-0.05, 1.0)
    ax_c.set_facecolor(PANEL_BG_GOOD)
    ax_c.axis("off")

    ax_c.set_title("C   Heuristic Robust Aggregation", fontsize=13, fontweight="bold",
                   loc="left", pad=6, color=CORRECT_GREEN)

    ax_c.text(0.5, 0.95, "Robust aggregate (effective weights)", ha="center", va="bottom",
              fontsize=10, color=DARK)

    _draw_consensus_bar(ax_c, 0.5, 0.56, data["robust"],
                        highlight_color=CORRECT_GREEN,
                        height_scale=height_scale)

    # argmax arrow pointing to correct state
    rob_peak = int(np.argmax(data["robust"]))
    bar_w_c = 0.55 / N_STATES
    rob_peak_x = 0.5 - 0.55 / 2 + (rob_peak + 0.5) * bar_w_c
    ax_c.annotate("", xy=(rob_peak_x, 0.83), xytext=(rob_peak_x, 0.89),
                  arrowprops={"arrowstyle": "->", "color": CORRECT_GREEN, "lw": 1.4})
    ax_c.text(rob_peak_x, 0.90, "argmax", ha="center", va="bottom",
              fontsize=9, color=CORRECT_GREEN)

    # Weight bars for each agent (5 small bars across the middle of panel C)
    agent_xs_c = np.linspace(0.12, 0.88, 5)
    weight_y = 0.27
    ax_c.text(0.02, weight_y + 0.035, "Agent influence weights:", ha="left", va="bottom",
              fontsize=9.5, color=DARK, style="italic")
    max_w = data["normalized_effective_weights"].max()
    for i, (x, w, adv, lbl) in enumerate(
        zip(agent_xs_c, data["normalized_effective_weights"], is_adv, all_labels)
    ):
        bh = (w / max_w) * 0.09
        color = ADVERSARIAL_RED if adv else HONEST_BLUE
        ax_c.bar(x, bh, bottom=weight_y - 0.09, width=0.07,
                 color=color, alpha=0.85, zorder=3)
        ax_c.text(x, weight_y - 0.09 + bh + 0.005, f"{w:.2f}",
                  ha="center", va="bottom", fontsize=8.5, color=DARK)
        ax_c.text(x, weight_y - 0.105, lbl, ha="center", va="top",
                  fontsize=8.5, color=color)

    rob_peak_state = rob_peak + 1  # 1-indexed
    ax_c.text(0.5, 0.045,
              f"True-state mass: {int(data['robust_acc'])}%\n"
              f"argmax → state {rob_peak_state}  (true: {TRUE_STATE + 1})",
              ha="center", va="center", fontsize=10.5, color=CORRECT_GREEN,
              fontweight="bold", linespacing=1.4,
              bbox={"boxstyle": "round,pad=0.4", "fc": "white",
                    "ec": CORRECT_GREEN, "alpha": 0.9})

    # ------------------------------------------------------------------
    # Dividers between panels
    # ------------------------------------------------------------------
    for x_pos in [0.355, 0.665]:
        fig.add_artist(
            plt.Line2D([x_pos, x_pos], [0.04, 0.96],
                       transform=fig.transFigure,
                       color=GREY, linewidth=0.8, linestyle=":", alpha=0.6)
        )

    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.08, wspace=0.05)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_dir = figures_dir(PROJECT_ROOT)
    png_path = out_dir / "system_overview.png"
    pdf_path = out_dir / "system_overview.pdf"

    fig.savefig(png_path, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(
        pdf_path,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)

    print(str(png_path))
    print(str(pdf_path))


if __name__ == "__main__":
    generate_system_overview()
