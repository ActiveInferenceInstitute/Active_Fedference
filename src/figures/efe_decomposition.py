"""Expected-free-energy decomposition figure: cost, value, and information gain.

The figure shows the exact identity returned by
:func:`fedference.expected_free_energy.decompose` (project gate ISC-19):

* ``G = risk + ambiguity``;
* ``G = -(pragmatic_value + epistemic_value)``.

The cost view is an additive stack. The value view is deliberately rendered as
a signed waterfall: ``-pragmatic_value`` rises from zero and ``-epistemic_value``
then corrects it downward to the same terminal ``G``. This avoids the false
visual implication that a signed cancellation is an ordinary positive stack.
The epistemic contribution is the state--outcome mutual information
``I(s;o|boldsymbol{pi})``. The analysis diagnostic uses an uncertainty-bearing uniform prior
so that this term is visible; the canonical point-mass sentinel prior is a
valid null case with zero information gain. Pure ``matplotlib`` (Agg); the four
terms come from the workflow.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fedference.expected_free_energy import EFE_IDENTITY_ATOL

from ._common import (
    COLOR_ACCENT,
    COLOR_AXIS,
    COLOR_MULTI_1,
    COLOR_MULTI_2,
    COLOR_MUTED,
    annotate_stats_box,
    apply_style,
    figures_dir,
    plt,
    save_figure,
)


def generate_efe_decomposition(
    risk: float,
    ambiguity: float,
    pragmatic_value: float,
    epistemic_value: float,
    *,
    project_root: Path | None = None,
    filename: str = "efe_decomposition.png",
) -> Path:
    """Render the EFE identity as an additive stack and signed waterfall.

    Args:
        risk: ``KL(q(o|boldsymbol{pi}) || p_C(o))`` term.
        ambiguity: ``E_q(s)[H[p(o|s)]]`` term.
        pragmatic_value: ``E_q(o)[ln p_C(o)]`` term (utility / exploitation).
        epistemic_value: state-outcome mutual information (information gain).
        project_root: Project root override.
        filename: Output PNG name under ``output/figures``.

    Returns:
        Path to the written PNG.

    Raises:
        ValueError: If any term is non-finite or violates the EFE identity.
    """
    terms = np.array([risk, ambiguity, pragmatic_value, epistemic_value], dtype=np.float64)
    if not np.all(np.isfinite(terms)):
        raise ValueError("all four EFE terms must be finite")
    r, amb, prag, epi = (float(v) for v in terms)

    total_ra = r + amb  # G = risk + ambiguity
    total_pe = -(prag + epi)  # G = -(pragmatic + epistemic)
    residual = total_ra - total_pe
    if abs(residual) > EFE_IDENTITY_ATOL:
        raise ValueError(
            "EFE terms violate risk + ambiguity == -(pragmatic + epistemic): "
            f"residual={residual:.3e}, tolerance={EFE_IDENTITY_ATOL:.3e}"
        )

    apply_style()
    fig, ax = plt.subplots(figsize=(10.8, 7.0))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.80, bottom=0.25)

    # Left stack: risk + ambiguity (both add toward G). The stacks use the
    # neutral multi-curve accents — this identity has no naive/robust contrast,
    # so the semantic COLOR_NAIVE/COLOR_ROBUST pair is deliberately avoided.
    ax.bar(
        0,
        r,
        color=COLOR_MULTI_1,
        alpha=0.88,
        width=0.52,
        edgecolor=COLOR_AXIS,
        linewidth=0.8,
        label="risk  $\\mathrm{KL}(q(o|\\boldsymbol{\\pi})\\|p_C(o))$",
    )
    ax.bar(
        0,
        amb,
        bottom=r,
        color=COLOR_MULTI_2,
        alpha=0.88,
        width=0.52,
        edgecolor=COLOR_AXIS,
        linewidth=0.8,
        label="ambiguity  $\\mathbb{E}[H[p(o|s)]]$",
    )

    pragmatic_magnitude = -prag
    epistemic_correction = -epi
    # The right view is a signed waterfall, not a conventional stack. Its
    # terminal cumulative endpoint is G; its upper extent is the intermediate
    # positive pragmatic magnitude before the epistemic correction.
    ax.bar(
        1,
        pragmatic_magnitude,
        color=COLOR_ACCENT,
        alpha=0.90,
        width=0.52,
        edgecolor=COLOR_AXIS,
        linewidth=0.8,
        label="$-$pragmatic value",
    )
    ax.bar(
        1,
        epistemic_correction,
        bottom=pragmatic_magnitude,
        color=COLOR_MUTED,
        alpha=0.90,
        width=0.52,
        edgecolor=COLOR_AXIS,
        linewidth=0.8,
        label="$-$epistemic correction  $I(s;o\\mid\\boldsymbol{\\pi})$",
    )

    def _segment_label(x: float, lower: float, upper: float, text: str, color: str) -> None:
        midpoint = lower + 0.5 * (upper - lower)
        if abs(upper - lower) >= 0.55:
            ax.text(
                x,
                midpoint,
                text,
                ha="center",
                va="center",
                fontsize=10.5,
                color=color,
                fontweight="bold",
            )

    _segment_label(0, 0.0, r, f"risk\n{r:.2f}", "white")
    _segment_label(0, r, total_ra, f"ambiguity\n{amb:.2f}", "white")
    _segment_label(1, 0.0, pragmatic_magnitude, f"$-$pragmatic\n{pragmatic_magnitude:.2f}", "white")
    _segment_label(1, pragmatic_magnitude, total_ra, f"$-$epistemic\n{epistemic_correction:.2f}", "white")

    g_level = total_ra
    ax.axhline(
        g_level,
        color=COLOR_AXIS,
        linestyle="--",
        linewidth=1.5,
        label=f"terminal $G(\\boldsymbol{{\\pi}})$ = {g_level:.3g} nats",
    )
    ax.scatter(
        [0.0, 1.0],
        [g_level, g_level],
        marker="D",
        s=58,
        color=COLOR_AXIS,
        edgecolor="white",
        linewidth=0.9,
        zorder=5,
    )
    ax.plot([0.27, 0.73], [g_level, g_level], color=COLOR_AXIS, linewidth=0.9, linestyle=":")
    ax.annotate(
        f"terminal endpoint = G = {g_level:.3g} nats",
        xy=(1.0, g_level),
        xytext=(1.30, g_level - 0.04 * max(g_level, 1.0)),
        ha="left",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color=COLOR_AXIS,
        arrowprops={"arrowstyle": "-", "color": COLOR_AXIS, "lw": 0.9},
    )
    positive_extent = max(0.0, pragmatic_magnitude, total_ra)
    span = max(positive_extent - min(0.0, g_level), 1.0)
    ax.set_ylim(min(0.0, g_level) - 0.08 * span, positive_extent + 0.25 * span)
    # Keep the two views visually comparable.  The endpoint annotation is
    # deliberately inside this fixed data window so tight bounding-box export
    # cannot shrink the plotting area around an overflowing note.
    ax.set_xlim(-0.55, 1.62)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        ["COST VIEW\nrisk + ambiguity", "VALUE VIEW\nsigned waterfall"],
        fontsize=11.5,
    )
    ax.set_ylabel("Expected free energy contribution (nats)", labelpad=8, fontsize=12.5)
    ax.set_title(
        "Expected-free-energy identity: additive cost and signed value views\n"
        "Categorical specialization of Friston et al. (2024), Eq. 2",
        fontsize=15,
        pad=14,
    )
    ax.annotate(
        f"identity residual = {residual:.2e} nats",
        xy=(0.5, 0.02),
        xycoords="axes fraction",
        ha="center",
        va="bottom",
        fontsize=10,
        color=COLOR_ACCENT,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLOR_ACCENT, alpha=0.7),
    )
    annotate_stats_box(
        ax,
        f"diagnostic prior: uniform\n"
        f"G = {g_level:.3g} nats\n"
        f"risk {r:.3g},  amb {amb:.3g}\n"
        f"prag {prag:.3g},  epi {epi:.3g}\n"
        "epi = $I(s;o\\mid\\boldsymbol{\\pi})$\n"
        "right endpoint, not top extent, equals G",
        loc="upper left",
        fontsize=10,
    )
    ax.text(
        0.5,
        -0.19,
        "Signed waterfall: $-$epistemic is a negative correction;\n"
        "terminal endpoint is $G$ (uniform prior makes information gain visible).",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        color=COLOR_AXIS,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=COLOR_MUTED, alpha=0.9),
    )
    ax.legend(
        fontsize=9.5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.34),
        ncol=2,
        borderaxespad=0.0,
    )

    return save_figure(fig, figures_dir(project_root) / filename)


__all__ = ["generate_efe_decomposition"]
