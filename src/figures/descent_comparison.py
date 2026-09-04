"""Descent-comparison figure: single-start capture vs multi-start escape.

Makes the iteration-4 fix visible. On a near-one-hot adversarial colony the
log-linear-pool seed is itself captured, so a single-start descent of the
variational free energy settles in a higher observed basin (the outlier keeps
its weight). One configured alternative start reaches a lower observed basin.
Plotting both free-energy trajectories on one axis shows the finite contrast
without certifying a global optimum. Pure ``matplotlib`` (Agg); the two histories
come from the analysis workflow, this module only draws.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._common import (
    COLOR_DARK,
    annotate_stats_box,
    apply_style,
    figures_dir,
    plt,
    save_figure,
    semantic_style,
)

MANUSCRIPT_WIDTH_FRACTION = 0.80


def generate_descent_comparison(
    single_history,
    multi_history,
    *,
    project_root: Path | None = None,
    filename: str = "descent_comparison.png",
) -> Path:
    """Render the single-start vs multi-start free-energy descent on one axis.

    Args:
        single_history: Free-energy values per iteration from the single
            (log-linear-pool) start — the higher observed basin on a near-vertex colony.
        multi_history: Free-energy values from the multi-start descent — the
            lower final $F$ among the configured starts.
        project_root: Project root override.
        filename: Output PNG name under ``output/figures``.

    Returns:
        Path to the written PNG.

    Raises:
        ValueError: If either history is empty or non-finite.
    """
    single = np.asarray(list(single_history), dtype=np.float64).ravel()
    multi = np.asarray(list(multi_history), dtype=np.float64).ravel()
    if single.size == 0 or multi.size == 0:
        raise ValueError("both histories must be non-empty")
    if not (np.all(np.isfinite(single)) and np.all(np.isfinite(multi))):
        raise ValueError("histories must be finite")

    apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    single_iters = np.arange(1, single.size + 1)
    multi_iters = np.arange(1, multi.size + 1)
    single_style = semantic_style("condition_reference")
    multi_style = semantic_style("condition_comparison")
    reference_rule = semantic_style("reference_rule")

    # Both curves are initialization conditions for the same variational
    # method.  Neutral condition styles prevent the single-start trajectory
    # from masquerading as the naive aggregation method and prevent the
    # multi-start trajectory from borrowing the heuristic-robust identity.
    ax.plot(
        single_iters,
        single,
        marker=single_style.marker,
        markersize=4,
        linewidth=single_style.linewidth,
        linestyle=single_style.dash,
        color=single_style.color,
        markerfacecolor=single_style.color,
        markeredgecolor=single_style.keyline,
        label=f"Single-start  (higher observed basin,  $F_\\infty$ = {single[-1]:.3g} nats)",
        zorder=3,
    )

    # The comparison condition uses an open diamond and dotted path, remaining
    # distinguishable in grayscale and when the legend is detached.
    ax.plot(
        multi_iters,
        multi,
        marker=multi_style.marker,
        markersize=5,
        linewidth=multi_style.linewidth,
        linestyle=multi_style.dash,
        alpha=1.0,
        color=multi_style.color,
        markerfacecolor="white",
        markeredgecolor=multi_style.keyline,
        label=f"Multi-start  (lower observed basin,  $F_\\infty$ = {multi[-1]:.3g} nats)",
        zorder=4,
    )

    # Highlight final points
    ax.scatter(
        [single_iters[-1]],
        [single[-1]],
        marker=single_style.marker,
        facecolor=single_style.color,
        edgecolor=single_style.keyline,
        s=55,
        zorder=5,
        label="Single-start terminus  (higher observed)",
    )
    ax.scatter(
        [multi_iters[-1]],
        [multi[-1]],
        marker=multi_style.marker,
        facecolor="white",
        edgecolor=multi_style.keyline,
        s=55,
        zorder=5,
        label="Multi-start terminus  (lower among configured starts)",
    )

    # Reference at the lower final value observed among configured starts.
    ax.axhline(
        multi[-1],
        color=reference_rule.color,
        linestyle=reference_rule.dash,
        linewidth=reference_rule.linewidth,
        label="Lower observed final level",
        zorder=2,
    )

    # Shade the gap between the two observed termini.
    gap = float(single[-1] - multi[-1])
    if gap > 0:
        ax.annotate(
            "",
            xy=(max(single_iters[-1], multi_iters[-1]) + 0.15, float(multi[-1])),
            xytext=(max(single_iters[-1], multi_iters[-1]) + 0.15, float(single[-1])),
            arrowprops={
                "arrowstyle": "<->",
                "color": multi_style.keyline,
                "lw": 1.4,
            },
        )
        ax.text(
            max(single_iters[-1], multi_iters[-1]) + 0.25,
            float(single[-1] + multi[-1]) / 2,
            f"ΔF={gap:.3g}",
            va="center",
            fontsize=9.5,
            color=multi_style.keyline,
        )

    ax.set_xlabel("Block-coordinate iteration", labelpad=6, color=COLOR_DARK)
    ax.set_ylabel("Variational free energy $F$  (nats)", labelpad=6, color=COLOR_DARK)
    ax.set_title(
        "Configured alternative start reaches a lower observed basin",
        pad=8,
        color=COLOR_DARK,
    )

    # Headroom above the captured-basin curve so the upper-right stats box sits
    # in empty space instead of on the single-start terminus marker.
    y_hi = float(max(single.max(), multi.max()))
    y_lo = float(min(single.min(), multi.min()))
    y_span = (y_hi - y_lo) if y_hi > y_lo else 1.0
    ax.set_ylim(top=y_hi + 0.42 * y_span)

    # Stats box
    stats_text = (
        f"Single-start final F: {single[-1]:.3g} nats\n"
        f"Multi-start final F: {multi[-1]:.3g} nats\n"
        f"Basin gap ΔF: {gap:.3g} nats\n"
        f"Iterations: single={single.size}, multi={multi.size}"
    )
    annotate_stats_box(ax, stats_text, loc="upper right", fontsize=10)

    # Legend in the empty band between the two flat trajectories, off the
    # multi-start curve.
    ax.legend(fontsize=10, loc="center left")

    return save_figure(
        fig,
        figures_dir(project_root) / filename,
        manuscript_width_fraction=MANUSCRIPT_WIDTH_FRACTION,
    )


__all__ = ["generate_descent_comparison"]
