"""Descent-comparison figure: single-start capture vs multi-start escape.

Makes the iteration-4 fix visible. On a near-one-hot adversarial colony the
log-linear-pool seed is itself captured, so a single-start descent of the
variational free energy settles in a high-$F$ capture basin (the outlier keeps
its weight). Multi-start descent (the default) reaches the genuinely lower-$F$
vetoing basin. Plotting both free-energy trajectories on one axis shows the
capture and the escape directly. Pure ``matplotlib`` (Agg); the two histories
come from the analysis workflow, this module only draws.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._common import (
    COLOR_ADVERSARY,
    COLOR_CORRECT,
    COLOR_DARK,
    COLOR_MUTED,
    COLOR_NAIVE,
    COLOR_ROBUST,
    annotate_stats_box,
    apply_style,
    figures_dir,
    plt,
    save_figure,
)


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
            (log-linear-pool) start — the captured basin on a near-vertex colony.
        multi_history: Free-energy values from the multi-start descent — the
            vetoing basin (lower final $F$).
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

    # Single-start: colored with COLOR_NAIVE (captured / naive baseline)
    ax.plot(
        single_iters,
        single,
        marker="s",
        markersize=4,
        linewidth=1.8,
        color=COLOR_NAIVE,
        label=f"Single-start  (captured basin,  $F_\\infty$ = {single[-1]:.3g} nats)",
        zorder=3,
    )

    # Multi-start: COLOR_ROBUST at full saturation, drawn above everything else
    # so the escape trajectory stays visible.
    ax.plot(
        multi_iters,
        multi,
        marker="o",
        markersize=5,
        linewidth=2.4,
        alpha=1.0,
        color=COLOR_ROBUST,
        label=f"Multi-start  (vetoing basin,  $F_\\infty$ = {multi[-1]:.3g} nats)",
        zorder=4,
    )

    # Highlight final points
    ax.scatter(
        [single_iters[-1]],
        [single[-1]],
        color=COLOR_ADVERSARY,
        s=55,
        zorder=5,
        label="Single-start terminus  (suboptimal)",
    )
    ax.scatter(
        [multi_iters[-1]],
        [multi[-1]],
        color=COLOR_CORRECT,
        s=55,
        zorder=5,
        label="Multi-start terminus  (optimal basin)",
    )

    # Floor reference for multi-start basin
    ax.axhline(
        multi[-1],
        color=COLOR_MUTED,
        linestyle=":",
        linewidth=1.0,
        label="Vetoing-basin floor",
        zorder=2,
    )

    # Shade gap between basins (shows the cost of capture)
    gap = float(single[-1] - multi[-1])
    if gap > 0:
        ax.annotate(
            "",
            xy=(max(single_iters[-1], multi_iters[-1]) + 0.15, float(multi[-1])),
            xytext=(max(single_iters[-1], multi_iters[-1]) + 0.15, float(single[-1])),
            arrowprops={
                "arrowstyle": "<->",
                "color": COLOR_ADVERSARY,
                "lw": 1.4,
            },
        )
        ax.text(
            max(single_iters[-1], multi_iters[-1]) + 0.25,
            float(single[-1] + multi[-1]) / 2,
            f"ΔF={gap:.3g}",
            va="center",
            fontsize=9.5,
            color=COLOR_ADVERSARY,
        )

    ax.set_xlabel("Block-coordinate iteration", labelpad=6, color=COLOR_DARK)
    ax.set_ylabel("Variational free energy $F$  (nats)", labelpad=6, color=COLOR_DARK)
    ax.set_title(
        "Multi-start escapes the near-vertex capture basin",
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

    return save_figure(fig, figures_dir(project_root) / filename)


__all__ = ["generate_descent_comparison"]
