"""Redescending-weight figure: variational vs naive sensitivity.

Visualizes the tested redescending effective-weight behavior of
:func:`fedference.aggregation.variational_aggregate`: as a single agent's belief
drifts away from the consensus, its normalized influence collapses toward zero,
along the configured path. This is not an estimator-level B-robustness proof.
The naive Friston pool, by contrast,
holds every agent at the fixed ``1/n`` influence however wrong it is — an
absence of influence suppression. The gap between the falling variational curve
and the flat naive line makes the bounded-influence weight control visible. Pure
``matplotlib`` (Agg); the weights come from the analysis workflow, this module
only draws.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._common import (
    COLOR_ADVERSARY,
    COLOR_CORRECT,
    COLOR_DARK,
    COLOR_NAIVE,
    COLOR_ROBUST,
    COLOR_VARIATE,
    annotate_stats_box,
    apply_style,
    figures_dir,
    plt,
    save_figure,
)


def generate_bounded_influence(
    drifts,
    variational_influence,
    naive_influence: float,
    *,
    project_root: Path | None = None,
    filename: str = "bounded_influence.png",
) -> Path:
    """Render outlier influence vs divergence: variational curve vs naive line.

    Args:
        drifts: Sequence of drift levels in ``[0, 1]`` — how far the probed agent
            has moved toward a confidently-wrong delta.
        variational_influence: The probed agent's normalized influence under
            :func:`fedference.aggregation.variational_aggregate` at each drift
            (same length as ``drifts``).
        naive_influence: The probed agent's fixed influence under the naive pool
            (``1/n`` — drawn as a horizontal reference line).
        project_root: Project root override.
        filename: Output PNG name under ``output/figures``.

    Returns:
        Path to the written PNG.

    Raises:
        ValueError: On length mismatch, empty input, or non-finite values.
    """
    x = np.asarray(list(drifts), dtype=np.float64).ravel()
    y = np.asarray(list(variational_influence), dtype=np.float64).ravel()
    if x.size == 0:
        raise ValueError("drifts must be non-empty")
    if x.size != y.size:
        raise ValueError("drifts and variational_influence must have equal length")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))
            and np.isfinite(naive_influence)):
        raise ValueError("all inputs must be finite")

    apply_style()
    # Wide enough that the long title and y-label render un-clipped.
    fig, ax = plt.subplots(figsize=(7.4, 4.4))

    # Variational curve (suppression in COLOR_ROBUST)
    ax.plot(
        x,
        y,
        marker="o",
        markersize=5,
        linewidth=1.8,
        color=COLOR_ROBUST,
        label="Variational  (redescending weight)",
        zorder=3,
    )

    # Highlight where variational influence drops below half of naive
    half_naive = naive_influence / 2.0
    crossings = np.where(y < half_naive)[0]
    if crossings.size > 0:
        cross_i = int(crossings[0])
        ax.scatter(
            [x[cross_i]],
            [y[cross_i]],
            color=COLOR_ADVERSARY,
            s=60,
            zorder=5,
            label=f"Drops below 50% of naive  (drift ≥ {x[cross_i]:.2f})",
        )

    # Highlight initial (consensus) point with COLOR_CORRECT
    ax.scatter(
        [x[0]],
        [y[0]],
        color=COLOR_CORRECT,
        s=60,
        zorder=5,
        label=f"At consensus  (influence = {y[0]:.3g})",
    )

    # Highlight minimum influence point
    min_idx = int(np.argmin(y))
    if min_idx != 0 and y[min_idx] < y[0] * 0.8:
        ax.scatter(
            [x[min_idx]],
            [y[min_idx]],
            color=COLOR_VARIATE,
            s=50,
            zorder=5,
            label=f"Minimum influence = {y[min_idx]:.3g}",
        )

    # Naive floor reference (COLOR_NAIVE)
    ax.axhline(
        float(naive_influence),
        color=COLOR_NAIVE,
        linestyle="--",
        linewidth=1.6,
        label=f"Naive pool  (fixed $1/n$ = {naive_influence:.3g})",
        zorder=2,
    )

    # Shade the robustness gap
    ax.fill_between(
        x,
        y,
        float(naive_influence),
        where=(y <= naive_influence).tolist(),
        color=COLOR_ROBUST,
        alpha=0.12,
        label="Robustness gap  (variational below naive)",
        zorder=1,
    )

    # Arrow annotation: final influence vs naive
    ax.annotate(
        f"Final influence\n{y[-1]:.3g}",
        xy=(x[-1], y[-1]),
        xytext=(-18, 36),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": COLOR_ROBUST, "lw": 1.0},
        fontsize=9.5,
        color=COLOR_ROBUST,
        ha="right",
    )
    # Label the flat naive line mid-axis, in the robustness gap below it,
    # well clear of the upper-right stats box.
    _label_x = x[0] + 0.45 * (x[-1] - x[0])
    ax.annotate(
        f"Naive fixed\n{naive_influence:.3g}",
        xy=(_label_x, naive_influence),
        xytext=(_label_x - 0.12 * (x[-1] - x[0]), naive_influence * 0.72),
        arrowprops={"arrowstyle": "->", "color": COLOR_NAIVE, "lw": 1.0},
        fontsize=9.5,
        color=COLOR_NAIVE,
        ha="center",
        va="top",
    )

    ax.set_xlabel(
        "Outlier drift toward confidently-wrong state  (0 = consensus, 1 = delta)",
        labelpad=6,
        color=COLOR_DARK,
    )
    ax.set_ylabel(
        "Normalized server weight of the probed agent",
        labelpad=6,
        color=COLOR_DARK,
    )
    ax.set_ylim(bottom=0.0)
    ax.set_title(
        "Variational server weight redescends on this path",
        pad=12,
        color=COLOR_DARK,
    )

    # Suppression factor (naive / diverged influence) for the stats box.
    total_gap = float(naive_influence - y[-1])
    if y[-1] > 0:
        suppression_line = (
            f"Influence suppressed {naive_influence / y[-1]:.0f}× vs naive"
        )
    else:
        suppression_line = "Influence fully suppressed (zero weight)"
    stats_text = (
        f"Naive fixed weight:  {naive_influence:.3g}\n"
        f"Variational @ max drift:  {y[-1]:.3g}\n"
        f"{suppression_line}\n"
        f"Max deterministic gap:  {total_gap:.3g}"
    )
    annotate_stats_box(ax, stats_text, loc="upper right", fontsize=10)

    ax.legend(fontsize=10, loc="lower left")
    return save_figure(fig, figures_dir(project_root) / filename)


__all__ = ["generate_bounded_influence"]
