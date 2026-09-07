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
    COLOR_DARK,
    apply_style,
    figures_dir,
    plt,
    save_figure,
    semantic_style,
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
    # Reserve a direct-label lane on the right and explicit perimeter margins.
    # Methodological qualifications and repeated numeric summaries live in the
    # self-contained caption rather than obscuring the plotted path.
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    fig.set_layout_engine("none")
    fig.subplots_adjust(left=0.14, right=0.96, top=0.84, bottom=0.19)
    variational_style = semantic_style("variational")
    naive_style = semantic_style("naive")
    adversarial_style = semantic_style("adversarial")

    # Variational curve: marker and dash distinguish it without colour.
    ax.plot(
        x,
        y,
        marker=variational_style.marker,
        markerfacecolor="white",
        markeredgecolor=variational_style.keyline,
        linestyle=variational_style.dash,
        markersize=5.5,
        linewidth=variational_style.linewidth,
        color=variational_style.color,
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
            edgecolor=adversarial_style.keyline,
            marker=adversarial_style.marker,
            s=60,
            zorder=5,
        )
        ax.annotate(
            f"below 0.5 × naive\nat drift {x[cross_i]:.2f}",
            xy=(x[cross_i], y[cross_i]),
            xytext=(12, 26),
            textcoords="offset points",
            arrowprops={"arrowstyle": "-", "color": adversarial_style.keyline, "lw": 0.9},
            fontsize=9.5,
            color=adversarial_style.keyline,
            ha="left",
            va="bottom",
        )

    # Naive fixed-weight reference.
    ax.axhline(
        float(naive_influence),
        color=naive_style.color,
        linestyle=naive_style.dash,
        linewidth=naive_style.linewidth,
        zorder=2,
    )

    # Shade the robustness gap
    ax.fill_between(
        x,
        y,
        float(naive_influence),
        where=(y <= naive_influence).tolist(),
        color=variational_style.color,
        alpha=0.12,
        zorder=1,
    )

    x_span = max(float(np.max(x) - np.min(x)), 1.0)
    y_top = max(float(np.max(y)), float(naive_influence), 1e-6) * 1.18
    label_x = float(np.max(x)) + 0.08 * x_span
    label_end_x = float(np.max(x)) + 0.31 * x_span
    variational_label_y = max(float(y[-1]), 0.09 * y_top)
    ax.plot(
        [float(x[-1]), label_x - 0.015 * x_span],
        [float(y[-1]), variational_label_y],
        color=variational_style.keyline,
        linewidth=0.9,
        clip_on=False,
    )
    ax.text(
        label_x,
        variational_label_y,
        f"variational weight\n{y[-1]:.3g} at final drift",
        ha="left",
        va="center",
        fontsize=9.5,
        color=variational_style.keyline,
    )
    ax.plot(
        [float(x[-1]), label_x - 0.015 * x_span],
        [float(naive_influence), float(naive_influence)],
        color=naive_style.keyline,
        linewidth=0.9,
        clip_on=False,
    )
    ax.text(
        label_x,
        float(naive_influence),
        f"naive fixed weight\n1/n = {naive_influence:.3g}",
        ha="left",
        va="center",
        fontsize=9.5,
        color=naive_style.keyline,
    )

    ax.set_xlabel(
        "Outlier drift (0 = consensus; 1 = confidently wrong)",
        labelpad=6,
        color=COLOR_DARK,
    )
    ax.set_ylabel(
        "Normalized weight of probed agent",
        labelpad=6,
        color=COLOR_DARK,
    )
    ax.set_xlim(float(np.min(x)) - 0.03 * x_span, label_end_x)
    ax.set_xticks(np.linspace(float(np.min(x)), float(np.max(x)), 6))
    # Leave room below zero so near-zero endpoint markers remain whole.
    ax.set_ylim(-0.03 * y_top, y_top)
    ax.set_title(
        "Normalized server weight along the configured drift path",
        pad=12,
        color=COLOR_DARK,
    )
    canonical = save_figure(
        fig,
        figures_dir(project_root) / filename,
        manuscript_width_fraction=0.80,
    )
    from ._presentation_diagnostics import influence_path_presentation

    influence_path_presentation(
        canonical, x, y, naive_influence,
        crossing_index=int(crossings[0]) if crossings.size else None,
    )
    return canonical


__all__ = ["generate_bounded_influence"]
