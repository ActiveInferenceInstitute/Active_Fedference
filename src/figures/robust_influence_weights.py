"""Robust-influence-weights figure: the heuristic down-weights saboteurs.

Draws the per-agent influence weights returned by a robust pooling round
(:func:`fedference.belief_sharing.share_round` with ``method="robust"`` ->
``SharingDiagnostics.normalized_effective_weights``; the old ``agent_weights``
property is a warned compatibility adapter). The contaminated (saboteur) agents are
highlighted: a working robust aggregator assigns them visibly *lower* influence
than the healthy majority, which is how the server-side consensus stays near the
truth under contamination.

**Three-robustness-axis honesty (HARD).** These weights belong to the *server-side*
``robust_aggregate`` divergence-reweighting heuristic. Its positive formal property
is the naive-recovery limit at robustness zero. A scoped no-go rejects one
declared separable objective class but does not certify another. This figure
makes NO claim that the down-weighting inherits the
per-agent beta/rcce FedGVI generalized-Bayes guarantees — those live in the
agent update, not in this pooling weight. The title and caption say "heuristic".
Pure ``matplotlib`` (Agg); the weights come from the analysis workflow.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ._common import (
    COLOR_ADVERSARY,
    COLOR_DARK,
    COLOR_MUTED,
    COLOR_ROBUST,
    annotate_stats_box,
    apply_style,
    figures_dir,
    plt,
    save_figure,
)


def generate_robust_influence_weights(
    normalized_effective_weights: Sequence[float] | None = None,
    contaminated_indices: Sequence[int] | None = None,
    *,
    project_root: Path | None = None,
    filename: str = "robust_influence_weights.png",
    **legacy: object,
) -> Path:
    """Render per-agent robust influence weights with saboteurs highlighted.

    Args:
        normalized_effective_weights: Length-n_agents normalized influence
            weights returned by the server.
        contaminated_indices: Indices of the contaminated (saboteur) agents.
        project_root: Project root override.
        filename: Output PNG name under ``output/figures``.

    Returns:
        Path to the written PNG.

    Raises:
        ValueError: If the influence vector is empty or an index is out of range.
    """
    if "agent_weights" in legacy:
        if normalized_effective_weights is not None:
            raise TypeError(
                "normalized_effective_weights and deprecated agent_weights "
                "cannot both be supplied"
            )
        normalized_effective_weights = legacy.pop("agent_weights")  # type: ignore[assignment]
        warnings.warn(
            "agent_weights is deprecated; use normalized_effective_weights",
            DeprecationWarning,
            stacklevel=2,
        )
    if legacy:
        names = ", ".join(sorted(legacy))
        raise TypeError(f"unexpected keyword argument(s): {names}")
    if normalized_effective_weights is None or contaminated_indices is None:
        raise TypeError(
            "normalized_effective_weights and contaminated_indices are required"
        )
    normalized_effective_weights_array = np.asarray(
        normalized_effective_weights, dtype=np.float64
    ).ravel()
    if normalized_effective_weights_array.size == 0:
        raise ValueError("normalized_effective_weights must be non-empty")
    contaminated = {int(i) for i in contaminated_indices}
    if any(
        not 0 <= i < normalized_effective_weights_array.size
        for i in contaminated
    ):
        raise ValueError(
            "contaminated_indices out of range for normalized_effective_weights"
        )

    apply_style()
    fig, ax = plt.subplots(figsize=(9.2, 5.2), facecolor="white")
    fig.subplots_adjust(left=0.12, right=0.98, top=0.82, bottom=0.24)
    x = np.arange(normalized_effective_weights_array.size)

    # Semantic palette: healthy agents in COLOR_ROBUST (sky blue), contaminated
    # saboteurs in COLOR_ADVERSARY (red). No special-casing of the top healthy
    # weight — with a tied healthy majority it would recolor every healthy bar.
    colors = [
        COLOR_ADVERSARY if i in contaminated else COLOR_ROBUST
        for i in range(normalized_effective_weights_array.size)
    ]

    bars = ax.bar(
        x,
        normalized_effective_weights_array,
        color=colors,
        alpha=0.88,
        edgecolor="white",
        linewidth=0.8,
    )

    equal_weight = 1.0 / normalized_effective_weights_array.size
    ax.axhline(
        equal_weight,
        color=COLOR_MUTED,
        linestyle=":",
        linewidth=1.4,
        label=f"equal-weight pool  (1/n = {equal_weight:.3g})",
        zorder=2,
    )

    # Highlight the gap: shade region where contaminated agents sit below equal weight
    for i in contaminated:
        if normalized_effective_weights_array[i] < equal_weight:
            # Offset to the right of the bar centre so the arrow does not
            # cross the bold value label printed above the bar.
            ax.annotate(
                "",
                xy=(
                    float(x[i]) + 0.28,
                    float(normalized_effective_weights_array[i]) + 0.002,
                ),
                xytext=(float(x[i]) + 0.28, equal_weight - 0.002),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": COLOR_ADVERSARY,
                    "lw": 1.2,
                },
            )

    for bar, weight in zip(bars, normalized_effective_weights_array):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            weight
            + max(normalized_effective_weights_array.max() * 0.025, 0.003),
            f"{weight:.3f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color=COLOR_DARK,
            fontweight="bold",
        )
    if contaminated:
        first = min(contaminated)
        last = max(contaminated)
        ax.axvspan(first - 0.45, last + 0.45, color=COLOR_ADVERSARY, alpha=0.07, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"a{i}\nadversary" if i in contaminated else f"a{i}\nhonest"
            for i in range(normalized_effective_weights_array.size)
        ],
        fontsize=9.5,
    )
    ax.set_xlabel("Agent  (role shown below label)", labelpad=6, color=COLOR_DARK)
    ax.set_ylabel("Normalized server weight", labelpad=6, color=COLOR_DARK)
    ax.set_title(
        "Heuristic server weights suppress adversarial broadcasts",
        pad=8,
        color=COLOR_DARK,
    )

    # Legend via proxy handles (must match exactly what is drawn)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(color=COLOR_ROBUST, alpha=0.85, label="Healthy agent"),
        Patch(color=COLOR_ADVERSARY, alpha=0.85, label="Contaminated agent"),
        Line2D(
            [0], [0], color=COLOR_MUTED, linestyle=":", linewidth=1.4,
            label=f"equal-weight pool (1/n = {equal_weight:.3g})",
        ),
    ]
    # Extra headroom so the legend clears the value labels above the tall bars.
    ax.set_ylim(
        0.0,
        max(normalized_effective_weights_array.max(), equal_weight) * 1.55,
    )
    ax.legend(handles=legend_handles, fontsize=9.5, loc="upper right")

    # Stats box: contamination fraction, weight gap
    n_contam = len(contaminated)
    n_total = normalized_effective_weights_array.size
    contam_pct = 100.0 * n_contam / n_total if n_total else 0.0
    healthy_mean = (
        float(
            np.mean(
                [
                    normalized_effective_weights_array[i]
                    for i in range(n_total)
                    if i not in contaminated
                ]
            )
        )
        if n_total > n_contam
        else 0.0
    )
    contam_mean = (
        float(np.mean([normalized_effective_weights_array[i] for i in contaminated]))
        if contaminated
        else 0.0
    )
    weight_gap = healthy_mean - contam_mean

    stats_text = (
        f"Agents contaminated: {n_contam}/{n_total} ({contam_pct:.0f}%)\n"
        f"Weight sum: {float(normalized_effective_weights_array.sum()):.3f}\n"
        f"Healthy mean weight: {healthy_mean:.3f}\n"
        f"Adversary mean weight: {contam_mean:.3f}\n"
        f"Suppression gap: {weight_gap:.3f}"
    )
    annotate_stats_box(ax, stats_text, loc="upper left", fontsize=10)

    return save_figure(fig, figures_dir(project_root) / filename)


__all__ = ["generate_robust_influence_weights"]
