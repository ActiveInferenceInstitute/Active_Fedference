"""Variational-aggregation free-energy descent figure (the rigor, shown).

Visualizes that :func:`fedference.aggregation.variational_aggregate` is genuine
block-coordinate descent on the stated objective
:func:`fedference.aggregation.aggregation_free_energy`: the recorded
``free_energy_history`` falls monotonically; a converged fixed point is
coordinatewise stationary. This is the
"show, not tell" companion to the axis-2-made-rigorous claim — the heuristic
:func:`fedference.aggregation.robust_aggregate` has no such curve because the
declared separable block-objective class is ruled out, while a broader objective
certificate is not claimed. Pure ``matplotlib`` (Agg); the history comes from
the analysis workflow, this module only draws.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._common import (
    COLOR_CORRECT,
    COLOR_DARK,
    COLOR_MUTED,
    COLOR_ROBUST,
    COLOR_VARIATE,
    annotate_stats_box,
    apply_style,
    figures_dir,
    plt,
    save_figure,
)


def generate_aggregation_descent(
    free_energy_history,
    *,
    converged: bool | None = None,
    project_root: Path | None = None,
    filename: str = "aggregation_descent.png",
) -> Path:
    """Render the variational free energy ``F`` against descent iteration.

    Args:
        free_energy_history: Sequence of ``F`` values, one per block-coordinate
            iteration (monotone non-increasing). Must be non-empty.
        converged: Whether the iteration met its tolerance (annotated if given).
        project_root: Project root override.
        filename: Output PNG name under ``output/figures``.

    Returns:
        Path to the written PNG.

    Raises:
        ValueError: If ``free_energy_history`` is empty or non-finite.
    """
    fe = np.asarray(list(free_energy_history), dtype=np.float64).ravel()
    if fe.size == 0:
        raise ValueError("free_energy_history must be non-empty")
    if not np.all(np.isfinite(fe)):
        raise ValueError("free_energy_history must be finite")

    iters = np.arange(1, fe.size + 1)
    drop = float(fe[0] - fe[-1])
    max_increase = float(np.max(np.diff(fe))) if fe.size > 1 else 0.0

    # Identify last significant step (iteration where most of the drop happened)
    diffs: np.ndarray = np.abs(np.diff(fe)) if fe.size > 1 else np.zeros(0)
    big_step_idx = int(np.argmax(diffs)) if diffs.size > 0 else 0

    apply_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    # Main descent line
    ax.plot(
        iters,
        fe,
        marker="o",
        markersize=5,
        linewidth=1.8,
        color=COLOR_ROBUST,
        label="Variational free energy $F(q, a)$",
        zorder=3,
    )

    # Highlight the converged point with COLOR_CORRECT
    ax.scatter(
        [iters[-1]],
        [fe[-1]],
        color=COLOR_CORRECT,
        s=60,
        zorder=5,
        label=f"Stationary point  $F^* = {fe[-1]:.4g}$ nats",
    )

    # Highlight the largest-descent step with COLOR_VARIATE
    if fe.size > 1:
        ax.scatter(
            [iters[big_step_idx], iters[big_step_idx + 1]],
            [fe[big_step_idx], fe[big_step_idx + 1]],
            color=COLOR_VARIATE,
            s=45,
            zorder=4,
            label=f"Largest descent step  (Δ = {float(diffs[big_step_idx]):.3g})",
        )

    # Stationary floor reference — no legend entry: the stationary-point marker
    # already carries the F* value, a second entry would duplicate it.
    ax.axhline(
        fe[-1],
        color=COLOR_MUTED,
        linestyle="--",
        linewidth=1.0,
        zorder=2,
    )

    ax.set_xlabel("Block-coordinate iteration", labelpad=6, color=COLOR_DARK)
    ax.set_ylabel("Free energy $F$  (nats)", labelpad=6, color=COLOR_DARK)
    ax.set_title(
        "Variational aggregation descends a stated objective",
        pad=8,
        color=COLOR_DARK,
    )
    if fe.size > 1:
        ax.set_xticks(iters)
        ax.set_xticklabels([str(int(i)) for i in iters])
    ax.set_ylim(top=float(fe[0]) * 1.08)

    convergence_label = ""
    if converged is True:
        convergence_label = "  ✓ converged"
    elif converged is False:
        convergence_label = "  ✗ max-iter reached"

    # Stats box (upper right). "Max ascent step" is the monotonicity witness —
    # the largest single-step *increase* in F, distinct from the largest
    # descent step highlighted on the curve.
    monotone = bool(np.all(np.diff(fe) <= 1e-10)) if fe.size > 1 else True
    stats_text = (
        f"Iterations: {fe.size}\n"
        f"Total drop ΔF = {drop:.3g} nats\n"
        f"Max ascent step = {max_increase:.1e} nats\n"
        f"Monotone: {'yes' if monotone else 'NO — violation!'}"
        + convergence_label
    )
    annotate_stats_box(ax, stats_text, loc="upper right", fontsize=10)

    # Legend at center right: clear of the stats box above and of the flat
    # converged tail of the curve along the bottom.
    ax.legend(fontsize=10, loc="center right")

    return save_figure(fig, figures_dir(project_root) / filename)


__all__ = ["generate_aggregation_descent"]
