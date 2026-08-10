"""Language-acquisition figure: a seed-aggregated categorical trajectory.

Draws the seed mean of ``KL(true A || learned A)`` from the categorical
language-learning protocol (ISC-24). The learning step is the ordered x-axis;
independent configured seeds are the replication unit for the pointwise
percentile-bootstrap interval. This is a source-mechanism analogue to the
language-acquisition estimand discussed by Friston et al. (2024), not an exact
source-protocol reconstruction. Pure ``matplotlib`` (Agg); the numbers come from
the analysis workflow, this module only draws.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ._common import (
    COLOR_ROBUST,
    annotate_stats_box,
    apply_style,
    figures_dir,
    plt,
    save_figure,
    shade_ci,
)


def generate_language_kl_decay(
    kl_trajectory: Sequence[float],
    *,
    trajectory_ci: tuple[Sequence[float], Sequence[float]] | None = None,
    monotone_decreasing: bool | None = None,
    n_seeds: int | None = None,
    project_root: Path | None = None,
    filename: str = "language_kl_decay.png",
) -> Path:
    """Render the language-acquisition KL learning curve.

    Args:
        kl_trajectory: Seed-mean ``KL(true || learned)`` per count batch, in
            order.
        trajectory_ci: Optional ``(lo, hi)`` pointwise 95% percentile-bootstrap
            intervals, with one bound per trajectory point. The bootstrap unit
            is the independent configured seed, never the ordered time points.
        monotone_decreasing: Optional monotonicity verdict, annotated on the plot.
        n_seeds: Optional number of independent seeds represented by the mean and
            interval.
        project_root: Project root override.
        filename: Output PNG name under ``output/figures``.

    Returns:
        Path to the written PNG.

    Raises:
        ValueError: If the trajectory or pointwise intervals are malformed.
    """
    kl = np.asarray(kl_trajectory, dtype=np.float64).ravel()
    if kl.size == 0 or not np.all(np.isfinite(kl)):
        raise ValueError("kl_trajectory must be non-empty and finite")
    if np.any(kl < 0.0):
        raise ValueError("KL trajectory cannot contain negative values")
    if n_seeds is not None and n_seeds < 1:
        raise ValueError("n_seeds must be positive when supplied")

    ci_lo: np.ndarray | None = None
    ci_hi: np.ndarray | None = None
    if trajectory_ci is not None:
        if len(trajectory_ci) != 2:
            raise ValueError("trajectory_ci must be a (lo, hi) pair")
        ci_lo = np.asarray(trajectory_ci[0], dtype=np.float64).ravel()
        ci_hi = np.asarray(trajectory_ci[1], dtype=np.float64).ravel()
        if ci_lo.size != kl.size or ci_hi.size != kl.size:
            raise ValueError("trajectory_ci bounds must match kl_trajectory length")
        if not (np.all(np.isfinite(ci_lo)) and np.all(np.isfinite(ci_hi))):
            raise ValueError("trajectory_ci bounds must be finite")
        if np.any(ci_lo < 0.0) or np.any(ci_hi < 0.0) or np.any(ci_lo > ci_hi):
            raise ValueError("trajectory_ci must be non-negative with lo <= hi")

    apply_style()
    steps = np.arange(kl.size)
    fig, ax = plt.subplots(figsize=(8.0, 5.2))

    if ci_lo is not None and ci_hi is not None:
        seed_label = f" (n={n_seeds} seeds)" if n_seeds is not None else ""
        shade_ci(
            ax,
            steps,
            ci_lo,
            ci_hi,
            COLOR_ROBUST,
            alpha=0.16,
        )
        ax.plot([], [], color=COLOR_ROBUST, alpha=0.42, linewidth=7.0,
                label=f"95% seed bootstrap CI{seed_label}")

    ax.plot(
        steps,
        kl,
        marker="o",
        markersize=4,
        linewidth=2.0,
        color=COLOR_ROBUST,
        label="seed mean KL(true ‖ learned)",
    )
    ax.set_xlabel("learning step (Dirichlet count batch)")
    ax.set_ylabel("KL divergence (nats)")
    ax.set_ylim(bottom=0.0)
    relation = "source-mechanism analogue to Friston et al. (2024), Fig. 7"
    ax.set_title(f"Categorical language acquisition\n({relation})", pad=12)

    if monotone_decreasing is not None:
        verdict = "monotone decreasing" if monotone_decreasing else "non-monotone"
        seed_text = f"\nSeeds: {n_seeds}" if n_seeds is not None else ""
        annotate_stats_box(
            ax,
            f"Verdict: {verdict}{seed_text}\n"
            f"Initial mean: {kl[0]:.3g} nats\n"
            f"Final mean: {kl[-1]:.3g} nats",
            loc="upper right",
            fontsize=10,
        )
    ax.legend(fontsize=10, loc="center right")

    return save_figure(fig, figures_dir(project_root) / filename)


__all__ = ["generate_language_kl_decay"]
