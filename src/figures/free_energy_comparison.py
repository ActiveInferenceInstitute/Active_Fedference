"""Free-energy comparison figure: communicating vs incommunicado colonies.

Renders a categorical source-mechanism analogue to Friston et al. (2024),
*Federated inference and belief sharing* (Neurosci. Biobehav. Rev. 156:105500),
Fig. 5 — "two heads are better than one" under the declared reduced protocol.
A colony that *communicates* (federated
belief sharing through the project's categorical Eq. 7 specialization) carries
a lower mean variational free energy than the same colony held
incommunicado. This does not reconstruct the complete source protocol. This
module draws the paired free-energy bars (and an optional per-seed scatter);
the values come from
:func:`fedference.experiments.run_belief_sharing` via the analysis workflow.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ._common import (
    COLOR_AXIS,
    COLOR_GRID,
    COLOR_MUTED,
    COLOR_NAIVE,
    COLOR_ROBUST,
    annotate_stats_box,
    apply_style,
    figures_dir,
    plt,
    save_figure,
)


def generate_free_energy_comparison(
    incommunicado: Sequence[float],
    communicating: Sequence[float],
    *,
    project_root: Path | None = None,
    filename: str = "free_energy_comparison.png",
) -> Path:
    """Render the free-energy gap between incommunicado and communicating colonies.

    Args:
        incommunicado: Per-seed mean free energy with no belief sharing.
        communicating: Per-seed mean free energy with federated sharing.
        project_root: Project root override.
        filename: Output PNG name under ``output/figures``.

    Returns:
        Path to the written PNG.

    Raises:
        ValueError: If the two sequences are empty or differ in length.
    """
    incom = np.asarray(incommunicado, dtype=np.float64).ravel()
    comm = np.asarray(communicating, dtype=np.float64).ravel()
    if incom.size == 0 or comm.size == 0:
        raise ValueError("both free-energy sequences must be non-empty")
    if incom.shape != comm.shape:
        raise ValueError("incommunicado and communicating must have equal length")

    apply_style()
    means = [float(incom.mean()), float(comm.mean())]
    errs = [float(incom.std(ddof=0)), float(comm.std(ddof=0))]
    labels = ["incommunicado", "communicating"]
    colors = [COLOR_NAIVE, COLOR_ROBUST]

    fig, ax = plt.subplots(figsize=(7.2, 5.4), facecolor="white")
    fig.subplots_adjust(left=0.19, right=0.97, top=0.78, bottom=0.18)
    x = np.arange(2)
    bars = ax.bar(x, means, yerr=errs, color=colors, capsize=6, alpha=0.85, width=0.55,
                  error_kw={"elinewidth": 1.4, "ecolor": COLOR_AXIS})
    # Overlay the per-seed points so the within-colony gap is visible.
    rng = np.random.default_rng(0)
    for i, data in enumerate((incom, comm)):
        jitter = rng.uniform(-0.08, 0.08, size=data.shape[0])
        ax.scatter(
            np.full_like(data, i) + jitter,
            data,
            color=COLOR_MUTED,
            s=22,
            zorder=3,
            alpha=0.75,
            label="per-seed value" if i == 0 else None,
        )
    # Value labels above each bar.
    for bar, mean in zip(bars, means, strict=True):
        ax.annotate(
            f"{mean:.3g} nats",
            xy=(bar.get_x() + bar.get_width() / 2.0, mean),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9.5,
            fontweight="bold",
        )
    gap = means[0] - means[1]
    ax.annotate(
        rf"$\Delta F$ = {gap:.3g} nats",
        xy=(0.5, 0.93),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": COLOR_GRID, "alpha": 0.85},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlabel("Colony communication condition", labelpad=6)
    ax.set_ylabel("mean variational free energy (nats)", labelpad=6)
    ax.set_title(
        "Belief sharing lowers colony free energy\n"
        "categorical protocol analogue (Friston et al., Fig. 5)",
        pad=10,
    )
    ax.legend(fontsize=9.5, loc="upper right")
    annotate_stats_box(ax, f"lower F = better\nn = {len(incom)} seeds",
                       loc="lower right")

    return save_figure(fig, figures_dir(project_root) / filename)


__all__ = ["generate_free_energy_comparison"]
