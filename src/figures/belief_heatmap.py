"""Belief-heatmap figure: a sentinel colony's posteriors over the shared grid.

Visualises the per-agent posterior beliefs over the creature's location on the
3x3 sentinel grid of Friston et al. (2024), *Federated inference and belief
sharing* (Neurosci. Biobehav. Rev. 156:105500), Fig. 1/4. Each row is one
agent's categorical pmf over the ``n_locations`` cells; the bottom row is the
project log-linear consensus (the documented categorical Eq. 7
specialization, not a complete source protocol). Pure ``matplotlib`` (Agg);
the numbers are supplied by the analysis workflow, this module only draws.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from ._common import (
    COLOR_ACCENT,
    annotate_stats_box,
    apply_style,
    figures_dir,
    plt,
    save_figure,
)

ArrayF = np.ndarray


def generate_belief_heatmap(
    local_posteriors: ArrayF | None = None,
    consensus: ArrayF | None = None,
    *,
    project_root: Path | None = None,
    filename: str = "belief_heatmap.png",
    **legacy: object,
) -> Path:
    """Render a colony's per-agent beliefs plus consensus as a heatmap.

    Args:
        local_posteriors: ``(n_agents, n_locations)`` array of categorical pmfs.
        consensus: length-``n_locations`` fused consensus pmf.
        project_root: Project root override (defaults to the repo root).
        filename: Output PNG name under ``output/figures``.

    Returns:
        Path to the written PNG.

    Raises:
        ValueError: If ``local_posteriors`` is not 2-D or ``consensus`` length
            mismatches the belief width.
    """
    if "agent_beliefs" in legacy:
        if local_posteriors is not None:
            raise TypeError(
                "local_posteriors and deprecated agent_beliefs cannot both be supplied"
            )
        local_posteriors = legacy.pop("agent_beliefs")  # type: ignore[assignment]
        warnings.warn(
            "agent_beliefs is deprecated; use local_posteriors",
            DeprecationWarning,
            stacklevel=2,
        )
    if legacy:
        names = ", ".join(sorted(legacy))
        raise TypeError(f"unexpected keyword argument(s): {names}")
    if local_posteriors is None or consensus is None:
        raise TypeError("local_posteriors and consensus are required")
    local_posteriors_matrix = np.asarray(local_posteriors, dtype=np.float64)
    cons = np.asarray(consensus, dtype=np.float64).ravel()
    if local_posteriors_matrix.ndim != 2:
        raise ValueError("local_posteriors must be a 2-D (n_agents, n_locations) array")
    if cons.shape[0] != local_posteriors_matrix.shape[1]:
        raise ValueError("consensus length must match the number of locations")

    apply_style()
    matrix = np.vstack([local_posteriors_matrix, cons])
    n_rows, n_cols = matrix.shape

    fig, ax = plt.subplots(figsize=(max(5.5, 0.75 * n_cols), max(3.0, 0.55 * n_rows)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    # Heatmaps look cleaner without the global axes grid lines.
    ax.grid(False)
    ax.set_xlabel("creature location (grid cell index)", labelpad=6)
    ax.set_ylabel("agent / consensus row", labelpad=6)
    ax.set_title("Sentinel colony beliefs over shared location", pad=8)

    labels = [
        f"agent {n}" for n in range(local_posteriors_matrix.shape[0])
    ] + ["consensus"]
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(labels)
    ax.set_xticks(range(n_cols))
    # Annotate the dominant cell in each row so the peak location is legible.
    for row in range(n_rows):
        peak = int(np.argmax(matrix[row]))
        val = matrix[row, peak]
        ax.text(
            peak,
            row,
            f"{val:.2f}",
            ha="center",
            va="center",
            fontsize=9.5,
            color="white" if val < 0.6 else "black",
            fontweight="bold",
        )
    # Separate the consensus row with a divider line.
    ax.axhline(
        local_posteriors_matrix.shape[0] - 0.5,
        color=COLOR_ACCENT,
        linewidth=2.0,
    )
    cbar = fig.colorbar(im, ax=ax, label="posterior probability mass")
    cbar.ax.tick_params(labelsize=9.5)
    annotate_stats_box(ax, f"{n_rows - 1} agents\n+ 1 consensus row", loc="lower right")

    return save_figure(fig, figures_dir(project_root) / filename)


__all__ = ["generate_belief_heatmap"]
