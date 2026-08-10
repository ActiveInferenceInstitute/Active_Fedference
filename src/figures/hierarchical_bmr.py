"""Hierarchical Bayesian-model-reduction figure (MAJ-7).

Draws, for each non-leaf level of an N-level POMDP, the per-level Bayesian
surprise ``KL(posterior || empirical prior)`` — the information the leaf
observation added at that level — as a horizontal bar chart with the prune
threshold marked. A non-gating meta-context (its states predict identical
children) earns ~zero surprise and is flagged prunable; an informative level
earns strictly positive surprise and is kept. The numbers come from
:func:`fedference.bayesian_model_reduction.hierarchical_reduce`; this module
only draws.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ._common import (
    COLOR_AXIS,
    COLOR_MUTED,
    COLOR_NAIVE,
    COLOR_ROBUST,
    apply_style,
    figures_dir,
    plt,
    save_figure,
)


def generate_hierarchical_bmr(
    degenerate: dict[str, Any],
    informative: dict[str, Any],
    *,
    surprise_tol: float = 1e-3,
    project_root: Path | None = None,
    filename: str = "hierarchical_bmr.png",
) -> Path:
    """Render the per-level Bayesian-surprise comparison for two worlds.

    Args:
        degenerate: ``hierarchical_reduce`` output for the world whose top
            meta-context level is non-gating (redundant).
        informative: ``hierarchical_reduce`` output for the world whose top
            level is informative.
        surprise_tol: The prune threshold; drawn as a vertical reference line.
        project_root: Project root override.
        filename: Output PNG name under ``output/figures``.

    Returns:
        Path to the written PNG.

    Raises:
        ValueError: If either result reports no non-leaf levels.
    """
    deg_levels = degenerate["levels"]
    inf_levels = informative["levels"]
    if not deg_levels or not inf_levels:
        raise ValueError("both results must report at least one non-leaf level")

    apply_style()
    labels = [f"L{lv['level']}\n{lv['label']}" for lv in inf_levels]
    y = np.arange(len(labels))
    height = 0.38
    # A tiny floor so an exactly-zero surprise still shows a sliver of bar.
    deg_vals = [max(float(lv["bayesian_surprise"]), 0.0) for lv in deg_levels]
    inf_vals = [max(float(lv["bayesian_surprise"]), 0.0) for lv in inf_levels]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.barh(y + height / 2, inf_vals, height, color=COLOR_ROBUST,
            edgecolor=COLOR_AXIS, linewidth=0.8, label="informative world", zorder=3)
    ax.barh(y - height / 2, deg_vals, height, color=COLOR_MUTED,
            edgecolor=COLOR_AXIS, linewidth=0.8, label="degenerate world", zorder=3)
    ax.axvline(surprise_tol, color=COLOR_NAIVE, linestyle="--", linewidth=1.2,
               label=f"prune threshold ({surprise_tol:g})")

    for yi, lv in zip(y + height / 2, inf_levels, strict=True):
        flag = "kept" if not lv["prunable"] else "prunable"
        ax.annotate(f"{lv['bayesian_surprise']:.3g} ({flag})",
                    xy=(max(float(lv["bayesian_surprise"]), 0.0), yi),
                    xytext=(4, 0), textcoords="offset points", va="center", fontsize=9.5)
    for yi, lv in zip(y - height / 2, deg_levels, strict=True):
        flag = "prunable" if lv["prunable"] else "kept"
        ax.annotate(f"{lv['bayesian_surprise']:.2g} ({flag})",
                    xy=(max(float(lv["bayesian_surprise"]), 0.0), yi),
                    xytext=(4, 0), textcoords="offset points", va="center", fontsize=9.5)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel("Bayesian surprise  $\\mathrm{KL}(q \\,\\|\\, \\text{prior})$ (nats)", labelpad=6)
    ax.set_title("Hierarchical model reduction: which level earns its keep", pad=8)
    ax.legend(fontsize=10, loc="lower right")
    # Head-room so the annotations do not clip the right spine.
    hi = max(inf_vals + deg_vals + [surprise_tol])
    ax.set_xlim(0.0, hi * 1.45 if hi > 0 else 1.0)

    return save_figure(fig, figures_dir(project_root) / filename)


__all__ = ["generate_hierarchical_bmr"]
