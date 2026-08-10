"""Emergence figure: structure emerges via Bayesian model reduction.

Draws the two free-energy differences recorded by
:func:`fedference.experiments.run_emergence` (ISC-25): pruning the *redundant*
state column (the one the data never support) yields ``delta_F_redundant > 0``
— the simpler reduced model wins — while pruning a *supported* column yields
``delta_F_supported < 0`` and is rejected (Friston & Penny 2011 / Friston 2024
Eq. 13). This is a categorical source-mechanism analogue related to Friston et
al. (2024), Fig. 9, not an exact source-protocol reconstruction. Pure ``matplotlib`` (Agg);
the numbers come from the analysis workflow, this module only draws.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._common import (
    COLOR_ACCENT,
    COLOR_AXIS,
    COLOR_MUTED,
    COLOR_ROBUST,
    annotate_stats_box,
    apply_style,
    figures_dir,
    plt,
    save_figure,
)


def generate_emergence_bmr(
    delta_F_redundant: float,
    delta_F_supported: float,
    *,
    convergence: bool | None = None,
    project_root: Path | None = None,
    filename: str = "emergence_bmr.png",
) -> Path:
    """Render the redundant-vs-supported model-reduction free-energy contrast.

    Args:
        delta_F_redundant: ``delta_F`` for pruning the redundant column (>0 wins).
        delta_F_supported: ``delta_F`` for pruning a supported column (<0 rejected).
        convergence: Optional emergence verdict (``df_redundant > 0 > df_supported``).
        project_root: Project root override.
        filename: Output PNG name under ``output/figures``.

    Returns:
        Path to the written PNG.

    Raises:
        ValueError: If either ``delta_F`` value is non-finite.
    """
    df_red = float(delta_F_redundant)
    df_sup = float(delta_F_supported)
    if not (np.isfinite(df_red) and np.isfinite(df_sup)):
        raise ValueError("both delta_F values must be finite")

    apply_style()
    labels = ["prune redundant\n(should win)", "prune supported\n(should be rejected)"]
    values = [df_red, df_sup]
    colors = [COLOR_ROBUST, COLOR_MUTED]

    fig, ax = plt.subplots(figsize=(7.2, 5.3), facecolor="white")
    fig.subplots_adjust(left=0.19, right=0.97, top=0.78, bottom=0.22)
    x = np.arange(2)
    # Full-opacity bars: the positive redundant-column bar is small relative to
    # the supported-column bar and must stay clearly visible.
    bars = ax.bar(x, values, color=colors, alpha=1.0, width=0.55,
                  edgecolor=[COLOR_ACCENT, COLOR_AXIS], linewidth=1.2, zorder=3)
    ax.axhline(0.0, color=COLOR_AXIS, linewidth=1.2)

    # Head/foot room so the bar-value labels never clip the axes edges or
    # collide with the tick labels (values stay computed from the data).
    span = max(df_red, 0.0) - min(df_sup, 0.0)
    span = span if span > 0 else 1.0
    ax.set_ylim(min(df_sup, 0.0) - 0.20 * span, max(df_red, 0.0) + 0.18 * span)

    for bar, val in zip(bars, values, strict=True):
        ax.annotate(
            f"ΔF = {val:.3g} nats",
            xy=(bar.get_x() + bar.get_width() / 2.0, val),
            xytext=(0, 8 if val >= 0 else -6),
            textcoords="offset points",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=9.5,
            fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlabel("Model-reduction target", labelpad=6)
    ax.set_ylabel("model-reduction free-energy gain  $\\Delta F$ (nats)", labelpad=6)
    ax.set_title(
        "Emergence via Bayesian model reduction\n"
        "categorical protocol analogue (Friston et al., Fig. 9)",
        pad=10,
    )

    if convergence is not None:
        verdict = "emergence confirmed" if convergence else "no emergence"
        # Upper right is empty (the supported-column bar hangs below zero
        # there), so the verdict box never occludes the positive bar.
        annotate_stats_box(ax, f"Verdict: {verdict}\ndFred = {df_red:.3g} nats\ndFsup = {df_sup:.3g} nats",  # noqa: E501
                           loc="upper right")

    return save_figure(fig, figures_dir(project_root) / filename)


__all__ = ["generate_emergence_bmr"]
