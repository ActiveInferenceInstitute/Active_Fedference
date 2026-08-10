"""Conditional-world/attack-geometry figure for the finite MED-1 grid."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from ._common import (
    COLOR_ACCENT,
    COLOR_MUTED,
    COLOR_ROBUST,
    MIN_QUANTITATIVE_FONT_SIZE,
    apply_style,
    figures_dir,
    plt,
    save_figure,
)


def generate_conditional_world(
    report: Mapping[str, object],
    *,
    project_root: Path | None = None,
    filename: str = "conditional_world.png",
) -> Path:
    """Render per-cell seed contrasts and finite-grid attack summaries."""
    raw = report.get("by_scenario")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("conditional-world report must contain by_scenario")
    cells = [cell for cell in raw.values() if isinstance(cell, Mapping)]
    if not cells:
        raise ValueError("conditional-world report has no scenario cells")
    attacks = ("clean", "confident_wrong", "permutation", "label_noise", "uniform")
    columns = ["s0_o45", "s0_o70", "s1_o45", "s1_o70"]
    heatmap = np.full((len(attacks), len(columns)), np.nan, dtype=np.float64)
    for cell in cells:
        attack = str(cell["attack"])
        key = f"s{int(cell['true_state'])}_o{int(float(cell['observability']) * 100)}"
        if attack in attacks and key in columns and float(cell["adversary_weight"]) == 1.0:
            heatmap[attacks.index(attack), columns.index(key)] = float(cell["contrast_mean"])
    attack_means: list[float] = []
    attack_min: list[float] = []
    attack_max: list[float] = []
    for attack in attacks:
        values = np.asarray(
            [float(cell["contrast_mean"]) for cell in cells if str(cell["attack"]) == attack],
            dtype=np.float64,
        )
        attack_means.append(float(values.mean()))
        attack_min.append(float(values.min()))
        attack_max.append(float(values.max()))

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.7), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=0.08, right=0.97, top=0.84, bottom=0.20, wspace=0.34)
    vmax = max(abs(float(np.nanmin(heatmap))), abs(float(np.nanmax(heatmap))), 1e-6)
    image = axes[0].imshow(heatmap, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    axes[0].set_xticks(
        range(len(columns)),
        ["true 0\nacuity .45", "true 0\nacuity .70", "true 1\nacuity .45", "true 1\nacuity .70"],
    )
    axes[0].set_yticks(range(len(attacks)), [attack.replace("_", " ") for attack in attacks])
    axes[0].set_xlabel("Declared world/observability cell")
    axes[0].set_ylabel("Attack mechanism")
    axes[0].set_title("Seed-level robust true-state-mass gain")
    for row in range(heatmap.shape[0]):
        for col in range(heatmap.shape[1]):
            value = heatmap[row, col]
            if np.isfinite(value):
                axes[0].text(col, row, f"{value:+.3f}", ha="center", va="center", fontsize=9.5)
    axes[0].axhline(-0.5, color="white", linewidth=0.8)
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04, label="naive error − robust error")

    x = np.arange(len(attacks))
    axes[1].errorbar(
        x,
        attack_means,
        yerr=np.vstack(
            (
                np.asarray(attack_means) - np.asarray(attack_min),
                np.asarray(attack_max) - np.asarray(attack_means),
            )
        ),
        fmt="o",
        color=COLOR_ROBUST,
        ecolor=COLOR_MUTED,
        capsize=4,
        linewidth=1.6,
        label="mean across finite grid cells ± min/max span",
    )
    axes[1].axhline(0.0, color=COLOR_ACCENT, linewidth=1.0, linestyle="--", label="no method contrast")
    axes[1].set_xticks(x, [attack.replace("_", "\n") for attack in attacks])
    axes[1].set_xlabel("Attack mechanism")
    axes[1].set_ylabel("True-state-mass gain")
    axes[1].set_title("Geometry-averaged conditional summary")
    axes[1].legend(fontsize=MIN_QUANTITATIVE_FONT_SIZE, loc="best")
    axes[1].text(
        0.03,
        0.56,
        "Unit: seed; agents/trials nested\nspan is grid variation, not a CI",
        transform=axes[1].transAxes,
        fontsize=MIN_QUANTITATIVE_FONT_SIZE,
        color=COLOR_MUTED,
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 2.5},
    )
    fig.suptitle(
        "Conditional robustness across hidden states, targets, acuity, and weights",
        fontweight="bold",
    )
    return save_figure(fig, figures_dir(project_root) / filename)


__all__ = ["generate_conditional_world"]
