"""Conditional-world/attack-geometry figure for the finite MED-1 grid."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

import numpy as np
from matplotlib.figure import Figure

from ._common import (
    COLOR_ACCENT,
    COLOR_MUTED,
    COLOR_ROBUST,
    MIN_QUANTITATIVE_FONT_SIZE,
    apply_style,
    figures_dir,
    plt,
    save_figure,
    semantic_style,
)

_ATTACKS = ("clean", "confident_wrong", "permutation", "label_noise", "uniform")
_COLUMNS = ["s0_o45", "s0_o70", "s1_o45", "s1_o70"]


class _ConditionalData(NamedTuple):
    heatmap: np.ndarray
    means: list[float]
    minima: list[float]
    maxima: list[float]


def generate_conditional_world(
    report: Mapping[str, object],
    *,
    project_root: Path | None = None,
    filename: str = "conditional_world.png",
) -> Path:
    """Render per-cell seed contrasts and finite-grid attack summaries."""
    data = _conditional_data(report)
    fig = _build_conditional_world(report, data=data)
    path = save_figure(fig, figures_dir(project_root) / filename)
    from ._presentation_estimates import conditional_presentation

    conditional_presentation(path, data.heatmap, data.means, data.minima, data.maxima, _ATTACKS, _COLUMNS)
    return path


def _conditional_data(report: Mapping[str, object]) -> _ConditionalData:
    """Derive the finite-grid display summaries once for both figure surfaces."""
    raw = report.get("by_scenario")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("conditional-world report must contain by_scenario")
    cells = [cell for cell in raw.values() if isinstance(cell, Mapping)]
    if not cells:
        raise ValueError("conditional-world report has no scenario cells")
    attacks, columns = _ATTACKS, _COLUMNS
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

    return _ConditionalData(heatmap, attack_means, attack_min, attack_max)


def _build_conditional_world(report: Mapping[str, object], *, data: _ConditionalData | None = None) -> Figure:
    """Compose the actual artists independently of the explicit file boundary."""
    heatmap, attack_means, attack_min, attack_max = data if data is not None else _conditional_data(report)
    attacks, columns = _ATTACKS, _COLUMNS
    apply_style()
    # The manuscript uses a 95%-width embed.  A vertical composition preserves
    # readable cell values and attack labels at that scale; the former 1x2
    # layout forced both panel headings and categorical ticks to collide.
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.0, 7.8),
        gridspec_kw={"height_ratios": [1.18, 1.0]},
    )
    fig.subplots_adjust(left=0.17, right=0.91, top=0.89, bottom=0.09, hspace=0.52)
    vmax = max(abs(float(np.nanmin(heatmap))), abs(float(np.nanmax(heatmap))), 1e-6)
    image = axes[0].imshow(heatmap, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    axes[0].set_xticks(range(len(columns)), ["s0 · .45", "s0 · .70", "s1 · .45", "s1 · .70"])
    axes[0].set_yticks(range(len(attacks)), [attack.replace("_", " ") for attack in attacks])
    axes[0].set_xlabel("World cell: true state s · observability")
    axes[0].set_ylabel("Attack mechanism")
    axes[0].set_title("A  Seed-level true-state-mass contrast", loc="left", pad=9)
    for row in range(heatmap.shape[0]):
        for col in range(heatmap.shape[1]):
            value = heatmap[row, col]
            if np.isfinite(value):
                axes[0].text(
                    col,
                    row,
                    f"{value:+.3f}",
                    ha="center",
                    va="center",
                    fontsize=10.5,
                    color=COLOR_ACCENT,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6},
                )
    axes[0].axhline(-0.5, color="white", linewidth=0.8)
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04, label="naive error − robust error")

    y = np.arange(len(attacks))
    axes[1].errorbar(
        attack_means,
        y,
        xerr=np.vstack(
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
        label="mean with asymmetric capped min–max range",
    )
    reference_rule = semantic_style("reference_rule")
    axes[1].axvline(
        0.0,
        color=reference_rule.color,
        linewidth=reference_rule.linewidth,
        linestyle=reference_rule.dash,
        label="zero: no method contrast",
    )
    axes[1].set_yticks(y, [attack.replace("_", " ") for attack in attacks])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("True-state-mass contrast: naive error − robust error")
    axes[1].set_ylabel("Attack mechanism")
    axes[1].set_title("B  Finite-grid means and capped min–max ranges", loc="left", pad=9)
    axes[1].legend(fontsize=MIN_QUANTITATIVE_FONT_SIZE, loc="lower left")
    fig.suptitle(
        "Conditional robustness on the declared finite grid",
        fontweight="bold",
    )
    return fig


__all__ = ["generate_conditional_world"]
