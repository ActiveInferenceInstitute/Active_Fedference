"""Sensitivity heatmap: signed accuracy contrasts by acuity and colony size.

Two vertically stacked panels show the accuracy gap
(hierarchical/communicating minus flat/isolated) as a 2-D color-annotated
heatmap over sensor acuity (y-axis) x colony size (x-axis). The upper panel is
communicating minus isolated accuracy; the lower panel is hierarchical minus
flat location accuracy.

Headless (Agg) matplotlib only; no infrastructure imports (layer contract).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from matplotlib.colors import to_hex

from experiment_config import (
    DEFAULT_SENSITIVITY_ACUITY,
    DEFAULT_SENSITIVITY_COLONY_SIZES,
    SENSITIVITY_NOISE_FLOOR,
)
from figures._common import (
    COLOR_GRID,
    apply_style,
    contrasting_text_color,
    figures_dir,
    plt,
    save_figure,
    semantic_style,
    signed_difference_colormap,
)


def generate_sensitivity_heatmap(
    project_root: str | Path | None = None,
    *,
    report: Mapping[str, object] | None = None,
    seed: int = 0,
    acuity_values: tuple[float, ...] = DEFAULT_SENSITIVITY_ACUITY,
    n_agents_values: tuple[int, ...] = DEFAULT_SENSITIVITY_COLONY_SIZES,
    n_trials: int = 20,
    filename: str = "sensitivity_heatmap.png",
) -> Path:
    """Render two signed accuracy-contrast heatmaps over acuity and colony size.

    The upper panel shows the belief-sharing federation gap (communicating
    minus isolated mean accuracy), and the lower panel shows the hierarchical
    POMDP location accuracy gap (hierarchical minus flat). Both are computed by
    the respective sensitivity sweep functions in
    :mod:`fedference.experiments`.

    Args:
        project_root: Project root directory; defaults to the project root
            inferred from this file's location.
        seed: RNG seed forwarded to the sweep functions.
        acuity_values: Sensor acuity levels (y-axis, top to bottom).
        n_agents_values: Colony sizes (x-axis, left to right).
        n_trials: Trials per cell inside each sweep call.
        filename: Output filename under ``output/figures/``.

    Returns:
        Path to the written PNG file.
    """
    apply_style()

    if report is None:
        from fedference.experiments import (
            run_belief_sharing_sensitivity,
            run_hierarchical_sensitivity,
        )

        bs_result: Mapping[str, object] = run_belief_sharing_sensitivity(
            seed,
            acuity_values=acuity_values,
            n_agents_values=n_agents_values,
            n_trials=n_trials,
        )
        hi_result: Mapping[str, object] = run_hierarchical_sensitivity(
            seed,
            acuity_values=acuity_values,
            n_agents_values=n_agents_values,
            n_trials=n_trials,
        )
    else:
        raw_bs = report.get("belief_sharing")
        raw_hi = report.get("hierarchical")
        raw_acuity = report.get("acuity_values")
        raw_agents = report.get("n_agents_values")
        if not isinstance(raw_bs, Mapping) or not isinstance(raw_hi, Mapping):
            raise ValueError("sensitivity report must contain belief_sharing and hierarchical mappings")
        if not isinstance(raw_acuity, list) or not isinstance(raw_agents, list):
            raise ValueError("sensitivity report must contain acuity_values and n_agents_values lists")
        bs_result = raw_bs
        hi_result = raw_hi
        acuity_values = tuple(float(value) for value in raw_acuity)
        n_agents_values = tuple(int(value) for value in raw_agents)
        raw_noise_floor = report.get("noise_floor")
        if isinstance(raw_noise_floor, bool) or not isinstance(raw_noise_floor, (int, float)):
            raise ValueError("sensitivity report must contain a numeric noise_floor")
        noise_floor = float(raw_noise_floor)

    bs_gap = np.array(bs_result["accuracy_gap_grid"], dtype=np.float64)
    hi_gap = np.array(hi_result["accuracy_gap_grid"], dtype=np.float64)
    acuity_labels = [f"{a:.2f}" for a in acuity_values]
    n_agents_labels = [str(n) for n in n_agents_values]

    # Symmetric signed map centred on zero. Its blue/near-white/brown endpoints
    # remain interpretable under common colour-vision deficiencies and in
    # grayscale; every cell also carries an exact signed value.
    vmax = float(max(np.abs(bs_gap).max(), np.abs(hi_gap).max(), 1e-6))
    vmin = -vmax

    # Stack the two maps so every five-column cell retains enough horizontal
    # room for a signed value at final page scale.  The earlier side-by-side
    # layout made adjacent negative labels read as one concatenated number.
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 9.5))
    fig.set_layout_engine("none")
    fig.subplots_adjust(left=0.14, right=0.86, top=0.91, bottom=0.17, hspace=0.80)

    # Hatching denotes a configured display band, not a claim of no effect.
    if report is None:
        noise_floor = SENSITIVITY_NOISE_FLOOR

    for ax, data, title in zip(
        axes,
        [bs_gap, hi_gap],
        ["Belief sharing\n(communicating − isolated)", "Hierarchical POMDP\n(hierarchical − flat)"],
    ):
        im = ax.imshow(
            data,
            cmap=signed_difference_colormap(),
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            origin="upper",
        )
        # Annotate cells; hatching denotes only the declared display band.
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                cell_background = to_hex(im.cmap(im.norm(val)), keep_alpha=False)
                text_color = contrasting_text_color(cell_background)
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color=text_color,
                    bbox={"facecolor": cell_background, "edgecolor": "none", "pad": 0.0},
                )
                if abs(val) <= noise_floor:
                    band_style = semantic_style("display_band")
                    rect = plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1.0,
                        1.0,
                        fill=False,
                        hatch=band_style.hatch,
                        edgecolor=band_style.keyline,
                        linewidth=0,
                        alpha=0.4,
                    )
                    ax.add_patch(rect)
        ax.set_xticks(range(len(n_agents_labels)))
        ax.set_xticklabels(n_agents_labels)
        ax.set_yticks(range(len(acuity_labels)))
        ax.set_yticklabels(acuity_labels)
        # Visible cell gutters prevent adjacent signed values from merging,
        # and remain useful when the colour map is viewed in grayscale.
        ax.set_xticks(np.arange(-0.5, data.shape[1], 1.0), minor=True)
        ax.set_yticks(np.arange(-0.5, data.shape[0], 1.0), minor=True)
        ax.grid(which="minor", color=COLOR_GRID, linewidth=0.8, alpha=0.72)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_xlabel("Colony size (n_agents)", labelpad=6)
        ax.set_ylabel("Sensor acuity", labelpad=6)
        ax.set_title(title, pad=8)
        max_gap = float(np.abs(data).max())
        pos_cells = int((data > noise_floor).sum())
        total_cells = data.size
        fig.colorbar(im, ax=ax, label="Accuracy difference")
        # Below the axes (under the x-label), fully outside the heatmap so the
        # box cannot cover the bottom-row cell value annotations.
        ax.text(
            0.5,
            -0.22,
            f"max |gap| = {max_gap:.3f}   ·   positive cells = {pos_cells}/{total_cells}",  # noqa: E501
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": COLOR_GRID, "alpha": 0.85},
        )

    # Legend note for hatching.
    fig.text(
        0.5,
        0.018,
        f"Hatching marks the declared display band |gap| ≤ {noise_floor:.2f}.\n"
        "It is not a CI, significance test, unreliability flag, or proof of zero effect.",
        ha="center",
        va="bottom",
        fontsize=10,
        linespacing=1.3,
        color=COLOR_GRID,
    )

    fig.suptitle(
        "Study 8 — Signed accuracy gaps across configured sensitivity cells",
        fontsize=12,
        y=0.985,
    )
    out = figures_dir(Path(project_root) if project_root is not None else None)
    path = save_figure(
        fig,
        out / filename,
        manuscript_width_fraction=0.90,
    )
    from ._presentation_estimates import sensitivity_presentation

    sensitivity_presentation(path, bs_gap, hi_gap, acuity_labels, n_agents_labels, noise_floor)
    return path


__all__ = ["generate_sensitivity_heatmap"]


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent.parent
    out = generate_sensitivity_heatmap(root)
    print(out)
