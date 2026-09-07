"""Contamination-gallery figure: robust vs naive across attack mechanisms.

Grouped bars of naive (standard log-linear pool) vs the pooled display robust
member's consensus
accuracy under each contamination mechanism from
:func:`fedference.experiments.run_contamination_gallery`. Directional attacks
(confident-wrong / byzantine / drift, which pull the consensus toward a wrong
state) and entropy attacks (uniform / label-noise) are both retained so the
configured report can show wins, near-ties, and reversals rather than assuming
an ordering. Pure ``matplotlib`` (Agg); the gallery dict comes from the analysis
workflow, and this module only draws.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._common import (
    COLOR_ACCENT,
    COLOR_MUTED,
    COLOR_NAIVE_LIGHT,
    apply_style,
    figures_dir,
    plt,
    save_figure,
    semantic_style,
)


def _robust_facecolors(reliable: list[bool], full_color: str) -> list[str]:
    """Encode the declared display flag truthfully in robust-bar fills."""
    return [full_color if is_flagged else COLOR_NAIVE_LIGHT for is_flagged in reliable]


def _selected_preset_annotation(method: str, win_fraction: float, reliable: bool) -> str:
    """Return a compact direct label for one selected-preset bar.

    The legend owns the repeated ``selected server preset`` role name.  Each
    category label therefore carries only the method, descriptive win
    fraction, and display disposition; this keeps five adjacent labels
    independently readable at manuscript scale.
    """
    mark = "flagged" if reliable else "below bar"
    return f"{method} · win {win_fraction:.2f}\n{mark}"


def generate_contamination_gallery(
    by_kind: dict,
    *,
    project_root: Path | None = None,
    filename: str = "contamination_gallery.png",
) -> Path:
    """Render naive vs pooled display robust accuracy per contamination mechanism.

    Args:
        by_kind: Mapping ``{kind: {naive_mean, naive_ci, robust_mean,
            robust_ci, win_fraction, reliably_beats, ...}}`` from
            :func:`fedference.experiments.run_contamination_gallery`.
            The source report supplies 95% seed-bootstrap intervals for both
            bars; older report-shaped mappings without those optional fields
            remain renderable without error bars. Mechanisms use lexical key
            order in both canonical and presentation assets, independent of
            mapping insertion order or JSON serialization.
        project_root: Project root override.
        filename: Output PNG name under ``output/figures``.

    Returns:
        Path to the written PNG.

    Raises:
        ValueError: If ``by_kind`` is empty or a cell lacks the required means.
    """
    if not by_kind:
        raise ValueError("by_kind must be non-empty")
    by_kind = dict(sorted(by_kind.items()))
    kinds = list(by_kind)
    naive = np.array([float(by_kind[k]["naive_mean"]) for k in kinds])
    robust = np.array([float(by_kind[k]["robust_mean"]) for k in kinds])
    reliable = [bool(by_kind[k].get("reliably_beats", False)) for k in kinds]
    wins = [float(by_kind[k].get("win_fraction", float("nan"))) for k in kinds]
    methods = [str(by_kind[k].get("best_robust_method", "robust")) for k in kinds]
    have_ci = all("naive_ci" in by_kind[k] and "robust_ci" in by_kind[k] for k in kinds)
    if have_ci:
        naive_ci = np.asarray([by_kind[k]["naive_ci"] for k in kinds], dtype=np.float64)
        robust_ci = np.asarray([by_kind[k]["robust_ci"] for k in kinds], dtype=np.float64)
        naive_yerr = np.vstack((naive - naive_ci[:, 0], naive_ci[:, 1] - naive))
        robust_yerr = np.vstack((robust - robust_ci[:, 0], robust_ci[:, 1] - robust))

    apply_style()
    fig, ax = plt.subplots(figsize=(7.4, 6.9))
    fig.set_layout_engine("none")
    fig.subplots_adjust(left=0.11, right=0.98, top=0.76, bottom=0.28)
    x = np.arange(len(kinds))
    w = 0.36
    naive_style = semantic_style("naive")
    robust_style = semantic_style("heuristic_robust")
    ax.bar(
        x - w / 2,
        naive,
        w,
        color="white",
        alpha=0.85,
        edgecolor=naive_style.keyline,
        linewidth=1.2,
        hatch=naive_style.hatch,
        label="reference log pool",
    )
    robust_bars = ax.bar(
        x + w / 2,
        robust,
        w,
        color=_robust_facecolors(reliable, robust_style.color),
        alpha=1.0,
        edgecolor=robust_style.keyline,
        linewidth=1.2,
        label="selected server preset",
    )
    for patch, is_flagged in zip(robust_bars.patches, reliable, strict=True):
        patch.set_hatch(robust_style.hatch if is_flagged else "..")
        patch.set_linewidth(1.6 if is_flagged else 1.0)
    if have_ci:
        ax.errorbar(
            x - w / 2,
            naive,
            yerr=naive_yerr,
            fmt="none",
            ecolor=COLOR_MUTED,
            elinewidth=1.0,
            capsize=3,
            zorder=4,
            label="95 % seed bootstrap interval",
        )
        ax.errorbar(
            x + w / 2,
            robust,
            yerr=robust_yerr,
            fmt="none",
            ecolor=COLOR_MUTED,
            elinewidth=1.0,
            capsize=3,
            zorder=4,
        )
    # Repeat the reference role inside every open bar so neither role identity
    # depends on hue or a remote legend. Reserve one aligned, compact direct-
    # label lane above the bars.  The shared role name remains in the legend;
    # repeating it five times here would turn the labels into one unreadable
    # text band at final manuscript scale.
    for i, height in enumerate(naive):
        ax.text(
            i - w / 2,
            min(0.08, max(0.025, float(height) * 0.08)),
            "reference\nlog pool",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=9.5,
            color=naive_style.keyline,
            fontweight="bold",
            zorder=5,
        )
    annotation_y = 1.12
    for i, (wf, rel, method) in enumerate(zip(wins, reliable, methods)):
        top = max(naive[i], robust[i])
        if have_ci:
            top = max(top, float(naive_ci[i, 1]), float(robust_ci[i, 1]))
        ax.annotate(
            _selected_preset_annotation(method, wf, rel),
            xy=(i + w / 2, top + 0.012),
            xytext=(i + w / 2, annotation_y),
            arrowprops={"arrowstyle": "-", "color": robust_style.keyline, "lw": 0.9},
            ha="center",
            va="bottom",
            fontsize=9.5,
            color=robust_style.keyline,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("_", "\n") for k in kinds], fontsize=10)
    ax.set_xlabel("Contamination mechanism", labelpad=6)
    ax.set_ylabel("Mean consensus accuracy $q(\\mathrm{true})$", labelpad=6)
    ax.set_ylim(0.0, 1.38)
    fig.suptitle(
        "Server-preset accuracy across contamination mechanisms",
        y=0.965,
        fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2)
    n_reliable = sum(reliable)
    fig.text(
        0.5,
        0.865,
        f"Display flags = {n_reliable}/{len(kinds)} · annotations are descriptive "
        "win fractions, not p-values",
        ha="center",
        va="center",
        fontsize=9.5,
        color=COLOR_ACCENT,
    )

    path = save_figure(
        fig,
        figures_dir(project_root) / filename,
        manuscript_width_fraction=0.85,
    )
    from ._presentation_studies import gallery_presentation

    gallery_presentation(path, by_kind)
    return path


__all__ = ["generate_contamination_gallery"]
