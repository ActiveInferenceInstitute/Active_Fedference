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
    apply_style,
    figures_dir,
    plt,
    save_figure,
    semantic_style,
)


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
            remain renderable without error bars.
        project_root: Project root override.
        filename: Output PNG name under ``output/figures``.

    Returns:
        Path to the written PNG.

    Raises:
        ValueError: If ``by_kind`` is empty or a cell lacks the required means.
    """
    if not by_kind:
        raise ValueError("by_kind must be non-empty")
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
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    fig.set_layout_engine("none")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.34)
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
        label="reference log pool (open fill)",
    )
    ax.bar(
        x + w / 2,
        robust,
        w,
        color=robust_style.color,
        alpha=0.9,
        edgecolor=robust_style.keyline,
        linewidth=1.2,
        hatch=robust_style.hatch,
        label="pooled display server preset (hatched)",
    )
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
    # annotate each mechanism with its across-seed win fraction and reliability.
    for i, (wf, rel, method) in enumerate(zip(wins, reliable, methods)):
        mark = "reliable" if rel else "not reliable"
        top = max(naive[i], robust[i])
        if have_ci:
            top = max(top, float(naive_ci[i, 1]), float(robust_ci[i, 1]))
        y = min(top + 0.035, 1.04)
        ax.annotate(
            f"{method}\nwin {wf:.2f}\n{mark}",
            xy=(i, y),
            ha="center",
            va="bottom",
            fontsize=9.5,
            color=COLOR_ACCENT,
            alpha=0.9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("_", "\n") for k in kinds], fontsize=10)
    ax.set_xlabel("Contamination mechanism", labelpad=6)
    ax.set_ylabel("Mean consensus accuracy $q(\\mathrm{true})$", labelpad=6)
    ax.set_ylim(0.0, 1.16)
    ax.set_title("Server-preset accuracy across contamination mechanisms", pad=8)
    ax.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2)
    # --- stats box: n mechanisms, reliable wins. Placed above the shortest
    # bar group so it never overlaps any bar (the lower-right corner sits on
    # the last group's bars).
    n_reliable = sum(reliable)
    tops = np.maximum(naive, robust)
    if have_ci:
        tops = np.maximum(tops, np.maximum(naive_ci[:, 1], robust_ci[:, 1]))
    i_min = int(np.argmin(tops))
    ax.text(
        float(x[i_min]),
        float(tops[i_min]) + 0.19,
        f"declared reliable contrasts = {n_reliable}/{len(kinds)}\n"
        "bars: means with 95% seed-bootstrap intervals",
        ha="center",
        va="bottom",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": COLOR_ACCENT, "alpha": 0.85},
    )
    fig.text(
        0.5,
        0.025,
        "Mixed positive, near-zero, and negative contrasts are retained; the finite "
        "configured mechanism grid does not establish a universal winner.",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color=COLOR_ACCENT,
        style="italic",
    )

    return save_figure(fig, figures_dir(project_root) / filename)


__all__ = ["generate_contamination_gallery"]
