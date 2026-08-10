"""Proper-score and reliability diagnostics for categorical beliefs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from ._common import (
    COLOR_ACCENT,
    COLOR_MUTED,
    COLOR_NAIVE,
    COLOR_ROBUST,
    MIN_QUANTITATIVE_FONT_SIZE,
    apply_style,
    figures_dir,
    plt,
    save_figure,
)


def generate_belief_quality(
    report: Mapping[str, object],
    *,
    project_root: Path | None = None,
    filename: str = "belief_quality.png",
) -> Path:
    """Render control log scores and reliability curves from the score report."""
    raw_controls = report.get("control_scores")
    if not isinstance(raw_controls, Mapping) or not raw_controls:
        raise ValueError("belief-quality report must contain control_scores")
    names = [name for name in ("oracle", "uniform", "confident_wrong") if name in raw_controls]
    if len(names) != 3:
        raise ValueError("belief-quality controls must include oracle, uniform, and confident_wrong")
    values = np.asarray([float(raw_controls[name]["mean_log_score"]) for name in names])
    intervals = np.asarray([raw_controls[name]["log_score_ci"] for name in names], dtype=np.float64)
    yerr = np.vstack((values - intervals[:, 0], intervals[:, 1] - values))

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.4))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.83, bottom=0.22, wspace=0.30)
    x = np.arange(len(names))
    axes[0].errorbar(
        x,
        values,
        yerr=yerr,
        fmt="o",
        color=COLOR_ACCENT,
        ecolor=COLOR_MUTED,
        capsize=4,
        linewidth=1.7,
    )
    axes[0].axhline(0.0, color=COLOR_MUTED, linewidth=0.8, linestyle=":")
    axes[0].set_xticks(x, [name.replace("_", "\n") for name in names])
    axes[0].set_ylabel("Mean categorical log score (nats)")
    axes[0].set_title("Primary score: higher is better")
    axes[0].text(
        0.03,
        0.03,
        "95% seed bootstrap intervals\nseed is the independent unit",
        transform=axes[0].transAxes,
        fontsize=9.5,
        color=COLOR_MUTED,
        va="bottom",
    )

    colors = {"oracle": COLOR_ROBUST, "uniform": COLOR_MUTED, "confident_wrong": COLOR_NAIVE}
    for name in names:
        curve = raw_controls[name]["reliability"]
        confidence = np.asarray(curve["mean_confidence"], dtype=np.float64)
        accuracy = np.asarray(curve["accuracy"], dtype=np.float64)
        mask = np.isfinite(confidence) & np.isfinite(accuracy)
        axes[1].plot(
            confidence[mask],
            accuracy[mask],
            marker="o",
            linewidth=1.7,
            color=colors[name],
            label=name.replace("_", " "),
        )
    axes[1].plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color=COLOR_ACCENT,
        linestyle="--",
        linewidth=1.0,
        label="perfect calibration",
    )
    axes[1].set_xlim(0.0, 1.02)
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_xlabel("Mean confidence")
    axes[1].set_ylabel("Empirical accuracy")
    axes[1].set_title("Reliability diagnostic")
    axes[1].legend(fontsize=MIN_QUANTITATIVE_FONT_SIZE, loc="best")
    fig.suptitle("Proper scoring and calibration controls", fontweight="bold")
    return save_figure(fig, figures_dir(project_root) / filename)


__all__ = ["generate_belief_quality"]
