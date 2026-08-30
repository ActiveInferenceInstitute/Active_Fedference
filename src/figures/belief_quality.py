"""Proper-score and reliability diagnostics for categorical beliefs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from ._common import (
    COLOR_MUTED,
    MIN_QUANTITATIVE_FONT_SIZE,
    apply_style,
    figures_dir,
    plt,
    save_figure,
    semantic_style,
)

_CONTROL_ROLES = {
    "oracle": "honest",
    "uniform": "operating_point_1",
    "confident_wrong": "adversarial",
}


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
    for index, name in enumerate(names):
        style = semantic_style(_CONTROL_ROLES[name])
        axes[0].errorbar(
            [x[index]],
            [values[index]],
            yerr=[[yerr[0, index]], [yerr[1, index]]],
            fmt=style.marker,
            color=style.color,
            markerfacecolor="white" if name != "confident_wrong" else style.color,
            markeredgecolor=style.keyline,
            ecolor=style.keyline,
            capsize=4,
            linewidth=style.linewidth,
            markersize=7,
        )
    rule = semantic_style("reference_rule")
    axes[0].axhline(
        0.0,
        color=rule.color,
        linewidth=rule.linewidth,
        linestyle=rule.dash,
    )
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

    for name in names:
        style = semantic_style(_CONTROL_ROLES[name])
        curve = raw_controls[name]["reliability"]
        confidence = np.asarray(curve["mean_confidence"], dtype=np.float64)
        accuracy = np.asarray(curve["accuracy"], dtype=np.float64)
        mask = np.isfinite(confidence) & np.isfinite(accuracy)
        axes[1].plot(
            confidence[mask],
            accuracy[mask],
            marker=style.marker,
            linestyle=style.dash,
            linewidth=style.linewidth,
            color=style.color,
            markerfacecolor="white" if name != "confident_wrong" else style.color,
            markeredgecolor=style.keyline,
            label=name.replace("_", " "),
        )
        if np.any(mask):
            axes[1].annotate(
                name.replace("_", " "),
                xy=(float(confidence[mask][-1]), float(accuracy[mask][-1])),
                xytext=(5, 0),
                textcoords="offset points",
                fontsize=9.5,
                color=style.keyline,
                ha="left",
                va="center",
                clip_on=False,
            )
    axes[1].plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color=rule.color,
        linestyle=rule.dash,
        linewidth=rule.linewidth,
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
