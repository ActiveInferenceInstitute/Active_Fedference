"""Paired communicating-versus-incommunicado free-energy estimation plot.

The values are produced by the declared reduced categorical protocol. This
module preserves seed pairing, displays the signed paired contrast
``incommunicado - communicating``, and consumes the report-owned paired mean
interval when supplied. It does not reconstruct the complete source protocol
or infer a communication-rate effect.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ._common import (
    COLOR_AXIS,
    COLOR_GRID,
    COLOR_MUTED,
    apply_style,
    figures_dir,
    plt,
    save_figure,
    semantic_style,
)


def generate_free_energy_comparison(
    incommunicado: Sequence[float],
    communicating: Sequence[float],
    *,
    paired_difference_mean: float | None = None,
    paired_difference_ci: Sequence[float] | None = None,
    difference_definition: str = "incommunicado_minus_communicating",
    analysis_unit: str = "seed-level paired colony mean",
    replication_unit: str = "seed",
    project_root: Path | None = None,
    filename: str = "free_energy_comparison.png",
) -> Path:
    """Render paired free energies and their signed seed-level differences.

    ``paired_difference_ci`` is optional for compatibility with older report
    snapshots. When absent, the figure shows the paired values and their mean
    without manufacturing an interval inside the plotter.
    """
    incom = np.asarray(incommunicado, dtype=np.float64).ravel()
    comm = np.asarray(communicating, dtype=np.float64).ravel()
    if incom.size == 0 or comm.size == 0:
        raise ValueError("both free-energy sequences must be non-empty")
    if incom.shape != comm.shape:
        raise ValueError("incommunicado and communicating must have equal length")
    if not (np.all(np.isfinite(incom)) and np.all(np.isfinite(comm))):
        raise ValueError("free-energy sequences must contain only finite values")
    if difference_definition != "incommunicado_minus_communicating":
        raise ValueError("difference_definition must be 'incommunicado_minus_communicating'")

    differences = incom - comm
    observed_mean = float(np.mean(differences))
    mean_difference = observed_mean if paired_difference_mean is None else float(paired_difference_mean)
    if not np.isfinite(mean_difference):
        raise ValueError("paired_difference_mean must be finite")
    if not np.isclose(mean_difference, observed_mean, rtol=1e-10, atol=1e-12):
        raise ValueError("paired_difference_mean disagrees with the paired values")

    interval: tuple[float, float] | None = None
    if paired_difference_ci is not None:
        if len(paired_difference_ci) != 2:
            raise ValueError("paired_difference_ci must contain [lo, hi]")
        lo, hi = (float(value) for value in paired_difference_ci)
        if not (np.isfinite(lo) and np.isfinite(hi) and lo <= mean_difference <= hi):
            raise ValueError("paired_difference_ci must be finite, ordered, and contain the mean")
        interval = (lo, hi)

    apply_style()
    incommunicado_style = semantic_style("condition_reference")
    communicating_style = semantic_style("condition_comparison")
    reference = semantic_style("reference_rule")
    fig, (ax_pairs, ax_difference) = plt.subplots(
        2,
        1,
        figsize=(6.5, 7.4),
        facecolor="white",
        gridspec_kw={"height_ratios": (1.08, 1.0)},
    )
    fig.set_layout_engine("none")
    fig.subplots_adjust(left=0.15, right=0.97, top=0.88, bottom=0.10, hspace=0.48)

    seed_alpha = max(0.08, min(0.34, 22.0 / incom.size))
    seed_size = max(14.0, min(34.0, 6000.0 / incom.size))
    for before, after in zip(incom, comm, strict=True):
        ax_pairs.plot(
            (0.0, 1.0),
            (before, after),
            color=COLOR_MUTED,
            linewidth=0.9,
            alpha=seed_alpha,
            zorder=1,
        )
    ax_pairs.scatter(
        np.zeros(incom.size),
        incom,
        color=incommunicado_style.color,
        edgecolor=incommunicado_style.keyline,
        marker=incommunicado_style.marker,
        s=seed_size,
        linewidth=0.8,
        alpha=0.78,
        label="incommunicado",
        zorder=2,
    )
    ax_pairs.scatter(
        np.ones(comm.size),
        comm,
        facecolor="white",
        edgecolor=communicating_style.keyline,
        marker=communicating_style.marker,
        s=seed_size,
        linewidth=1.2,
        label="communicating",
        zorder=3,
    )
    pair_means = (float(incom.mean()), float(comm.mean()))
    ax_pairs.plot(
        (0.0, 1.0),
        pair_means,
        color=COLOR_AXIS,
        linewidth=2.5,
        marker="D",
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=1.4,
        label="paired-condition means",
        zorder=4,
    )
    ax_pairs.set_xticks((0.0, 1.0), ("incommunicado", "communicating"))
    ax_pairs.set_xlim(-0.28, 1.28)
    ax_pairs.set_ylabel("Seed-level colony mean free energy (nats)")
    ax_pairs.set_title("A  Matched conditions", loc="left")
    ax_pairs.legend(loc="best", fontsize=9.5)

    violin = ax_difference.violinplot(
        differences,
        positions=[0.0],
        orientation="horizontal",
        widths=0.30,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in violin["bodies"]:
        body.set_facecolor(communicating_style.color)
        body.set_edgecolor(communicating_style.keyline)
        body.set_alpha(0.16)
        body.set_linewidth(1.0)
    jitter = np.random.default_rng(0).uniform(-0.12, 0.12, size=differences.size)
    ax_difference.scatter(
        differences,
        jitter,
        marker=communicating_style.marker,
        facecolor="white",
        edgecolor=communicating_style.keyline,
        linewidth=1.1,
        s=max(12.0, min(30.0, 4200.0 / differences.size)),
        alpha=max(0.22, min(0.75, 36.0 / differences.size)),
        label="one paired difference per seed",
        zorder=3,
    )
    ax_difference.axvline(
        0.0,
        color=reference.color,
        linestyle=reference.dash,
        linewidth=reference.linewidth,
        label="zero difference",
        zorder=1,
    )
    if interval is None:
        ax_difference.scatter(
            [mean_difference],
            [-0.26],
            marker="D",
            s=72,
            color=COLOR_AXIS,
            label="paired mean; interval unavailable",
            zorder=4,
        )
    else:
        lo, hi = interval
        ax_difference.errorbar(
            [mean_difference],
            [-0.26],
            xerr=[[mean_difference - lo], [hi - mean_difference]],
            fmt="D",
            markersize=7,
            markerfacecolor="white",
            markeredgecolor=COLOR_AXIS,
            color=COLOR_AXIS,
            linewidth=2.0,
            capsize=5,
            label="paired mean with 95% interval",
            zorder=4,
        )
    ax_difference.set_ylim(-0.38, 0.21)
    ax_difference.set_yticks(())
    ax_difference.set_xlabel("$\\Delta F$ (nats)\nincommunicado − communicating")
    ax_difference.set_title("B  Seed-level paired differences", loc="left")
    ax_difference.legend(loc="upper left", fontsize=9.5)
    ax_difference.text(
        0.02,
        0.03,
        rf"mean $\Delta F$ = {mean_difference:+.3g} nats"
        "\n"
        f"n = {differences.size} {replication_unit}s\n"
        "positive = lower F with communication",
        transform=ax_difference.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": COLOR_GRID},
    )

    fig.suptitle(
        "Paired colony free-energy comparison",
        y=0.985,
        fontsize=15,
        fontweight="bold",
    )
    ax_difference.text(
        0.98,
        0.96,
        f"{analysis_unit}\nindependent unit: {replication_unit}",
        transform=ax_difference.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        color=COLOR_MUTED,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": COLOR_GRID},
    )
    return save_figure(
        fig,
        figures_dir(project_root) / filename,
        manuscript_width_fraction=0.80,
    )


__all__ = ["generate_free_energy_comparison"]
