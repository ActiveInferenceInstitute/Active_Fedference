"""Disjoint-FOV world figure (V4): communication necessity and EFE navigation benefit.

Headless (``Agg``) matplotlib only; no ``infrastructure.*`` imports (layer
contract). Shares the project palette via :mod:`figures._common`.

Two-panel figure:

* Left: grouped bar chart comparing isolated vs communicating accuracy across
  seeds, illustrating that communication is necessary when agents have
  disjoint fields of view.
* Right: grouped bar chart comparing EFE-guided vs random movement accuracy
  across seeds, illustrating the benefit of active inference navigation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from figures._common import (
    COLOR_AXIS,
    COLOR_MUTED,
    COLOR_NAIVE,
    COLOR_ROBUST,
    annotate_stats_box,
    apply_style,
    figures_dir,
    plt,
    save_figure,
)


def _load_disjoint_report(report: dict | None, project_root: Path | None) -> dict:
    if report is not None:
        return report
    root = (
        Path(project_root)
        if project_root is not None
        else Path(__file__).resolve().parent.parent.parent
    )
    path = root / "output" / "reports" / "disjoint_fov_world.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    from fedference.experiments import disjoint_fov_report

    return disjoint_fov_report(0)


def generate_disjoint_fov_figure(
    report: dict | None = None,
    *,
    project_root: str | Path | None = None,
    filename: str = "disjoint_fov_world.png",
) -> Path:
    """Generate the disjoint-FOV two-panel figure for the V4 manuscript.

    Args:
        report: Precomputed report from :func:`fedference.experiments.disjoint_fov_report`
            or ``output/reports/disjoint_fov_world.json``. When omitted, loads
            the JSON report or recomputes the default bundle.
        project_root: Project root when loading the JSON report from disk.
        filename: Output filename under ``output/figures/``.

    Returns:
        Absolute path of the saved PNG as a :class:`pathlib.Path`.
    """
    payload = _load_disjoint_report(report, Path(project_root) if project_root else None)
    ms = payload["multiseed"]
    iso_ms = ms["isolated"]
    comm_ms = ms["communicating"]
    efe_ms = ms["efe_guided"]
    rnd_ms = ms["random"]

    apply_style()

    means_left = [float(iso_ms["mean"]), float(comm_ms["mean"])]
    stds_left = [float(iso_ms["std"]), float(comm_ms["std"])]
    means_right = [float(efe_ms["mean"]), float(rnd_ms["mean"])]
    stds_right = [float(efe_ms["std"]), float(rnd_ms["std"])]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11.5, 5.0), facecolor="white")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.78, bottom=0.18, wspace=0.28)
    fig.suptitle(
        "Disjoint-FOV extension: communication necessity and EFE navigation\n"
        "source-inspired original project protocol",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    bar_width = 0.35
    x_left = np.array([0.0, 1.0])
    colors_left = [COLOR_NAIVE, COLOR_ROBUST]
    labels_left = ["Isolated", "Communicating"]

    for x, mean, std, color, label in zip(
        x_left, means_left, stds_left, colors_left, labels_left
    ):
        ax_left.bar(
            x,
            mean,
            width=bar_width,
            color=color,
            label=label,
            yerr=std,
            capsize=4,
            error_kw={"elinewidth": 1.5, "ecolor": COLOR_AXIS},
        )

    ax_left.set_xticks(x_left)
    ax_left.set_xticklabels(labels_left)
    ax_left.set_xlabel("Condition", labelpad=6)
    ax_left.set_ylabel("Mean accuracy (fraction correct)", labelpad=6)
    ax_left.set_title("Communication necessity", pad=8)
    ax_left.set_ylim(0.0, min(1.05, max(means_left) * 1.4 + 0.05))
    # No legend: the x tick labels already name the two conditions.
    comm_gain = float(comm_ms["mean"]) - float(iso_ms["mean"])
    annotate_stats_box(
                        ax_left,
                        f"comm gain = {comm_gain:+.3f}\niso = {float(iso_ms['mean']):.3f}\ncomm = {float(comm_ms['mean']):.3f}",  # noqa: E501
                        loc="lower right",
                        fontsize=9.5,
                    )

    x_right = np.array([0.0, 1.0])
    colors_right = [COLOR_ROBUST, COLOR_MUTED]
    labels_right = ["EFE-guided", "Random"]

    for x, mean, std, color, label in zip(
        x_right, means_right, stds_right, colors_right, labels_right
    ):
        ax_right.bar(
            x,
            mean,
            width=bar_width,
            color=color,
            label=label,
            yerr=std,
            capsize=4,
            error_kw={"elinewidth": 1.5, "ecolor": COLOR_AXIS},
        )

    ax_right.set_xticks(x_right)
    ax_right.set_xticklabels(labels_right)
    ax_right.set_xlabel("Movement policy", labelpad=6)
    ax_right.set_ylabel("Final accuracy (fraction correct)", labelpad=6)
    ax_right.set_title("EFE vs random navigation\n(null result)", pad=8)
    ax_right.set_ylim(0.0, min(1.05, max(means_right) * 1.4 + 0.05))
    # Legend in the empty column between the two near-ceiling bars so it never
    # occludes the EFE-guided bar or its error bar.
    ax_right.legend(
        fontsize=9.5,
        loc="upper center",
        handlelength=1.2,
        handletextpad=0.5,
        borderpad=0.3,
    )
    efe_gain = float(efe_ms["mean"]) - float(rnd_ms["mean"])
    annotate_stats_box(
                        ax_right,
                        f"EFE gain = {efe_gain:+.3f}\nefe = {float(efe_ms['mean']):.3f}\nrandom = {float(rnd_ms['mean']):.3f}",  # noqa: E501
                        loc="lower right",
                        fontsize=9.5,
                    )

    out_path = figures_dir(Path(project_root) if project_root else None) / filename
    save_figure(fig, out_path)
    return out_path


__all__ = ["generate_disjoint_fov_figure"]
