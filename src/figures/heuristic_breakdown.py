"""Empirical characterization figure for robust_aggregate (MAJ-1).

Two panels drawn from :func:`fedference.experiments.run_heuristic_characterization`:

* Left — numerical influence: the perturbed agent's normalized pooling weight as
  it is dragged toward a contamination point, for the naive pool (flat 1/n) and
  the robust heuristic (down-weighting). Labeled "empirical, at these settings —
  not a guarantee".
* Right — measured breakdown points: the number of colluding adversaries that
  captures each aggregator's argmax (a finite bar for each), making visible that
  the sharp heuristic can be overwhelmed.
* Optional third panel — attack-mechanism coverage across the declared MAJ-1 grid;
  finite-search frequency is shown, never as a global breakdown probability.

The numbers come from the analysis workflow; this module only draws.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._common import (
    COLOR_AXIS,
    COLOR_GRID,
    COLOR_NAIVE,
    COLOR_ROBUST,
    COLOR_VARIATE,
    apply_style,
    figures_dir,
    plt,
    save_figure,
)


def generate_heuristic_breakdown(
    report: dict[str, Any],
    *,
    project_root: Path | None = None,
    filename: str = "heuristic_breakdown.png",
) -> Path:
    """Render the influence + breakdown characterization of ``robust_aggregate``.

    Args:
        report: ``run_heuristic_characterization`` output.
        project_root: project root override.
        filename: output PNG name under ``output/figures``.
    """
    naive = report["influence_naive"]
    robust = report["influence_robust"]
    breakdown = report["breakdown"]

    apply_style()
    grid = report.get("grid")
    if grid:
        fig = plt.figure(figsize=(8.2, 7.4))
        layout = fig.add_gridspec(2, 2, height_ratios=(1.0, 0.88))
        axl = fig.add_subplot(layout[0, 0])
        axr = fig.add_subplot(layout[0, 1])
        axg = fig.add_subplot(layout[1, :])
        fig.subplots_adjust(
            left=0.11,
            right=0.97,
            top=0.86,
            bottom=0.11,
            hspace=0.58,
            wspace=0.42,
        )
    else:
        fig, (axl, axr) = plt.subplots(1, 2, figsize=(8.2, 4.8))
        fig.subplots_adjust(left=0.11, right=0.97, top=0.78, bottom=0.20, wspace=0.42)

    eps = naive["eps_grid"]
    naive_weights = (
        naive["normalized_effective_weights"]
        if "normalized_effective_weights" in naive
        else naive["agent_weight"]
    )
    robust_weights = (
        robust["normalized_effective_weights"]
        if "normalized_effective_weights" in robust
        else robust["agent_weight"]
    )
    axl.plot(eps, naive_weights, "o-", color=COLOR_NAIVE, label="naive pool (flat 1/n)", linewidth=1.8)
    axl.plot(eps, robust_weights, "s--", color=COLOR_ROBUST, label="robust heuristic", linewidth=1.8)
    axl.axhline(1.0 / naive["n_agents"], color=COLOR_AXIS, linewidth=0.8, linestyle=":")
    axl.set_xlabel(r"Perturbation fraction $\epsilon$")
    axl.set_ylabel("Normalized agent weight")
    axl.set_title("A  Numerical influence", loc="left", fontsize=12)
    final_drop = float(naive_weights[-1] - robust_weights[-1])
    axl.annotate(
        f"$\\Delta w$ = {final_drop:.3f}",
        xy=(float(eps[-1]), float(robust_weights[-1])),
        xytext=(-4, 65),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=9.5,
        color=COLOR_AXIS,
        arrowprops={"arrowstyle": "->", "color": COLOR_AXIS, "lw": 0.8},
    )
    axl.legend(fontsize=9.5, loc="lower left")

    labels = ["robust\nheuristic", "variational\n(objective-backed)"]
    ks = [breakdown["robust_breakdown_k"], breakdown["variational_breakdown_k"]]
    ks = [k if k is not None else 0 for k in ks]
    bars = axr.bar(
        [0, 1], ks, 0.55, color=[COLOR_ROBUST, COLOR_VARIATE], edgecolor=COLOR_AXIS, linewidth=0.9, zorder=3
    )
    observed_ks = [breakdown["robust_breakdown_k"], breakdown["variational_breakdown_k"]]
    for b, k, observed, hatch in zip(bars, ks, observed_ks, ("//", ".."), strict=True):
        b.set_hatch(hatch)
        capture_label = f"captured\nat k={k}" if observed is not None else "not observed\nin search"
        axr.annotate(
            capture_label,
            xy=(b.get_x() + b.get_width() / 2, k),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9.5,
        )
    axr.set_xticks([0, 1])
    axr.set_xticklabels(labels, fontsize=10)
    axr.set_ylabel("Adversaries $k$ at argmax capture")
    axr.set_title("B  Finite-search capture", loc="left", fontsize=12)
    axr.set_ylim(0, max(ks) + 1.4)
    axr.grid(axis="x", visible=False)

    if grid:
        rows = grid["rows"]
        attacks = grid["parameter_grid"]["attacks"]
        fractions: list[float] = []
        for attack in attacks:
            attack_rows = [row for row in rows if row["attack"] == attack]
            finite = sum(row["robust_breakdown_k"] is not None for row in attack_rows)
            fractions.append(finite / len(attack_rows) if attack_rows else 0.0)
        bars = axg.bar(
            range(len(attacks)), fractions, color=COLOR_ROBUST, edgecolor=COLOR_AXIS, linewidth=0.9, zorder=3
        )
        for bar, fraction in zip(bars, fractions, strict=True):
            axg.annotate(
                f"{fraction:.0%}",
                xy=(bar.get_x() + bar.get_width() / 2, fraction),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=9.5,
            )
        axg.set_xticks(range(len(attacks)))
        axg.set_xticklabels([str(attack).replace("_", "\n") for attack in attacks], fontsize=9.5)
        axg.set_ylim(0, 1.12)
        axg.set_xlabel("Attack mechanism")
        axg.set_ylabel("Rows with finite capture\n(fraction)")
        axg.axhline(0.5, color=COLOR_GRID, linestyle=":", linewidth=0.8)
        axg.set_title("C  Declared-grid capture frequency", loc="left", fontsize=12)

    fig.suptitle(
        "Server-rule finite-grid characterization",
        fontsize=14,
        fontweight="bold",
    )

    path = save_figure(fig, figures_dir(project_root) / filename)
    from ._presentation_studies import heuristic_presentation

    heuristic_presentation(path, report)
    return path


__all__ = ["generate_heuristic_breakdown"]
