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
    annotate_stats_box,
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
        fig, (axl, axr, axg) = plt.subplots(1, 3, figsize=(15.2, 4.8))
    else:
        fig, (axl, axr) = plt.subplots(1, 2, figsize=(10.4, 4.8))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.20, wspace=0.32)

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
    axl.plot(eps, naive_weights, "o-", color=COLOR_NAIVE,
             label="naive pool (flat 1/n)", linewidth=1.8)
    axl.plot(eps, robust_weights, "s--", color=COLOR_ROBUST,
             label="robust heuristic", linewidth=1.8)
    axl.axhline(1.0 / naive["n_agents"], color=COLOR_AXIS, linewidth=0.8, linestyle=":")
    axl.set_xlabel("perturbation toward contamination  (eps)")
    axl.set_ylabel("perturbed agent normalized weight")
    axl.set_title("Numerical influence\n(empirical at these settings, not a guarantee)", fontsize=12)
    final_drop = float(naive_weights[-1] - robust_weights[-1])
    annotate_stats_box(
        axl,
        f"final naive-robust weight gap = {final_drop:.3f}\nprobe path: one agent",
        loc="upper right",
        fontsize=9.5,
    )
    axl.legend(fontsize=9.5, loc="lower left")

    labels = ["robust\nheuristic", "variational\n(objective-backed)"]
    ks = [breakdown["robust_breakdown_k"], breakdown["variational_breakdown_k"]]
    ks = [k if k is not None else 0 for k in ks]
    bars = axr.bar([0, 1], ks, 0.55, color=[COLOR_ROBUST, COLOR_VARIATE],
                   edgecolor=COLOR_AXIS, linewidth=0.9, zorder=3)
    for b, k in zip(bars, ks, strict=True):
        axr.annotate(f"captured\nat k={k}", xy=(b.get_x() + b.get_width() / 2, k),
                     xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9.5)
    axr.set_xticks([0, 1])
    axr.set_xticklabels(labels, fontsize=10)
    axr.set_ylabel("colluding adversaries to capture argmax")
    axr.set_title("Measured breakdown point\nfinite search, not unconditional recovery", fontsize=12)
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
        bars = axg.bar(range(len(attacks)), fractions, color=COLOR_ROBUST,
                       edgecolor=COLOR_AXIS, linewidth=0.9, zorder=3)
        for bar, fraction in zip(bars, fractions, strict=True):
            axg.annotate(f"{fraction:.0%}",
                         xy=(bar.get_x() + bar.get_width() / 2, fraction),
                         xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9.5)
        axg.set_xticks(range(len(attacks)))
        axg.set_xticklabels([str(attack).replace("_", "\n") for attack in attacks], fontsize=9.5)
        axg.set_ylim(0, 1.12)
        axg.set_ylabel("rows with finite capture in search budget")
        axg.axhline(0.5, color=COLOR_GRID, linestyle=":", linewidth=0.8)
        axg.set_title("Declared attack grid\nfinite-search frequency, not a bound", fontsize=12)

    return save_figure(fig, figures_dir(project_root) / filename)


__all__ = ["generate_heuristic_breakdown"]
