"""Cross-study summary figure grouped by native metric units.

The nine studies do not share one numerical estimand: several report accuracy
fractions, two report nats, and parameter recovery reports :math:`R^2`. The
renderer therefore uses one horizontal facet per native unit rather than
putting incompatible quantities on a common axis. Every interval is a
seed-level percentile bootstrap interval.

Headless (Agg) matplotlib only; no infrastructure imports (layer contract).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from figures._common import (
    COLOR_AXIS,
    COLOR_GRID,
    COLOR_MUTED,
    COLOR_NAIVE,
    COLOR_ROBUST,
    apply_style,
    figures_dir,
    plt,
    save_figure,
)

_UNIT_ORDER: tuple[str, ...] = ("fraction", "nats", "R-sq")
_UNIT_TITLES: dict[str, str] = {
    "fraction": "Accuracy gaps (fraction units)",
    "nats": "Information / free-energy changes (nats)",
    "R-sq": r"Parameter-recovery fit ($R^2$, unitless)",
}


def _load_cross_study_report(
    project_root: Path | None,
    report: dict | None,
    *,
    seed: int,
    n_seeds: int,
) -> dict:
    if report is not None:
        return report
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parent.parent.parent
    path = root / "output" / "reports" / "cross_study_summary.json"
    if path.exists():
        import json

        return json.loads(path.read_text(encoding="utf-8"))
    from experiment_config import load_experiment_config
    from fedference.experiments import summarize_cross_study

    cfg = load_experiment_config(root)
    return summarize_cross_study(
        seed=seed,
        n_seeds=n_seeds,
        n_trials=cfg.cross_study_n_trials,
    )


def generate_cross_study_summary(
    report: dict | None = None,
    project_root: str | Path | None = None,
    *,
    seed: int = 0,
    n_seeds: int = 64,
    filename: str = "cross_study_summary.png",
) -> Path:
    """Render native-unit facets for the nine-study summary.

    Args:
        report: Precomputed cross-study report dict (from
            :func:`fedference.experiments.summarize_cross_study` or
            ``output/reports/cross_study_summary.json``). When omitted, loads
            the JSON report from ``project_root`` or recomputes a small default.
        project_root: Project root directory.
        seed: Starting RNG seed when recomputing (ignored when ``report`` given).
        n_seeds: Seed count when recomputing (ignored when ``report`` given).
        filename: Output filename under ``output/figures/``.

    Returns:
        Path to the written PNG file.
    """
    apply_style()

    payload = _load_cross_study_report(
        Path(project_root) if project_root is not None else None,
        report,
        seed=seed,
        n_seeds=n_seeds,
    )
    studies = payload["studies"]
    report_n_seeds = int(payload.get("n_seeds", n_seeds))

    grouped: dict[str, list[dict]] = {unit: [] for unit in _UNIT_ORDER}
    for study in studies:
        unit = str(study.get("unit", ""))
        if unit not in grouped:
            raise ValueError(
                f"cross-study figure requires a known native unit; got {unit!r}"
            )
        grouped[unit].append(study)
    if any(not entries for entries in grouped.values()):
        missing = [unit for unit, entries in grouped.items() if not entries]
        raise ValueError(f"cross-study report is missing native-unit facet(s): {missing}")

    max_rows = max(len(entries) for entries in grouped.values())
    fig, axes = plt.subplots(
        len(_UNIT_ORDER),
        1,
        figsize=(11.5, 2.15 * max_rows + 4.1),
        gridspec_kw={"height_ratios": [len(grouped[unit]) for unit in _UNIT_ORDER]},
    )
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.28, right=0.97, top=0.88, bottom=0.12, hspace=0.48)

    threshold = 1e-3
    for ax, unit in zip(axes, _UNIT_ORDER, strict=True):
        entries = grouped[unit]
        means = np.asarray([float(s["mean"]) for s in entries], dtype=np.float64)
        ci_lo = np.asarray([float(s["ci_lo"]) for s in entries], dtype=np.float64)
        ci_hi = np.asarray([float(s["ci_hi"]) for s in entries], dtype=np.float64)
        if not (np.all(np.isfinite(means)) and np.all(np.isfinite(ci_lo)) and np.all(np.isfinite(ci_hi))):
            raise ValueError(f"cross-study report contains non-finite values in {unit!r}")
        if np.any(ci_lo > means) or np.any(means > ci_hi):
            raise ValueError(f"cross-study intervals do not contain means in {unit!r}")

        y = np.arange(len(entries))
        colors = [
            COLOR_ROBUST if value > threshold else
            (COLOR_NAIVE if value < -threshold else COLOR_MUTED)
            for value in means
        ]
        ax.barh(
            y,
            means,
            xerr=[means - ci_lo, ci_hi - means],
            color=colors,
            alpha=0.88,
            error_kw={
                "elinewidth": 2.0,
                "capsize": 5,
                "capthick": 2.0,
                "ecolor": COLOR_AXIS,
            },
        )
        ax.axvline(0.0, color=COLOR_GRID, linewidth=0.9, linestyle="--")

        x_lo = min(0.0, float(ci_lo.min()))
        x_hi = max(0.0, float(ci_hi.max()))
        span = max(x_hi - x_lo, 1e-6)
        ax.set_xlim(x_lo - 0.05 * span, x_hi + 0.18 * span)
        for index, (entry, value) in enumerate(zip(entries, means, strict=True)):
            endpoint = float(ci_hi[index] if value >= 0.0 else ci_lo[index])
            sign = 1.0 if value >= 0.0 else -1.0
            ax.text(
                endpoint + sign * 0.025 * span,
                index,
                f"{value:+.3f}",
                va="center",
                ha="left" if sign > 0 else "right",
                fontsize=10.5,
                fontweight="bold",
            )
        ax.set_yticks(y)
        ax.set_yticklabels([str(entry["label"]) for entry in entries], fontsize=10.5)
        ax.set_xlabel(_UNIT_TITLES[unit], labelpad=7, fontsize=12)
        ax.set_title(_UNIT_TITLES[unit], loc="left", pad=7, fontsize=14)
        ax.invert_yaxis()
        ax.tick_params(axis="x", labelsize=10.5)
        ax.text(
            0.99,
            0.04,
            f"n = {report_n_seeds} seeds; whiskers = 95% seed bootstrap CI",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9.5,
            color=COLOR_AXIS,
        )

    fig.suptitle(
        "Cross-study metrics grouped by native unit\n"
        "Panels are not directly comparable; intervals resample independent seeds",
        fontsize=16,
        fontweight="bold",
    )

    out = figures_dir(Path(project_root) if project_root is not None else None)
    return save_figure(fig, out / filename)


__all__ = ["generate_cross_study_summary"]


if __name__ == "__main__":
    out = generate_cross_study_summary(project_root=Path(__file__).resolve().parent.parent.parent)
    print(out)
