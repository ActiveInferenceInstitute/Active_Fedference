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
    "fraction": "Signed accuracy contrast (fraction)",
    "nats": "Signed information / free-energy\ncontrast (nats)",
    "R-sq": r"Parameter-recovery fit ($R^2$, unitless)",
}
_UNIT_AXIS_LABELS: dict[str, str] = {
    "fraction": "Signed contrast (fraction)",
    "nats": "Signed contrast (nats)",
    "R-sq": r"Parameter-recovery fit ($R^2$)",
}


def _display_label(entry: dict) -> str:
    """Return the visible study label with selection status made explicit."""
    label = str(entry["label"]).replace(
        "Emergence (BMR)",
        "Configured BMR\nsign control",
    )
    if int(entry.get("study", -1)) == 4:
        return "Study 4\nWithin-run\ndisplay-selected maximum"
    return label


def _value_annotation_position(
    value: float,
    endpoint: float,
    span: float,
) -> tuple[float, str, dict[str, object] | None]:
    """Choose a legible data-label lane for a horizontal bar.

    Small negative effects end immediately to the left of the zero line. If
    their interval endpoint is labelled on that side, the text can collide
    with the long study labels in the left margin. Put those labels in a
    dedicated, lightly boxed lane just to the right of zero; the sign and bar
    still carry the direction, while the printed value remains readable.
    """
    if value < 0.0 and abs(endpoint) < 0.12 * span:
        return (
            0.02 * span,
            "left",
            {"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.6},
        )
    sign = 1.0 if value >= 0.0 else -1.0
    return (
        endpoint + sign * 0.025 * span,
        "left" if sign > 0 else "right",
        None,
    )


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
            raise ValueError(f"cross-study figure requires a known native unit; got {unit!r}")
        grouped[unit].append(study)
    if any(not entries for entries in grouped.values()):
        missing = [unit for unit, entries in grouped.items() if not entries]
        raise ValueError(f"cross-study report is missing native-unit facet(s): {missing}")

    # Use a page-compatible 2 x 2 composition: the six-row accuracy family
    # occupies the full left column, while the two shorter native-unit facets
    # occupy the right column.  This preserves separate estimands without the
    # unusually tall portrait page previously required by a three-panel stack.
    fig = plt.figure(figsize=(9.2, 7.4), facecolor="white")
    fig.set_layout_engine("none")
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.48, 1.0),
        height_ratios=(1.0, 1.0),
        left=0.235,
        right=0.97,
        top=0.79,
        bottom=0.16,
        wspace=0.48,
        hspace=0.74,
    )
    axes_by_unit = {
        "fraction": fig.add_subplot(grid[:, 0]),
        "nats": fig.add_subplot(grid[0, 1]),
        "R-sq": fig.add_subplot(grid[1, 1]),
    }

    threshold = 1e-3
    for panel_index, unit in enumerate(_UNIT_ORDER):
        ax = axes_by_unit[unit]
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
            COLOR_ROBUST if value > threshold else (COLOR_NAIVE if value < -threshold else COLOR_MUTED)
            for value in means
        ]
        bars = ax.barh(
            y,
            means,
            xerr=[means - ci_lo, ci_hi - means],
            color=colors,
            alpha=0.88,
            edgecolor=COLOR_AXIS,
            linewidth=0.8,
            error_kw={
                "elinewidth": 2.0,
                "capsize": 5,
                "capthick": 2.0,
                "ecolor": COLOR_AXIS,
            },
        )
        for bar, value in zip(bars, means, strict=True):
            bar.set_hatch("///" if value > threshold else ("xxx" if value < -threshold else ".."))
        ax.axvline(0.0, color=COLOR_GRID, linewidth=1.0, linestyle=":")

        x_lo = min(0.0, float(ci_lo.min()))
        x_hi = max(0.0, float(ci_hi.max()))
        span = max(x_hi - x_lo, 1e-6)
        ax.set_xlim(x_lo - 0.05 * span, x_hi + 0.18 * span)
        for index, (entry, value) in enumerate(zip(entries, means, strict=True)):
            endpoint = float(ci_hi[index] if value >= 0.0 else ci_lo[index])
            annotation_x, alignment, bbox = _value_annotation_position(value, endpoint, span)
            ax.text(
                annotation_x,
                index,
                f"{value:+.3f}",
                va="center",
                ha=alignment,
                fontsize=11.2,
                fontweight="bold",
                bbox=bbox,
            )
        ax.set_yticks(y)
        display_labels = [_display_label(entry) for entry in entries]
        if unit == "fraction":
            ax.set_yticklabels(display_labels, fontsize=11.2)
        else:
            # The compact right-hand facets carry study names inside their
            # bars.  Long external tick labels otherwise intrude into Panel A
            # and collide with its direct value annotations.
            ax.set_yticklabels(["" for _ in display_labels])
            label_x = max(0.0, x_lo) + 0.035 * span
            for index, label in enumerate(display_labels):
                ax.text(
                    label_x,
                    index,
                    label,
                    ha="left",
                    va="center",
                    fontsize=10.8,
                    color=COLOR_AXIS,
                    linespacing=1.05,
                    bbox={
                        "boxstyle": "round,pad=0.14",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.86,
                    },
                )
        ax.set_xlabel(_UNIT_AXIS_LABELS[unit], labelpad=5, fontsize=11.8)
        panel_letter = chr(ord("A") + panel_index)
        ax.set_title(
            f"{panel_letter}  {_UNIT_TITLES[unit]}",
            loc="left",
            pad=7,
            fontsize=12.8,
        )
        ax.invert_yaxis()
        ax.tick_params(axis="x", labelsize=11.0)

    fig.suptitle(
        "Cross-study estimands by native unit",
        fontsize=16.5,
        fontweight="bold",
        y=0.980,
    )
    fig.text(
        0.5,
        0.902,
        f"Separate harmonized seed-level rerun · n = {report_n_seeds} seeds · seed is the independent unit · "
        "whiskers are 95% percentile-bootstrap intervals",
        ha="center",
        va="center",
        fontsize=10.8,
        color=COLOR_AXIS,
        wrap=True,
    )
    fig.text(
        0.5,
        0.045,
        "/// positive signed estimand    xxx negative signed estimand    ·· near zero\n"
        "A spans the left column; B and C occupy the upper and lower right. "
        "Direction is metric-specific; no cross-unit ranking.",
        ha="center",
        va="center",
        fontsize=11.0,
        color=COLOR_AXIS,
    )

    out = figures_dir(Path(project_root) if project_root is not None else None)
    return save_figure(fig, out / filename, manuscript_width_fraction=0.95)


__all__ = ["generate_cross_study_summary"]


if __name__ == "__main__":
    out = generate_cross_study_summary(project_root=Path(__file__).resolve().parent.parent.parent)
    print(out)
