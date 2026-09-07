"""Complexity accounting plus measured scaling diagnostics.

The figure keeps symbolic implementation orders and machine timings visually
separate. Lines show median wall-clock observations; faint dotted companions
are normalized reference slopes with the expected exponent from the analytic
catalog. Error bars span the repeated timing observations' min--max range, not
a confidence interval.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from figures._common import (
    COLOR_DARK,
    MIN_QUANTITATIVE_FONT_SIZE,
    apply_style,
    figures_dir,
    plt,
    save_figure,
    semantic_style,
)
from figures._presentation_estimates import complexity_presentation

__all__ = ["generate_complexity_scaling"]


_METHOD_LABELS = {
    "log_linear_pool": "log-linear pool",
    "robust_aggregate": "robust aggregate",
    "variational_aggregate": "variational aggregate",
    "share_round_naive": "self-excluding sharing",
    "share_round_robust": "robust self-excluding sharing",
    "infer_states": "state inference",
}
_METHOD_ROLES = {
    "log_linear_pool": "naive",
    "robust_aggregate": "heuristic_robust",
    "variational_aggregate": "variational",
    "share_round_naive": "operating_point_1",
    "share_round_robust": "operating_point_2",
    "infer_states": "operating_point_3",
}
_METHOD_DIRECT_LABELS = {
    "log_linear_pool": "log pool",
    "robust_aggregate": "heuristic robust",
    "variational_aggregate": "variational",
    "share_round_naive": "sharing: reference",
    "share_round_robust": "sharing: robust",
    "infer_states": "state inference",
}


def _measurement(
    report: Mapping[str, object],
    *,
    method: str,
    axis: str,
) -> Mapping[str, object]:
    """Return one measurement row or raise a loud report-shape error."""
    raw = report.get("measurements")
    if not isinstance(raw, list):
        raise ValueError("complexity report must contain a measurement list")
    for row in raw:
        if isinstance(row, Mapping) and row.get("method") == method and row.get("axis") == axis:
            return row
    raise ValueError(f"complexity report is missing measurement {method}/{axis}")


def _numbers(row: Mapping[str, object], key: str) -> np.ndarray:
    """Read a finite one-dimensional numeric measurement field."""
    raw = row.get(key)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"complexity measurement field {key!r} must be a sequence")
    values: list[float] = []
    for value in raw:
        if not isinstance(value, (int, float, str)):
            raise ValueError(f"complexity measurement field {key!r} contains a non-number")
        values.append(float(value))
    values_array = np.asarray(values, dtype=np.float64)
    if values_array.ndim != 1 or values_array.size < 2 or not np.all(np.isfinite(values_array)):
        raise ValueError(f"complexity measurement field {key!r} must contain finite values")
    return values_array


def _plot_measurements(
    ax: "plt.Axes",
    rows: Sequence[Mapping[str, object]],
    *,
    xlabel: str,
    title: str,
) -> None:
    """Draw median/error-bar observations and normalized reference slopes."""
    for row in rows:
        method = str(row["method"])
        sizes = _numbers(row, "sizes")
        medians = _numbers(row, "median_seconds")
        minima = _numbers(row, "min_seconds")
        maxima = _numbers(row, "max_seconds")
        expected_raw = row["expected_exponent"]
        if not isinstance(expected_raw, (int, float, str)):
            raise ValueError(f"expected exponent is not numeric for {method}")
        expected = float(expected_raw)
        if not (sizes.size == medians.size == minima.size == maxima.size):
            raise ValueError(f"complexity measurement arrays have inconsistent lengths for {method}")
        if np.any(sizes <= 0.0) or np.any(medians <= 0.0) or np.any(minima <= 0.0):
            raise ValueError(f"complexity measurement values must be positive for {method}")
        style = semantic_style(_METHOD_ROLES.get(method, "reference"))
        label = _METHOD_LABELS.get(method, method)
        yerr = np.vstack((medians - minima, maxima - medians))
        ax.errorbar(
            sizes,
            medians,
            yerr=yerr,
            color=style.color,
            marker=style.marker,
            markerfacecolor="white" if method != "log_linear_pool" else style.color,
            markeredgecolor=style.keyline,
            linestyle=style.dash,
            linewidth=style.linewidth,
            markersize=5,
            capsize=2.5,
            label=label,
            zorder=3,
        )
        reference = medians[0] * (sizes / sizes[0]) ** expected
        rule = semantic_style("reference_rule")
        ax.plot(
            sizes,
            reference,
            color=rule.color,
            linewidth=rule.linewidth,
            linestyle=rule.dash,
            alpha=0.65,
        )
        ax.annotate(
            _METHOD_DIRECT_LABELS.get(method, label),
            xy=(float(sizes[-1]), float(medians[-1])),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=MIN_QUANTITATIVE_FONT_SIZE,
            color=style.keyline,
            ha="left",
            va="center",
            clip_on=False,
        )
        slope_raw = row["observed_log_log_slope"]
        if not isinstance(slope_raw, (int, float, str)):
            raise ValueError(f"observed complexity slope is not numeric for {method}")
        slope = float(slope_raw)
        if not np.isfinite(slope):
            raise ValueError(f"observed complexity slope is non-finite for {method}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Median wall time (s)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=MIN_QUANTITATIVE_FONT_SIZE, loc="best")
    x_values = np.concatenate([_numbers(row, "sizes") for row in rows])
    ax.set_xlim(float(x_values.min()) / 1.08, float(x_values.max()) * 1.65)


def generate_complexity_scaling(
    report: Mapping[str, object],
    *,
    project_root: str | Path | None = None,
    filename: str = "complexity_scaling.png",
) -> Path:
    """Generate the four-panel complexity/scaling diagnostic figure.

    Args:
        report: Validated ``complexity_scaling.json`` payload.
        project_root: Project root; defaults to the inferred repository root.
        filename: PNG filename under ``output/figures``.

    Returns:
        Path to the written PNG (with a vector PDF companion).
    """
    analytic = report.get("analytic_specs")
    benchmark = report.get("benchmark")
    if not isinstance(analytic, list) or not isinstance(benchmark, Mapping):
        raise ValueError("complexity report must contain analytic_specs and benchmark")
    agent_rows = [
        _measurement(report, method=method, axis="agents")
        for method in ("log_linear_pool", "robust_aggregate", "variational_aggregate")
    ]
    sharing_rows = [
        _measurement(report, method=method, axis="agents")
        for method in ("share_round_naive", "share_round_robust")
    ]
    state_rows = [
        _measurement(report, method=method, axis="states")
        for method in ("log_linear_pool", "robust_aggregate", "variational_aggregate")
    ]
    modality_row = _measurement(report, method="infer_states", axis="modalities")

    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 8.6), facecolor="white")
    _plot_measurements(
        axes[0, 0],
        agent_rows,
        xlabel="Number of agents $N$",
        title="Aggregation scaling with agents",
    )
    _plot_measurements(
        axes[0, 1],
        sharing_rows,
        xlabel="Number of agents $N$",
        title="Self-excluding sharing round",
    )
    _plot_measurements(
        axes[1, 0],
        state_rows,
        xlabel="Number of states $S$",
        title="Aggregation scaling with states",
    )
    _plot_measurements(
        axes[1, 1],
        [modality_row],
        xlabel="Number of modalities $M$",
        title="State inference scaling",
    )

    repeats = benchmark.get("repeats", "N/A")
    seed = report.get("seed", "N/A")
    fig.suptitle(
        "Implementation complexity and machine scaling",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        0.005,
        (
            f"Seeded inputs; medians over {repeats} repeats (min–max bars); slope "
            f"references are normalized Θ-order guides; seed={seed}"
        ),
        ha="center",
        va="bottom",
        fontsize=9.5,
        color=COLOR_DARK,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.91, wspace=0.25, hspace=0.30)
    out = figures_dir(Path(project_root) if project_root is not None else None)
    path = save_figure(fig, out / filename)
    complexity_presentation(path, dict(report))
    return path
