"""Visualization for the expanded, source-bound robustness review grid."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from matplotlib.colors import to_hex

from ._common import (
    COLOR_MUTED,
    MIN_QUANTITATIVE_FONT_SIZE,
    SemanticStyle,
    apply_style,
    contrasting_text_color,
    figures_dir,
    plt,
    save_figure,
    semantic_style,
    signed_difference_colormap,
)

_PRESET_STYLE_ROLES = (
    "heuristic_robust",
    "operating_point_1",
    "operating_point_2",
    "operating_point_3",
    "operating_point_4",
)


def _spread_endpoint_labels(
    values: list[float],
    *,
    minimum_separation: float,
    lower: float,
    upper: float,
) -> list[float]:
    """Return ordered endpoint-label lanes without changing curve endpoints."""
    if not values:
        return []
    order = sorted(range(len(values)), key=values.__getitem__)
    positions = list(values)
    for previous, current in zip(order, order[1:]):
        positions[current] = max(positions[current], positions[previous] + minimum_separation)
    overflow = positions[order[-1]] - upper
    if overflow > 0.0:
        positions = [position - overflow for position in positions]
    underflow = lower - positions[order[0]]
    if underflow > 0.0:
        positions = [position + underflow for position in positions]
    return positions


def _as_mapping(value: object, *, label: str) -> Mapping[str, object]:
    """Return a mapping or raise a named fail-closed figure-contract error."""
    if not isinstance(value, Mapping):
        raise ValueError(f"review-grid {label} must be a mapping")
    return value


def _as_finite_number(value: object, *, label: str) -> float:
    """Return one finite scalar after preserving the figure's public contract."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"review-grid {label} must be a finite number")
    return float(value)


def _as_finite_number_list(value: object, *, label: str) -> list[float]:
    """Return finite numeric rows instead of treating arbitrary objects as arrays."""
    if not isinstance(value, list):
        raise ValueError(f"review-grid {label} must be a list")
    return [_as_finite_number(item, label=f"{label}[{index}]") for index, item in enumerate(value)]


def _as_string_list(value: object, *, label: str) -> list[str]:
    """Return a non-empty string sequence or fail before plotting it."""
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"review-grid {label} must be a list of strings")
    return value


def _as_interval(value: object, *, label: str) -> tuple[float, float]:
    """Return a finite two-endpoint interval for a method-rate curve."""
    values = _as_finite_number_list(value, label=label)
    if len(values) != 2:
        raise ValueError(f"review-grid {label} must contain exactly two endpoints")
    return values[0], values[1]


def generate_robustness_review_grid(
    report: Mapping[str, object],
    *,
    project_root: Path | None = None,
    filename: str = "robustness_review_grid.png",
) -> Path:
    """Render conditional cells and every predeclared directional method curve.

    The rate panels draw one signed robust-minus-KLD curve per configured
    non-KLD method, with its own seed-bootstrap interval.  They deliberately do
    not consume the legacy pooled-display selection fields retained in the onset
    report, so neither visual uncertainty nor inference is post-selection.
    """
    # ``analysis.__init__`` intentionally re-exports its workflow, which in
    # turn imports the public figures barrel.  Resolve the schema validator at
    # call time, after that barrel is initialized, rather than forming an
    # import-time figures -> analysis -> workflow -> figures cycle.
    from analysis.report_schemas import validate_report

    # The figure is a public direct API as well as a workflow consumer. Validate
    # the full nested report here, rather than trusting a workflow-only shallow
    # contract and silently dropping malformed method/CI rows.
    validate_report("robustness_review_grid", report)

    conditional = _as_mapping(report["conditional_world"], label="conditional_world")
    statistics = _as_mapping(report["statistics"], label="statistics")
    cells_raw = _as_mapping(conditional["by_scenario"], label="conditional cells")
    statistics_by_kind = _as_mapping(statistics["by_mechanism"], label="statistics.by_mechanism")
    rates = np.asarray(_as_finite_number_list(report["rates"], label="rates"), dtype=np.float64)
    directional_kinds = _as_string_list(report["directional_mechanisms"], label="directional mechanisms")
    robust_methods = [
        method for method in _as_string_list(report["divergences"], label="divergences") if method != "KLD"
    ]
    if rates.size == 0 or not directional_kinds or not robust_methods:
        raise ValueError("review-grid report has no rate, mechanism, or robust-method rows")

    cells = [
        _as_mapping(cell, label=f"conditional cell {scenario_id!r}")
        for scenario_id, cell in cells_raw.items()
    ]
    attacks = sorted({str(cell["attack"]) for cell in cells})
    weights = (0.5, 1.0)
    matrix = np.full((len(attacks), len(weights)), np.nan, dtype=np.float64)
    spans = np.zeros_like(matrix)
    for row, attack in enumerate(attacks):
        for col, weight in enumerate(weights):
            values = np.asarray(
                [
                    _as_finite_number(cell["contrast_mean"], label="conditional contrast mean")
                    for cell in cells
                    if str(cell["attack"]) == attack
                    and _as_finite_number(cell["adversary_weight"], label="conditional adversary weight")
                    == weight
                ],
                dtype=np.float64,
            )
            if values.size:
                matrix[row, col] = float(values.mean())
                spans[row, col] = float(values.max() - values.min()) / 2.0

    apply_style()
    figure_height = max(6.2, 2.45 * len(directional_kinds))
    fig = plt.figure(figsize=(13.8, figure_height))
    # This layout uses a spanning GridSpec plus colorbar; disable the global
    # autolayout policy so Matplotlib does not attempt an incompatible second
    # tight-layout pass while saving the deterministic PNG/PDF companions.
    fig.set_layout_engine("none")
    grid = fig.add_gridspec(
        len(directional_kinds),
        2,
        width_ratios=(0.92, 1.6),
        left=0.07,
        right=0.985,
        top=0.91,
        bottom=0.10,
        hspace=0.37,
        wspace=0.43,
    )
    heatmap_axis = fig.add_subplot(grid[:, 0])
    rate_axes = [fig.add_subplot(grid[index, 1]) for index in range(len(directional_kinds))]

    vmax = max(float(np.nanmax(np.abs(matrix))), 1e-6)
    image = heatmap_axis.imshow(
        matrix,
        cmap=signed_difference_colormap(),
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
    )
    heatmap_axis.set_xticks(range(len(weights)), ["half adversary\nweight", "full adversary\nweight"])
    heatmap_axis.set_yticks(range(len(attacks)), [attack.replace("_", " ") for attack in attacks])
    heatmap_axis.set_xlabel("Declared conditional cell")
    heatmap_axis.set_ylabel("Attack mechanism")
    heatmap_axis.set_title("A  Conditional preset-minus-reference means", loc="left")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if np.isfinite(matrix[row, col]):
                mapped = np.asarray(image.cmap(image.norm(matrix[row, col])), dtype=np.float64).reshape(4)
                background = to_hex(
                    (
                        float(mapped[0]),
                        float(mapped[1]),
                        float(mapped[2]),
                        float(mapped[3]),
                    )
                )
                heatmap_axis.text(
                    col,
                    row,
                    f"{matrix[row, col]:+.3f}\n±{spans[row, col]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=9.5,
                    color=contrasting_text_color(background),
                    bbox={"facecolor": background, "edgecolor": "none", "pad": 0.0},
                )
    fig.colorbar(
        image,
        ax=heatmap_axis,
        fraction=0.046,
        pad=0.04,
        label="true-state mass gain",
    )
    heatmap_axis.text(
        0.02,
        -0.12,
        "Second line is half the finite-grid min/max span, not a CI.",
        transform=heatmap_axis.transAxes,
        fontsize=MIN_QUANTITATIVE_FONT_SIZE,
        color=COLOR_MUTED,
    )

    for kind_index, (kind, axis) in enumerate(zip(directional_kinds, rate_axes)):
        kind_statistics = _as_mapping(statistics_by_kind[kind], label=f"statistics for {kind!r}")
        by_rate = _as_mapping(kind_statistics["by_rate"], label=f"rate rows for {kind!r}")
        endpoint_rows: list[tuple[float, str, SemanticStyle]] = []
        interval_values: list[float] = [0.0]
        for method_index, method in enumerate(robust_methods):
            means: list[float] = []
            ci_lo: list[float] = []
            ci_hi: list[float] = []
            for rate in rates:
                rate_row = _as_mapping(
                    by_rate[f"{float(rate):g}"],
                    label=f"statistics row {kind!r}/{rate:g}",
                )
                methods = _as_mapping(rate_row["methods"], label=f"method rows {kind!r}/{rate:g}")
                method_row = _as_mapping(methods[method], label=f"method {method!r} at {kind!r}/{rate:g}")
                summary = _as_mapping(
                    method_row["summary"],
                    label=f"summary {method!r} at {kind!r}/{rate:g}",
                )
                ci_lower, ci_upper = _as_interval(
                    method_row["contrast_ci"],
                    label=f"contrast CI {method!r} at {kind!r}/{rate:g}",
                )
                means.append(_as_finite_number(summary["mean"], label=f"mean {method!r}"))
                ci_lo.append(ci_lower)
                ci_hi.append(ci_upper)
            interval_values.extend(ci_lo)
            interval_values.extend(ci_hi)
            style = semantic_style(_PRESET_STYLE_ROLES[method_index % len(_PRESET_STYLE_ROLES)])
            label = f"server preset {method_index + 1}"
            axis.fill_between(
                rates,
                ci_lo,
                ci_hi,
                color=style.color,
                alpha=0.13,
                linewidth=0,
            )
            axis.plot(
                rates,
                means,
                marker=style.marker,
                markerfacecolor="white",
                markeredgecolor=style.keyline,
                linestyle=style.dash,
                linewidth=style.linewidth,
                markersize=5.0,
                color=style.color,
                label=label,
            )
            endpoint_rows.append((float(means[-1]), label, style))
        series_lower = min(interval_values)
        series_upper = max(interval_values)
        series_span = max(series_upper - series_lower, 0.10)
        label_lower = series_lower - 0.04 * series_span
        label_upper = series_upper + 0.04 * series_span
        label_positions = _spread_endpoint_labels(
            [row[0] for row in endpoint_rows],
            minimum_separation=max(0.018, 0.055 * series_span),
            lower=label_lower,
            upper=label_upper,
        )
        label_x = float(rates[-1]) + 0.025
        for (endpoint, label, style), label_y in zip(
            endpoint_rows,
            label_positions,
            strict=True,
        ):
            axis.plot(
                [float(rates[-1]), label_x],
                [endpoint, label_y],
                color=style.keyline,
                linewidth=0.8,
                clip_on=False,
            )
            axis.text(
                label_x,
                label_y,
                label,
                fontsize=MIN_QUANTITATIVE_FONT_SIZE,
                color=style.keyline,
                ha="left",
                va="center",
                clip_on=False,
            )
        axis.set_ylim(label_lower, label_upper)
        reference = semantic_style("reference_rule")
        axis.axhline(
            0.0,
            color=reference.color,
            linestyle=reference.dash,
            linewidth=reference.linewidth,
        )
        panel_letter = chr(ord("B") + kind_index)
        axis.set_title(
            f"{panel_letter}  {kind.replace('_', ' ')}: all predeclared presets",
            loc="left",
        )
        axis.set_ylabel("Preset − reference\ntrue-state mass")
        axis.set_xlim(float(rates.min()), float(rates.max()) + 0.24)
        axis.grid(axis="y", alpha=0.20)
        if kind_index + 1 == len(rate_axes):
            axis.set_xlabel("Contamination rate $\\epsilon$")
        else:
            axis.tick_params(labelbottom=False)
        if kind_index == 0:
            axis.text(
                0.02,
                0.04,
                "bands: 95% seed-bootstrap intervals",
                transform=axis.transAxes,
                fontsize=MIN_QUANTITATIVE_FONT_SIZE,
                color=COLOR_MUTED,
                ha="left",
                va="bottom",
            )

    fig.suptitle(
        "Source-bound robustness review grid: signed, selection-free server contrasts",
        fontweight="bold",
    )
    fig.text(
        0.52,
        0.025,
        "Mixed signs are retained; no universal winning server preset is inferred.",
        ha="center",
        va="bottom",
        fontsize=MIN_QUANTITATIVE_FONT_SIZE,
        color=COLOR_MUTED,
    )
    return save_figure(fig, figures_dir(project_root) / filename)


__all__ = ["generate_robustness_review_grid"]
