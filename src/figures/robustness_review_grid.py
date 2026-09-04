"""Visualization for the expanded, source-bound robustness review grid."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from matplotlib.colors import to_hex

from analysis.figure_exact_values import conditional_display_summaries

from ._common import (
    COLOR_MUTED,
    COLOR_PANEL_BG,
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
    """Return bounded label-lane positions with a guaranteed separation."""
    if not values:
        return []
    if minimum_separation <= 0.0:
        raise ValueError("minimum_separation must be positive")
    if upper <= lower:
        raise ValueError("endpoint-label bounds must be ordered")
    required_span = minimum_separation * (len(values) - 1)
    if required_span > upper - lower + 1e-12:
        raise ValueError("endpoint-label lane is too narrow for the requested separation")
    order = sorted(range(len(values)), key=values.__getitem__)
    ordered = [float(np.clip(values[index], lower, upper)) for index in order]
    for index in range(1, len(ordered)):
        ordered[index] = max(ordered[index], ordered[index - 1] + minimum_separation)
    if ordered[-1] > upper:
        ordered[-1] = upper
        for index in range(len(ordered) - 2, -1, -1):
            ordered[index] = min(ordered[index], ordered[index + 1] - minimum_separation)
    if ordered[0] < lower:
        ordered[0] = lower
        for index in range(1, len(ordered)):
            ordered[index] = max(ordered[index], ordered[index - 1] + minimum_separation)
    positions = [0.0] * len(values)
    for original_index, position in zip(order, ordered, strict=True):
        positions[original_index] = position
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

    display_summaries = conditional_display_summaries(cells_raw)
    attacks = sorted({summary["attack"] for summary in display_summaries})
    weights = (0.5, 1.0)
    matrix = np.full((len(attacks), len(weights)), np.nan, dtype=np.float64)
    spans = np.zeros_like(matrix)
    attack_index = {attack: index for index, attack in enumerate(attacks)}
    weight_index = {weight: index for index, weight in enumerate(weights)}
    for display_summary in display_summaries:
        row = attack_index[display_summary["attack"]]
        col = weight_index[display_summary["adversary_weight"]]
        matrix[row, col] = display_summary["mean_contrast"]
        spans[row, col] = display_summary["half_min_max_span"]

    apply_style()
    figure_height = max(8.6, 3.0 + 2.4 * len(directional_kinds))
    fig = plt.figure(figsize=(8.6, figure_height))
    # Stack the heatmap and rate panels at manuscript width. A side-by-side
    # canvas forced both the interval curves and their endpoint labels below
    # the effective page-scale font floor.
    fig.set_layout_engine("none")
    grid = fig.add_gridspec(
        len(directional_kinds) + 1,
        1,
        height_ratios=(1.18, *(1.0 for _ in directional_kinds)),
        left=0.11,
        right=0.95,
        top=0.90,
        bottom=0.07,
        hspace=0.66,
    )
    heatmap_axis = fig.add_subplot(grid[0, 0])
    rate_axes = [
        fig.add_subplot(grid[index + 1, 0])
        for index in range(len(directional_kinds))
    ]

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
        fraction=0.028,
        pad=0.025,
        label="true-state mass gain",
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
                method_summary = _as_mapping(
                    method_row["summary"],
                    label=f"summary {method!r} at {kind!r}/{rate:g}",
                )
                ci_lower, ci_upper = _as_interval(
                    method_row["contrast_ci"],
                    label=f"contrast CI {method!r} at {kind!r}/{rate:g}",
                )
                means.append(_as_finite_number(method_summary["mean"], label=f"mean {method!r}"))
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
        y_lower = series_lower - 0.12 * series_span
        y_upper = series_upper + 0.12 * series_span
        axis.set_ylim(y_lower, y_upper)
        endpoint_fractions = [
            (endpoint - y_lower) / (y_upper - y_lower)
            for endpoint, _, _ in endpoint_rows
        ]
        lane_separation = min(
            0.16,
            0.72 / max(len(endpoint_rows) - 1, 1),
        )
        label_fractions = _spread_endpoint_labels(
            endpoint_fractions,
            minimum_separation=lane_separation,
            lower=0.11,
            upper=0.89,
        )
        rate_span = max(float(rates[-1] - rates[0]), 0.50)
        elbow_x = float(rates[-1]) + 0.035 * rate_span
        label_x = float(rates[-1]) + 0.105 * rate_span
        label_end_x = float(rates[-1]) + 0.44 * rate_span
        for (endpoint, label, style), label_fraction in zip(
            endpoint_rows,
            label_fractions,
            strict=True,
        ):
            label_y = y_lower + label_fraction * (y_upper - y_lower)
            axis.plot(
                [float(rates[-1]), elbow_x, label_x - 0.012 * rate_span],
                [endpoint, endpoint, label_y],
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
                bbox={"facecolor": COLOR_PANEL_BG, "edgecolor": "none", "pad": 0.08},
            )
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
        axis.set_xlim(float(rates.min()), label_end_x)
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
    return save_figure(
        fig,
        figures_dir(project_root) / filename,
        manuscript_width_fraction=0.98,
    )


__all__ = ["generate_robustness_review_grid"]
