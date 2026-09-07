"""Source-generated exact-value fallbacks for visually encoded figures."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TypedDict

EXACT_VALUE_REPORT_BINDINGS: tuple[tuple[str, str], ...] = (
    ("belief_quality", "belief_quality"),
    ("bnn_robustness", "bnn_robustness"),
    ("complexity_scaling", "complexity_scaling"),
    ("conditional_world", "conditional_world"),
    ("cross_study_summary", "cross_study_summary"),
    ("robustness_review_grid", "robustness_review_grid"),
    ("sensitivity_heatmap", "sensitivity"),
)
EXACT_VALUE_GENERATORS: tuple[str, ...] = tuple(
    generator for generator, _report in EXACT_VALUE_REPORT_BINDINGS
)
_EXACT_VALUE_DISPLAY_TITLES = {
    "bnn_robustness": "Composite two-configuration logistic-regression proxy",
}


class ExactValueTable(TypedDict):
    """One identifier-addressable table derived from a plotted report."""

    identifier: str
    generator: str
    source_report: str
    columns: list[str]
    rows: list[dict[str, object]]
    note: str


class FigureExactValuesPayload(TypedDict):
    """Complete accessibility fallback artifact for the declared figure set."""

    schema_version: str
    generated_by: str
    tables: list[ExactValueTable]


class ConditionalDisplaySummary(TypedDict):
    """One grouped descriptive cell shared by a figure and its fallback."""

    attack: str
    adversary_weight: float
    mean_contrast: float
    half_min_max_span: float
    cell_count: int


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a sequence")
    return value


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field} must be a finite number")
    return numeric


def conditional_display_summaries(
    cells_raw: Mapping[str, object],
) -> list[ConditionalDisplaySummary]:
    """Return the review grid's grouped means and half min/max spans.

    The source-owned helper is consumed by both the plotted annotation and its
    exact-value fallback. It is a finite descriptive grouping over registered
    cells, not an uncertainty interval or a new statistical estimand.
    Explicit accurate summation keeps the display independent of the built-in
    float-summation algorithm, which changed in Python 3.12.
    """

    cells = [
        _mapping(cell, field=f"conditional cell {scenario_id!r}")
        for scenario_id, cell in sorted(cells_raw.items())
    ]
    attacks = sorted({str(cell["attack"]) for cell in cells})
    summaries: list[ConditionalDisplaySummary] = []
    for attack in attacks:
        for weight in (0.5, 1.0):
            values = [
                _number(cell["contrast_mean"], field="conditional contrast mean")
                for cell in cells
                if str(cell["attack"]) == attack
                and _number(
                    cell["adversary_weight"],
                    field="conditional adversary weight",
                )
                == weight
            ]
            if values:
                summaries.append(
                    {
                        "attack": attack,
                        "adversary_weight": weight,
                        "mean_contrast": math.fsum(values) / len(values),
                        "half_min_max_span": (max(values) - min(values)) / 2.0,
                        "cell_count": len(values),
                    }
                )
    return summaries


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _number_key(value: object, *, field: str) -> float:
    """Parse a JSON object key that canonically represents a finite number."""
    if isinstance(value, str):
        try:
            numeric = float(value)
        except ValueError as exc:
            raise ValueError(f"{field} must be a finite numeric key") from exc
        if not math.isfinite(numeric):
            raise ValueError(f"{field} must be a finite numeric key")
        return numeric
    return _number(value, field=field)


def _interval(value: object, *, field: str) -> tuple[float, float]:
    sequence = _sequence(value, field=field)
    if len(sequence) != 2:
        raise ValueError(f"{field} must contain two endpoints")
    return (
        _number(sequence[0], field=f"{field}[0]"),
        _number(sequence[1], field=f"{field}[1]"),
    )


def _belief_quality_table(report: Mapping[str, object]) -> ExactValueTable:
    controls = _mapping(report.get("control_scores"), field="belief_quality.control_scores")
    rows: list[dict[str, object]] = []
    for name in ("oracle", "uniform", "confident_wrong"):
        control = _mapping(controls.get(name), field=f"belief_quality.control_scores.{name}")
        ci_lo, ci_hi = _interval(control.get("log_score_ci"), field=f"{name}.log_score_ci")
        rows.append(
            {
                "series": name,
                "row_kind": "log_score_summary",
                "bin_center": None,
                "mean_confidence": None,
                "empirical_accuracy": None,
                "mean_log_score_nats": _number(
                    control["mean_log_score"],
                    field=f"{name}.mean_log_score",
                ),
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
            }
        )
        reliability = _mapping(control.get("reliability"), field=f"{name}.reliability")
        bin_centers = _sequence(reliability.get("bin_center"), field=f"{name}.bin_center")
        confidences = _sequence(
            reliability.get("mean_confidence"),
            field=f"{name}.mean_confidence",
        )
        accuracies = _sequence(reliability.get("accuracy"), field=f"{name}.accuracy")
        if not (len(bin_centers) == len(confidences) == len(accuracies)):
            raise ValueError(f"{name} reliability arrays have inconsistent lengths")
        for center, confidence, accuracy in zip(bin_centers, confidences, accuracies, strict=True):
            if confidence is None or accuracy is None:
                continue
            rows.append(
                {
                    "series": name,
                    "row_kind": "reliability_bin",
                    "bin_center": _number(center, field=f"{name}.bin_center"),
                    "mean_confidence": _number(confidence, field=f"{name}.mean_confidence"),
                    "empirical_accuracy": _number(accuracy, field=f"{name}.accuracy"),
                    "mean_log_score_nats": None,
                    "ci_lo": None,
                    "ci_hi": None,
                }
            )
    return {
        "identifier": "fig-values:belief-quality",
        "generator": "belief_quality",
        "source_report": "belief_quality.json",
        "columns": [
            "series",
            "row_kind",
            "bin_center",
            "mean_confidence",
            "empirical_accuracy",
            "mean_log_score_nats",
            "ci_lo",
            "ci_hi",
        ],
        "rows": rows,
        "note": "Log-score intervals resample configured seeds; empty reliability bins are omitted.",
    }


def _bnn_table(report: Mapping[str, object]) -> ExactValueTable:
    levels = _sequence(report.get("contamination_levels"), field="bnn.contamination_levels")
    curves = _mapping(report.get("accuracy_by_config"), field="bnn.accuracy_by_config")
    intervals = _mapping(report.get("accuracy_ci_by_config"), field="bnn.accuracy_ci_by_config")
    rows: list[dict[str, object]] = []
    for config, raw_curve in sorted(curves.items()):
        curve = _sequence(raw_curve, field=f"bnn.accuracy_by_config.{config}")
        raw_intervals = _sequence(intervals.get(config), field=f"bnn.accuracy_ci_by_config.{config}")
        if not (len(levels) == len(curve) == len(raw_intervals)):
            raise ValueError(f"BNN curve {config!r} does not match the contamination axis")
        for level, accuracy, interval in zip(levels, curve, raw_intervals, strict=True):
            ci_lo, ci_hi = _interval(interval, field=f"bnn interval {config!r}/{level}")
            rows.append(
                {
                    "configuration": str(config),
                    "contamination_fraction": _number(level, field="bnn.contamination_level"),
                    "held_out_accuracy": _number(accuracy, field=f"bnn.accuracy.{config}"),
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                }
            )
    return {
        "identifier": "fig-values:bnn-robustness",
        "generator": "bnn_robustness",
        "source_report": "bnn_robustness.json",
        "columns": ["configuration", "contamination_fraction", "held_out_accuracy", "ci_lo", "ci_hi"],
        "rows": rows,
        "note": (
            "Exploratory point-estimate proxy comparing joint NLL/L2=0.05 and RCCE/L2=0.10 "
            "configurations; because loss and shrinkage change together, the contrast cannot "
            "identify an RCCE-only effect. Intervals resample synthetic-data seeds."
        ),
    }


def _complexity_table(report: Mapping[str, object]) -> ExactValueTable:
    measurements = _sequence(report.get("measurements"), field="complexity.measurements")
    rows: list[dict[str, object]] = []
    for raw_measurement in measurements:
        measurement = _mapping(raw_measurement, field="complexity.measurement")
        sizes = _sequence(measurement.get("sizes"), field="complexity.sizes")
        medians = _sequence(measurement.get("median_seconds"), field="complexity.median_seconds")
        minima = _sequence(measurement.get("min_seconds"), field="complexity.min_seconds")
        maxima = _sequence(measurement.get("max_seconds"), field="complexity.max_seconds")
        if not (len(sizes) == len(medians) == len(minima) == len(maxima)):
            raise ValueError("complexity measurement arrays have inconsistent lengths")
        for size, median, minimum, maximum in zip(sizes, medians, minima, maxima, strict=True):
            rows.append(
                {
                    "method": str(measurement["method"]),
                    "axis": str(measurement["axis"]),
                    "size": _number(size, field="complexity.size"),
                    "median_seconds": _number(median, field="complexity.median_seconds"),
                    "min_seconds": _number(minimum, field="complexity.min_seconds"),
                    "max_seconds": _number(maximum, field="complexity.max_seconds"),
                    "expected_exponent": _number(
                        measurement["expected_exponent"],
                        field="complexity.expected_exponent",
                    ),
                    "observed_log_log_slope": _number(
                        measurement["observed_log_log_slope"],
                        field="complexity.observed_log_log_slope",
                    ),
                }
            )
    return {
        "identifier": "fig-values:complexity-scaling",
        "generator": "complexity_scaling",
        "source_report": "complexity_scaling.json",
        "columns": [
            "method",
            "axis",
            "size",
            "median_seconds",
            "min_seconds",
            "max_seconds",
            "expected_exponent",
            "observed_log_log_slope",
        ],
        "rows": rows,
        "note": "Min-max spans are timing-repeat ranges, not confidence intervals.",
    }


def _conditional_table(report: Mapping[str, object]) -> ExactValueTable:
    scenarios = _mapping(report.get("by_scenario"), field="conditional.by_scenario")
    rows: list[dict[str, object]] = []
    for scenario_id, raw_cell in sorted(scenarios.items()):
        cell = _mapping(raw_cell, field=f"conditional.by_scenario.{scenario_id}")
        ci_lo, ci_hi = _interval(cell.get("contrast_ci"), field=f"{scenario_id}.contrast_ci")
        rows.append(
            {
                "scenario_id": str(scenario_id),
                "attack": str(cell["attack"]),
                "true_state": _integer(cell["true_state"], field=f"{scenario_id}.true_state"),
                "target_state": _integer(cell["target_state"], field=f"{scenario_id}.target_state"),
                "observability": _number(cell["observability"], field=f"{scenario_id}.observability"),
                "adversary_weight": _number(
                    cell["adversary_weight"],
                    field=f"{scenario_id}.adversary_weight",
                ),
                "contrast_mean": _number(cell["contrast_mean"], field=f"{scenario_id}.contrast_mean"),
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
            }
        )
    return {
        "identifier": "fig-values:conditional-world",
        "generator": "conditional_world",
        "source_report": "conditional_world.json",
        "columns": [
            "scenario_id",
            "attack",
            "true_state",
            "target_state",
            "observability",
            "adversary_weight",
            "contrast_mean",
            "ci_lo",
            "ci_hi",
        ],
        "rows": rows,
        "note": (
            "Contrast is robust-minus-reference true-state mass; the seeded world/scenario row is the "
            "independent unit and trials are nested within that row."
        ),
    }


def _review_grid_table(report: Mapping[str, object]) -> ExactValueTable:
    conditional = _mapping(report.get("conditional_world"), field="review.conditional_world")
    cells = _mapping(conditional.get("by_scenario"), field="review.conditional_world.by_scenario")
    rows: list[dict[str, object]] = []
    for scenario_id, raw_cell in sorted(cells.items()):
        cell = _mapping(raw_cell, field=f"review cell {scenario_id}")
        rows.append(
            {
                "row_kind": "conditional_cell",
                "mechanism": str(cell["attack"]),
                "rate": None,
                "preset": None,
                "scenario_id": str(scenario_id),
                "adversary_weight": _number(
                    cell["adversary_weight"],
                    field=f"review.{scenario_id}.adversary_weight",
                ),
                "mean_contrast": _number(
                    cell["contrast_mean"],
                    field=f"review.{scenario_id}.contrast_mean",
                ),
                "half_min_max_span": None,
                "group_cell_count": None,
                "ci_lo": None,
                "ci_hi": None,
            }
        )
    for display_summary in conditional_display_summaries(cells):
        rows.append(
            {
                "row_kind": "conditional_display_summary",
                "mechanism": display_summary["attack"],
                "rate": None,
                "preset": None,
                "scenario_id": None,
                "adversary_weight": display_summary["adversary_weight"],
                "mean_contrast": display_summary["mean_contrast"],
                "half_min_max_span": display_summary["half_min_max_span"],
                "group_cell_count": display_summary["cell_count"],
                "ci_lo": None,
                "ci_hi": None,
            }
        )
    statistics = _mapping(report.get("statistics"), field="review.statistics")
    by_mechanism = _mapping(statistics.get("by_mechanism"), field="review.statistics.by_mechanism")
    directional = _sequence(report.get("directional_mechanisms"), field="review.directional_mechanisms")
    presets = [str(value) for value in _sequence(report.get("divergences"), field="review.divergences")]
    for mechanism in directional:
        mechanism_name = str(mechanism)
        mechanism_row = _mapping(
            by_mechanism.get(mechanism_name),
            field=f"review mechanism {mechanism_name}",
        )
        by_rate = _mapping(mechanism_row.get("by_rate"), field=f"review mechanism {mechanism}.by_rate")
        for rate_key, raw_rate in sorted(
            by_rate.items(),
            key=lambda item: _number_key(
                item[0],
                field=f"review rate {mechanism_name}",
            ),
        ):
            rate_row = _mapping(raw_rate, field=f"review rate {mechanism}/{rate_key}")
            methods = _mapping(rate_row.get("methods"), field=f"review methods {mechanism}/{rate_key}")
            for preset in presets:
                if preset == "KLD":
                    continue
                method = _mapping(methods.get(preset), field=f"review preset {preset}")
                method_summary = _mapping(
                    method.get("summary"),
                    field=f"review summary {preset}",
                )
                ci_lo, ci_hi = _interval(method.get("contrast_ci"), field=f"review CI {preset}")
                rows.append(
                    {
                        "row_kind": "rate_profile",
                        "mechanism": str(mechanism),
                        "rate": _number_key(
                            rate_key,
                            field=f"review rate {mechanism_name}",
                        ),
                        "preset": preset,
                        "scenario_id": None,
                        "adversary_weight": None,
                        "mean_contrast": _number(
                            method_summary["mean"],
                            field=f"review summary {mechanism_name}/{rate_key}/{preset}",
                        ),
                        "half_min_max_span": None,
                        "group_cell_count": None,
                        "ci_lo": ci_lo,
                        "ci_hi": ci_hi,
                    }
                )
    return {
        "identifier": "fig-values:robustness-review-grid",
        "generator": "robustness_review_grid",
        "source_report": "robustness_review_grid.json",
        "columns": [
            "row_kind",
            "mechanism",
            "rate",
            "preset",
            "scenario_id",
            "adversary_weight",
            "mean_contrast",
            "half_min_max_span",
            "group_cell_count",
            "ci_lo",
            "ci_hi",
        ],
        "rows": rows,
        "note": (
            "Conditional display-summary rows reproduce the heatmap's grouped mean and half min-max "
            "span; all predeclared preset-rate rows are retained without result-dependent selection."
        ),
    }


def _sensitivity_table(report: Mapping[str, object]) -> ExactValueTable:
    acuity = _sequence(report.get("acuity_values"), field="sensitivity.acuity_values")
    agents = _sequence(report.get("n_agents_values"), field="sensitivity.n_agents_values")
    rows: list[dict[str, object]] = []
    for panel in ("belief_sharing", "hierarchical"):
        result = _mapping(report.get(panel), field=f"sensitivity.{panel}")
        grid = _sequence(result.get("accuracy_gap_grid"), field=f"sensitivity.{panel}.accuracy_gap_grid")
        for row_index, raw_row in enumerate(grid):
            values = _sequence(raw_row, field=f"sensitivity.{panel}.row[{row_index}]")
            for column_index, value in enumerate(values):
                rows.append(
                    {
                        "panel": panel,
                        "acuity": _number(acuity[row_index], field="sensitivity.acuity"),
                        "n_agents": _integer(agents[column_index], field="sensitivity.n_agents"),
                        "accuracy_gap": _number(value, field=f"sensitivity.{panel}.accuracy_gap"),
                        "inside_display_band": abs(_number(value, field=f"sensitivity.{panel}.accuracy_gap"))
                        <= _number(report["noise_floor"], field="sensitivity.noise_floor"),
                    }
                )
    return {
        "identifier": "fig-values:sensitivity-heatmap",
        "generator": "sensitivity_heatmap",
        "source_report": "sensitivity.json",
        "columns": ["panel", "acuity", "n_agents", "accuracy_gap", "inside_display_band"],
        "rows": rows,
        "note": (
            "Display-band membership is not a confidence interval, significance test, or zero-effect claim."
        ),
    }


def _cross_study_table(report: Mapping[str, object]) -> ExactValueTable:
    studies = _sequence(report.get("studies"), field="cross_study.studies")
    rows: list[dict[str, object]] = []
    for raw_study in studies:
        study = _mapping(raw_study, field="cross_study.study")
        rows.append(
            {
                "study": _integer(study["study"], field="cross_study.study"),
                "label": str(study["label"]),
                "metric": str(study["metric"]),
                "unit": str(study["unit"]),
                "mean": _number(study["mean"], field="cross_study.mean"),
                "ci_lo": _number(study["ci_lo"], field="cross_study.ci_lo"),
                "ci_hi": _number(study["ci_hi"], field="cross_study.ci_hi"),
            }
        )
    return {
        "identifier": "fig-values:cross-study-summary",
        "generator": "cross_study_summary",
        "source_report": "cross_study_summary.json",
        "columns": ["study", "label", "metric", "unit", "mean", "ci_lo", "ci_hi"],
        "rows": rows,
        "note": (
            "Native units remain separate; intervals come from the harmonized seed-level rerun. "
            "Study 4 is a within-run display-selected maximum across non-reference server presets "
            "at the declared worst rate, not a preselected method or inferential winner."
        ),
    }


def build_figure_exact_values(
    reports: Mapping[str, Mapping[str, object]],
) -> FigureExactValuesPayload:
    """Build every declared exact-value fallback from the reports used to plot it."""
    report_by_generator = dict(EXACT_VALUE_REPORT_BINDINGS)
    required_reports = set(report_by_generator.values())
    if set(reports) != required_reports:
        raise ValueError(
            "exact-value report inventory mismatch: "
            f"missing={sorted(required_reports - set(reports))}, "
            f"extra={sorted(set(reports) - required_reports)}"
        )
    tables = [
        _belief_quality_table(reports["belief_quality"]),
        _bnn_table(reports["bnn_robustness"]),
        _complexity_table(reports["complexity_scaling"]),
        _conditional_table(reports["conditional_world"]),
        _cross_study_table(reports["cross_study_summary"]),
        _review_grid_table(reports["robustness_review_grid"]),
        _sensitivity_table(reports["sensitivity"]),
    ]
    return {
        "schema_version": "1.0",
        "generated_by": "analysis.figure_exact_values.build_figure_exact_values",
        "tables": sorted(tables, key=lambda table: table["identifier"]),
    }


def render_figure_exact_values_markdown(payload: FigureExactValuesPayload) -> str:
    """Render deterministic, human-readable Markdown tables from exact values."""
    lines = [
        "# Exact values for accessibility fallbacks",
        "",
        "These tables are generated from the same typed reports consumed by the figures. "
        "They preserve numerical access without changing the caption or claim boundary.",
        "",
    ]
    for table in payload["tables"]:
        anchor = table["identifier"].replace(":", "-")
        title = _EXACT_VALUE_DISPLAY_TITLES.get(
            table["generator"],
            table["generator"].replace("_", " ").title(),
        )
        lines.extend(
            [
                f"## {title} {{#{anchor}}}",
                "",
                f"Source report: `{table['source_report']}`. {table['note']}",
                "",
                "| " + " | ".join(table["columns"]) + " |",
                "| " + " | ".join("---" for _ in table["columns"]) + " |",
            ]
        )
        for row in table["rows"]:
            cells = [_format_markdown_value(row.get(column)) for column in table["columns"]]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def _format_markdown_value(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


__all__ = [
    "EXACT_VALUE_GENERATORS",
    "EXACT_VALUE_REPORT_BINDINGS",
    "ExactValueTable",
    "FigureExactValuesPayload",
    "build_figure_exact_values",
    "render_figure_exact_values_markdown",
]
