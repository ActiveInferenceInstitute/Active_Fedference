"""Exact-value accessibility fallbacks remain bound to plotted reports."""

from __future__ import annotations

import json
from copy import deepcopy
from fractions import Fraction
from itertools import permutations
from pathlib import Path

import pytest

from analysis.figure_exact_values import (
    EXACT_VALUE_GENERATORS,
    EXACT_VALUE_REPORT_BINDINGS,
    _bnn_table,
    _cross_study_table,
    _review_grid_table,
    build_figure_exact_values,
    conditional_display_summaries,
    render_figure_exact_values_markdown,
)
from analysis.report_schemas import ReportSchemaError, validate_report
from figures._metadata import FIGURE_METADATA

_ROOT = Path(__file__).resolve().parents[2]
_REPORT_NAMES = {
    "belief_quality",
    "bnn_robustness",
    "complexity_scaling",
    "conditional_world",
    "cross_study_summary",
    "robustness_review_grid",
    "sensitivity",
}


def _source_reports() -> dict[str, dict[str, object]]:
    return {
        name: json.loads((_ROOT / "output" / "reports" / f"{name}.json").read_text(encoding="utf-8"))
        for name in _REPORT_NAMES
    }


def test_exact_value_inventory_matches_figure_metadata() -> None:
    assert tuple(generator for generator, _report in EXACT_VALUE_REPORT_BINDINGS) == (EXACT_VALUE_GENERATORS)
    declared = {
        generator: metadata["exact_value_fallback"]
        for generator, metadata in FIGURE_METADATA.items()
        if "exact_value_fallback" in metadata
    }
    assert declared == {
        generator: f"fig-values:{generator.replace('_', '-')}" for generator in EXACT_VALUE_GENERATORS
    }


def test_exact_values_are_independent_of_report_mapping_insertion_order() -> None:
    reports = _source_reports()
    reordered = deepcopy(reports)
    for report_name, mapping_name in (
        ("bnn_robustness", "accuracy_by_config"),
        ("bnn_robustness", "accuracy_ci_by_config"),
        ("conditional_world", "by_scenario"),
    ):
        mapping = reordered[report_name][mapping_name]
        assert isinstance(mapping, dict)
        reordered[report_name][mapping_name] = dict(reversed(mapping.items()))

    review = reordered["robustness_review_grid"]
    conditional = review["conditional_world"]
    assert isinstance(conditional, dict)
    cells = conditional["by_scenario"]
    assert isinstance(cells, dict)
    conditional["by_scenario"] = dict(reversed(cells.items()))
    statistics = review["statistics"]
    assert isinstance(statistics, dict)
    by_mechanism = statistics["by_mechanism"]
    assert isinstance(by_mechanism, dict)
    for mechanism in by_mechanism.values():
        assert isinstance(mechanism, dict)
        by_rate = mechanism["by_rate"]
        assert isinstance(by_rate, dict)
        mechanism["by_rate"] = dict(reversed(by_rate.items()))

    expected = build_figure_exact_values(reports)
    reordered_payload = build_figure_exact_values(reordered)
    round_tripped = json.loads(json.dumps(reordered, sort_keys=True))
    assert reordered_payload == expected
    assert build_figure_exact_values(round_tripped) == expected
    assert render_figure_exact_values_markdown(reordered_payload) == (
        render_figure_exact_values_markdown(expected)
    )
    for table in expected["tables"]:
        assert len(table["columns"]) == len(set(table["columns"]))


@pytest.mark.parametrize("values", list(permutations((0.5, 2.0**-54, -0.5))))
def test_display_mean_preserves_small_residual_under_cancellation(
    values: tuple[float, ...],
) -> None:
    """A rational reference detects version-dependent float-sum cancellation."""
    cells = {
        f"cell-{index}": {
            "attack": "directional",
            "adversary_weight": 0.5,
            "contrast_mean": value,
        }
        for index, value in enumerate(values)
    }
    original = deepcopy(cells)
    exact_mean = float(sum(Fraction.from_float(value) for value in values) / len(values))

    summaries = conditional_display_summaries(cells)

    assert summaries == [
        {
            "attack": "directional",
            "adversary_weight": 0.5,
            "mean_contrast": exact_mean,
            "half_min_max_span": 0.5,
            "cell_count": 3,
        }
    ]
    assert summaries[0]["mean_contrast"] > 0.0
    assert cells == original


def test_review_grid_fallback_reuses_displayed_group_mean_and_half_span() -> None:
    report = _source_reports()["robustness_review_grid"]
    conditional = report["conditional_world"]
    assert isinstance(conditional, dict)
    cells = conditional["by_scenario"]
    assert isinstance(cells, dict)
    renderer_rows = conditional_display_summaries(cells)
    table = _review_grid_table(report)
    fallback_rows = [row for row in table["rows"] if row["row_kind"] == "conditional_display_summary"]

    assert fallback_rows == [
        {
            "row_kind": "conditional_display_summary",
            "mechanism": row["attack"],
            "rate": None,
            "preset": None,
            "scenario_id": None,
            "adversary_weight": row["adversary_weight"],
            "mean_contrast": row["mean_contrast"],
            "half_min_max_span": row["half_min_max_span"],
            "group_cell_count": row["cell_count"],
            "ci_lo": None,
            "ci_hi": None,
        }
        for row in renderer_rows
    ]


def test_cross_study_fallback_discloses_study_four_display_selection() -> None:
    table = _cross_study_table(_source_reports()["cross_study_summary"])
    assert "Study 4 is a within-run display-selected maximum" in table["note"]
    study_four = next(row for row in table["rows"] if row["study"] == 4)
    assert study_four["label"]


def test_bnn_fallback_names_the_composite_configuration_identification_boundary() -> None:
    table = _bnn_table(
        {
            "contamination_levels": [0.0],
            "accuracy_by_config": {
                "nll / L2=0.05 (standard proxy)": [0.8],
                "rcce / L2=0.10 (exploratory proxy)": [0.82],
            },
            "accuracy_ci_by_config": {
                "nll / L2=0.05 (standard proxy)": [[0.75, 0.85]],
                "rcce / L2=0.10 (exploratory proxy)": [[0.77, 0.87]],
            },
        }
    )

    assert "joint NLL/L2=0.05 and RCCE/L2=0.10 configurations" in table["note"]
    assert "cannot identify an RCCE-only effect" in table["note"]


@pytest.mark.publication
def test_exact_values_are_rebuilt_from_the_reports_used_by_figures() -> None:
    expected = build_figure_exact_values(_source_reports())
    actual = json.loads(
        (_ROOT / "output" / "figures" / "figure_exact_values.json").read_text(encoding="utf-8")
    )
    assert actual == expected
    validate_report("figure_exact_values", actual)


@pytest.mark.publication
def test_exact_value_markdown_is_deterministic_and_identifier_addressable() -> None:
    payload = build_figure_exact_values(_source_reports())
    expected = render_figure_exact_values_markdown(payload)
    actual = (_ROOT / "output" / "figures" / "figure_exact_values.md").read_text(encoding="utf-8")
    assert actual == expected
    assert "## Composite two-configuration logistic-regression proxy {#fig-values-bnn-robustness}" in actual
    for table in payload["tables"]:
        assert f"{{#{table['identifier'].replace(':', '-')}}}" in actual


@pytest.mark.publication
def test_exact_value_table_tamper_is_rejected() -> None:
    payload = build_figure_exact_values(_source_reports())
    damaged = deepcopy(payload)
    del damaged["tables"][0]["rows"][0][damaged["tables"][0]["columns"][0]]
    with pytest.raises(ReportSchemaError, match="does not match columns"):
        validate_report("figure_exact_values", damaged)


@pytest.mark.publication
def test_exact_value_inventory_rejects_missing_or_duplicate_tables() -> None:
    payload = build_figure_exact_values(_source_reports())
    missing = deepcopy(payload)
    missing["tables"].pop()
    with pytest.raises(ReportSchemaError, match="generator inventory mismatch"):
        validate_report("figure_exact_values", missing)

    duplicated = deepcopy(payload)
    duplicated["tables"].append(deepcopy(duplicated["tables"][-1]))
    with pytest.raises(ReportSchemaError, match="sorted and unique"):
        validate_report("figure_exact_values", duplicated)

    wrong_source = deepcopy(payload)
    wrong_source["tables"][0]["source_report"] = "unrelated.json"
    with pytest.raises(ReportSchemaError, match="source_report mismatch"):
        validate_report("figure_exact_values", wrong_source)
