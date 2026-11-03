"""Bnn validation report contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping

from analysis.schemas.primitives import ReportSchemaError, _mapping, _number, _number_list


def _check_bnn_robustness(payload: Mapping[str, object]) -> None:
    """Keep the legacy BNN stem bound to the point-estimate exploratory model."""
    if payload["model_family"] != "point_estimate_logistic_regression":
        raise ReportSchemaError("bnn_robustness model_family must be 'point_estimate_logistic_regression'")
    if payload["study_status"] != "exploratory_conditional_synthetic_sweep":
        raise ReportSchemaError("bnn_robustness study_status must remain exploratory and conditional")
    selection_disclosure = str(payload["selection_disclosure"])
    required_selection_fragments = (
        "peak_margin_contamination is selected within the displayed contamination sweep",
        "composite-configuration held-out accuracy margin",
        "robust_loss_param and n_per are configured inputs, not selected by this report",
    )
    if any(fragment not in selection_disclosure for fragment in required_selection_fragments):
        raise ReportSchemaError(
            "bnn_robustness selection_disclosure must limit selection to peak contamination"
        )
    expected_design_strings = {
        "analysis_unit": "synthetic-data seed within contamination operating point",
        "replication_unit": "synthetic-data seed",
        "interval_method": "percentile bootstrap across synthetic-data seeds",
    }
    for field_name, expected in expected_design_strings.items():
        if payload[field_name] != expected:
            raise ReportSchemaError(f"bnn_robustness {field_name} must remain {expected!r}")
    ci_percent = payload["ci_percent"]
    n_bootstrap = payload["n_bootstrap"]
    n_per = payload["n_per"]
    assert isinstance(ci_percent, int) and not isinstance(ci_percent, bool)
    assert isinstance(n_bootstrap, int) and not isinstance(n_bootstrap, bool)
    assert isinstance(n_per, int) and not isinstance(n_per, bool)
    if ci_percent != 95 or n_bootstrap != 5000:
        raise ReportSchemaError("bnn_robustness intervals must retain the 95-percent, 5000-resample design")
    if n_per < 1:
        raise ReportSchemaError("bnn_robustness n_per must be a positive points-per-class count")
    robust_loss_param = _number(
        payload["robust_loss_param"],
        schema="bnn_robustness",
        field="robust_loss_param",
    )
    if not 0.0 <= robust_loss_param <= 1.0:
        raise ReportSchemaError("bnn_robustness robust_loss_param must lie in [0, 1]")
    boundary = str(payload["claim_boundary"])
    required_boundaries = (
        "no posterior-uncertainty BNN",
        "leakage-free calibration",
        "universal robustness",
        "source-protocol BNN replication",
        "legacy AR argument selects stronger L2",
        "does not evaluate an Alpha-Renyi objective",
    )
    if any(fragment not in boundary for fragment in required_boundaries):
        raise ReportSchemaError("bnn_robustness claim_boundary must retain every declared no-claim")
    levels = _number_list(
        payload["contamination_levels"], schema="bnn_robustness", field="contamination_levels"
    )
    if not levels or len(set(levels)) != len(levels) or any(level < 0.0 or level > 1.0 for level in levels):
        raise ReportSchemaError("bnn_robustness contamination levels must be unique and lie in [0, 1]")
    curves = _mapping(payload["accuracy_by_config"], schema="bnn_robustness", field="accuracy_by_config")
    intervals = _mapping(
        payload["accuracy_ci_by_config"], schema="bnn_robustness", field="accuracy_ci_by_config"
    )
    seed_values = _mapping(
        payload["accuracy_seed_values_by_config"],
        schema="bnn_robustness",
        field="accuracy_seed_values_by_config",
    )
    configuration_boundary = str(payload["configuration_boundary"])
    required_configuration_fragments = (
        "composite two-configuration contrast",
        "changes both client loss and L2 shrinkage",
        "L2 coefficient 0.05",
        "L2 coefficient 0.10",
        "cannot identify an RCCE-only effect",
        "neither branch computes a weight-space divergence",
    )
    if any(fragment not in configuration_boundary for fragment in required_configuration_fragments):
        raise ReportSchemaError("bnn_robustness configuration_boundary must expose the proxy's L2 semantics")
    expected_configs = {
        "nll / L2=0.05 (standard proxy)",
        "rcce / L2=0.10 (exploratory proxy)",
    }
    if (
        set(curves) != expected_configs
        or set(intervals) != expected_configs
        or set(seed_values) != expected_configs
    ):
        raise ReportSchemaError("bnn_robustness configuration mappings must retain both declared baselines")
    n_seeds = payload["n_seeds"]
    assert isinstance(n_seeds, int) and not isinstance(n_seeds, bool)
    if n_seeds < 2:
        raise ReportSchemaError("bnn_robustness n_seeds must be >= 2")
    for config in expected_configs:
        curve = _number_list(curves[config], schema="bnn_robustness", field=f"accuracy_by_config.{config}")
        config_intervals = intervals[config]
        config_seed_values = seed_values[config]
        if (
            len(curve) != len(levels)
            or not isinstance(config_intervals, list)
            or not isinstance(config_seed_values, list)
            or len(config_intervals) != len(levels)
            or len(config_seed_values) != len(levels)
        ):
            raise ReportSchemaError(
                "bnn_robustness curves, intervals, and seed rows must match operating points"
            )
        for index, value in enumerate(curve):
            if not 0.0 <= value <= 1.0:
                raise ReportSchemaError("bnn_robustness accuracy must lie in [0, 1]")
            interval = _number_list(
                config_intervals[index],
                schema="bnn_robustness",
                field=f"accuracy_ci_by_config.{config}[{index}]",
            )
            row = _number_list(
                config_seed_values[index],
                schema="bnn_robustness",
                field=f"accuracy_seed_values_by_config.{config}[{index}]",
            )
            seed_row_out_of_bounds = any(seed_accuracy < 0.0 or seed_accuracy > 1.0 for seed_accuracy in row)
            if len(row) != n_seeds or seed_row_out_of_bounds:
                raise ReportSchemaError(
                    "bnn_robustness seed accuracy row must match n_seeds and lie in [0, 1]"
                )
            row_mean = math.fsum(row) / len(row)
            if not math.isclose(value, row_mean, rel_tol=0.0, abs_tol=1e-12):
                raise ReportSchemaError("bnn_robustness reported mean disagrees with its seed accuracy row")
            if (
                len(interval) != 2
                or interval[0] < 0.0
                or interval[1] > 1.0
                or interval[0] > value
                or value > interval[1]
            ):
                raise ReportSchemaError(
                    "bnn_robustness interval must lie in [0, 1] and contain its seed-row mean"
                )
    standard = _number_list(
        curves["nll / L2=0.05 (standard proxy)"],
        schema="bnn_robustness",
        field="standard",
    )
    robust = _number_list(
        curves["rcce / L2=0.10 (exploratory proxy)"],
        schema="bnn_robustness",
        field="robust",
    )
    gaps = _number_list(
        payload["robust_minus_standard"], schema="bnn_robustness", field="robust_minus_standard"
    )
    expected_gaps = [rob - std for rob, std in zip(robust, standard)]
    if len(gaps) != len(levels) or any(
        not math.isclose(gap, expected, rel_tol=0.0, abs_tol=1e-12)
        for gap, expected in zip(gaps, expected_gaps)
    ):
        raise ReportSchemaError("bnn_robustness robust_minus_standard disagrees with accuracy curves")
    if "peak_margin" in payload or "peak_margin_contamination" in payload:
        if "peak_margin" not in payload or "peak_margin_contamination" not in payload:
            raise ReportSchemaError("bnn_robustness peak selection fields must appear together")
        peak_index = max(range(len(gaps)), key=lambda index: gaps[index])
        if not math.isclose(
            _number(payload["peak_margin"], schema="bnn_robustness", field="peak_margin"),
            gaps[peak_index],
            rel_tol=0.0,
            abs_tol=1e-12,
        ) or not math.isclose(
            _number(
                payload["peak_margin_contamination"],
                schema="bnn_robustness",
                field="peak_margin_contamination",
            ),
            levels[peak_index],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ReportSchemaError("bnn_robustness peak fields disagree with within-sweep selection")
