"""Parameter recovery report contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TypedDict

from analysis.schemas.primitives import ReportSchemaError, SchemaDefinition, _number, _number_list


class ParameterRecoveryReport(TypedDict):
    """Top-level payload written to ``parameter_recovery.json``."""

    abs_error: list[float]
    acuity_grid: list[float]
    analysis_unit: str
    claim_boundary: str
    estimator: str
    fit_bounds: list[float]
    fit_grid: list[float]
    fit_resolution: int
    interval_method: str
    interval_percent: int
    mean_abs_error: float
    n_grid_points: int
    n_independent_trials: int
    n_observations: int
    n_trials: int
    r_squared: float
    recovered_acuity: list[float]
    recovered_acuity_by_trial: list[object]
    recovered_acuity_ci_hi: list[float]
    recovered_acuity_ci_lo: list[float]
    replication_unit: str
    schema_version: str
    seed: int
    seed_role: str
    scope: str
    true_acuity: list[float]


def _check_parameter_recovery(payload: Mapping[str, object]) -> None:
    """Validate the complete finite-grid recovery design and trial summaries."""
    true_values = _number_list(payload["true_acuity"], schema="parameter_recovery", field="true_acuity")
    acuity_grid = _number_list(payload["acuity_grid"], schema="parameter_recovery", field="acuity_grid")
    recovered = _number_list(
        payload["recovered_acuity"], schema="parameter_recovery", field="recovered_acuity"
    )
    ci_lo = _number_list(
        payload["recovered_acuity_ci_lo"], schema="parameter_recovery", field="recovered_acuity_ci_lo"
    )
    ci_hi = _number_list(
        payload["recovered_acuity_ci_hi"], schema="parameter_recovery", field="recovered_acuity_ci_hi"
    )
    abs_error = _number_list(payload["abs_error"], schema="parameter_recovery", field="abs_error")
    fit_grid = _number_list(payload["fit_grid"], schema="parameter_recovery", field="fit_grid")
    fit_bounds = _number_list(payload["fit_bounds"], schema="parameter_recovery", field="fit_bounds")
    n_grid_points = payload["n_grid_points"]
    n_trials = payload["n_trials"]
    fit_resolution = payload["fit_resolution"]
    assert isinstance(n_grid_points, int) and not isinstance(n_grid_points, bool)
    assert isinstance(n_trials, int) and not isinstance(n_trials, bool)
    assert isinstance(fit_resolution, int) and not isinstance(fit_resolution, bool)
    if n_grid_points < 1 or n_trials < 1 or fit_resolution < 2:
        raise ReportSchemaError("parameter_recovery grid, trial, and fit budgets must be positive")
    if true_values != acuity_grid:
        raise ReportSchemaError("parameter_recovery true_acuity must preserve acuity_grid exactly")
    grid_fields = (true_values, recovered, ci_lo, ci_hi, abs_error)
    if any(len(values) != n_grid_points for values in grid_fields):
        raise ReportSchemaError("parameter_recovery per-grid fields must match n_grid_points")
    if len(fit_grid) != fit_resolution or fit_grid != sorted(set(fit_grid)):
        raise ReportSchemaError("parameter_recovery fit_grid must be strictly increasing at fit_resolution")
    if len(fit_bounds) != 2 or fit_bounds != [fit_grid[0], fit_grid[-1]]:
        raise ReportSchemaError("parameter_recovery fit_bounds must equal the fit_grid endpoints")
    trial_rows = payload["recovered_acuity_by_trial"]
    assert isinstance(trial_rows, list)
    if len(trial_rows) != n_grid_points:
        raise ReportSchemaError("parameter_recovery recovered_acuity_by_trial has wrong grid count")
    for index, row_value in enumerate(trial_rows):
        row = _number_list(
            row_value,
            schema="parameter_recovery",
            field=f"recovered_acuity_by_trial[{index}]",
        )
        if len(row) != n_trials:
            raise ReportSchemaError("parameter_recovery trial row has wrong n_trials")
        mean = sum(row) / n_trials
        if not math.isclose(recovered[index], mean, rel_tol=0.0, abs_tol=1e-12):
            raise ReportSchemaError("parameter_recovery recovered_acuity disagrees with trial values")
        if ci_lo[index] > mean or mean > ci_hi[index]:
            raise ReportSchemaError("parameter_recovery empirical interval is not ordered around its mean")
        expected_abs = sum(abs(value - true_values[index]) for value in row) / n_trials
        if not math.isclose(abs_error[index], expected_abs, rel_tol=0.0, abs_tol=1e-12):
            raise ReportSchemaError("parameter_recovery abs_error disagrees with trial values")
    if payload["n_independent_trials"] != n_grid_points * n_trials:
        raise ReportSchemaError(
            "parameter_recovery n_independent_trials disagrees with grid and trial counts"
        )
    if not math.isclose(
        _number(
            payload["mean_abs_error"],
            schema="parameter_recovery",
            field="mean_abs_error",
        ),
        sum(abs_error) / n_grid_points,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ReportSchemaError("parameter_recovery mean_abs_error disagrees with grid errors")
    if payload["estimator"] != "finite_grid_maximum_marginal_likelihood":
        raise ReportSchemaError("parameter_recovery estimator must identify finite-grid MML")
    if payload["replication_unit"] != "synthetic trial":
        raise ReportSchemaError("parameter_recovery replication_unit must be 'synthetic trial'")
    if "seed is not" not in str(payload["seed_role"]):
        raise ReportSchemaError("parameter_recovery seed_role must distinguish seed from analysis unit")
    if "finite" not in str(payload["scope"]) or "no continuous-parameter" not in str(
        payload["claim_boundary"]
    ):
        raise ReportSchemaError("parameter_recovery scope and claim boundary must remain finite-grid bounded")


_PARAMETER_RECOVERY_SCHEMA = SchemaDefinition(
    required={
        "abs_error": "list",
        "acuity_grid": "list",
        "analysis_unit": "str",
        "claim_boundary": "str",
        "estimator": "str",
        "fit_bounds": "list",
        "fit_grid": "list",
        "fit_resolution": "int",
        "interval_method": "str",
        "interval_percent": "int",
        "mean_abs_error": "number",
        "n_grid_points": "int",
        "n_independent_trials": "int",
        "n_observations": "int",
        "n_trials": "int",
        "r_squared": "number",
        "recovered_acuity": "list",
        "recovered_acuity_by_trial": "list",
        "recovered_acuity_ci_hi": "list",
        "recovered_acuity_ci_lo": "list",
        "replication_unit": "str",
        "schema_version": "str",
        "seed": "int",
        "seed_role": "str",
        "scope": "str",
        "true_acuity": "list",
    },
)
