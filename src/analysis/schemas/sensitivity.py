"""Sensitivity report contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict

from analysis.schemas.primitives import ReportSchemaError, SchemaDefinition, _is_number, _number, _number_list


class SensitivityReport(TypedDict):
    """Source-bound payload for the two configured sensitivity grids."""

    schema_version: str
    seed: int
    n_trials: int
    acuity_values: list[float]
    n_agents_values: list[int]
    noise_floor: float
    belief_sharing: dict[str, object]
    hierarchical: dict[str, object]
    estimands: dict[str, str]
    unit: str
    analysis_unit: str
    replication_unit: str
    uncertainty: str
    claim_boundary: str


def _check_sensitivity(payload: Mapping[str, object]) -> None:
    """Keep both displayed sensitivity grids aligned with their declared axes."""
    schema = "sensitivity"
    acuity = _number_list(payload["acuity_values"], schema=schema, field="acuity_values")
    raw_agents = payload["n_agents_values"]
    if (
        not isinstance(raw_agents, list)
        or not raw_agents
        or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in raw_agents)
    ):
        raise ReportSchemaError(f"{schema} n_agents_values must contain positive integers")
    if not acuity or any(value <= 0.0 or value > 1.0 for value in acuity):
        raise ReportSchemaError(f"{schema} acuity_values must lie in (0, 1]")
    if acuity != sorted(set(acuity)):
        raise ReportSchemaError(f"{schema} acuity_values must be strictly increasing and unique")
    if raw_agents != sorted(set(raw_agents)):
        raise ReportSchemaError(f"{schema} n_agents_values must be strictly increasing and unique")
    noise_floor = _number(payload["noise_floor"], schema=schema, field="noise_floor")
    if noise_floor < 0.0 or noise_floor > 1.0:
        raise ReportSchemaError(f"{schema} noise_floor must lie in [0, 1]")
    n_trials = payload["n_trials"]
    if not isinstance(n_trials, int) or isinstance(n_trials, bool) or n_trials <= 0:
        raise ReportSchemaError(f"{schema} n_trials must be positive")

    for panel_name in ("belief_sharing", "hierarchical"):
        nested = payload[panel_name]
        assert isinstance(nested, Mapping)
        if nested.get("acuity_values") != payload["acuity_values"]:
            raise ReportSchemaError(f"{schema} {panel_name} acuity axis does not match the top-level axis")
        if nested.get("n_agents_values") != raw_agents:
            raise ReportSchemaError(
                f"{schema} {panel_name} colony-size axis does not match the top-level axis"
            )
        if nested.get("seed") != payload["seed"] or nested.get("n_trials") != n_trials:
            raise ReportSchemaError(f"{schema} {panel_name} execution identity does not match the report")
        raw_grid = nested.get("accuracy_gap_grid")
        if not isinstance(raw_grid, list) or len(raw_grid) != len(acuity):
            raise ReportSchemaError(f"{schema} {panel_name} accuracy_gap_grid has the wrong row count")
        for row_index, row in enumerate(raw_grid):
            if (
                not isinstance(row, list)
                or len(row) != len(raw_agents)
                or any(not _is_number(value) for value in row)
            ):
                raise ReportSchemaError(
                    f"{schema} {panel_name} accuracy_gap_grid row {row_index} does not match the colony axis"
                )
            if any(abs(float(value)) > 1.0 for value in row):
                raise ReportSchemaError(f"{schema} {panel_name} accuracy_gap_grid values must lie in [-1, 1]")
    estimands = payload["estimands"]
    if not isinstance(estimands, Mapping) or set(estimands) != {"belief_sharing", "hierarchical"}:
        raise ReportSchemaError(f"{schema} estimands must name both displayed panels")
    if payload["unit"] != "fraction":
        raise ReportSchemaError(f"{schema} unit must be 'fraction'")
    if "not a confidence interval" not in str(payload["claim_boundary"]):
        raise ReportSchemaError(f"{schema} claim_boundary must explain the hatch semantics")


_SENSITIVITY_SCHEMA = SchemaDefinition(
    required={
        "schema_version": "str",
        "seed": "int",
        "n_trials": "int",
        "acuity_values": "list",
        "n_agents_values": "list",
        "noise_floor": "number",
        "belief_sharing": "dict",
        "hierarchical": "dict",
        "estimands": "dict",
        "unit": "str",
        "analysis_unit": "str",
        "replication_unit": "str",
        "uncertainty": "str",
        "claim_boundary": "str",
    }
)
