"""Observer worlds report contracts."""

from __future__ import annotations

from typing import TypedDict

from analysis.schemas.primitives import SchemaDefinition


class DisjointFovWorldReport(TypedDict):
    """Top-level payload written to ``disjoint_fov_world.json``."""

    communicating_accuracy: float
    efe_navigation: dict[str, object]
    fov_width: int
    gap: float
    isolated_accuracy: float
    multiseed: dict[str, object]
    n_agents: int
    n_positions: int
    n_steps: int
    n_trials: int
    seed: int


class ComplexityScalingReport(TypedDict):
    """Top-level payload written to ``complexity_scaling.json``."""

    analytic_specs: list[object]
    benchmark: dict[str, object]
    claim_boundary: str
    machine: dict[str, object]
    measurements: list[object]
    schema_version: str
    seed: int
    status: str


class ConditionalWorldReport(TypedDict):
    """Top-level payload written to ``conditional_world.json``."""

    by_scenario: dict[str, object]
    claim_status: str
    controls: dict[str, object]
    grid: list[object]
    independent_unit: str
    n_agents: int
    n_seeds: int
    n_states: int
    n_trials: int
    primary_estimand: str
    robustness: float
    schema_version: str
    seed: int


_DISJOINT_FOV_WORLD_SCHEMA = SchemaDefinition(
    required={
        "communicating_accuracy": "number",
        "efe_navigation": "dict",
        "fov_width": "int",
        "gap": "number",
        "isolated_accuracy": "number",
        "multiseed": "dict",
        "n_agents": "int",
        "n_positions": "int",
        "n_steps": "int",
        "n_trials": "int",
        "seed": "int",
    }
)


_COMPLEXITY_SCALING_SCHEMA = SchemaDefinition(
    required={
        "analytic_specs": "list",
        "benchmark": "dict",
        "claim_boundary": "str",
        "machine": "dict",
        "measurements": "list",
        "schema_version": "str",
        "seed": "int",
        "status": "str",
    }
)


_CONDITIONAL_WORLD_SCHEMA = SchemaDefinition(
    required={
        "by_scenario": "dict",
        "claim_status": "str",
        "controls": "dict",
        "grid": "list",
        "independent_unit": "str",
        "n_agents": "int",
        "n_seeds": "int",
        "n_states": "int",
        "n_trials": "int",
        "primary_estimand": "str",
        "robustness": "number",
        "schema_version": "str",
        "seed": "int",
    }
)
