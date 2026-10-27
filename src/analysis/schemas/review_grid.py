"""Review grid report contracts."""

from __future__ import annotations

from typing import TypedDict

from analysis.schemas.primitives import SchemaDefinition


class ReviewGridReport(TypedDict):
    """Top-level payload written to ``robustness_review_grid.json``."""

    analysis_profile: str
    attack_mechanisms: list[str]
    conditional_world: dict[str, object]
    controls: dict[str, object]
    divergences: list[str]
    directional_mechanisms: list[str]
    entropy_controls: list[str]
    independent_unit: str
    n_agents: int
    n_seeds: int
    n_trials: int
    primary_estimand: str
    precision_plan: dict[str, object]
    rates: list[float]
    rate_profiles: dict[str, object]
    robustness: float
    schema_version: str
    seed: int
    seed_schedule: dict[str, object]
    selection_status: str
    statistics: dict[str, object]
    trial_structure: str


_ROBUSTNESS_REVIEW_GRID_SCHEMA = SchemaDefinition(
    required={
        "analysis_profile": "str",
        "attack_mechanisms": "list",
        "conditional_world": "dict",
        "controls": "dict",
        "divergences": "list",
        "directional_mechanisms": "list",
        "entropy_controls": "list",
        "independent_unit": "str",
        "n_agents": "int",
        "n_seeds": "int",
        "n_trials": "int",
        "primary_estimand": "str",
        "precision_plan": "dict",
        "rates": "list",
        "rate_profiles": "dict",
        "robustness": "number",
        "schema_version": "str",
        "seed": "int",
        "seed_schedule": "dict",
        "selection_status": "str",
        "statistics": "dict",
        "trial_structure": "str",
    }
)
