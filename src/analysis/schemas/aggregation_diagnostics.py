"""Aggregation diagnostics report contracts."""

from __future__ import annotations

from typing import TypedDict

from typing_extensions import NotRequired

from analysis.schemas.primitives import SchemaDefinition


class HeuristicCharacterizationReport(TypedDict):
    """Top-level payload written to ``heuristic_characterization.json``."""

    breakdown: dict[str, object]
    claim_level: str
    formal_no_go: dict[str, object]
    grid: dict[str, object]
    independent_unit: str
    influence_naive: dict[str, object]
    influence_robust: dict[str, object]
    no_claim: str
    schema_version: str
    seed: int
    theory_status: str


class RobustInfluenceWeightsReport(TypedDict):
    """Top-level payload written to ``robust_influence_weights.json``."""

    normalized_effective_weights: list[float]
    agent_weights: NotRequired[list[float]]
    contaminated_indices: list[int]
    n_agents: int
    n_contaminated: int
    robustness: float
    schema_version: str
    true_state: int


class VariationalAggregationReport(TypedDict):
    """Top-level payload written to ``variational_aggregation.json``."""

    capture_gap: NotRequired[float]
    converged: bool
    drifts: list[float]
    free_energy_history: list[float]
    iterations: int
    multi_start_final_f: NotRequired[float]
    multi_start_history: list[float]
    n_agents: int
    n_contaminated: int
    naive_influence: float
    robustness: float
    single_start_final_f: NotRequired[float]
    single_start_history: list[float]
    true_state: int
    variational_influence: list[float]


_HEURISTIC_CHARACTERIZATION_SCHEMA = SchemaDefinition(
    required={
        "breakdown": "dict",
        "claim_level": "str",
        "formal_no_go": "dict",
        "grid": "dict",
        "independent_unit": "str",
        "influence_naive": "dict",
        "influence_robust": "dict",
        "no_claim": "str",
        "schema_version": "str",
        "seed": "int",
        "theory_status": "str",
    }
)


_ROBUST_INFLUENCE_WEIGHTS_SCHEMA = SchemaDefinition(
    required={
        "normalized_effective_weights": "list",
        "contaminated_indices": "list",
        "n_agents": "int",
        "n_contaminated": "int",
        "robustness": "number",
        "schema_version": "str",
        "true_state": "int",
    }
)


_VARIATIONAL_AGGREGATION_SCHEMA = SchemaDefinition(
    required={
        "converged": "bool",
        "drifts": "list",
        "free_energy_history": "list",
        "iterations": "int",
        "multi_start_history": "list",
        "n_agents": "int",
        "n_contaminated": "int",
        "naive_influence": "number",
        "robustness": "number",
        "single_start_history": "list",
        "true_state": "int",
        "variational_influence": "list",
    },
    optional={
        "capture_gap": "number",
        "multi_start_final_f": "number",
        "single_start_final_f": "number",
    },
)
