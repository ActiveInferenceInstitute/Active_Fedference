"""Learning report contracts."""

from __future__ import annotations

from typing import TypedDict

from analysis.schemas.primitives import SchemaDefinition


class LanguageAcquisitionReport(TypedDict):
    """Top-level payload written to ``language_acquisition.json``."""

    final_kl: float
    initial_kl: float
    kl_trajectory: list[float]
    kl_trajectory_by_seed: list[object]
    monotone_decreasing: bool
    n_points: int
    n_seeds: int
    num_steps: int
    seed: int
    trajectory_ci_hi: list[float]
    trajectory_ci_lo: list[float]


class EmergenceReport(TypedDict):
    """Top-level payload written to ``emergence.json``."""

    convergence: bool
    delta_F_redundant: float
    delta_F_supported: float
    n: int
    n_states: int
    seed: int


class EfeDecompositionReport(TypedDict):
    """Top-level payload written to ``efe_decomposition.json``."""

    ambiguity: float
    epistemic_value: float
    identity_residual: float
    policy: list[int]
    pragmatic_value: float
    prior_type: str
    risk: float
    total: float


_LANGUAGE_ACQUISITION_SCHEMA = SchemaDefinition(
    required={
        "final_kl": "number",
        "initial_kl": "number",
        "kl_trajectory": "list",
        "kl_trajectory_by_seed": "list",
        "monotone_decreasing": "bool",
        "n_points": "int",
        "n_seeds": "int",
        "num_steps": "int",
        "seed": "int",
        "trajectory_ci_hi": "list",
        "trajectory_ci_lo": "list",
    }
)


_EMERGENCE_SCHEMA = SchemaDefinition(
    required={
        "convergence": "bool",
        "delta_F_redundant": "number",
        "delta_F_supported": "number",
        "n": "int",
        "n_states": "int",
        "seed": "int",
    }
)


_EFE_DECOMPOSITION_SCHEMA = SchemaDefinition(
    required={
        "ambiguity": "number",
        "epistemic_value": "number",
        "identity_residual": "number",
        "policy": "list",
        "pragmatic_value": "number",
        "prior_type": "str",
        "risk": "number",
        "total": "number",
    }
)
