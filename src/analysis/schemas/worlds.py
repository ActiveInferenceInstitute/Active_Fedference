"""Worlds report contracts."""

from __future__ import annotations

from typing import TypedDict

from analysis.schemas.primitives import SchemaDefinition


class MovingWorldReport(TypedDict):
    """Top-level payload written to ``moving_world.json``."""

    accuracy: dict[str, object]
    free_energy_gap: dict[str, object]
    multiseed: dict[str, object]
    n_agents: int
    n_positions: int
    n_steps: int
    n_steps_to_consensus: dict[str, object]
    n_trials: int
    seed: int


class HierarchicalWorldReport(TypedDict):
    """Top-level payload written to ``hierarchical_world.json``."""

    acuity: float
    context_accuracy: float
    free_energy_gap: dict[str, object]
    location_accuracy: dict[str, object]
    location_accuracy_gap: float
    multiseed: dict[str, object]
    n_agents: int
    n_contexts: int
    n_iters: int
    n_trials: int
    seed: int


class NLevel3WorldReport(TypedDict):
    """Top-level payload written to ``nlevel3_world.json``."""

    acuity: float
    context_accuracy: float
    free_energy_gap: dict[str, object]
    location_accuracy: dict[str, object]
    location_accuracy_gap: float
    meta_context_accuracy: float
    multiseed: dict[str, object]
    n_agents: int
    n_contexts: int
    n_iters: int
    n_levels: int
    n_meta_contexts: int
    n_trials: int
    seed: int


_MOVING_WORLD_SCHEMA = SchemaDefinition(
    required={
        "accuracy": "dict",
        "free_energy_gap": "dict",
        "multiseed": "dict",
        "n_agents": "int",
        "n_positions": "int",
        "n_steps": "int",
        "n_steps_to_consensus": "dict",
        "n_trials": "int",
        "seed": "int",
    }
)


_HIERARCHICAL_WORLD_SCHEMA = SchemaDefinition(
    required={
        "acuity": "number",
        "context_accuracy": "number",
        "free_energy_gap": "dict",
        "location_accuracy": "dict",
        "location_accuracy_gap": "number",
        "multiseed": "dict",
        "n_agents": "int",
        "n_contexts": "int",
        "n_iters": "int",
        "n_trials": "int",
        "seed": "int",
    }
)


_NLEVEL3_WORLD_SCHEMA = SchemaDefinition(
    required={
        "acuity": "number",
        "context_accuracy": "number",
        "free_energy_gap": "dict",
        "location_accuracy": "dict",
        "location_accuracy_gap": "number",
        "meta_context_accuracy": "number",
        "multiseed": "dict",
        "n_agents": "int",
        "n_contexts": "int",
        "n_iters": "int",
        "n_levels": "int",
        "n_meta_contexts": "int",
        "n_trials": "int",
        "seed": "int",
    }
)
