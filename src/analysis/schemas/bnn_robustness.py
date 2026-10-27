"""Bnn robustness report contracts."""

from __future__ import annotations

from typing import TypedDict

from typing_extensions import NotRequired

from analysis.schemas.primitives import SchemaDefinition


class BnnRobustnessReport(TypedDict):
    """Top-level payload written to ``bnn_robustness.json``."""

    accuracy_by_config: dict[str, object]
    accuracy_ci_by_config: dict[str, object]
    accuracy_seed_values_by_config: dict[str, object]
    analysis_unit: str
    ci_percent: int
    claim_boundary: str
    configuration_boundary: str
    contamination_levels: list[float]
    interval_method: str
    model_family: str
    n_bootstrap: int
    n_per: int
    n_seeds: int
    peak_margin: NotRequired[float]
    peak_margin_contamination: NotRequired[float]
    replication_unit: str
    robust_loss_param: float
    robust_minus_standard: list[float]
    schema_version: str
    seed: int
    selection_disclosure: str
    study_status: str


_BNN_ROBUSTNESS_SCHEMA = SchemaDefinition(
    required={
        "accuracy_by_config": "dict",
        "accuracy_ci_by_config": "dict",
        "accuracy_seed_values_by_config": "dict",
        "analysis_unit": "str",
        "ci_percent": "int",
        "claim_boundary": "str",
        "configuration_boundary": "str",
        "contamination_levels": "list",
        "interval_method": "str",
        "model_family": "str",
        "n_bootstrap": "int",
        "n_per": "int",
        "n_seeds": "int",
        "replication_unit": "str",
        "robust_loss_param": "number",
        "robust_minus_standard": "list",
        "schema_version": "str",
        "seed": "int",
        "selection_disclosure": "str",
        "study_status": "str",
    },
    optional={
        "peak_margin": "number",
        "peak_margin_contamination": "number",
    },
)
