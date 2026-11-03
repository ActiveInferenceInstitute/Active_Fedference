"""Study summaries report contracts."""

from __future__ import annotations

from typing import TypedDict

from analysis.schemas.primitives import SchemaDefinition


class CrossStudySummaryReport(TypedDict):
    """Top-level payload written to ``cross_study_summary.json``."""

    n_seeds: int
    n_trials: int
    seed: int
    studies: list[object]


class BeliefQualityReport(TypedDict):
    """Top-level payload written to ``belief_quality.json``."""

    by_scenario: dict[str, object]
    control_scores: dict[str, object]
    controls: dict[str, object]
    independent_unit: str
    n_agents: int
    n_seeds: int
    n_trials: int
    primary_estimand: str
    robustness: float
    schema_version: str
    seed: int


_CROSS_STUDY_SUMMARY_SCHEMA = SchemaDefinition(
    required={
        "n_seeds": "int",
        "n_trials": "int",
        "seed": "int",
        "studies": "list",
    }
)


_BELIEF_QUALITY_SCHEMA = SchemaDefinition(
    required={
        "by_scenario": "dict",
        "control_scores": "dict",
        "controls": "dict",
        "independent_unit": "str",
        "n_agents": "int",
        "n_seeds": "int",
        "n_trials": "int",
        "primary_estimand": "str",
        "robustness": "number",
        "schema_version": "str",
        "seed": "int",
    }
)
