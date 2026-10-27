"""Hierarchical bmr report contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TypedDict

from analysis.schemas.primitives import ReportSchemaError, SchemaDefinition, _is_number, _mapping, _number


class HierarchicalBmrReport(TypedDict):
    """Top-level payload written to ``hierarchical_bmr.json``."""

    acuity: float
    claim_boundary: str
    configured_control_passed: bool
    decision_rule: str
    degenerate: dict[str, object]
    degenerate_recommends_prune_top: bool
    degenerate_top_surprise: float
    informative: dict[str, object]
    informative_keeps_top: bool
    informative_top_surprise: float
    n_iters: int
    n_levels: int
    obs: int
    primary_control: str
    schema_version: str
    secondary_diagnostic: str
    seed: int
    seed_role: str
    study_status: str
    surprise_tol: float


def _check_hierarchical_bmr(payload: Mapping[str, object]) -> None:
    """Bind the configured control verdict to its declared surprise threshold."""
    tolerance = _number(payload["surprise_tol"], schema="hierarchical_bmr", field="surprise_tol")
    if tolerance <= 0.0:
        raise ReportSchemaError("hierarchical_bmr surprise_tol must be positive")
    worlds: dict[str, Mapping[str, object]] = {}
    for world_name in ("degenerate", "informative"):
        world = _mapping(payload[world_name], schema="hierarchical_bmr", field=world_name)
        levels = world.get("levels")
        if not isinstance(levels, list) or not levels:
            raise ReportSchemaError(f"hierarchical_bmr {world_name} must contain non-leaf levels")
        for index, level_value in enumerate(levels):
            level = _mapping(level_value, schema="hierarchical_bmr", field=f"{world_name}.levels[{index}]")
            surprise = level.get("bayesian_surprise")
            prunable = level.get("prunable")
            if not _is_number(surprise) or not isinstance(prunable, bool):
                raise ReportSchemaError(
                    f"hierarchical_bmr {world_name} level must report surprise and prunable"
                )
            surprise_value = _number(
                surprise,
                schema="hierarchical_bmr",
                field=f"{world_name}.levels[{index}].bayesian_surprise",
            )
            if prunable is not (surprise_value < tolerance):
                raise ReportSchemaError(
                    f"hierarchical_bmr {world_name} prunable flag disagrees with surprise_tol"
                )
        worlds[world_name] = world
    deg_levels = worlds["degenerate"]["levels"]
    inf_levels = worlds["informative"]["levels"]
    assert isinstance(deg_levels, list) and isinstance(inf_levels, list)
    deg_top = next(
        (
            _mapping(level, schema="hierarchical_bmr", field="degenerate top")
            for level in deg_levels
            if isinstance(level, Mapping) and level.get("level") == 0
        ),
        None,
    )
    inf_top = next(
        (
            _mapping(level, schema="hierarchical_bmr", field="informative top")
            for level in inf_levels
            if isinstance(level, Mapping) and level.get("level") == 0
        ),
        None,
    )
    if deg_top is None or inf_top is None:
        raise ReportSchemaError("hierarchical_bmr controls must report top level 0")
    if not math.isclose(
        _number(
            payload["degenerate_top_surprise"],
            schema="hierarchical_bmr",
            field="degenerate_top_surprise",
        ),
        _number(
            deg_top["bayesian_surprise"],
            schema="hierarchical_bmr",
            field="degenerate.bayesian_surprise",
        ),
        rel_tol=0.0,
        abs_tol=1e-12,
    ) or not math.isclose(
        _number(
            payload["informative_top_surprise"],
            schema="hierarchical_bmr",
            field="informative_top_surprise",
        ),
        _number(
            inf_top["bayesian_surprise"],
            schema="hierarchical_bmr",
            field="informative.bayesian_surprise",
        ),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ReportSchemaError("hierarchical_bmr headline surprise disagrees with nested level")
    deg_prunes = worlds["degenerate"].get("recommended_prune") == 0
    inf_keeps = worlds["informative"].get("recommended_prune") != 0
    if payload["degenerate_recommends_prune_top"] is not deg_prunes:
        raise ReportSchemaError("hierarchical_bmr degenerate verdict disagrees with nested result")
    if payload["informative_keeps_top"] is not inf_keeps:
        raise ReportSchemaError("hierarchical_bmr informative verdict disagrees with nested result")
    if payload["configured_control_passed"] is not (deg_prunes and inf_keeps):
        raise ReportSchemaError("hierarchical_bmr configured_control_passed disagrees with verdicts")
    if payload["study_status"] != "configured_deterministic_control":
        raise ReportSchemaError("hierarchical_bmr study_status must identify a configured control")
    if "does not determine" not in str(payload["secondary_diagnostic"]):
        raise ReportSchemaError("hierarchical_bmr secondary diagnostic must not own the decision")
    if "no universal" not in str(payload["claim_boundary"]):
        raise ReportSchemaError("hierarchical_bmr claim_boundary must deny universal emergence")


_HIERARCHICAL_BMR_SCHEMA = SchemaDefinition(
    required={
        "acuity": "number",
        "claim_boundary": "str",
        "configured_control_passed": "bool",
        "decision_rule": "str",
        "degenerate": "dict",
        "degenerate_recommends_prune_top": "bool",
        "degenerate_top_surprise": "number",
        "informative": "dict",
        "informative_keeps_top": "bool",
        "informative_top_surprise": "number",
        "n_iters": "int",
        "n_levels": "int",
        "obs": "int",
        "primary_control": "str",
        "schema_version": "str",
        "secondary_diagnostic": "str",
        "seed": "int",
        "seed_role": "str",
        "study_status": "str",
        "surprise_tol": "number",
    }
)
