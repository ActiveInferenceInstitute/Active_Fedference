"""Robustness sweep report contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict

from analysis.schemas.primitives import ReportSchemaError, SchemaDefinition, _mapping, _number, _number_list


class RobustnessSweepReport(TypedDict):
    """Top-level payload written to ``robustness_sweep.json``."""

    schema_version: str
    accuracy_at_verdict_rate: dict[str, object]
    accuracy_by_method_and_rate: dict[str, object]
    accuracy_threshold: float
    any_robust_wins: bool
    attack_target_state: int
    divergences: list[str]
    fdr_alpha: float
    headline_method: str
    headline_selection_rule: str
    headline_tie_break: str
    headline_tie_set: list[str]
    headline_is_display_selection: bool
    largest_mean_difference_method: str
    headline_n_for_target_power: int
    headline_power: float
    kind: str
    n: int
    n_agents: int
    n_contaminated: int
    n_trials: int
    naive_degrades_with_rate: bool
    naive_verdict_rate_mean: float
    paired_tests_by_rate: dict[str, object]
    per_rate_summary: dict[str, object]
    power_alpha: float
    power_alternative: str
    prospective_n_for_target_power: int
    rates: list[float]
    robust_above_threshold_at_worst_rate: bool
    seed: int
    server_robustness_by_label: dict[str, object]
    target_power: float
    true_state: int
    verdict: dict[str, object]
    verdict_rate: float
    trial_structure: str
    analysis_unit: str
    worst_rate_best_method: str
    worst_rate: float
    paired_test_alternative: str
    fdr_family_ownership: str
    d_equivalent_status: str


def _check_robustness_sweep(payload: Mapping[str, object]) -> None:
    """Bind the display sweep to matched trials and actual server presets."""
    divergences = payload["divergences"]
    rates = payload["rates"]
    if (
        not isinstance(divergences, list)
        or not divergences
        or any(not isinstance(item, str) for item in divergences)
    ):
        raise ReportSchemaError("robustness_sweep divergences must be a non-empty string list")
    if len(set(divergences)) != len(divergences) or "KLD" not in divergences:
        raise ReportSchemaError("robustness_sweep divergences must be unique and include KLD")
    rate_values = _number_list(rates, schema="robustness_sweep", field="rates")
    if not rate_values or len(set(rate_values)) != len(rate_values):
        raise ReportSchemaError("robustness_sweep rates must be non-empty and unique")
    presets = _mapping(
        payload["server_robustness_by_label"],
        schema="robustness_sweep",
        field="server_robustness_by_label",
    )
    if set(presets) != set(divergences):
        raise ReportSchemaError("robustness_sweep server presets must match divergence labels")
    for label, value in presets.items():
        preset = _number(
            value,
            schema="robustness_sweep",
            field=f"server_robustness_by_label.{label}",
        )
        if (label == "KLD" and preset != 0.0) or (label != "KLD" and preset <= 0.0):
            raise ReportSchemaError(
                "robustness_sweep server presets must expose naive zero and positive robust values"
            )
    if payload["headline_is_display_selection"] is not True:
        raise ReportSchemaError("robustness_sweep headline must remain disclosed as display selection")
    if "matched trial" not in str(payload["analysis_unit"]):
        raise ReportSchemaError("robustness_sweep analysis_unit must identify matched trials")
    if "true state" not in str(payload["trial_structure"]) or "attack target" not in str(
        payload["trial_structure"]
    ):
        raise ReportSchemaError("robustness_sweep trial_structure must identify fixed world variables")
    n_trials = payload["n_trials"]
    assert isinstance(n_trials, int) and not isinstance(n_trials, bool)
    profiles = _mapping(payload["per_rate_summary"], schema="robustness_sweep", field="per_rate_summary")
    expected_rate_keys = {f"{rate:g}" for rate in rate_values}
    if set(profiles) != expected_rate_keys:
        raise ReportSchemaError("robustness_sweep per_rate_summary must cover every declared rate")
    for rate_key, profile_value in profiles.items():
        profile = _mapping(profile_value, schema="robustness_sweep", field=f"per_rate_summary.{rate_key}")
        methods = _mapping(
            profile.get("methods"), schema="robustness_sweep", field=f"per_rate_summary.{rate_key}.methods"
        )
        if profile.get("n") != n_trials or set(methods) != set(divergences):
            raise ReportSchemaError(
                "robustness_sweep per-rate summaries must retain matched trial budgets and presets"
            )


_ROBUSTNESS_SWEEP_SCHEMA = SchemaDefinition(
    required={
        "schema_version": "str",
        "accuracy_at_verdict_rate": "dict",
        "accuracy_by_method_and_rate": "dict",
        "accuracy_threshold": "number",
        "any_robust_wins": "bool",
        "attack_target_state": "int",
        "divergences": "list",
        "fdr_alpha": "number",
        "headline_method": "str",
        "headline_selection_rule": "str",
        "headline_tie_break": "str",
        "headline_tie_set": "list",
        "headline_is_display_selection": "bool",
        "largest_mean_difference_method": "str",
        "headline_n_for_target_power": "int",
        "headline_power": "number",
        "kind": "str",
        "n": "int",
        "n_agents": "int",
        "n_contaminated": "int",
        "n_trials": "int",
        "naive_degrades_with_rate": "bool",
        "naive_verdict_rate_mean": "number",
        "paired_tests_by_rate": "dict",
        "per_rate_summary": "dict",
        "power_alpha": "number",
        "power_alternative": "str",
        "prospective_n_for_target_power": "int",
        "rates": "list",
        "robust_above_threshold_at_worst_rate": "bool",
        "seed": "int",
        "server_robustness_by_label": "dict",
        "target_power": "number",
        "true_state": "int",
        "verdict": "dict",
        "verdict_rate": "number",
        "trial_structure": "str",
        "analysis_unit": "str",
        "worst_rate_best_method": "str",
        "worst_rate": "number",
        "paired_test_alternative": "str",
        "fdr_family_ownership": "str",
        "d_equivalent_status": "str",
    }
)
