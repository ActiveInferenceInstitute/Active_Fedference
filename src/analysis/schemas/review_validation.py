"""Review validation report contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping

from analysis.schemas.primitives import ReportSchemaError
from analysis.schemas.review_values import _review_list, _review_mapping, _review_number, _review_string_set


def _check_review_grid(payload: Mapping[str, object]) -> None:
    """Fail closed on nested data consumed by the review-grid public figure.

    A shallow top-level schema is insufficient here: a missing interval or a
    malformed method row once caused the figure to silently omit uncertainty.
    This validator binds the complete signed method × rate surface, its seed
    replication unit, and the observed (not merely declared) attack controls.
    """
    n_seeds = payload["n_seeds"]
    n_trials = payload["n_trials"]
    assert isinstance(n_seeds, int) and not isinstance(n_seeds, bool)
    assert isinstance(n_trials, int) and not isinstance(n_trials, bool)
    if n_seeds < 2 or n_trials < 2:
        raise ReportSchemaError("robustness_review_grid requires n_seeds and n_trials >= 2")

    divergences = _review_list(payload["divergences"], field="divergences")
    if any(not isinstance(method, str) for method in divergences) or "KLD" not in divergences:
        raise ReportSchemaError("robustness_review_grid divergences must be strings including 'KLD'")
    robust_methods = [str(method) for method in divergences if method != "KLD"]
    if not robust_methods or len(set(robust_methods)) != len(robust_methods):
        raise ReportSchemaError("robustness_review_grid must declare one or more unique non-KLD methods")

    rates_raw = _review_list(payload["rates"], field="rates")
    rates = [_review_number(rate, field="rates[]") for rate in rates_raw]
    if not rates or any(rate < 0.0 or rate > 1.0 for rate in rates):
        raise ReportSchemaError("robustness_review_grid rates must be non-empty and lie in [0, 1]")
    if rates != sorted(rates) or len(set(rates)) != len(rates):
        raise ReportSchemaError("robustness_review_grid rates must be strictly increasing")
    rate_keys = {f"{rate:g}" for rate in rates}
    directional = _review_string_set(payload["directional_mechanisms"], field="directional_mechanisms")
    declared_attacks = _review_string_set(payload["attack_mechanisms"], field="attack_mechanisms")

    conditional = _review_mapping(payload["conditional_world"], field="conditional_world")
    conditional_rows = _review_mapping(conditional.get("by_scenario"), field="conditional_world.by_scenario")
    if not conditional_rows:
        raise ReportSchemaError("robustness_review_grid conditional_world has no cells")
    conditional_attacks: set[str] = set()
    for scenario_id, row_value in conditional_rows.items():
        row = _review_mapping(row_value, field=f"conditional_world.by_scenario.{scenario_id}")
        attack = row.get("attack")
        if not isinstance(attack, str):
            raise ReportSchemaError(f"robustness_review_grid conditional cell {scenario_id!r} lacks attack")
        conditional_attacks.add(attack)
        if row.get("n_seeds") != n_seeds or row.get("n_trials") != n_trials:
            raise ReportSchemaError(
                f"robustness_review_grid conditional cell {scenario_id!r} has a mismatched budget"
            )
        contrast = _review_list(
            row.get("contrast_by_seed"),
            field=f"conditional_world.by_scenario.{scenario_id}.contrast_by_seed",
        )
        if len(contrast) != n_seeds:
            raise ReportSchemaError(
                f"robustness_review_grid conditional cell {scenario_id!r} has wrong seed count"
            )
        for index, value in enumerate(contrast):
            _review_number(
                value,
                field=f"conditional_world.by_scenario.{scenario_id}.contrast_by_seed[{index}]",
            )
        interval = _review_list(
            row.get("contrast_ci"),
            field=f"conditional_world.by_scenario.{scenario_id}.contrast_ci",
        )
        if len(interval) != 2:
            raise ReportSchemaError(f"robustness_review_grid conditional cell {scenario_id!r} has invalid CI")
        lo = _review_number(interval[0], field="conditional contrast CI lower")
        hi = _review_number(interval[1], field="conditional contrast CI upper")
        mean = _review_number(row.get("contrast_mean"), field="conditional contrast mean")
        if lo > mean or mean > hi:
            raise ReportSchemaError(
                f"robustness_review_grid conditional cell {scenario_id!r} has unordered CI"
            )

    profiles = _review_mapping(payload["rate_profiles"], field="rate_profiles")
    by_kind = _review_mapping(profiles.get("by_kind"), field="rate_profiles.by_kind")
    if set(by_kind) != directional:
        raise ReportSchemaError(
            "robustness_review_grid rate profile mechanisms differ from directional_mechanisms"
        )
    for mechanism, profile_value in by_kind.items():
        profile = _review_mapping(profile_value, field=f"rate_profiles.by_kind.{mechanism}")
        profile_rates = [
            _review_number(rate, field=f"rate_profiles.by_kind.{mechanism}.rates[]")
            for rate in _review_list(profile.get("rates"), field=f"rate_profiles.by_kind.{mechanism}.rates")
        ]
        if profile_rates != rates:
            raise ReportSchemaError(
                f"robustness_review_grid profile {mechanism!r} has rates inconsistent with top level"
            )

    statistics = _review_mapping(payload["statistics"], field="statistics")
    if statistics.get("selection_free") is not True:
        raise ReportSchemaError("robustness_review_grid statistics must declare selection_free=true")
    for alpha_field in ("fdr_alpha", "power_alpha"):
        alpha = _review_number(statistics.get(alpha_field), field=f"statistics.{alpha_field}")
        if not 0.0 < alpha < 1.0:
            raise ReportSchemaError(f"robustness_review_grid statistics {alpha_field} must lie in (0, 1)")
    planning_alternative = statistics.get("planning_alternative")
    if planning_alternative not in ("greater", "less", "two-sided"):
        raise ReportSchemaError("robustness_review_grid statistics planning_alternative is unsupported")
    by_mechanism = _review_mapping(statistics.get("by_mechanism"), field="statistics.by_mechanism")
    if set(by_mechanism) != directional:
        raise ReportSchemaError(
            "robustness_review_grid statistics mechanisms differ from directional_mechanisms"
        )
    observed_cell_mcses: list[float] = []
    for mechanism, mechanism_value in by_mechanism.items():
        mechanism_rows = _review_mapping(mechanism_value, field=f"statistics.by_mechanism.{mechanism}")
        methods = _review_list(
            mechanism_rows.get("methods"), field=f"statistics.by_mechanism.{mechanism}.methods"
        )
        if methods != robust_methods:
            raise ReportSchemaError(f"robustness_review_grid statistics methods differ for {mechanism!r}")
        rate_rows = _review_mapping(
            mechanism_rows.get("by_rate"),
            field=f"statistics.by_mechanism.{mechanism}.by_rate",
        )
        if set(rate_rows) != rate_keys:
            raise ReportSchemaError(f"robustness_review_grid statistics rate rows differ for {mechanism!r}")
        for rate_key, rate_row_value in rate_rows.items():
            rate_row = _review_mapping(
                rate_row_value,
                field=f"statistics.by_mechanism.{mechanism}.by_rate.{rate_key}",
            )
            if rate_row.get("n_seeds") != n_seeds:
                raise ReportSchemaError(
                    "robustness_review_grid statistics row has a mismatched seed count: "
                    f"{mechanism!r}/{rate_key!r}"
                )
            method_rows = _review_mapping(
                rate_row.get("methods"),
                field=f"statistics.by_mechanism.{mechanism}.by_rate.{rate_key}.methods",
            )
            if set(method_rows) != set(robust_methods):
                raise ReportSchemaError(
                    "robustness_review_grid statistics row has missing or unexpected methods: "
                    f"{mechanism!r}/{rate_key!r}"
                )
            for method, method_value in method_rows.items():
                method_row = _review_mapping(
                    method_value,
                    field=(f"statistics.by_mechanism.{mechanism}.by_rate.{rate_key}.methods.{method}"),
                )
                contrast = _review_list(
                    method_row.get("contrast_by_seed"),
                    field=f"statistics contrast_by_seed {mechanism}/{rate_key}/{method}",
                )
                if len(contrast) != n_seeds:
                    raise ReportSchemaError(
                        "robustness_review_grid method contrast has a wrong seed count: "
                        f"{mechanism!r}/{rate_key!r}/{method!r}"
                    )
                for index, value in enumerate(contrast):
                    _review_number(
                        value,
                        field=(f"statistics contrast_by_seed {mechanism}/{rate_key}/{method}[{index}]"),
                    )
                summary = _review_mapping(
                    method_row.get("summary"),
                    field=f"statistics summary {mechanism}/{rate_key}/{method}",
                )
                if summary.get("n") != n_seeds:
                    raise ReportSchemaError(
                        "robustness_review_grid summary has a mismatched seed count: "
                        f"{mechanism!r}/{rate_key!r}/{method!r}"
                    )
                summary_mean = _review_number(summary.get("mean"), field="statistics summary mean")
                summary_lo = _review_number(summary.get("ci_lo"), field="statistics summary ci_lo")
                summary_hi = _review_number(summary.get("ci_hi"), field="statistics summary ci_hi")
                mcse = _review_number(summary.get("mcse"), field="statistics summary mcse")
                mde = _review_number(summary.get("mde"), field="statistics summary mde")
                if summary_lo > summary_mean or summary_mean > summary_hi or mcse < 0 or mde < 0:
                    raise ReportSchemaError(
                        "robustness_review_grid summary has invalid uncertainty ordering: "
                        f"{mechanism!r}/{rate_key!r}/{method!r}"
                    )
                observed_cell_mcses.append(mcse)
                interval = _review_list(
                    method_row.get("contrast_ci"),
                    field=f"statistics contrast_ci {mechanism}/{rate_key}/{method}",
                )
                if len(interval) != 2:
                    raise ReportSchemaError(
                        "robustness_review_grid method interval must have exactly two values: "
                        f"{mechanism!r}/{rate_key!r}/{method!r}"
                    )
                interval_lo = _review_number(interval[0], field="statistics contrast_ci lower")
                interval_hi = _review_number(interval[1], field="statistics contrast_ci upper")
                if not (
                    math.isclose(interval_lo, summary_lo, abs_tol=1e-12)
                    and math.isclose(interval_hi, summary_hi, abs_tol=1e-12)
                ):
                    raise ReportSchemaError(
                        "robustness_review_grid method interval disagrees with its summary: "
                        f"{mechanism!r}/{rate_key!r}/{method!r}"
                    )

    controls = _review_mapping(payload["controls"], field="controls")
    observed = _review_string_set(
        controls.get("observed_attack_mechanisms"), field="controls.observed_attack_mechanisms"
    )
    if observed != declared_attacks or observed != conditional_attacks | directional:
        raise ReportSchemaError(
            "robustness_review_grid observed attack controls do not match the emitted components"
        )
    if controls.get("all_declared_attack_controls_present") is not True:
        raise ReportSchemaError("robustness_review_grid controls must prove declared attack coverage")
    if controls.get("conditional_zero_robustness_control_passed") is not True:
        raise ReportSchemaError("robustness_review_grid controls must retain the zero-robustness check")

    precision = _review_mapping(payload["precision_plan"], field="precision_plan")
    observed_mcse = _review_number(
        precision.get("observed_max_mcse"), field="precision_plan.observed_max_mcse"
    )
    if observed_mcse < 0.0:
        raise ReportSchemaError("robustness_review_grid precision plan must report a non-negative MCSE")
    expected_signed_cells = len(directional) * len(rates) * len(robust_methods)
    signed_cells = precision.get("n_signed_method_rate_cells")
    if (
        isinstance(signed_cells, bool)
        or not isinstance(signed_cells, int)
        or signed_cells != expected_signed_cells
    ):
        raise ReportSchemaError(
            "robustness_review_grid precision plan has a mismatched signed method-rate cell count"
        )
    if not observed_cell_mcses or not math.isclose(
        observed_mcse,
        max(observed_cell_mcses),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ReportSchemaError(
            "robustness_review_grid precision plan maximum MCSE disagrees with signed method-rate summaries"
        )
    target = precision.get("target_max_mcse")
    target_status = precision.get("target_status")
    target_met = precision.get("target_met")
    if target is not None:
        target_value = _review_number(target, field="precision_plan.target_max_mcse")
        if (
            target_value <= 0.0
            or target_status != "met"
            or target_met is not True
            or observed_mcse > target_value
        ):
            raise ReportSchemaError("robustness_review_grid precision plan has an unmet MCSE target")
    elif target_status != "not_evaluated" or target_met is not None:
        raise ReportSchemaError(
            "robustness_review_grid precision plan must mark an absent target as not_evaluated"
        )
