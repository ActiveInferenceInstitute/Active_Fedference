"""Bounded source-bound review grid for the manuscript red-team pass.

This module composes the already declared finite conditional-world and onset
mechanisms. It adds replication budget and a stable seed schedule, but does not
add an external dataset, a new attack implementation, or a tuning loop. The
returned payload keeps seeds as the inferential unit and trials nested within a
seed/cell.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..statistics import (
    bh_fdr,
    interpret_effect_size,
    paired_test,
    power_analysis,
    summary_statistics,
)
from ._common import _N_BOOT, _finite_d_equivalent
from .conditional_world import run_conditional_world_generalization
from .gallery import _DIRECTIONAL_KINDS, _ENTROPY_KINDS, run_robustness_onset

DEFAULT_REVIEW_GRID_RATES: tuple[float, ...] = (
    0.0,
    0.2,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
)
REVIEW_GRID_ATTACKS: tuple[str, ...] = (
    "clean",
    "confident_wrong",
    "permutation",
    "byzantine",
    "drift",
    "label_noise",
    "uniform",
)


def _paired_seed_summary(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    """Return a deterministic paired summary with an explicit all-tied branch."""
    if np.all(first == second):
        return {"statistic": 0.0, "pvalue": 1.0, "effect_size": 0.0}
    return paired_test(first, second)


def _review_statistics(
    onset: dict[str, Any],
    *,
    seed: int,
    divergences: tuple[str, ...],
    fdr_alpha: float,
    power_alpha: float,
    planning_alternative: str,
) -> dict[str, Any]:
    """Derive selection-free seed contrasts and BH-owned rate families."""
    robust_methods = [method for method in divergences if method != "KLD"]
    by_mechanism: dict[str, Any] = {}
    for kind_index, kind in enumerate(onset["kinds"]):
        cells = onset["by_kind"][kind]
        method_pvalues: dict[str, list[float]] = {method: [] for method in robust_methods}
        method_rows: dict[str, dict[str, Any]] = {}
        for rate_index, rate in enumerate(onset["rates"]):
            rate_key = f"{float(rate):g}"
            cell = cells["by_rate"][rate_key]
            naive = np.asarray(cell["per_seed_naive"], dtype=np.float64)
            method_rows[rate_key] = {
                "n_seeds": int(naive.size),
                "methods": {},
            }
            for method_index, method in enumerate(robust_methods):
                robust = np.asarray(cell["per_seed_robust"][method], dtype=np.float64)
                contrasts = robust - naive
                summary = summary_statistics(
                    contrasts,
                    n_boot=_N_BOOT,
                    rng=np.random.default_rng(
                        seed + 4_000_000 + kind_index * 100_000 + rate_index * 1_000 + method_index
                    ),
                )
                test = _paired_seed_summary(naive, robust)
                d_equivalent = _finite_d_equivalent(test["effect_size"])
                planning = power_analysis(
                    d_equivalent,
                    int(naive.size),
                    alpha=power_alpha,
                    alternative=planning_alternative,
                )
                method_pvalues[method].append(float(test["pvalue"]))
                method_rows[rate_key]["methods"][method] = {
                    "contrast_by_seed": contrasts.tolist(),
                    "summary": summary,
                    "contrast_ci": [
                        float(summary["ci_lo"]),
                        float(summary["ci_hi"]),
                    ],
                    "paired_test": test,
                    "d_equivalent": float(d_equivalent),
                    "effect_label": interpret_effect_size(d_equivalent),
                    "planning_power": float(planning["power"]),
                    "planning_n_for_80_power": int(planning["n_for_80_power"]),
                    "bh_family_id": f"{kind}:{method}:declared_rates",
                }
        for method in robust_methods:
            family = bh_fdr(np.asarray(method_pvalues[method]), alpha=fdr_alpha)
            for rate_index, rate in enumerate(onset["rates"]):
                rate_key = f"{float(rate):g}"
                row = method_rows[rate_key]["methods"][method]
                row["raw_pvalue"] = float(family["pvalues"][rate_index])
                row["qvalue"] = float(family["qvalues"][rate_index])
                row["rejected"] = bool(family["rejected"][rate_index])
        by_mechanism[kind] = {
            "rates": [float(rate) for rate in onset["rates"]],
            "methods": robust_methods,
            "by_rate": method_rows,
            "bh_family_ownership": (
                "one BH family per attack mechanism and robust method across "
                "the declared rates; no families are pooled across cells"
            ),
        }
    return {
        "selection_free": True,
        "estimand": "seed-level robust-minus-naive true-state mass",
        "replication_unit": "seed-level mean over nested trials",
        "paired_test_alternative": "two-sided",
        "fdr_alpha": float(fdr_alpha),
        "power_alpha": float(power_alpha),
        "planning_alternative": str(planning_alternative),
        "bootstrap_interval": "95% percentile bootstrap over seeds",
        "d_equivalent_status": (
            "secondary rank-biserial-derived display transform; finite saturation "
            "sentinels are not literal million-scale effects"
        ),
        "mde_definition": (
            "two-sided normal-approximation observed-design MDE from the "
            "seed-level contrast standard deviation"
        ),
        "bh_family_ownership": (
            "BH is applied within each attack-mechanism × method rate family; "
            "cells sharing design structure are not treated as independent families"
        ),
        "by_mechanism": by_mechanism,
    }


def _observed_attack_controls(conditional: dict[str, Any], onset: dict[str, Any]) -> dict[str, Any]:
    """Derive, rather than assert, coverage of the registered attack controls.

    The review grid composes two producers.  Its control receipt must therefore
    inspect their emitted scenario/profile labels; constants alone would make a
    tautological coverage claim if either producer silently dropped a cell.
    """
    conditional_rows = conditional.get("by_scenario")
    if not isinstance(conditional_rows, dict) or not conditional_rows:
        raise ValueError("review grid conditional component has no scenario rows")
    conditional_attacks: set[str] = set()
    for scenario_id, row in conditional_rows.items():
        if not isinstance(row, dict) or not isinstance(row.get("attack"), str):
            raise ValueError(f"review grid conditional scenario {scenario_id!r} lacks an attack label")
        conditional_attacks.add(row["attack"])

    profile_rows = onset.get("by_kind")
    if not isinstance(profile_rows, dict) or not profile_rows:
        raise ValueError("review grid rate component has no mechanism profiles")
    rate_profile_attacks = {str(kind) for kind in profile_rows}

    observed = conditional_attacks | rate_profile_attacks
    declared = set(REVIEW_GRID_ATTACKS)
    if observed != declared:
        missing = sorted(declared - observed)
        unexpected = sorted(observed - declared)
        raise ValueError(
            "review grid attack coverage differs from its declared finite grid: "
            f"missing={missing}, unexpected={unexpected}"
        )
    conditional_controls = conditional.get("controls")
    if not isinstance(conditional_controls, dict) or (
        conditional_controls.get("robustness_zero_recovers_log_pool") is not True
    ):
        raise ValueError("review grid conditional component did not carry the zero-robustness control")
    return {
        "conditional_attack_mechanisms": sorted(conditional_attacks),
        "rate_profile_attack_mechanisms": sorted(rate_profile_attacks),
        "observed_attack_mechanisms": sorted(observed),
        "all_declared_attack_controls_present": True,
        "conditional_zero_robustness_control_passed": True,
    }


def _precision_plan(statistics: dict[str, Any], *, target_max_mcse: float | None) -> dict[str, Any]:
    """Summarize the registered MCSE stopping rule over every signed contrast."""
    by_mechanism = statistics.get("by_mechanism")
    if not isinstance(by_mechanism, dict) or not by_mechanism:
        raise ValueError("review grid statistics has no mechanism rows")
    mcse_values: list[float] = []
    for mechanism, mechanism_rows in by_mechanism.items():
        if not isinstance(mechanism_rows, dict):
            raise ValueError(f"review grid statistics mechanism {mechanism!r} is malformed")
        by_rate = mechanism_rows.get("by_rate")
        if not isinstance(by_rate, dict) or not by_rate:
            raise ValueError(f"review grid statistics mechanism {mechanism!r} has no rate rows")
        for rate, rate_row in by_rate.items():
            if not isinstance(rate_row, dict):
                raise ValueError(f"review grid statistics row {mechanism!r}/{rate!r} is malformed")
            methods = rate_row.get("methods")
            if not isinstance(methods, dict) or not methods:
                raise ValueError(f"review grid statistics row {mechanism!r}/{rate!r} has no methods")
            for method, method_row in methods.items():
                if not isinstance(method_row, dict):
                    raise ValueError(
                        f"review grid statistics method row is malformed: {mechanism!r}/{rate!r}/{method!r}"
                    )
                summary = method_row.get("summary")
                if not isinstance(summary, dict):
                    raise ValueError(
                        f"review grid statistics method summary is missing: {mechanism!r}/{rate!r}/{method!r}"
                    )
                mcse = float(summary.get("mcse", float("nan")))
                if not np.isfinite(mcse) or mcse < 0.0:
                    raise ValueError(
                        f"review grid statistics method MCSE is invalid: {mechanism!r}/{rate!r}/{method!r}"
                    )
                mcse_values.append(mcse)
    observed = max(mcse_values)
    met: bool | None = None if target_max_mcse is None else observed <= target_max_mcse
    return {
        "replication_unit": "seed-level contrast with trials nested within seed/cell",
        "target_max_mcse": target_max_mcse,
        "observed_max_mcse": observed,
        "target_met": met,
        "target_status": "not_evaluated" if target_max_mcse is None else ("met" if met else "unmet"),
        "n_signed_method_rate_cells": len(mcse_values),
    }


def run_review_grid(
    seed: int = 0,
    *,
    n_seeds: int = 64,
    n_trials: int = 12,
    n_agents: int = 7,
    robustness: float = 1.5,
    rates: tuple[float, ...] = DEFAULT_REVIEW_GRID_RATES,
    divergences: tuple[str, ...] = ("KLD", "RKL", "AR", "beta", "rcce"),
    target_max_mcse: float | None = None,
    fdr_alpha: float = 0.05,
    power_alpha: float = 0.05,
    planning_alternative: str = "greater",
) -> dict[str, Any]:
    """Run the expanded finite review grid with selection-free payloads.

    The conditional component is the existing 40-cell grid expanded to the
    declared seed budget. The rate component is the existing directional
    onset profile over the declared rates; clean, entropy, and permutation
    controls remain in the conditional component. Robust operating points are
    fixed by the existing experiment mapping and are never selected from the
    resulting contrasts.
    """
    if n_seeds < 2 or n_trials < 2:
        raise ValueError("n_seeds must be >= 2 and n_trials must be >= 2")
    if not rates or tuple(rates) != tuple(sorted(rates)):
        raise ValueError("rates must be a non-empty sorted tuple")
    if len(set(rates)) != len(rates):
        raise ValueError("rates must not contain duplicates")
    if any(not 0.0 <= float(rate) <= 1.0 for rate in rates):
        raise ValueError("rates must lie in [0, 1]")
    if not divergences or any(not isinstance(method, str) or not method for method in divergences):
        raise ValueError("divergences must be non-empty strings")
    if len(set(divergences)) != len(divergences):
        raise ValueError("divergences must not contain duplicates")
    if divergences.count("KLD") != 1:
        raise ValueError("divergences must include KLD exactly once")
    for name, alpha in (("fdr_alpha", fdr_alpha), ("power_alpha", power_alpha)):
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not np.isfinite(alpha):
            raise ValueError(f"{name} must be finite and lie in (0, 1)")
        if not 0.0 < float(alpha) < 1.0:
            raise ValueError(f"{name} must be finite and lie in (0, 1)")
    if planning_alternative not in ("greater", "less", "two-sided"):
        raise ValueError("planning_alternative must be 'greater', 'less' or 'two-sided'")
    fdr_alpha = float(fdr_alpha)
    power_alpha = float(power_alpha)
    if target_max_mcse is not None:
        if (
            isinstance(target_max_mcse, bool)
            or not isinstance(target_max_mcse, (int, float))
            or not np.isfinite(target_max_mcse)
            or target_max_mcse <= 0.0
        ):
            raise ValueError("target_max_mcse must be finite and > 0 when supplied")
        target_max_mcse = float(target_max_mcse)

    # Give the two composed surfaces disjoint numeric seed ranges. The
    # resulting streams are reproducible and non-overlapping by construction;
    # the report still refuses to treat cells sharing design structure as
    # independent scientific populations.
    conditional_seed = int(seed) + 100_000_000
    rate_seed = int(seed) + 200_000_000
    conditional = run_conditional_world_generalization(
        conditional_seed,
        n_seeds=n_seeds,
        n_trials=n_trials,
        n_agents=n_agents,
        robustness=robustness,
    )
    onset = run_robustness_onset(
        rate_seed,
        kinds=_DIRECTIONAL_KINDS,
        rates=rates,
        n_agents=n_agents,
        n_contaminated=max(1, n_agents // 3),
        divergences=divergences,
        n_trials=n_trials,
        n_seeds=n_seeds,
        include_seed_data=True,
    )
    review_statistics = _review_statistics(
        onset,
        seed=seed,
        divergences=divergences,
        fdr_alpha=fdr_alpha,
        power_alpha=power_alpha,
        planning_alternative=planning_alternative,
    )
    control_receipt = _observed_attack_controls(conditional, onset)
    precision_plan = _precision_plan(review_statistics, target_max_mcse=target_max_mcse)
    if target_max_mcse is not None and precision_plan["target_met"] is not True:
        raise ValueError(
            "review grid precision target was not met: observed maximum MCSE "
            f"{precision_plan['observed_max_mcse']:.6g} exceeds "
            f"{target_max_mcse:.6g}"
        )
    return {
        "schema_version": "1.1",
        "analysis_profile": (
            "source_bound_review_grid" if target_max_mcse is not None else "diagnostic_review_grid"
        ),
        "seed": int(seed),
        "n_seeds": int(n_seeds),
        "n_trials": int(n_trials),
        "n_agents": int(n_agents),
        "robustness": float(robustness),
        "rates": [float(rate) for rate in rates],
        "divergences": [str(label) for label in divergences],
        "attack_mechanisms": list(REVIEW_GRID_ATTACKS),
        "directional_mechanisms": list(_DIRECTIONAL_KINDS),
        "entropy_controls": list(_ENTROPY_KINDS),
        "primary_estimand": "seed-level robust minus naive true-state mass within each declared cell",
        "independent_unit": "seed within a declared scenario or rate cell",
        "trial_structure": (
            "n_trials nested within each seed/cell; no trial is promoted to an independent world"
        ),
        "selection_status": (
            "selection-free source payload; every configured non-KLD method is "
            "reported at every directional rate and no winner is used for inference"
        ),
        "seed_schedule": {
            "master_seed": int(seed),
            "conditional_component_seed": conditional_seed,
            "rate_component_seed": rate_seed,
            "conditional_seed_stride": 1_000_003,
            "scenario_seed_stride": 10_007,
            "rate_seed_schedule": (
                "existing onset schedule rate_component_seed .. rate_component_seed + n_seeds - 1"
            ),
            "independence_boundary": (
                "seed streams are deterministic and disjoint within cells; "
                "design cells are not assumed independent"
            ),
        },
        "conditional_world": conditional,
        "rate_profiles": onset,
        "statistics": review_statistics,
        "precision_plan": precision_plan,
        "controls": {
            "finite_existing_attack_union": control_receipt["all_declared_attack_controls_present"],
            "clean_control_present": "clean" in control_receipt["observed_attack_mechanisms"],
            "entropy_controls_present": all(
                attack in control_receipt["observed_attack_mechanisms"] for attack in _ENTROPY_KINDS
            ),
            "permutation_control_present": "permutation" in control_receipt["observed_attack_mechanisms"],
            **control_receipt,
            "robustness_zero_identity_checked_elsewhere": True,
            "external_data_used": False,
            "tuning_from_outcomes": False,
        },
    }


__all__ = [
    "DEFAULT_REVIEW_GRID_RATES",
    "REVIEW_GRID_ATTACKS",
    "run_review_grid",
]
