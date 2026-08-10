"""Seeded conditional-world and proper-score extensions (MED-1/MED-2 slices).

The original heuristic characterization fixes one hidden state, attack target,
observability level, and weight geometry.  This module expands that witness to a
pre-registered finite grid while preserving the seed as the independent unit.
Each seed averages its nested trial rows before contrasts and bootstrap
intervals are formed.

The paired primary estimand for the world grid is the reduction in attacked
belief error, ``(1 - q_naive(true)) - (1 - q_robust(true))``.  The companion
proper-score report uses the paired per-seed difference in categorical log score
as its primary belief-quality estimand; Brier score and ECE remain secondary.
Neither report grants a universal robustness, calibration, objective, or
multi-machine claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from ..aggregation import log_linear_pool, robust_aggregate
from ..scoring import deterministic_score_controls, reliability_curve, summarize_scores
from ..statistics import bootstrap_ci, paired_test

ArrayF = np.ndarray
Attack = Literal["clean", "confident_wrong", "permutation", "label_noise", "uniform"]
_N_STATES = 4
_BOOTSTRAP_ALPHA = 0.05
_N_BOOT = 2000


@dataclass(frozen=True)
class ConditionalScenario:
    """One preregistered world/target/observability/weighting cell."""

    scenario_id: str
    true_state: int
    target_state: int
    observability: float
    attack: Attack
    adversary_weight: float
    n_contaminated: int


def conditional_scenario_grid() -> tuple[ConditionalScenario, ...]:
    """Return the source-owned finite grid before results are inspected."""
    rows: list[ConditionalScenario] = []
    attacks: tuple[Attack, ...] = (
        "clean",
        "confident_wrong",
        "permutation",
        "label_noise",
        "uniform",
    )
    for true_state in (0, 1):
        target = (true_state + 1) % _N_STATES
        for observability in (0.45, 0.70):
            for attack in attacks:
                for adversary_weight in (0.5, 1.0):
                    n_contaminated = 0 if attack == "clean" else 2
                    suffix = f"s{true_state}_t{target}_o{int(observability * 100)}"
                    weight_label = "half" if adversary_weight < 1.0 else "full"
                    rows.append(
                        ConditionalScenario(
                            scenario_id=f"{suffix}_{attack}_{weight_label}",
                            true_state=true_state,
                            target_state=target,
                            observability=observability,
                            attack=attack,
                            adversary_weight=adversary_weight,
                            n_contaminated=n_contaminated,
                        )
                    )
    return tuple(rows)


def _honest_local_posteriors(
    rng: np.random.Generator,
    *,
    n_agents: int,
    true_state: int,
    observability: float,
) -> ArrayF:
    """Draw a jittered honest colony with declared observation acuity."""
    base = np.full((_N_STATES,), (1.0 - observability) / (_N_STATES - 1))
    base[true_state] = observability
    rows = np.tile(base, (n_agents, 1))
    rows *= rng.uniform(0.97, 1.03, size=rows.shape)
    rows = np.clip(rows, 1e-12, None)
    return rows / rows.sum(axis=1, keepdims=True)


def _attack_belief(scenario: ConditionalScenario) -> ArrayF:
    """Construct one declared contamination mechanism."""
    true_state = scenario.true_state
    target = scenario.target_state
    if scenario.attack == "clean":
        belief = np.full(_N_STATES, (1.0 - scenario.observability) / (_N_STATES - 1))
        belief[true_state] = scenario.observability
        return belief
    if scenario.attack == "uniform":
        return np.full(_N_STATES, 1.0 / _N_STATES)
    if scenario.attack == "label_noise":
        belief = np.full(_N_STATES, 1e-12)
        belief[true_state] += 0.5
        belief[target] += 0.5
        return belief / belief.sum()
    if scenario.attack == "permutation":
        belief = np.full(_N_STATES, (1.0 - scenario.observability) / (_N_STATES - 1))
        belief[true_state] = scenario.observability
        belief[[true_state, target]] = belief[[target, true_state]]
        return belief / belief.sum()
    if scenario.attack == "confident_wrong":
        belief = np.full(_N_STATES, 0.01 / (_N_STATES - 1))
        belief[target] = 0.99
        return belief
    raise ValueError(f"unknown attack mechanism: {scenario.attack}")


def _trial(
    rng: np.random.Generator,
    scenario: ConditionalScenario,
    *,
    n_agents: int,
    robustness: float,
) -> dict[str, Any]:
    honest = _honest_local_posteriors(
        rng,
        n_agents=n_agents - scenario.n_contaminated,
        true_state=scenario.true_state,
        observability=scenario.observability,
    )
    if scenario.n_contaminated:
        attack = np.tile(_attack_belief(scenario), (scenario.n_contaminated, 1))
        local_posteriors = np.vstack([honest, attack])
    else:
        local_posteriors = honest
    base_weights = np.concatenate(
        [
            np.ones(honest.shape[0], dtype=np.float64),
            np.full(scenario.n_contaminated, scenario.adversary_weight, dtype=np.float64),
        ]
    )
    naive = log_linear_pool(
        local_posteriors=local_posteriors,
        base_weights=base_weights,
    )
    robust = robust_aggregate(
        local_posteriors=local_posteriors,
        base_weights=base_weights,
        robustness=robustness,
    ).consensus
    true_state = scenario.true_state
    zero = robust_aggregate(
        local_posteriors=local_posteriors,
        base_weights=base_weights,
        robustness=0.0,
    ).consensus
    return {
        "naive": naive,
        "robust": robust,
        "naive_error": float(1.0 - naive[true_state]),
        "robust_error": float(1.0 - robust[true_state]),
        # Primary estimand is naive error minus robust error, which is
        # algebraically the robust-minus-naive true-state mass contrast.
        "contrast": float(robust[true_state] - naive[true_state]),
        "zero_control_error": float(np.max(np.abs(zero - naive))),
    }


def _paired_summary(first: ArrayF, second: ArrayF) -> dict[str, float]:
    """Return a stable paired-test summary, including the all-tied control."""
    if np.allclose(first, second, atol=0.0, rtol=0.0):
        return {"statistic": 0.0, "pvalue": 1.0, "effect_size": 0.0}
    return paired_test(first, second)


def _validate_run_parameters(
    n_seeds: int, n_trials: int, n_agents: int, robustness: float
) -> None:
    """Validate shared budgets before either experiment allocates work."""
    if isinstance(n_seeds, bool) or not isinstance(n_seeds, int) or n_seeds < 2:
        raise ValueError("n_seeds must be an integer >= 2")
    if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials < 1:
        raise ValueError("n_trials must be a positive integer")
    if isinstance(n_agents, bool) or not isinstance(n_agents, int) or n_agents < 3:
        raise ValueError("n_agents must be an integer >= 3")
    if not np.isfinite(robustness) or robustness < 0.0:
        raise ValueError("robustness must be finite and non-negative")


def run_conditional_world_generalization(
    seed: int = 0,
    *,
    n_seeds: int = 16,
    n_trials: int = 12,
    n_agents: int = 7,
    robustness: float = 1.5,
) -> dict[str, Any]:
    """Run the pre-registered world/target/observability attack grid.

    The returned ``by_scenario`` cells contain one value per independent seed;
    trial-level observations remain nested and are not treated as replicates.
    ``contrast`` is positive when robust aggregation assigns more true-state
    mass than the naive pool.
    """
    _validate_run_parameters(n_seeds, n_trials, n_agents, robustness)

    scenarios = conditional_scenario_grid()
    by_scenario: dict[str, dict[str, Any]] = {}
    max_zero_error = 0.0
    for scenario_index, scenario in enumerate(scenarios):
        naive_by_seed: list[float] = []
        robust_by_seed: list[float] = []
        contrast_by_seed: list[float] = []
        for seed_index in range(n_seeds):
            rng = np.random.default_rng(
                int(seed) + seed_index * 1_000_003 + scenario_index * 10_007
            )
            trials = [
                _trial(rng, scenario, n_agents=n_agents, robustness=robustness)
                for _ in range(n_trials)
            ]
            naive_errors = np.asarray([row["naive_error"] for row in trials])
            robust_errors = np.asarray([row["robust_error"] for row in trials])
            naive_by_seed.append(float(naive_errors.mean()))
            robust_by_seed.append(float(robust_errors.mean()))
            contrast_by_seed.append(float(np.mean([row["contrast"] for row in trials])))
            max_zero_error = max(max_zero_error, max(row["zero_control_error"] for row in trials))
        contrast_array = np.asarray(contrast_by_seed, dtype=np.float64)
        ci = bootstrap_ci(
            contrast_array,
            alpha=_BOOTSTRAP_ALPHA,
            n_boot=_N_BOOT,
            rng=np.random.default_rng(int(seed) + scenario_index + 99_991),
        )
        by_scenario[scenario.scenario_id] = {
            **asdict(scenario),
            "n_seeds": int(n_seeds),
            "n_trials": int(n_trials),
            "naive_error_by_seed": naive_by_seed,
            "robust_error_by_seed": robust_by_seed,
            "contrast_by_seed": contrast_by_seed,
            "contrast_mean": float(contrast_array.mean()),
            "contrast_ci": [float(ci[0]), float(ci[1])],
            "paired_error_test": _paired_summary(
                np.asarray(robust_by_seed), np.asarray(naive_by_seed)
            ),
            "target_is_distinct": bool(scenario.target_state != scenario.true_state),
            "finite_grid_cell": True,
        }

    return {
        "schema_version": "1.0",
        "seed": int(seed),
        "n_seeds": int(n_seeds),
        "n_trials": int(n_trials),
        "n_agents": int(n_agents),
        "n_states": _N_STATES,
        "robustness": float(robustness),
        "primary_estimand": "naive true-state error minus robust true-state error",
        "independent_unit": "seeded world/scenario row",
        "claim_status": "conditional_finite_grid",
        "grid": [asdict(scenario) for scenario in scenarios],
        "by_scenario": by_scenario,
        "controls": {
            "robustness_zero_recovers_log_pool": bool(max_zero_error <= 1e-12),
            "max_zero_control_error": float(max_zero_error),
            "all_attack_targets_distinct": bool(
                all(row["target_is_distinct"] for row in by_scenario.values())
            ),
            "seed_is_independent_unit": True,
        },
    }


def run_belief_quality_sensitivity(
    seed: int = 0,
    *,
    n_seeds: int = 16,
    n_trials: int = 12,
    n_agents: int = 7,
    robustness: float = 1.5,
) -> dict[str, Any]:
    """Score naive and robust consensus beliefs on a fixed conditional subset.

    The log-score contrast is primary; Brier and ECE are paired secondary
    diagnostics. Deterministic oracle/uniform/confident-wrong controls are
    included in the same report so score ordering can be falsified independently
    of the attack experiment.
    """
    _validate_run_parameters(n_seeds, n_trials, n_agents, robustness)
    scenarios = tuple(
        scenario
        for scenario in conditional_scenario_grid()
        if scenario.observability == 0.70
        and scenario.adversary_weight == 1.0
        and scenario.attack in ("clean", "confident_wrong", "uniform")
    )
    by_scenario: dict[str, dict[str, Any]] = {}
    for scenario_index, scenario in enumerate(scenarios):
        naive_log: list[float] = []
        robust_log: list[float] = []
        naive_brier: list[float] = []
        robust_brier: list[float] = []
        naive_ece: list[float] = []
        robust_ece: list[float] = []
        for seed_index in range(n_seeds):
            rng = np.random.default_rng(
                int(seed) + seed_index * 1_000_003 + scenario_index * 10_007
            )
            naive_rows: list[ArrayF] = []
            robust_rows: list[ArrayF] = []
            labels: list[int] = []
            for _ in range(n_trials):
                trial = _trial(rng, scenario, n_agents=n_agents, robustness=robustness)
                naive_rows.append(trial["naive"])
                robust_rows.append(trial["robust"])
                labels.append(scenario.true_state)
            states = np.asarray(labels, dtype=np.int64)
            naive_summary = summarize_scores(np.vstack(naive_rows), states)
            robust_summary = summarize_scores(np.vstack(robust_rows), states)
            naive_log.append(float(naive_summary["mean_log_score"]))
            robust_log.append(float(robust_summary["mean_log_score"]))
            naive_brier.append(float(naive_summary["mean_brier_score"]))
            robust_brier.append(float(robust_summary["mean_brier_score"]))
            naive_ece.append(float(naive_summary["ece"]))
            robust_ece.append(float(robust_summary["ece"]))
        log_contrast = np.asarray(robust_log) - np.asarray(naive_log)
        ci = bootstrap_ci(
            log_contrast,
            alpha=_BOOTSTRAP_ALPHA,
            n_boot=_N_BOOT,
            rng=np.random.default_rng(int(seed) + scenario_index + 199_991),
        )
        by_scenario[scenario.scenario_id] = {
            **asdict(scenario),
            "primary_score": "categorical_log_score",
            "naive_log_score_by_seed": naive_log,
            "robust_log_score_by_seed": robust_log,
            "log_score_contrast_by_seed": log_contrast.tolist(),
            "log_score_contrast_mean": float(log_contrast.mean()),
            "log_score_contrast_ci": [float(ci[0]), float(ci[1])],
            "paired_log_score_test": _paired_summary(np.asarray(naive_log), np.asarray(robust_log)),
            "naive_brier_by_seed": naive_brier,
            "robust_brier_by_seed": robust_brier,
            "naive_ece_by_seed": naive_ece,
            "robust_ece_by_seed": robust_ece,
            "n_seeds": int(n_seeds),
            "n_trials": int(n_trials),
            "score_unit": "seed-level mean over nested trials",
        }

    control_scores: dict[str, dict[str, Any]] = {}
    for control_name in ("oracle", "uniform", "confident_wrong"):
        seed_scores: list[float] = []
        seed_brier: list[float] = []
        seed_ece: list[float] = []
        all_rows: list[ArrayF] = []
        all_states: list[int] = []
        for seed_index in range(n_seeds):
            states = np.arange(n_trials, dtype=np.int64) % _N_STATES
            rows = np.vstack(
                [deterministic_score_controls(_N_STATES, int(state))[control_name] for state in states]
            )
            all_rows.append(rows)
            all_states.extend(int(state) for state in states)
            summary = summarize_scores(rows, states)
            seed_scores.append(float(summary["mean_log_score"]))
            seed_brier.append(float(summary["mean_brier_score"]))
            seed_ece.append(float(summary["ece"]))
        ci = bootstrap_ci(
            np.asarray(seed_scores),
            alpha=_BOOTSTRAP_ALPHA,
            n_boot=_N_BOOT,
            rng=np.random.default_rng(int(seed) + 299_991 + n_seeds),
        )
        control_scores[control_name] = {
            "mean_log_score": float(np.mean(seed_scores)),
            "log_score_ci": [float(ci[0]), float(ci[1])],
            "mean_brier_score": float(np.mean(seed_brier)),
            "mean_ece": float(np.mean(seed_ece)),
            "log_score_by_seed": seed_scores,
            "reliability": reliability_curve(
                np.vstack(all_rows), np.asarray(all_states, dtype=np.int64), n_bins=10
            ),
            "n_seeds": int(n_seeds),
            "n_trials": int(n_trials),
        }
    controls = {
        "oracle_best_log_score": bool(
            control_scores["oracle"]["mean_log_score"]
            > control_scores["uniform"]["mean_log_score"]
            > control_scores["confident_wrong"]["mean_log_score"]
        ),
        "confident_wrong_worse_than_uniform": bool(
            control_scores["confident_wrong"]["mean_log_score"]
            < control_scores["uniform"]["mean_log_score"]
        ),
        "seed_is_independent_unit": True,
    }
    return {
        "schema_version": "1.0",
        "seed": int(seed),
        "n_seeds": int(n_seeds),
        "n_trials": int(n_trials),
        "n_agents": int(n_agents),
        "robustness": float(robustness),
        "primary_estimand": "paired seed-level robust minus naive categorical log score",
        "independent_unit": "seed",
        "by_scenario": by_scenario,
        "controls": controls,
        "control_scores": control_scores,
    }


__all__ = [
    "ConditionalScenario",
    "conditional_scenario_grid",
    "run_belief_quality_sensitivity",
    "run_conditional_world_generalization",
]
