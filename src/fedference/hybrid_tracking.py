"""Minimal discrete-context continuous target-tracking task (MAJ-3).

The physical state contains position and velocity. Agents observe both with
Gaussian noise and infer a discrete dynamics context; each
:class:`~fedference.hybrid.HybridBelief` represents the next-position
distribution conditional on that context. The fused predictive belief drives a
bounded acceleration action.

This is a compact recovery/evaluation task, not evidence for general continuous
control or an objective for the robust hybrid heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .hybrid import HybridBelief, hybrid_aggregate, hybrid_log_linear_pool

TrackingMethod = Literal[
    "naive",
    "robust",
    "discrete-only",
    "continuous-only",
    "oracle-context",
]
_EPS = 1e-12


@dataclass(frozen=True)
class HybridTrackingConfig:
    """Deterministic task and observation settings."""

    n_agents: int = 5
    horizon: int = 24
    target_position: float = 1.0
    position_observation_var: float = 0.04
    velocity_observation_var: float = 0.01
    process_var: float = 0.0025
    context_accuracy: float = 0.85
    action_gain: float = 0.6
    max_acceleration: float = 0.4
    robustness: float = 1.5
    outlier_bias: float = 1.5

    def __post_init__(self) -> None:
        if (
            isinstance(self.n_agents, bool)
            or not isinstance(self.n_agents, (int, np.integer))
            or isinstance(self.horizon, bool)
            or not isinstance(self.horizon, (int, np.integer))
            or self.n_agents < 2
            or self.horizon < 2
        ):
            raise ValueError("n_agents and horizon must both be at least 2")
        object.__setattr__(self, "n_agents", int(self.n_agents))
        object.__setattr__(self, "horizon", int(self.horizon))
        for name in (
            "target_position",
            "position_observation_var",
            "velocity_observation_var",
            "process_var",
            "action_gain",
            "max_acceleration",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value)
                or (name != "target_position" and value <= 0.0)
            ):
                requirement = "finite" if name == "target_position" else "finite and positive"
                raise ValueError(f"{name} must be {requirement}")
            object.__setattr__(self, name, float(value))
        if (
            isinstance(self.context_accuracy, bool)
            or not isinstance(
                self.context_accuracy,
                (int, float, np.integer, np.floating),
            )
            or not np.isfinite(self.context_accuracy)
            or not 0.5 < self.context_accuracy < 1.0
        ):
            raise ValueError("context_accuracy must lie in (0.5, 1)")
        object.__setattr__(self, "context_accuracy", float(self.context_accuracy))
        if (
            isinstance(self.robustness, bool)
            or not isinstance(
                self.robustness,
                (int, float, np.integer, np.floating),
            )
            or not np.isfinite(self.robustness)
            or self.robustness < 0.0
        ):
            raise ValueError("robustness must be finite and non-negative")
        object.__setattr__(self, "robustness", float(self.robustness))
        if (
            isinstance(self.outlier_bias, bool)
            or not isinstance(
                self.outlier_bias,
                (int, float, np.integer, np.floating),
            )
            or not np.isfinite(self.outlier_bias)
        ):
            raise ValueError("outlier_bias must be finite")
        object.__setattr__(self, "outlier_bias", float(self.outlier_bias))


def _normal_density(value: float, mean: float, variance: float) -> float:
    return float(np.exp(-0.5 * (value - mean) ** 2 / variance) / np.sqrt(2.0 * np.pi * variance))


def _mixture_log_score(belief: HybridBelief, value: float) -> float:
    density = sum(
        float(belief.discrete[index])
        * _normal_density(
            value,
            float(belief.gaussian_mean[index]),
            float(belief.gaussian_var[index]),
        )
        for index in range(belief.n_components)
    )
    return float(np.log(max(density, _EPS)))


def _agent_belief(
    rng: np.random.Generator,
    *,
    position: float,
    velocity: float,
    context: int,
    config: HybridTrackingConfig,
    is_outlier: bool,
) -> HybridBelief:
    observed_position = position + rng.normal(0.0, np.sqrt(config.position_observation_var))
    observed_velocity = velocity + rng.normal(0.0, np.sqrt(config.velocity_observation_var))
    cue_correct = bool(rng.random() < config.context_accuracy)
    observed_context = context if cue_correct else 1 - context
    if is_outlier:
        observed_position += config.outlier_bias
        observed_context = 1 - context
    discrete = np.full(2, 1.0 - config.context_accuracy)
    discrete[observed_context] = config.context_accuracy
    context_acceleration = np.asarray([0.0, 0.12])
    means = observed_position + 0.9 * observed_velocity + context_acceleration
    variances = np.full(
        2,
        config.position_observation_var + 0.9**2 * config.velocity_observation_var + config.process_var,
    )
    return HybridBelief(discrete, means, variances)


def _consensus_for_method(
    local_posteriors: list[HybridBelief],
    *,
    method: TrackingMethod,
    context: int,
    config: HybridTrackingConfig,
) -> tuple[HybridBelief, np.ndarray]:
    """Build one member of the preregistered matched comparison family."""
    if method == "naive":
        return hybrid_log_linear_pool(local_posteriors), np.full(
            config.n_agents, 1.0 / config.n_agents
        )
    if method == "robust":
        result = hybrid_aggregate(local_posteriors, robustness=config.robustness)
        return result.consensus, result.normalized_effective_weights

    uniform = np.full(config.n_agents, 1.0 / config.n_agents)
    if method == "discrete-only":
        discrete = hybrid_log_linear_pool(local_posteriors).discrete
        means = np.asarray([np.mean(row.gaussian_mean) for row in local_posteriors])
        variances = np.asarray([np.mean(row.gaussian_var) for row in local_posteriors])
        mean = float(np.mean(means))
        variance = float(np.mean(variances + (means - mean) ** 2))
        return HybridBelief(discrete, np.full(2, mean), np.full(2, variance)), uniform
    if method == "continuous-only":
        component_means = np.concatenate([row.gaussian_mean for row in local_posteriors])
        component_variances = np.concatenate([row.gaussian_var for row in local_posteriors])
        mean = float(np.mean(component_means))
        variance = float(np.mean(component_variances + (component_means - mean) ** 2))
        return HybridBelief(np.full(2, 0.5), np.full(2, mean), np.full(2, variance)), uniform
    if method == "oracle-context":
        means = np.asarray([row.gaussian_mean[context] for row in local_posteriors])
        variances = np.asarray([row.gaussian_var[context] for row in local_posteriors])
        return HybridBelief(
            np.eye(2, dtype=np.float64)[context],
            np.full(2, float(np.mean(means))),
            np.full(2, float(np.mean(variances))),
        ), uniform
    raise ValueError(f"unknown tracking method {method!r}")


def run_hybrid_tracking(
    seed: int = 0,
    *,
    method: TrackingMethod = "naive",
    contaminate_one_agent: bool = True,
    config: HybridTrackingConfig | None = None,
) -> dict[str, object]:
    """Run one seeded closed-loop hybrid tracking episode."""
    if config is not None and not isinstance(config, HybridTrackingConfig):
        raise ValueError("config must be a HybridTrackingConfig or None")
    cfg = config or HybridTrackingConfig()
    if method not in ("naive", "robust", "discrete-only", "continuous-only", "oracle-context"):
        raise ValueError(
            "method must be 'naive', 'robust', 'discrete-only', 'continuous-only', or 'oracle-context'"
        )
    if not isinstance(contaminate_one_agent, bool):
        raise ValueError("contaminate_one_agent must be a boolean")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    position = -1.0
    velocity = 0.05
    log_scores: list[float] = []
    errors: list[float] = []
    known_context_component_errors: list[float] = []
    control_costs: list[float] = []
    predictive_risk_surrogates: list[float] = []
    contexts: list[int] = []
    trajectory: list[dict[str, float | int]] = []

    for step in range(cfg.horizon):
        context = int(step >= cfg.horizon // 2)
        contexts.append(context)
        local_posteriors = [
            _agent_belief(
                rng,
                position=position,
                velocity=velocity,
                context=context,
                config=cfg,
                is_outlier=contaminate_one_agent and agent == cfg.n_agents - 1,
            )
            for agent in range(cfg.n_agents)
        ]
        consensus, effective_weights = _consensus_for_method(
            local_posteriors,
            method=method,
            context=context,
            config=cfg,
        )
        uncontrolled_prediction = float(
            np.sum(consensus.discrete * consensus.gaussian_mean)
        )
        acceleration = float(
            np.clip(
                cfg.action_gain * (cfg.target_position - uncontrolled_prediction),
                -cfg.max_acceleration,
                cfg.max_acceleration,
            )
        )
        controlled_consensus = HybridBelief(
            consensus.discrete,
            consensus.gaussian_mean + acceleration,
            consensus.gaussian_var,
        )
        predicted_position = float(
            np.sum(
                controlled_consensus.discrete
                * controlled_consensus.gaussian_mean
            )
        )
        known_context_component_prediction = float(
            controlled_consensus.gaussian_mean[context]
        )
        context_drift = 0.0 if context == 0 else 0.12
        velocity = 0.9 * velocity + acceleration + context_drift + rng.normal(0.0, np.sqrt(cfg.process_var))
        position = position + velocity
        log_scores.append(_mixture_log_score(controlled_consensus, position))
        errors.append((predicted_position - position) ** 2)
        known_context_component_errors.append((known_context_component_prediction - position) ** 2)
        control_costs.append(acceleration**2)
        predictive_variance = float(
            np.sum(
                controlled_consensus.discrete
                * (
                    controlled_consensus.gaussian_var
                    + (controlled_consensus.gaussian_mean - predicted_position) ** 2
                )
            )
        )
        predictive_risk_surrogates.append(
            (predicted_position - cfg.target_position) ** 2 + predictive_variance
        )
        trajectory.append(
            {
                "step": step,
                "context": context,
                "position": float(position),
                "velocity": float(velocity),
                "uncontrolled_predicted_position": uncontrolled_prediction,
                "predicted_position": predicted_position,
                "acceleration": acceleration,
                "outlier_weight": float(effective_weights[-1]),
            }
        )

    return {
        "status": "ok",
        "method": method,
        "seed": int(seed),
        "n_agents": cfg.n_agents,
        "horizon": cfg.horizon,
        "contaminate_one_agent": contaminate_one_agent,
        "primary_estimand": ("mean on-policy next-position log score within one closed-loop episode"),
        "independent_unit": "seeded tracking world",
        "on_policy_mean_log_score": float(np.mean(log_scores)),
        "held_out_posterior_predictive_log_score": float(np.mean(log_scores)),
        "position_rmse": float(np.sqrt(np.mean(errors))),
        "known_context_component_rmse": float(np.sqrt(np.mean(known_context_component_errors))),
        "mean_control_cost": float(np.mean(control_costs)),
        "mean_predictive_risk_surrogate": float(np.mean(predictive_risk_surrogates)),
        "action_success": bool(abs(position - cfg.target_position) <= 1.0),
        "contexts": contexts,
        "trajectory": trajectory,
        "no_claim": (
            "minimal hybrid tracking does not establish general continuous "
            "control, held-out prediction, an oracle-control comparison, "
            "expected-free-energy equivalence, or a robust-server theorem"
        ),
    }


def run_hybrid_tracking_comparison(
    seed: int = 0,
    *,
    contaminate_one_agent: bool = True,
    config: HybridTrackingConfig | None = None,
) -> dict[str, object]:
    """Run the matched hybrid-control family for one seeded tracking world.

    The controls intentionally share the same task configuration and seed. A
    positive result would still be conditional on this small tracking world;
    the report therefore records recovery and singular-covariance gates before
    any method comparison.
    """
    cfg = config or HybridTrackingConfig()
    methods: tuple[TrackingMethod, ...] = (
        "naive",
        "robust",
        "discrete-only",
        "continuous-only",
        "oracle-context",
    )
    rows = {
        method: run_hybrid_tracking(
            seed,
            method=method,
            contaminate_one_agent=contaminate_one_agent,
            config=cfg,
        )
        for method in methods
    }
    singular_control: dict[str, object]
    try:
        HybridBelief(np.asarray([0.5, 0.5]), np.zeros(2), np.asarray([0.0, 1.0]))
    except ValueError as exc:
        singular_control = {
            "status": "rejected",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
    else:  # pragma: no cover - the constructor is required to fail closed
        singular_control = {"status": "accepted_unexpectedly"}
    return {
        "status": "ok",
        "seed": int(seed),
        "methods": rows,
        "method_order": list(methods),
        "primary_estimand": "held-out posterior-predictive log score per seeded tracking world",
        "independent_unit": "seeded tracking world",
        "recovery_gate": "hybrid_aggregate robustness=0 equals hybrid_log_linear_pool",
        "singular_covariance_control": singular_control,
        "no_claim": (
            "this matched pilot does not establish general continuous control, "
            "an objective for the robust heuristic, or estimator-level robustness"
        ),
    }


__all__ = [
    "HybridTrackingConfig",
    "TrackingMethod",
    "run_hybrid_tracking",
    "run_hybrid_tracking_comparison",
]
