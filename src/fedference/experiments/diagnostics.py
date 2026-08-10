"""Diagnostic experiment reports for analysis workflow and figures."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..aggregation import variational_aggregate
from ..belief_sharing import share_round
from ..bnn_baseline import fed_gvi_logreg
from ..colonies import healthy_colony
from ..contamination import contaminate
from ..expected_free_energy import decompose
from ..pomdp import N_LOCATIONS, build_sentinel_world
from ..statistics import bootstrap_ci

_INFLUENCE_DRIFTS: tuple[float, ...] = (0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 0.99)
_VARIATIONAL_ROBUSTNESS: float = 1.5
_BNN_CONTAMINATION_LEVELS: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.35, 0.4)
_BNN_N_PER: int = 200
_BNN_ROBUST_LOSS_PARAM: float = 1.0
_BNN_N_SEEDS: int = 20
_BNN_BOOTSTRAP_ALPHA: float = 0.05
_BNN_BOOTSTRAP_N: int = 5000
_DEFAULT_COLONY_CONFIDENCE: float = 0.45


def run_efe_decomposition_report(seed: int) -> dict[str, Any]:
    """Closed-form EFE decomposition of one sentinel policy.

    The canonical sentinel world's D prior is a point mass at the den.
    That prior is appropriate for the recovery and inference studies, but it
    makes the state--outcome mutual-information term identically zero: there is
    no state uncertainty left for an observation to resolve. This diagnostic
    therefore uses a uniform prior deliberately, so Figure 6 exposes the
    epistemic component of the EFE identity. The choice is recorded in the
    report and does not alter the world builder or any study that uses D.
    """
    rng = np.random.default_rng(seed)
    world = build_sentinel_world(rng, acuity=0.6)
    A = np.asarray(world["A"][0], dtype=np.float64)  # type: ignore[index]
    B = np.asarray(world["B"][0], dtype=np.float64)  # type: ignore[index]
    C = np.asarray(world["C"][0], dtype=np.float64).ravel()  # type: ignore[index]
    source_prior = np.asarray(world["D"][0], dtype=np.float64).ravel()  # type: ignore[index]
    prior = np.full_like(source_prior, 1.0 / source_prior.size)
    n_a = int(B.shape[2])
    policy = [int(rng.integers(0, n_a)) for _ in range(3)]
    terms = decompose(A, B, C, prior, policy)
    return {
        "risk": float(terms.risk),
        "ambiguity": float(terms.ambiguity),
        "pragmatic_value": float(terms.pragmatic_value),
        "epistemic_value": float(terms.epistemic_value),
        "total": float(terms.total),
        "identity_residual": float(terms.identity_residual),
        "policy": policy,
        "prior_type": "uniform diagnostic prior",
    }


def run_influence_weights_report(
    seed: int,
    *,
    n_agents: int,
    robustness: float = 1.5,
    colony_confidence: float = _DEFAULT_COLONY_CONFIDENCE,
) -> dict[str, Any]:
    """Server-side robust pooling influence weights on a contaminated colony."""
    rng = np.random.default_rng(seed)
    n_s = int(N_LOCATIONS)
    n_contaminated = max(1, n_agents // 3)
    true_state = int(rng.integers(0, n_s))
    wrong_state = int((true_state + n_s // 2) % n_s)

    colony = healthy_colony(
        true_state, n_agents, n_s, colony_confidence, rng=rng, jitter=0.0
    )
    for k in range(n_contaminated):
        colony[k] = contaminate(
            colony[k], kind="confident_wrong", rate=0.8, rng=rng, wrong_state=wrong_state
        )

    diag = share_round(
        colony, method="robust", robustness=robustness, exclude_self=False, true_state=true_state
    )
    normalized_effective_weights = np.asarray(
        diag.normalized_effective_weights,
        dtype=np.float64,
    ).ravel()
    return {
        "schema_version": "2.0",
        "normalized_effective_weights": [
            float(weight) for weight in normalized_effective_weights
        ],
        # Federation transport retains the agent_weights wire-level
        # compatibility key. This report alias is not a new interpretation.
        "agent_weights": [
            float(weight) for weight in normalized_effective_weights
        ],
        "contaminated_indices": list(range(n_contaminated)),
        "n_agents": int(n_agents),
        "n_contaminated": int(n_contaminated),
        "true_state": true_state,
        "robustness": float(robustness),
    }


def run_variational_aggregation_report(
    seed: int,
    *,
    n_agents: int,
    robustness: float = _VARIATIONAL_ROBUSTNESS,
    colony_confidence: float = _DEFAULT_COLONY_CONFIDENCE,
) -> dict[str, Any]:
    """Diagnostics for the objective-backed variational aggregator."""
    rng = np.random.default_rng(seed)
    n_s = int(N_LOCATIONS)
    n_contaminated = max(1, n_agents // 3)
    true_state = int(rng.integers(0, n_s))
    wrong_state = int((true_state + n_s // 2) % n_s)

    def _healthy() -> np.ndarray:
        return healthy_colony(
            true_state, n_agents, n_s, colony_confidence, rng=rng, jitter=0.0
        )

    colony = _healthy()
    for k in range(n_contaminated):
        colony[k] = contaminate(
            colony[k], kind="confident_wrong", rate=0.8, rng=rng, wrong_state=wrong_state
        )
    res = variational_aggregate(colony, robustness=robustness, multistart=False)

    influence: list[float] = []
    for drift in _INFLUENCE_DRIFTS:
        probed = _healthy()
        probed[0] = contaminate(
            probed[0],
            kind="confident_wrong",
            rate=float(drift),
            rng=rng,
            wrong_state=wrong_state,
        )
        diag = variational_aggregate(probed, robustness=robustness)
        influence.append(float(diag.normalized_effective_weights[0]))

    n_capture_honest = 2
    capture = np.empty((n_capture_honest + 1, n_s), dtype=np.float64)
    for i in range(n_capture_honest + 1):
        belief = np.full(n_s, (1.0 - colony_confidence) / (n_s - 1), dtype=np.float64)
        belief[true_state] = colony_confidence
        capture[i] = belief
    capture[n_capture_honest] = contaminate(
        capture[n_capture_honest],
        kind="confident_wrong",
        rate=0.999,
        rng=rng,
        wrong_state=wrong_state,
    )
    single = variational_aggregate(
        capture, robustness=robustness, multistart=False, max_iter=128
    )
    multi = variational_aggregate(
        capture, robustness=robustness, multistart=True, max_iter=128
    )

    return {
        "free_energy_history": [float(v) for v in res.free_energy_history],
        "converged": bool(res.converged),
        "iterations": int(res.iterations),
        "drifts": [float(d) for d in _INFLUENCE_DRIFTS],
        "variational_influence": influence,
        "naive_influence": float(1.0 / n_agents),
        "robustness": float(robustness),
        "n_agents": int(n_agents),
        "n_contaminated": int(n_contaminated),
        "true_state": true_state,
        "single_start_history": [float(v) for v in single.free_energy_history],
        "multi_start_history": [float(v) for v in multi.free_energy_history],
        "single_start_final_f": float(single.free_energy_history[-1]),
        "multi_start_final_f": float(multi.free_energy_history[-1]),
        "capture_gap": float(single.free_energy_history[-1] - multi.free_energy_history[-1]),
    }


def run_bnn_robustness_report(
    seed: int,
    *,
    n_seeds: int = _BNN_N_SEEDS,
    n_per: int = _BNN_N_PER,
    robust_loss_param: float = _BNN_ROBUST_LOSS_PARAM,
    contamination_levels: tuple[float, ...] = _BNN_CONTAMINATION_LEVELS,
) -> dict[str, Any]:
    """BNN held-out accuracy vs label contamination for standard vs robust clients."""
    levels = list(contamination_levels)
    # The public seed is the base of the independent replicate stream.  The
    # previous range(n_seeds) construction made ``seed`` affect only bootstrap
    # resampling, so changing it silently reused the same simulated datasets.
    seeds = list(range(seed, seed + n_seeds))
    rng = np.random.default_rng(seed)

    def _seeded_curve(
        *, loss: str, divergence: str, loss_param: float
    ) -> tuple[list[float], list[list[float]], list[list[float]]]:
        means: list[float] = []
        intervals: list[list[float]] = []
        values_by_level: list[list[float]] = []
        for c in levels:
            values = [
                float(
                    fed_gvi_logreg(
                        n_per=n_per,
                        contamination=c,
                        loss=loss,
                        loss_param=loss_param,
                        divergence=divergence,
                        seed=s,
                    )["test_accuracy"]
                )
                for s in seeds
            ]
            lo, hi = bootstrap_ci(
                values,
                alpha=_BNN_BOOTSTRAP_ALPHA,
                n_boot=_BNN_BOOTSTRAP_N,
                rng=rng,
            )
            means.append(float(np.mean(values)))
            intervals.append([float(lo), float(hi)])
            values_by_level.append(values)
        return means, intervals, values_by_level

    standard, standard_ci, standard_seed_values = _seeded_curve(
        loss="nll", divergence="KLD", loss_param=0.0
    )
    robust, robust_ci, robust_seed_values = _seeded_curve(
        loss="rcce", divergence="AR", loss_param=robust_loss_param
    )
    gaps = [float(r - s) for r, s in zip(robust, standard)]
    peak_idx = int(max(range(len(gaps)), key=lambda i: gaps[i]))
    return {
        "contamination_levels": levels,
        "accuracy_by_config": {
            "nll / KLD (standard)": standard,
            "rcce / AR (robust)": robust,
        },
        "accuracy_ci_by_config": {
            "nll / KLD (standard)": standard_ci,
            "rcce / AR (robust)": robust_ci,
        },
        "accuracy_seed_values_by_config": {
            "nll / KLD (standard)": standard_seed_values,
            "rcce / AR (robust)": robust_seed_values,
        },
        "robust_minus_standard": gaps,
        "peak_margin": float(gaps[peak_idx]),
        "peak_margin_contamination": float(levels[peak_idx]),
        "ci_percent": int(round((1.0 - _BNN_BOOTSTRAP_ALPHA) * 100)),
        "n_bootstrap": _BNN_BOOTSTRAP_N,
        "n_seeds": len(seeds),
        "n_per": n_per,
        "robust_loss_param": robust_loss_param,
        "seed": int(seed),
    }


__all__ = [
    "run_bnn_robustness_report",
    "run_efe_decomposition_report",
    "run_influence_weights_report",
    "run_variational_aggregation_report",
]
