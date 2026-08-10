"""Empirical characterization of the server-side ``robust_aggregate`` heuristic (MAJ-1 slice).

This module MEASURES the sharp reverse-KL reweighting heuristic's empirical
behavior.  Its formal companion in :mod:`fedference.server_theory` proves a
separate, scoped no-go for a declared separable objective class; this module
records that witness as report metadata without treating the finite grid as a
theorem prover. ``robust_aggregate`` retains its heuristic label: the no-go is
not a universal no-objective result, and the implementation's only positive
formal recovery property remains its bit-identical log-pool corner at
``robustness = 0``.

The empirical instruments are careful to (a) anchor every measurement to that
proven corner and (b) exhibit the heuristic's **finite breakdown point**, i.e.
a witness that it CAN be captured by a colluding majority and therefore has no
unconditional server truth-recovery claim. This finite capture result is not
itself an influence-function theorem and cannot transfer or refute the distinct,
source-conditional bounded-influence result for the client-side FedGVI update.
The objective-backed ``variational_aggregate`` has a raw-weight bound, not an
estimator-level bounded-influence guarantee.

Two instruments:

* :func:`numerical_influence_function` — a finite-difference empirical influence
  curve: perturb one agent's belief a fraction ``eps`` toward a contamination
  point and record the response of the consensus (and that agent's converged
  weight). At ``robustness = 0`` this must reproduce the log-linear pool's
  analytic per-agent influence (equal ``1/n`` weights; consensus sensitivity =
  the closed-form softmax Jacobian), tying the instrument to the proven corner.
* :func:`empirical_breakdown` — sweep the number of colluding confident-wrong
  adversaries and report the smallest count at which the heuristic's consensus
  argmax is *captured* (flips to the adversaries' target), for both
  ``robust_aggregate`` and ``variational_aggregate`` on the *same* colonies.

Nothing here claims an objective, a bound, or a guarantee. The honest headline
the module earns is the breakdown witness: a measured, finite capture threshold.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np

from ..aggregation import log_linear_pool, robust_aggregate, variational_aggregate
from ..server_theory import (
    construct_normalized_weight_no_go_witness,
    construct_raw_log_pool_no_go_witness,
)

ArrayF = np.ndarray
AttackKind = Literal["confident_wrong", "label_noise", "uniform", "permutation"]
_ATTACK_KINDS: tuple[AttackKind, ...] = (
    "confident_wrong",
    "label_noise",
    "uniform",
    "permutation",
)


def _formal_no_go_report() -> dict[str, Any]:
    """Return the source-bound summary of the separate MAJ-1 proposition."""
    raw = construct_raw_log_pool_no_go_witness()
    normalized = construct_normalized_weight_no_go_witness()
    return {
        "status": "proved_for_declared_class",
        "objective_class": "sum_n a_n KL(q || s_n) + R(a, w) + G(q)",
        "raw_q_block_requirement": (
            "softmax(sum_n a_n log s_n) is the q-coordinate minimizer for every "
            "interior raw a and local-posterior matrix"
        ),
        "raw_q_block_witness": {
            "scales": [float(scale) for scale in raw.scales],
            "max_q_block_error": raw.max_q_block_error,
            "tangential_contradiction_norm": raw.tangential_contradiction_norm,
        },
        "normalized_weight_companion": {
            "robustness": normalized.robustness,
            "normalized_weight_max_absolute_gap": (
                normalized.normalized_weight_max_absolute_gap
            ),
            "forward_difference_gap": normalized.forward_difference_gap,
        },
        "limitations": (
            "The proposition does not rule out nonseparable q-a couplings, "
            "source-dependent terms, non-C1 objectives, or objectives that encode "
            "selected fixed points without reproducing both update blocks."
        ),
    }


def _honest_colony(
    n_agents: int, n_states: int, true_state: int, confidence: float, rng: np.random.Generator
) -> ArrayF:
    """A colony of soft honest beliefs peaked (weakly) on the true state."""
    local_posteriors = np.full(
        (n_agents, n_states),
        (1.0 - confidence) / (n_states - 1),
        dtype=np.float64,
    )
    local_posteriors[:, true_state] = confidence
    # Small per-agent jitter so the colony is not degenerate-identical.
    jitter = rng.uniform(0.97, 1.03, size=local_posteriors.shape)
    local_posteriors = local_posteriors * jitter
    return local_posteriors / local_posteriors.sum(axis=1, keepdims=True)


def _confident_wrong(n_states: int, target: int, sharpness: float) -> ArrayF:
    """A confidently-wrong broadcast concentrated on ``target``."""
    b = np.full(n_states, (1.0 - sharpness) / (n_states - 1), dtype=np.float64)
    b[target] = sharpness
    return b / b.sum()


def _attack_belief(
    n_states: int,
    true_state: int,
    target: int,
    sharpness: float,
    attack: AttackKind,
) -> ArrayF:
    """Construct one declared contamination mechanism.

    These are deliberately simple diagnostic mechanisms, not a Byzantine threat
    model.  Keeping the mechanism name in every report row prevents a positive
    result for one attack from being promoted into a universal robustness claim.
    """
    if attack == "confident_wrong":
        return _confident_wrong(n_states, target, sharpness)
    if attack == "uniform":
        return np.full(n_states, 1.0 / n_states, dtype=np.float64)
    if attack == "label_noise":
        belief = np.full(n_states, 1e-12, dtype=np.float64)
        belief[true_state] += 0.5
        belief[target] += 0.5
        return belief / belief.sum()
    if attack == "permutation":
        belief = np.full(n_states, (1.0 - sharpness) / (n_states - 1), dtype=np.float64)
        belief[true_state] = sharpness
        belief[[true_state, target]] = belief[[target, true_state]]
        return belief / belief.sum()
    raise ValueError(f"unknown attack mechanism: {attack}")


def numerical_influence_function(
    local_posteriors: ArrayF | None = None,
    agent: int | None = None,
    contamination_point: ArrayF | None = None,
    *,
    robustness: float,
    eps_grid: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
    **legacy: Any,
) -> dict[str, Any]:
    """Finite-difference empirical influence of one agent under ``robust_aggregate``.

    For each ``eps`` in ``eps_grid`` we mix agent ``agent``'s belief a fraction
    ``eps`` toward ``contamination_point`` (a convex mix, renormalized), re-run
    the aggregator, and record the L1 shift of the consensus from the clean
    (``eps = 0``) consensus and the agent's converged normalized weight.

    Returns ``consensus_shift`` (per-eps L1 distance) and
    ``normalized_effective_weights`` (per-eps normalized influence). At
    ``robustness = 0`` the normalized weight is a flat ``1/n`` at every eps
    (the naive pool never down-weights), which the
    negative-control test binds to the analytic value.
    """
    if "beliefs" in legacy:
        if local_posteriors is not None:
            raise TypeError(
                "local_posteriors and deprecated beliefs cannot both be supplied"
            )
        local_posteriors = legacy.pop("beliefs")  # type: ignore[assignment]
        warnings.warn(
            "beliefs is deprecated; use local_posteriors",
            DeprecationWarning,
            stacklevel=2,
        )
    if legacy:
        names = ", ".join(sorted(legacy))
        raise TypeError(f"unexpected keyword argument(s): {names}")
    if local_posteriors is None:
        raise TypeError("local_posteriors is required")
    if agent is None:
        raise TypeError("agent is required")
    if contamination_point is None:
        raise TypeError("contamination_point is required")
    base = np.asarray(local_posteriors, dtype=np.float64).copy()
    n = base.shape[0]
    cp = np.asarray(contamination_point, dtype=np.float64)
    clean = robust_aggregate(base, robustness=robustness).consensus
    shifts: list[float] = []
    normalized_effective_weights: list[float] = []
    for eps in eps_grid:
        perturbed = base.copy()
        mixed = (1.0 - eps) * base[agent] + eps * cp
        perturbed[agent] = mixed / mixed.sum()
        res = robust_aggregate(perturbed, robustness=robustness)
        shifts.append(float(np.sum(np.abs(res.consensus - clean))))
        normalized_effective_weights.append(
            float(res.normalized_effective_weights[agent])
        )
    return {
        "eps_grid": list(eps_grid),
        "consensus_shift": shifts,
        "normalized_effective_weights": normalized_effective_weights,
        # Compatibility key for pre-canonical characterization reports.
        "agent_weight": normalized_effective_weights,
        "robustness": float(robustness),
        "n_agents": int(n),
    }


class _PoolResult:
    """Minimal ``.consensus``-carrying adapter so :func:`_capture_count` can
    drive the plain log-linear pool as an aggregator (negative control)."""

    def __init__(self, consensus: ArrayF) -> None:
        self.consensus = consensus


def _log_pool_aggregator(
    local_posteriors: ArrayF,
    *,
    base_weights: Any = None,
    robustness: float = 0.0,
) -> _PoolResult:
    """The naive pool wearing the aggregator calling convention; ``robustness``
    is accepted and ignored — the pool has no robustness knob by definition."""
    del robustness
    return _PoolResult(
        log_linear_pool(
            local_posteriors=local_posteriors, base_weights=base_weights
        )
    )


def _grid_budget(n_honest: int) -> int:
    """The finite adversary search budget :func:`characterization_grid` uses."""
    return max(8, 2 * n_honest)


def _capture_count(
    aggregate_fn: Any,
    n_honest: int,
    confidence: float,
    n_states: int,
    true_state: int,
    target: int,
    sharpness: float,
    robustness: float,
    seed: int,
    max_adversaries: int,
    attack: AttackKind = "confident_wrong",
    honest_weight: float = 1.0,
    adversary_weight: float = 1.0,
) -> int | None:
    """Smallest number of colluding confident-wrong adversaries that captures
    the aggregator's consensus argmax (flips it to ``target``). ``None`` if no
    count up to ``max_adversaries`` captures it."""
    rng = np.random.default_rng(seed)
    honest = _honest_colony(n_honest, n_states, true_state, confidence, rng)
    liar = _attack_belief(n_states, true_state, target, sharpness, attack)
    for k in range(0, max_adversaries + 1):
        colony = np.vstack([honest] + [liar] * k) if k else honest
        base_weights = [honest_weight] * n_honest + [adversary_weight] * k
        consensus = aggregate_fn(
            colony, base_weights=base_weights, robustness=robustness
        ).consensus
        if int(np.argmax(consensus)) == target:
            return k
    return None


def empirical_breakdown(
    seed: int = 0,
    *,
    n_honest: int = 5,
    confidence: float = 0.45,
    n_states: int = 8,
    true_state: int = 3,
    target: int = 6,
    sharpness: float = 0.97,
    robustness: float = 1.5,
    max_adversaries: int = 12,
    attack: AttackKind = "confident_wrong",
    honest_weight: float = 1.0,
    adversary_weight: float = 1.0,
) -> dict[str, Any]:
    """Measure the breakdown point of both server aggregators on the same colony.

    Adds colluding confident-wrong adversaries (all broadcasting ``target``) to a
    fixed honest colony until each aggregator's consensus argmax is captured. The
    returned counts are the measured breakdown points — finite in the declared
    fixture when either server rule is overwhelmed by a colluding majority.
    This is evidence against unconditional truth recovery, not a global
    breakdown bound or an estimator influence-function theorem.

    Returns the two capture counts and the study parameters.
    """
    if honest_weight <= 0.0 or adversary_weight <= 0.0:
        raise ValueError("honest_weight and adversary_weight must be positive")
    robust_k = _capture_count(
        robust_aggregate,
        n_honest,
        confidence,
        n_states,
        true_state,
        target,
        sharpness,
        robustness,
        seed,
        max_adversaries,
        attack,
        honest_weight,
        adversary_weight,
    )
    variational_k = _capture_count(
        variational_aggregate,
        n_honest,
        confidence,
        n_states,
        true_state,
        target,
        sharpness,
        robustness,
        seed,
        max_adversaries,
        attack,
        honest_weight,
        adversary_weight,
    )
    return {
        "robust_breakdown_k": robust_k,
        "variational_breakdown_k": variational_k,
        "n_honest": int(n_honest),
        "n_states": int(n_states),
        "true_state": int(true_state),
        "target": int(target),
        "sharpness": float(sharpness),
        "robustness": float(robustness),
        "confidence": float(confidence),
        "seed": int(seed),
        # A finite k records capture in this fixture; it does not classify
        # estimator-level B-robustness or transfer the client-side theorem.
        "robust_has_finite_breakdown": robust_k is not None,
        "attack": attack,
        "honest_weight": float(honest_weight),
        "adversary_weight": float(adversary_weight),
    }


def _negative_controls(rows: list[dict[str, Any]]) -> dict[str, bool]:
    """Compute the grid's negative-control flags from the actual rows.

    Every flag is derived, never asserted by construction, so a regression in
    the corresponding mechanism flips it to ``False``:

    * ``robustness_zero_recovers_log_pool`` — for every ``robustness == 0`` row,
      re-measure the capture count under the plain :func:`log_linear_pool` (via
      the row's recorded seed/params) and require it to equal BOTH recorded
      aggregator counts. At ``c = 0`` both aggregators are the pool exactly, so
      any reweighting leak at the Friston corner breaks the equality.
    * ``clean_and_permutation_are_separate_mechanisms`` — for every permutation
      row, rebuild the permutation attack belief and the clean (un-swapped)
      confident-true belief at the row's parameters and require them to differ;
      a no-op permutation (e.g. a swap-index bug) collapses them.
    * ``finite_search_is_not_a_global_breakdown_bound`` — every recorded
      breakdown count is either ``None`` (not captured *within the finite
      budget* — no global bound claimed) or lies inside the row's finite search
      budget; a sentinel value or out-of-budget count would falsify the
      finite-search interpretation.
    """
    zero_rows = [row for row in rows if row["robustness"] == 0.0]
    pool_recovered = bool(zero_rows) and all(
        row["robust_breakdown_k"]
        == row["variational_breakdown_k"]
        == _capture_count(
            _log_pool_aggregator,
            row["n_honest"],
            row["confidence"],
            row["n_states"],
            row["true_state"],
            row["target"],
            row["sharpness"],
            0.0,
            row["seed"],
            _grid_budget(row["n_honest"]),
            row["attack"],
            row["honest_weight"],
            row["adversary_weight"],
        )
        for row in zero_rows
    )

    perm_rows = [row for row in rows if row["attack"] == "permutation"]
    mechanisms_separate = all(
        not np.allclose(
            _attack_belief(
                row["n_states"],
                row["true_state"],
                row["target"],
                row["sharpness"],
                "permutation",
            ),
            _confident_wrong(row["n_states"], row["true_state"], row["sharpness"]),
        )
        for row in perm_rows
    )

    finite_search_honest = all(
        k is None or 0 <= k <= _grid_budget(row["n_honest"])
        for row in rows
        for k in (row["robust_breakdown_k"], row["variational_breakdown_k"])
    )

    return {
        "robustness_zero_recovers_log_pool": bool(pool_recovered),
        "clean_and_permutation_are_separate_mechanisms": bool(mechanisms_separate),
        "finite_search_is_not_a_global_breakdown_bound": bool(finite_search_honest),
    }


def characterization_grid(
    seed: int = 0,
    *,
    n_states_grid: tuple[int, ...] = (4, 8),
    n_honest_grid: tuple[int, ...] = (3, 5),
    robustness_grid: tuple[float, ...] = (0.0, 1.5),
    attacks: tuple[AttackKind, ...] = _ATTACK_KINDS,
    weight_scenarios: tuple[tuple[str, float, float], ...] = (
        ("balanced", 1.0, 1.0),
        ("adversary_downweighted", 1.0, 0.5),
    ),
) -> dict[str, Any]:
    """Run a small declared MAJ-1 scenario grid.

    The grid is an evidence ladder, not a theorem prover.  It records finite
    breakdown witnesses and their absence within a search budget, while the
    report explicitly labels the result as a scoped implementation diagnostic.
    ``seed + row_index`` gives each row a deterministic but distinct colony.
    """
    if not n_states_grid or not n_honest_grid or not robustness_grid or not attacks:
        raise ValueError("characterization grids must be non-empty")
    if any(n < 3 for n in n_states_grid):
        raise ValueError("n_states_grid values must be >= 3")
    if any(n < 2 for n in n_honest_grid):
        raise ValueError("n_honest_grid values must be >= 2")
    if any(c < 0.0 for c in robustness_grid):
        raise ValueError("robustness_grid values must be non-negative")
    unknown = set(attacks) - set(_ATTACK_KINDS)
    if unknown:
        raise ValueError(f"unknown attack mechanisms: {sorted(unknown)}")

    rows: list[dict[str, Any]] = []
    row_seed = int(seed)
    for n_states in n_states_grid:
        true_state = n_states // 3
        target = (true_state + max(1, n_states // 2)) % n_states
        for n_honest in n_honest_grid:
            for robustness in robustness_grid:
                for attack in attacks:
                    for scenario, honest_weight, adversary_weight in weight_scenarios:
                        row = empirical_breakdown(
                            seed=row_seed,
                            n_honest=n_honest,
                            n_states=n_states,
                            true_state=true_state,
                            target=target,
                            robustness=robustness,
                            max_adversaries=_grid_budget(n_honest),
                            attack=attack,
                            honest_weight=honest_weight,
                            adversary_weight=adversary_weight,
                        )
                        row["weight_scenario"] = scenario
                        rows.append(row)
                        row_seed += 1
    finite = [row for row in rows if row["robust_breakdown_k"] is not None]
    return {
        "claim_level": "scoped_implementation_fact",
        "theory_status": "open_no_global_objective",
        "independent_unit": "declared seeded scenario row",
        # Each control is COMPUTED from the grid rows (a regression flips it to
        # False); none is a hardcoded literal.
        "negative_controls": _negative_controls(rows),
        "parameter_grid": {
            "n_states": [int(v) for v in n_states_grid],
            "n_honest": [int(v) for v in n_honest_grid],
            "robustness": [float(v) for v in robustness_grid],
            "attacks": list(attacks),
            "weight_scenarios": [
                {"name": name, "honest": float(hw), "adversary": float(aw)}
                for name, hw, aw in weight_scenarios
            ],
        },
        "n_rows": len(rows),
        "finite_robust_breakdown_rows": len(finite),
        "rows": rows,
    }


def run_heuristic_characterization(seed: int = 0) -> dict[str, Any]:
    """JSON-serialisable characterization report: breakdown points plus a
    numerical influence sweep of one contaminating agent at the study settings.
    """
    breakdown = empirical_breakdown(seed)
    rng = np.random.default_rng(seed)
    colony = _honest_colony(6, 8, 3, 0.45, rng)
    contamination = _confident_wrong(8, 6, 0.97)
    if_naive = numerical_influence_function(colony, 0, contamination, robustness=0.0)
    if_robust = numerical_influence_function(colony, 0, contamination, robustness=1.5)
    grid = characterization_grid(seed)
    formal_no_go = _formal_no_go_report()
    return {
        "status": "scoped_implementation_fact",
        "breakdown": breakdown,
        "influence_naive": if_naive,
        "influence_robust": if_robust,
        "claim_level": "scoped_implementation_fact",
        "schema_version": "2.0",
        "theory_status": "open_no_global_objective",
        "primary_estimand": "declared finite influence and no-go diagnostics",
        "independent_unit": "declared seeded scenario row",
        "no_claim": (
            "Finite breakdown and influence witnesses do not prove a global objective, "
            "bounded influence, Byzantine tolerance, or universal attack robustness; "
            "the adjacent proposition excludes only its declared separable class."
        ),
        "formal_no_go": formal_no_go,
        "grid": grid,
        "seed": int(seed),
    }


__all__ = [
    "empirical_breakdown",
    "characterization_grid",
    "numerical_influence_function",
    "run_heuristic_characterization",
]
