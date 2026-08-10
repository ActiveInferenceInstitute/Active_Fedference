"""Empirical characterization of robust_aggregate (MAJ-1) — no mocks, real runs.

Two falsifiable numeric bindings, both tied to INDEPENDENT references so the
instrument cannot be green-by-construction:

1. Negative-control identity — at ``robustness = 0`` the numerical influence
   function reproduces the log-linear pool: every agent weight is exactly
   ``1/n`` and the clean consensus is bit-identical to ``log_linear_pool``. A
   wrong reweighting at c=0 breaks this.
2. Breakdown witness — the measured smallest number of colluding confident-wrong
   adversaries that captures each aggregator's argmax is a specific integer
   (pinned to the real seeded run), and at that count the argmax flips while at
   one fewer it does not — a genuine finite conditional capture witness. It
   establishes no unconditional truth-recovery guarantee, not an
   influence-function or B-robustness result.
"""

from __future__ import annotations

import numpy as np

from fedference.aggregation import log_linear_pool, robust_aggregate
from fedference.experiments.heuristic_characterization import (
    _confident_wrong,
    _honest_colony,
    characterization_grid,
    empirical_breakdown,
    numerical_influence_function,
    run_heuristic_characterization,
)


def test_influence_at_zero_robustness_is_the_naive_pool() -> None:
    """Negative control: c=0 reproduces the log-linear pool exactly — flat 1/n
    weights at every perturbation, and a bit-identical clean consensus."""
    rng = np.random.default_rng(0)
    colony = _honest_colony(6, 8, 3, 0.45, rng)
    cp = _confident_wrong(8, 6, 0.97)
    out = numerical_influence_function(colony, 0, cp, robustness=0.0)
    # The naive pool never down-weights: every agent weight is exactly 1/n,
    # independent of how far the perturbed agent is dragged.
    for w in out["agent_weight"]:
        assert abs(w - 1.0 / 6.0) < 1e-12
    # And the clean (eps=0) consensus is the log-linear pool, bit-identical.
    clean = robust_aggregate(colony, robustness=0.0).consensus
    assert np.array_equal(clean, log_linear_pool(colony))


def test_positive_robustness_downweights_the_dragged_agent() -> None:
    """At c>0 a heavily-dragged agent nets a strictly lower influence than at the
    clean point. NOTE: the heuristic is NOT monotone at tiny eps (a small drag
    can briefly raise the weight before the divergence penalty dominates) — that
    honest non-monotonicity is left in place, not asserted away. We pin the
    meaningful property: net down-weighting once the agent is substantially
    contaminated, and a monotone decline from the weight's peak onward."""
    rng = np.random.default_rng(0)
    colony = _honest_colony(6, 8, 3, 0.45, rng)
    cp = _confident_wrong(8, 6, 0.97)
    out = numerical_influence_function(colony, 0, cp, robustness=1.5)
    weights = out["agent_weight"]
    # Net: the fully-dragged agent carries strictly less influence than clean.
    assert weights[-1] < weights[0] - 0.03
    # From the peak onward the decline is monotone (the penalty dominates).
    peak = int(np.argmax(weights))
    tail = weights[peak:]
    assert all(tail[i] >= tail[i + 1] - 1e-9 for i in range(len(tail) - 1))


def test_breakdown_point_is_finite_and_measured() -> None:
    """Breakdown witness: both aggregators are captured by a finite number of
    colluders (no unconditional guarantee); the objective-backed variational
    rule withstands strictly MORE colluders than the sharp heuristic."""
    b = empirical_breakdown(0)
    assert b["robust_breakdown_k"] == 2
    assert b["variational_breakdown_k"] == 4
    assert b["robust_has_finite_breakdown"] is True
    # The heuristic breaks down no later than the objective-backed rule —
    # exactly the honest ordering (sharp-but-unguaranteed vs conservative).
    assert b["robust_breakdown_k"] <= b["variational_breakdown_k"]


def test_capture_threshold_actually_flips_the_argmax() -> None:
    """At k = breakdown the argmax is the adversary target; at k-1 it is not —
    the measured threshold is a real boundary, not a labelling artifact."""
    b = empirical_breakdown(0)
    k = b["robust_breakdown_k"]
    rng = np.random.default_rng(0)
    honest = _honest_colony(b["n_honest"], b["n_states"], b["true_state"], b["confidence"], rng)
    liar = _confident_wrong(b["n_states"], b["target"], b["sharpness"])
    at_k = robust_aggregate(np.vstack([honest] + [liar] * k), robustness=b["robustness"]).consensus
    at_km1 = robust_aggregate(np.vstack([honest] + [liar] * (k - 1)), robustness=b["robustness"]).consensus
    assert int(np.argmax(at_k)) == b["target"]
    assert int(np.argmax(at_km1)) != b["target"]


def test_report_is_json_serialisable_and_deterministic() -> None:
    import json

    a = run_heuristic_characterization(0)
    b = run_heuristic_characterization(0)
    assert json.dumps(a) == json.dumps(b)
    assert a["breakdown"]["robust_breakdown_k"] == 2
    formal = a["formal_no_go"]
    assert formal["status"] == "proved_for_declared_class"
    assert formal["raw_q_block_witness"]["tangential_contradiction_norm"] > 0.0
    assert formal["normalized_weight_companion"]["forward_difference_gap"] > 0.0


def test_characterization_grid_records_attack_and_weight_controls() -> None:
    report = characterization_grid(
        3,
        n_states_grid=(4,),
        n_honest_grid=(3,),
        robustness_grid=(0.0, 1.5),
        attacks=("confident_wrong", "uniform"),
        weight_scenarios=(("balanced", 1.0, 1.0),),
    )
    assert report["claim_level"] == "scoped_implementation_fact"
    assert report["theory_status"] == "open_no_global_objective"
    assert report["n_rows"] == 4
    assert {row["attack"] for row in report["rows"]} == {"confident_wrong", "uniform"}
    assert {row["robustness"] for row in report["rows"]} == {0.0, 1.5}
    # The negative-control flags are COMPUTED from the grid rows (SYN-7): the
    # c=0 rows must re-measure identically under the plain log-linear pool, the
    # permutation mechanism must differ from the clean belief, and every
    # breakdown count must respect the finite search budget. A regression in
    # any of those mechanisms flips the corresponding flag to False here.
    controls = report["negative_controls"]
    assert controls["robustness_zero_recovers_log_pool"] is True
    assert controls["clean_and_permutation_are_separate_mechanisms"] is True
    assert controls["finite_search_is_not_a_global_breakdown_bound"] is True


def test_characterization_grid_rejects_unknown_attack() -> None:
    import pytest

    with pytest.raises(ValueError, match="unknown attack"):
        characterization_grid(0, attacks=("not_an_attack",))  # type: ignore[arg-type]
