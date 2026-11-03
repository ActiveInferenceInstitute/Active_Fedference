"""Runtime numerical controls and fail-closed retired domain interfaces."""

from __future__ import annotations

import importlib
import json

import numpy as np
import pytest

from fedference.aggregation import AggregationConfig, log_linear_pool, robust_aggregate, variational_aggregate
from fedference.aggregation_comparators import linear_opinion_pool
from fedference.generalized_bayes import cavity, generalized_posterior, update_factor
from fedference.losses import rcce
from fedference.statistics import d_equivalent_from_rank_biserial


def test_canonical_calls_and_scientific_comparators() -> None:
    # Given raw exponent weights with a non-unit sum.
    rows = np.array([[0.8, 0.2], [0.5, 0.5]])
    weights = np.array([2.0, 1.0])
    expected = np.array([16 / 17, 1 / 17])
    # When the retained scientific controls consume canonical inputs.
    pooled = log_linear_pool(rows, base_weights=weights)
    robust = robust_aggregate(rows, base_weights=weights, robustness=0)
    variational = variational_aggregate(rows, base_weights=weights, robustness=0)
    mixture = linear_opinion_pool(rows, base_weights=weights)
    # Then the product and arithmetic controls retain distinct known values.
    np.testing.assert_allclose(pooled, expected)
    np.testing.assert_allclose(robust.consensus, expected)
    np.testing.assert_allclose(variational.consensus, expected)
    np.testing.assert_allclose(mixture, [0.7, 0.3])
    np.testing.assert_allclose(robust.raw_effective_weights, weights)
    np.testing.assert_allclose(robust.normalized_effective_weights, [2 / 3, 1 / 3])
    print(json.dumps({
        "raw_exponents": weights.tolist(), "expected": expected.tolist(),
        "pool": pooled.tolist(), "robust_zero": robust.consensus.tolist(),
        "variational_zero": variational.consensus.tolist(), "linear": mixture.tolist(),
        "normalized_effective_weights": robust.normalized_effective_weights.tolist(),
    }, sort_keys=True))


def test_canonical_client_and_effect_size_numerics() -> None:
    # Given simple finite states with analytic updates.
    prior = np.log([0.5, 0.5])
    # When applying canonical client updates and losses.
    posterior = generalized_posterior(prior, loss_by_state=np.log([1.0, 4.0]), tau=0.5)
    cav = cavity(global_posterior=np.array([0.8, 0.2]), site_factor=np.array([0.5, 0.5]))
    factor = update_factor(
        old_site_factor=np.array([0.5, 0.5]),
        old_global_posterior=np.array([0.5, 0.5]),
        new_global_posterior=np.array([0.8, 0.2]),
    )
    # Then independent closed forms hold.
    np.testing.assert_allclose(posterior, [2 / 3, 1 / 3])
    np.testing.assert_allclose(cav, [0.8, 0.2])
    np.testing.assert_allclose(factor, [0.8, 0.2])
    assert rcce(np.array([0.25, 0.75]), 0, q_loss=0.5) == pytest.approx(1.0)
    assert d_equivalent_from_rank_biserial(0.6) == pytest.approx(1.5)


@pytest.mark.parametrize(
    ("module", "name", "retired"),
    [
        ("aggregation", name, retired)
        for name in (
            "aggregate",
            "aggregate_result",
            "log_linear_pool",
            "robust_aggregate",
            "variational_aggregate",
        )
        for retired in ("beliefs", "weights")
    ]
    + [
        ("aggregation_comparators", name, retired)
        for name in ("linear_opinion_pool", "clr_geometric_median_pool")
        for retired in ("beliefs", "weights")
    ]
    + [
        ("belief_sharing", name, retired)
        for name in ("share_round", "share_round_result")
        for retired in ("agent_beliefs", "weights")
    ],
)
def test_retired_keyword_and_export_rejected(module: str, name: str, retired: str) -> None:
    # Given valid input that the former alias accepted.
    rows = np.array([[0.8, 0.2], [0.5, 0.5]])
    imported = importlib.import_module(f"fedference.{module}")
    if name in ("aggregate", "share_round"):
        with pytest.raises(AttributeError):
            getattr(imported, name)
        canonical = "aggregate_result" if name == "aggregate" else "share_round_result"
        function = getattr(imported, canonical)
    else:
        function = getattr(imported, name)
    arguments = {retired: [2.0, 1.0] if retired == "weights" else rows}
    if retired == "weights":
        arguments["local_posteriors"] = rows
    # When a retired spelling is used instead of its canonical name.
    with pytest.raises(TypeError):
        function(**arguments)
    # Then the rejected call leaves its input unchanged.
    np.testing.assert_array_equal(rows, [[0.8, 0.2], [0.5, 0.5]])


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("statistics", "cohens_d_from_rank_biserial"),
        ("experiments", "_finite_cohens_d"),
        ("experiments._common", "_finite_cohens_d"),
        ("aggregation", "_resolve_compat_alias"),
    ],
)
def test_retired_exports_cannot_be_imported(module: str, name: str) -> None:
    with pytest.raises(ImportError):
        exec(f"from fedference.{module} import {name}", {})


@pytest.mark.parametrize(
    ("module", "name", "retired"),
    [("aggregation", "aggregation_free_energy", old) for old in ("beliefs", "consensus", "agent_weights")]
    + [("generalized_bayes", "generalized_posterior", old) for old in ("learning_rate", "loss_vec")]
    + [("generalized_bayes", "cavity", old) for old in ("posterior", "factor")]
    + [
        ("generalized_bayes", "update_factor", old)
        for old in ("old_factor", "old_posterior", "new_posterior")
    ]
    + [
        ("hybrid", name, old)
        for name in ("hybrid_log_linear_pool", "hybrid_aggregate")
        for old in ("beliefs", "weights")
    ]
    + [
        ("server_theory", name, "beliefs")
        for name in ("objective_weight_block", "heuristic_weight_block", "construct_orientation_witness")
    ]
    + [
        ("calibration", "CalibrationEpisode", "beliefs"),
        ("losses", "rcce", "q"),
        ("pomdp", "efe_policy_select", "beliefs"),
        ("experiments.heuristic_characterization", "numerical_influence_function", "beliefs"),
    ],
)
def test_retired_domain_keywords_fail_before_dispatch(module: str, name: str, retired: str) -> None:
    from fedference.hybrid import HybridBelief
    from fedference.pomdp import build_moving_world

    rows = np.array([[0.8, 0.2], [0.5, 0.5]])
    imported = importlib.import_module(f"fedference.{module}")
    if name in ("aggregate", "share_round"):
        with pytest.raises(AttributeError):
            getattr(imported, name)
        canonical = "aggregate_result" if name == "aggregate" else "share_round_result"
        function = getattr(imported, canonical)
    else:
        function = getattr(imported, name)
    # Each case replaces one canonical argument in an otherwise valid call.
    match name:
        case "aggregation_free_energy":
            arguments = dict(
                consensus_posterior=rows[0],
                raw_effective_weights=[1.0, 1.0],
                local_posteriors=rows,
                robustness=1.0,
            )
            canonical = {
                "beliefs": "local_posteriors",
                "consensus": "consensus_posterior",
                "agent_weights": "raw_effective_weights",
            }[retired]
        case "generalized_posterior":
            arguments = dict(log_prior=np.log([0.5, 0.5]), loss_by_state=np.log([1.0, 4.0]), tau=0.5)
            canonical = {"learning_rate": "tau", "loss_vec": "loss_by_state"}[retired]
        case "cavity":
            arguments = dict(global_posterior=rows[0], site_factor=rows[1])
            canonical = {"posterior": "global_posterior", "factor": "site_factor"}[retired]
        case "update_factor":
            arguments = dict(
                old_site_factor=rows[1], old_global_posterior=rows[1], new_global_posterior=rows[0]
            )
            canonical = {
                "old_factor": "old_site_factor",
                "old_posterior": "old_global_posterior",
                "new_posterior": "new_global_posterior",
            }[retired]
        case "hybrid_log_linear_pool" | "hybrid_aggregate":
            beliefs = [HybridBelief(row, np.zeros(2), np.ones(2)) for row in rows]
            arguments = dict(local_posteriors=beliefs, base_weights=[2.0, 1.0])
            canonical = {"beliefs": "local_posteriors", "weights": "base_weights"}[retired]
        case "objective_weight_block" | "heuristic_weight_block" | "construct_orientation_witness":
            arguments = dict(consensus=rows[0], local_posteriors=rows, robustness=1.0)
            canonical = "local_posteriors"
        case "CalibrationEpisode":
            arguments = dict(episode_id="episode", world_family="clean", local_posteriors=rows, true_state=0)
            canonical = "local_posteriors"
        case "rcce":
            arguments = dict(p=rows[0], o=0, q_loss=0.5)
            canonical = "q_loss"
        case "efe_policy_select":
            arguments = dict(local_posteriors=rows, world_dict=build_moving_world())
            canonical = "local_posteriors"
        case "numerical_influence_function":
            arguments = dict(local_posteriors=rows, agent=0, contamination_point=rows[1], robustness=0.0)
            canonical = "local_posteriors"
        case _:
            raise AssertionError(f"Unbound runtime retirement case: {name}")
    arguments[retired] = arguments.pop(canonical)
    with pytest.raises(TypeError, match="unexpected keyword"):
        function(**arguments)
    np.testing.assert_array_equal(rows, [[0.8, 0.2], [0.5, 0.5]])


def test_retired_assimilation_keyword_leaves_state_unchanged() -> None:
    from fedference.agents import SentinelEnsemble

    ensemble = SentinelEnsemble.from_world(2, seed=11)
    before = ensemble.broadcast().copy()
    heard = log_linear_pool(before)
    ensemble.assimilate(heard, tau=0.0)
    np.testing.assert_allclose(ensemble.broadcast(), before)
    with pytest.raises(TypeError, match="unexpected keyword"):
        ensemble.assimilate(heard, **{"learning_rate": 0.0})
    np.testing.assert_allclose(ensemble.broadcast(), before)


def test_retired_result_properties_are_absent() -> None:
    from fedference.belief_sharing import share_round_result
    from fedference.calibration import CalibrationEpisode
    from fedference.hybrid import HybridBelief, hybrid_aggregate

    rows = np.array([[0.8, 0.2], [0.5, 0.5]])
    aggregation = robust_aggregate(rows, robustness=0.0)
    shared = share_round_result(
        rows, config=AggregationConfig(method="naive", max_iter=32, tol=1e-9), exclude_self=False
    )
    episode = CalibrationEpisode("episode", "clean", rows, 0)
    belief = HybridBelief(np.array([0.8, 0.2]), np.zeros(2), np.ones(2))
    hybrid = hybrid_aggregate([belief], robustness=0.0)
    np.testing.assert_allclose(shared.consensus, [0.8, 0.2])
    np.testing.assert_array_equal(episode.local_posteriors, rows)
    np.testing.assert_allclose(hybrid.consensus.discrete, [0.8, 0.2])
    for result, name in (
        (aggregation, "agent_weights"),
        (shared, "agent_weights"),
        (shared, "shared_beliefs"),
        (episode, "beliefs"),
        (hybrid, "agent_weights"),
    ):
        with pytest.raises(AttributeError):
            getattr(result, name)
