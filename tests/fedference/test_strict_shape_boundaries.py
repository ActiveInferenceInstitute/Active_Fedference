"""Fail-closed categorical-vector shape checks at every public boundary."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from fedference import (
    AggregationConfig,
    AggregationResult,
    LabeledAggregationRequest,
    aggregate,
    aggregate_result,
    aggregation_free_energy,
    log_linear_pool,
    robust_aggregate,
    share_round,
    share_round_result,
    variational_aggregate,
)
from fedference.calibration import CalibrationEpisode
from fedference.divergences import (
    alpha_renyi_divergence,
    divergence,
    kl_divergence,
    renyi_divergence,
    reverse_kl,
    total_variation,
)
from fedference.federation import (
    run_multiprocess_round,
    run_multiprocess_round_result,
    run_socket_round,
    run_socket_round_result,
)
from fedference.generalized_bayes import generalized_posterior, softmax

ArrayOperation = Callable[[np.ndarray], Any]
DivergenceOperation = Callable[[np.ndarray, np.ndarray], float]

_VALID_POSTERIOR = np.asarray([0.4, 0.6])
_LOCAL_POSTERIORS = (
    np.asarray([0.7, 0.3]),
    np.asarray([0.2, 0.8]),
)
_ROBUST_CONFIG = AggregationConfig(method="robust", robustness=1.0)

_BAD_CATEGORICAL_VECTORS = (
    pytest.param(np.asarray([[0.7, 0.3]]), id="row-matrix"),
    pytest.param(np.asarray([[0.7], [0.3]]), id="column-matrix"),
    pytest.param(np.asarray([[[0.7, 0.3]]]), id="rank-three"),
)


@pytest.mark.parametrize("bad_posterior", _BAD_CATEGORICAL_VECTORS)
@pytest.mark.parametrize(
    ("boundary", "operation"),
    (
        (
            "log_linear_pool",
            lambda bad: log_linear_pool((bad, _VALID_POSTERIOR)),
        ),
        (
            "robust_aggregate",
            lambda bad: robust_aggregate(
                (bad, _VALID_POSTERIOR), robustness=1.0
            ),
        ),
        (
            "variational_aggregate",
            lambda bad: variational_aggregate(
                (bad, _VALID_POSTERIOR), robustness=1.0
            ),
        ),
        (
            "aggregate_result",
            lambda bad: aggregate_result((bad, _VALID_POSTERIOR)),
        ),
        (
            "aggregate compatibility wrapper",
            lambda bad: aggregate((bad, _VALID_POSTERIOR)),
        ),
        (
            "share_round_result",
            lambda bad: share_round_result((bad, _VALID_POSTERIOR)),
        ),
        (
            "share_round compatibility wrapper",
            lambda bad: share_round((bad, _VALID_POSTERIOR)),
        ),
        (
            "run_multiprocess_round_result",
            lambda bad: run_multiprocess_round_result(
                (bad, _VALID_POSTERIOR)
            ),
        ),
        (
            "run_multiprocess_round compatibility wrapper",
            lambda bad: run_multiprocess_round((bad, _VALID_POSTERIOR)),
        ),
        (
            "run_socket_round_result",
            lambda bad: run_socket_round_result((bad, _VALID_POSTERIOR)),
        ),
        (
            "run_socket_round compatibility wrapper",
            lambda bad: run_socket_round((bad, _VALID_POSTERIOR)),
        ),
    ),
)
def test_advertised_aggregation_and_federation_boundaries_reject_matrix_rows(
    boundary: str,
    operation: ArrayOperation,
    bad_posterior: np.ndarray,
) -> None:
    del boundary  # The parametrized id names the failing public boundary.
    with pytest.raises(ValueError, match="one-dimensional"):
        operation(bad_posterior)


@pytest.mark.parametrize("bad_posterior", _BAD_CATEGORICAL_VECTORS)
@pytest.mark.parametrize(
    ("boundary", "operation"),
    (
        ("kl_divergence", kl_divergence),
        ("reverse_kl", reverse_kl),
        (
            "renyi_divergence",
            lambda q, p: renyi_divergence(q, p, alpha=0.5),
        ),
        (
            "alpha_renyi_divergence",
            lambda q, p: alpha_renyi_divergence(q, p, alpha=0.5),
        ),
        ("total_variation", total_variation),
        ("divergence KLD", lambda q, p: divergence("KLD", q, p)),
        ("divergence RKL", lambda q, p: divergence("RKL", q, p)),
        ("divergence AR", lambda q, p: divergence("AR", q, p, param=0.5)),
        ("divergence TV", lambda q, p: divergence("TV", q, p)),
    ),
)
def test_every_categorical_divergence_boundary_rejects_matrix_vectors(
    boundary: str,
    operation: DivergenceOperation,
    bad_posterior: np.ndarray,
) -> None:
    del boundary
    with pytest.raises(ValueError, match="one-dimensional"):
        operation(bad_posterior, _VALID_POSTERIOR)
    with pytest.raises(ValueError, match="one-dimensional"):
        operation(_VALID_POSTERIOR, bad_posterior)


@pytest.mark.parametrize("bad_posterior", _BAD_CATEGORICAL_VECTORS)
def test_labeled_python_boundary_rejects_matrix_agent_posteriors(
    bad_posterior: np.ndarray,
) -> None:
    raw = {
        "schema_version": "1.0",
        "state_labels": ["no", "yes"],
        "agents": [
            {"agent_id": "agent-0", "posterior": bad_posterior},
            {"agent_id": "agent-1", "posterior": [0.4, 0.6]},
        ],
        "aggregation_config": AggregationConfig().as_dict(),
    }
    with pytest.raises(ValueError, match="one-dimensional"):
        LabeledAggregationRequest.from_dict(raw)


@pytest.mark.parametrize("bad_vector", _BAD_CATEGORICAL_VECTORS)
def test_log_potential_boundaries_reject_implicit_flattening(
    bad_vector: np.ndarray,
) -> None:
    valid = np.asarray([0.0, -1.0])
    with pytest.raises(ValueError, match="one-dimensional"):
        softmax(bad_vector)
    with pytest.raises(ValueError, match="one-dimensional"):
        generalized_posterior(bad_vector, valid)
    with pytest.raises(ValueError, match="one-dimensional"):
        generalized_posterior(valid, bad_vector)


@pytest.mark.parametrize("bad_posterior", _BAD_CATEGORICAL_VECTORS)
def test_calibration_episode_rejects_matrix_rows(
    bad_posterior: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        CalibrationEpisode(
            "strict-shape",
            "test-world",
            (bad_posterior, _VALID_POSTERIOR),
            0,
        )


@pytest.mark.parametrize("bad_consensus", _BAD_CATEGORICAL_VECTORS)
def test_free_energy_rejects_matrix_consensus(
    bad_consensus: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="consensus must be one-dimensional"):
        aggregation_free_energy(
            consensus_posterior=bad_consensus,
            raw_effective_weights=np.asarray([1.0, 1.0]),
            local_posteriors=_LOCAL_POSTERIORS,
            base_weights=np.asarray([1.0, 1.0]),
            robustness=1.0,
        )


@pytest.mark.parametrize("bad_weights", _BAD_CATEGORICAL_VECTORS)
def test_free_energy_rejects_matrix_effective_weights(
    bad_weights: np.ndarray,
) -> None:
    with pytest.raises(
        ValueError,
        match="raw_effective_weights must be one-dimensional",
    ):
        aggregation_free_energy(
            consensus_posterior=np.asarray([0.5, 0.5]),
            raw_effective_weights=bad_weights,
            local_posteriors=_LOCAL_POSTERIORS,
            base_weights=np.asarray([1.0, 1.0]),
            robustness=1.0,
        )


@pytest.mark.parametrize("bad_vector", _BAD_CATEGORICAL_VECTORS)
@pytest.mark.parametrize(
    "field_name",
    ("consensus", "raw_effective_weights", "normalized_effective_weights"),
)
def test_aggregation_result_rejects_matrix_public_vectors(
    field_name: str,
    bad_vector: np.ndarray,
) -> None:
    values: dict[str, Any] = {
        "consensus": np.asarray([0.5, 0.5]),
        "raw_effective_weights": np.asarray([1.0, 1.0]),
        "normalized_effective_weights": np.asarray([0.5, 0.5]),
        "iterations": 0,
        "converged": True,
    }
    values[field_name] = bad_vector
    with pytest.raises(ValueError, match="one-dimensional"):
        AggregationResult(**values)


@pytest.mark.parametrize("bad_weights", _BAD_CATEGORICAL_VECTORS)
@pytest.mark.parametrize(
    ("boundary", "operation"),
    (
        (
            "log_linear_pool",
            lambda bad: log_linear_pool(_LOCAL_POSTERIORS, base_weights=bad),
        ),
        (
            "robust_aggregate",
            lambda bad: robust_aggregate(
                _LOCAL_POSTERIORS,
                base_weights=bad,
                robustness=1.0,
            ),
        ),
        (
            "variational_aggregate",
            lambda bad: variational_aggregate(
                _LOCAL_POSTERIORS,
                base_weights=bad,
                robustness=1.0,
            ),
        ),
        (
            "aggregate_result",
            lambda bad: aggregate_result(
                _LOCAL_POSTERIORS,
                config=_ROBUST_CONFIG,
                base_weights=bad,
            ),
        ),
        (
            "aggregate compatibility wrapper",
            lambda bad: aggregate(
                _LOCAL_POSTERIORS,
                config=_ROBUST_CONFIG,
                base_weights=bad,
            ),
        ),
        (
            "share_round_result",
            lambda bad: share_round_result(
                _LOCAL_POSTERIORS,
                base_weights=bad,
            ),
        ),
        (
            "share_round compatibility wrapper",
            lambda bad: share_round(
                _LOCAL_POSTERIORS,
                base_weights=bad,
            ),
        ),
        (
            "aggregation_free_energy",
            lambda bad: aggregation_free_energy(
                consensus_posterior=np.asarray([0.5, 0.5]),
                raw_effective_weights=np.asarray([1.0, 1.0]),
                local_posteriors=_LOCAL_POSTERIORS,
                base_weights=bad,
                robustness=1.0,
            ),
        ),
    ),
)
def test_advertised_weight_boundaries_reject_matrix_weights(
    boundary: str,
    operation: ArrayOperation,
    bad_weights: np.ndarray,
) -> None:
    del boundary  # The parametrized id names the failing public boundary.
    with pytest.raises(ValueError, match="base_weights must be one-dimensional"):
        operation(bad_weights)


def test_direct_aggregation_preserves_outer_generator_support() -> None:
    rows = (row for row in _LOCAL_POSTERIORS)
    assert log_linear_pool(rows).shape == (2,)


def _outer_generator():
    return (row.copy() for row in _LOCAL_POSTERIORS)


def test_outer_generators_cross_aggregate_and_sharing_boundaries() -> None:
    direct = aggregate_result(_outer_generator(), config=_ROBUST_CONFIG)
    compatibility = aggregate(_outer_generator(), config=_ROBUST_CONFIG)
    sharing = share_round_result(
        _outer_generator(),
        config=_ROBUST_CONFIG,
        exclude_self=False,
    )
    legacy_sharing = share_round(
        _outer_generator(),
        config=_ROBUST_CONFIG,
        exclude_self=False,
    )
    assert np.array_equal(compatibility, direct.consensus)
    assert np.array_equal(sharing.consensus, direct.consensus)
    assert np.array_equal(legacy_sharing.consensus, direct.consensus)


def test_outer_generators_cross_process_and_socket_boundaries() -> None:
    direct = aggregate_result(_outer_generator(), config=_ROBUST_CONFIG)
    process = run_multiprocess_round_result(
        _outer_generator(),
        config=_ROBUST_CONFIG,
    )
    socket = run_socket_round_result(
        _outer_generator(),
        config=_ROBUST_CONFIG,
        round_id="outer-generator",
    )
    assert np.array_equal(process.consensus, direct.consensus)
    assert np.array_equal(socket.consensus, direct.consensus)


def test_outer_generator_crosses_calibration_episode_boundary() -> None:
    episode = CalibrationEpisode(
        "outer-generator",
        "test-world",
        _outer_generator(),
        0,
    )
    assert episode.local_posteriors.shape == (2, 2)
