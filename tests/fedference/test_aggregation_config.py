"""Public aggregation configuration and rich-dispatch compatibility tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from fedference import (
    AggregationConfig,
    AggregationResult,
    aggregate,
    aggregate_result,
    log_linear_pool,
    robust_aggregate,
    share_round,
    variational_aggregate,
)

BELIEFS = np.asarray(
    [
        [0.70, 0.20, 0.10],
        [0.60, 0.30, 0.10],
        [0.10, 0.10, 0.80],
    ]
)


def test_config_is_canonical_and_fingerprint_changes_with_method() -> None:
    first = AggregationConfig(method="robust", robustness=1.5)
    second = AggregationConfig(method="robust", robustness=1.5)
    variational = AggregationConfig(method="variational", robustness=1.5)
    assert first.as_dict() == second.as_dict()
    assert first.fingerprint == second.fingerprint
    assert first.fingerprint != variational.fingerprint
    assert len(first.fingerprint) == 64


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"method": "bogus"}, "unknown method"),
        ({"robustness": float("nan")}, "robustness"),
        ({"entropy_weight": -0.1}, "entropy_weight"),
        ({"max_iter": 0}, "max_iter"),
        ({"max_iter": float("inf")}, "max_iter"),
        ({"robustness": "1.0"}, "robustness"),
        ({"tol": float("inf")}, "tol"),
        ({"tol": None}, "tol"),
        ({"multistart": 1}, "multistart"),
    ],
)
def test_config_rejects_invalid_controls(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        AggregationConfig(**kwargs)


def test_rich_dispatch_matches_each_low_level_method() -> None:
    naive = aggregate_result(BELIEFS, config=AggregationConfig(method="naive"))
    robust = aggregate_result(
        BELIEFS,
        config=AggregationConfig(method="robust", robustness=1.5),
    )
    variational = aggregate_result(
        BELIEFS,
        config=AggregationConfig(
            method="variational",
            robustness=1.5,
            entropy_weight=0.8,
        ),
    )
    assert isinstance(naive, AggregationResult)
    assert np.array_equal(naive.consensus, log_linear_pool(BELIEFS))
    assert np.array_equal(
        robust.consensus,
        robust_aggregate(BELIEFS, robustness=1.5, max_iter=64).consensus,
    )
    assert np.array_equal(
        variational.consensus,
        variational_aggregate(
            BELIEFS,
            robustness=1.5,
            entropy_weight=0.8,
        ).consensus,
    )
    assert naive.iterations == 0
    assert np.isclose(naive.normalized_effective_weights.sum(), 1.0)


def test_rich_dispatch_accepts_positional_config_and_weights() -> None:
    config = AggregationConfig(method="naive")
    weights = np.asarray([1.0, 2.0, 3.0])
    positional = aggregate_result(BELIEFS, config, weights)
    keyword = aggregate_result(BELIEFS, config=config, base_weights=weights)
    assert np.array_equal(positional.consensus, keyword.consensus)
    assert np.array_equal(
        positional.normalized_effective_weights,
        keyword.normalized_effective_weights,
    )


def test_zero_entropy_configuration_has_a_deterministic_boundary_solution() -> None:
    config = AggregationConfig(
        method="variational",
        robustness=0.0,
        entropy_weight=0.0,
    )
    result = aggregate_result(BELIEFS, config=config)
    pooled_logits = np.log(BELIEFS).sum(axis=0)
    expected = (pooled_logits == pooled_logits.max()).astype(np.float64)
    expected /= expected.sum()
    assert np.array_equal(result.consensus, expected)


def test_legacy_dispatch_and_config_path_are_compatible() -> None:
    legacy = aggregate(BELIEFS, method="robust", robustness=1.5)
    configured = aggregate(
        BELIEFS,
        config=AggregationConfig(method="robust", robustness=1.5),
    )
    assert np.array_equal(legacy, configured)
    with pytest.raises(ValueError, match="mutually exclusive"):
        aggregate(
            BELIEFS,
            method="robust",
            config=AggregationConfig(method="robust", robustness=1.5),
        )
    with pytest.raises(ValueError, match="compatibility solver"):
        aggregate(
            BELIEFS,
            config=AggregationConfig(method="robust", robustness=1.5),
            robustness=2.0,
        )


def test_share_round_accepts_config_and_rejects_mixed_paths() -> None:
    config = AggregationConfig(
        method="variational",
        robustness=1.5,
        entropy_weight=0.8,
    )
    configured = share_round(BELIEFS, config=config, exclude_self=False)
    direct = aggregate_result(BELIEFS, config=config)
    assert np.array_equal(configured.consensus, direct.consensus)
    assert np.array_equal(
        configured.normalized_effective_weights,
        direct.normalized_effective_weights,
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        share_round(BELIEFS, method="robust", config=config)


@pytest.mark.parametrize(
    "config",
    (
        AggregationConfig(method="naive"),
        AggregationConfig(method="robust", robustness=1.5),
        AggregationConfig(method="variational", robustness=1.5),
    ),
)
def test_share_round_global_consensus_is_bit_identical_to_direct_dispatch(
    config: AggregationConfig,
) -> None:
    base_weights = np.asarray([0.5, 1.0, 2.0])
    shared = share_round(
        BELIEFS,
        config=config,
        base_weights=base_weights,
        exclude_self=True,
    )
    direct = aggregate_result(BELIEFS, config=config, base_weights=base_weights)
    assert np.array_equal(shared.consensus, direct.consensus)
    if shared.normalized_effective_weights is not None:
        assert np.array_equal(
            shared.normalized_effective_weights,
            direct.normalized_effective_weights,
        )


@pytest.mark.parametrize(
    "config",
    (
        AggregationConfig(method="naive"),
        AggregationConfig(method="robust", robustness=1.5),
        AggregationConfig(method="variational", robustness=1.5),
    ),
)
def test_share_round_generators_preserve_exact_global_and_leave_one_out_dispatch(
    config: AggregationConfig,
) -> None:
    """One-shot iterables must not introduce a second normalization pass."""
    base_weights = np.asarray([0.5, 1.0, 2.0])
    shared = share_round(
        (row for row in BELIEFS),
        config=config,
        base_weights=(float(weight) for weight in base_weights),
        exclude_self=True,
    )
    direct = aggregate_result(BELIEFS, config=config, base_weights=base_weights)
    assert np.array_equal(shared.consensus, direct.consensus)

    all_indices = np.arange(len(BELIEFS))
    for recipient in all_indices:
        included = all_indices[all_indices != recipient]
        expected = aggregate_result(
            BELIEFS[included],
            config=config,
            base_weights=base_weights[included],
        )
        assert np.array_equal(shared.shared_posteriors[recipient], expected.consensus)


def test_dispatch_rejects_non_configuration_objects() -> None:
    invalid: Any = {"method": "naive"}
    with pytest.raises(ValueError, match="config must be an AggregationConfig"):
        aggregate_result(BELIEFS, config=invalid)
    with pytest.raises(ValueError, match="config must be an AggregationConfig"):
        share_round(BELIEFS, config=invalid)
