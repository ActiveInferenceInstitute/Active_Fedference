"""Recovery and validation tests for the mixed discrete/Gaussian slice."""

from __future__ import annotations

import numpy as np
import pytest

from fedference.hybrid import (
    HybridBelief,
    _hybrid_divergence,
    hybrid_aggregate,
    hybrid_log_linear_pool,
)


def _beliefs() -> list[HybridBelief]:
    return [
        HybridBelief(np.array([0.8, 0.2]), np.array([-1.0, 1.0]), np.array([0.5, 0.8])),
        HybridBelief(np.array([0.6, 0.4]), np.array([-0.5, 1.5]), np.array([0.7, 0.6])),
    ]


def test_zero_robustness_is_exact_hybrid_log_pool():
    beliefs = _beliefs()
    direct = hybrid_log_linear_pool(beliefs)
    result = hybrid_aggregate(beliefs, robustness=0.0)
    np.testing.assert_allclose(result.consensus.discrete, direct.discrete)
    np.testing.assert_allclose(result.consensus.gaussian_mean, direct.gaussian_mean)
    np.testing.assert_allclose(result.consensus.gaussian_var, direct.gaussian_var)
    assert result.iterations == 0
    assert result.converged


def test_one_component_hybrid_is_gaussian_precision_pool():
    beliefs = [
        HybridBelief(np.array([1.0]), np.array([0.0]), np.array([1.0])),
        HybridBelief(np.array([1.0]), np.array([2.0]), np.array([0.5])),
    ]
    result = hybrid_log_linear_pool(beliefs)
    precision = 0.5 / 1.0 + 0.5 / 0.5
    assert result.gaussian_mean[0] == pytest.approx((0.5 * 0.0 + 0.5 * 2.0 / 0.5) / precision)
    assert result.gaussian_var[0] == pytest.approx(1.0 / precision)


def test_positive_robustness_returns_valid_hybrid_and_reweighting():
    result = hybrid_aggregate(_beliefs(), robustness=1.0)
    assert np.isclose(result.consensus.discrete.sum(), 1.0)
    assert np.all(result.consensus.gaussian_var > 0.0)
    assert np.isclose(result.agent_weights.sum(), 1.0)
    assert np.all(result.agent_weights >= 0.0)


def test_hybrid_rejects_mismatched_or_invalid_components():
    with pytest.raises(ValueError, match="match discrete"):
        HybridBelief(np.array([0.5, 0.5]), np.array([0.0]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match="strictly positive"):
        HybridBelief(np.array([0.5, 0.5]), np.array([0.0, 1.0]), np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="same component"):
        hybrid_log_linear_pool(
            [_beliefs()[0], HybridBelief(np.array([1.0]), np.array([0.0]), np.array([1.0]))]
        )


def test_hybrid_joint_kl_weights_conditionals_by_the_reporting_context() -> None:
    row = HybridBelief(
        np.asarray([0.9, 0.1]),
        np.asarray([10.0, 0.0]),
        np.ones(2),
    )
    reference = HybridBelief(
        np.asarray([0.1, 0.9]),
        np.zeros(2),
        np.ones(2),
    )
    categorical = float(
        np.sum(row.discrete * np.log(row.discrete / reference.discrete))
    )
    expected = categorical + 0.9 * 50.0
    assert _hybrid_divergence(row, reference) == pytest.approx(expected)


def test_hybrid_values_and_result_weights_are_owned_and_read_only() -> None:
    discrete = np.asarray([0.8, 0.2])
    means = np.asarray([0.0, 1.0])
    belief = HybridBelief(discrete, means, np.ones(2))
    discrete[:] = [0.1, 0.9]
    means[:] = [4.0, 5.0]
    assert np.array_equal(belief.discrete, np.asarray([0.8, 0.2]))
    assert np.array_equal(belief.gaussian_mean, np.asarray([0.0, 1.0]))
    with pytest.raises(ValueError, match="read-only"):
        belief.discrete[0] = 0.5

    result = hybrid_aggregate(_beliefs(), robustness=1.0)
    with pytest.raises(ValueError, match="read-only"):
        result.agent_weights[0] = 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"robustness": True}, "robustness"),
        ({"robustness": "1.0"}, "robustness"),
        ({"max_iter": 1.0}, "max_iter"),
        ({"tol": True}, "tol"),
    ),
)
def test_hybrid_controls_reject_coercive_types(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        hybrid_aggregate(_beliefs(), **kwargs)
