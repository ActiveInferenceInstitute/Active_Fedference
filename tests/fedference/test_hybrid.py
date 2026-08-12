"""Recovery and validation tests for the mixed discrete/Gaussian slice."""

from __future__ import annotations

import numpy as np
import pytest

from fedference.hybrid import (
    HybridAggregationResult,
    HybridBelief,
    _hybrid_divergence,
    _normalise_weights,
    _stack,
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


def test_hybrid_stack_and_weights_fail_closed() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _stack([])
    with pytest.raises(TypeError, match="HybridBelief"):
        _stack([object()])
    with pytest.raises(ValueError, match="positive"):
        _normalise_weights([0.0, 0.0], 2)
    np.testing.assert_allclose(_normalise_weights(None, 2), [0.5, 0.5])


@pytest.mark.parametrize(
    ("discrete", "mean", "variance", "message"),
    (
        ([0.5, 0.5], [0.0, np.nan], [1.0, 1.0], "finite"),
        ([0.5, 0.5], [0.0, 1.0], [1.0, np.nan], "finite"),
        ([0.5, 0.5], [0.0, 1.0], [1.0, -1.0], "strictly positive"),
    ),
)
def test_hybrid_belief_rejects_nonfinite_parameters(discrete, mean, variance, message) -> None:
    with pytest.raises(ValueError, match=message):
        HybridBelief(np.asarray(discrete), np.asarray(mean), np.asarray(variance))


def test_hybrid_legacy_aliases_and_unknown_arguments_are_explicit() -> None:
    with pytest.warns(DeprecationWarning, match="beliefs"):
        pooled = hybrid_log_linear_pool(beliefs=_beliefs())
    assert pooled.n_components == 2
    with pytest.warns(DeprecationWarning, match="weights"):
        weighted = hybrid_log_linear_pool(beliefs=_beliefs(), weights=[1.0, 3.0])
    assert weighted.n_components == 2
    with pytest.raises(TypeError, match="unexpected keyword"):
        hybrid_log_linear_pool(_beliefs(), unknown=True)
    with pytest.raises(TypeError, match="cannot both"):
        hybrid_log_linear_pool(_beliefs(), beliefs=_beliefs())
    with pytest.raises(TypeError, match="cannot both"):
        hybrid_log_linear_pool(_beliefs(), base_weights=[1.0, 1.0], weights=[1.0, 1.0])


def test_hybrid_result_validates_weights_iterations_and_convergence() -> None:
    belief = _beliefs()[0]
    for weights, message in (([], "probability"), ([0.2, 0.2], "probability"), ([np.nan], "probability")):
        with pytest.raises(ValueError, match=message):
            HybridAggregationResult(belief, weights, 0, True)
    with pytest.raises(ValueError, match="non-negative integer"):
        HybridAggregationResult(belief, [1.0], -1, True)
    with pytest.raises(ValueError, match="boolean"):
        HybridAggregationResult(belief, [1.0], 0, 1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"robustness": -0.1}, "robustness"),
        ({"max_iter": 0}, "max_iter"),
        ({"tol": 0.0}, "tol"),
    ),
)
def test_hybrid_aggregate_rejects_invalid_controls(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        hybrid_aggregate(_beliefs(), **kwargs)


def test_hybrid_aggregate_legacy_aliases_and_argument_conflicts() -> None:
    with pytest.warns(DeprecationWarning, match="beliefs"):
        result = hybrid_aggregate(beliefs=_beliefs(), robustness=0.0)
    assert result.converged
    with pytest.warns(DeprecationWarning, match="weights"):
        result = hybrid_aggregate(beliefs=_beliefs(), weights=[1.0, 2.0], robustness=0.0)
    assert result.iterations == 0
    with pytest.raises(TypeError, match="cannot both"):
        hybrid_aggregate(_beliefs(), beliefs=_beliefs())
    with pytest.raises(TypeError, match="unexpected keyword"):
        hybrid_aggregate(_beliefs(), unknown=True)
