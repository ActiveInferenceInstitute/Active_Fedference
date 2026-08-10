"""Experimental MAJ-1 belief-space comparator controls."""

from __future__ import annotations

import numpy as np

from fedference.aggregation import log_linear_pool
from fedference.aggregation_comparators import (
    clr_geometric_median_pool,
    linear_opinion_pool,
)


def test_linear_pool_is_weighted_arithmetic_mixture() -> None:
    beliefs = np.asarray([[0.8, 0.2], [0.2, 0.8]])
    result = linear_opinion_pool(beliefs, weights=[3.0, 1.0])
    assert np.allclose(result, [0.65, 0.35])


def test_clr_median_is_valid_deterministic_and_permutation_invariant() -> None:
    beliefs = np.asarray([[0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    first = clr_geometric_median_pool(beliefs)
    second = clr_geometric_median_pool(beliefs[::-1])
    assert first.converged and second.converged
    assert np.allclose(first.consensus, second.consensus, atol=1e-9)
    assert np.isclose(first.consensus.sum(), 1.0)
    assert np.all(first.consensus > 0.0)


def test_identical_beliefs_recover_that_belief_and_differ_from_no_rule() -> None:
    belief = np.asarray([0.6, 0.3, 0.1])
    result = clr_geometric_median_pool([belief, belief, belief])
    assert np.allclose(result.consensus, belief)
    assert np.allclose(log_linear_pool([belief]), belief)


def test_duplicate_majority_is_recognized_as_an_exact_weighted_median() -> None:
    majority = np.asarray([0.7, 0.2, 0.1])
    outlier = np.asarray([0.1, 0.2, 0.7])
    result = clr_geometric_median_pool(
        [majority, majority, outlier],
        weights=[0.26, 0.26, 0.48],
        max_iter=4,
        tol=1e-12,
    )
    assert result.converged
    assert result.iterations == 0
    assert np.allclose(result.consensus, majority, atol=1e-12)
