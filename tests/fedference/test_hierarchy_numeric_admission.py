"""Reject invalid hierarchy probabilities without repairing caller inputs."""

from __future__ import annotations

import numpy as np
import pytest

from fedference.pomdp import (
    LayerSpec,
    build_3level_world,
    build_hierarchical_world,
    build_nlevel_world,
)


@pytest.fixture(params=[
    [1.1, -.1], [1.0 + 1e-12, -1e-12], [np.nan, .5],
    [np.inf, .5], [-np.inf, .5], [1e308, 1e308], [0., 0.], [.2, .3], [1.],
])
def invalid_pair(request: pytest.FixtureRequest) -> np.ndarray:
    return np.array(request.param)


def test_nlevel_rejects_invalid_default_prior(invalid_pair: np.ndarray) -> None:
    layers = [LayerSpec(2, default_prior=invalid_pair), LayerSpec(9)]
    with pytest.raises(ValueError, match="default_prior"):
        build_nlevel_world(layers)


def test_nlevel_rejects_invalid_conditioned_prior(invalid_pair: np.ndarray) -> None:
    child = np.pad(invalid_pair, (0, 7))
    layers = [LayerSpec(2, conditioned_priors=[child, child]), LayerSpec(9)]
    with pytest.raises(ValueError, match="conditioned_prior"):
        build_nlevel_world(layers)


def test_context_rejects_invalid_prior(invalid_pair: np.ndarray) -> None:
    with pytest.raises(ValueError, match="context_prior"):
        build_hierarchical_world(context_prior=tuple(invalid_pair))


def test_context_rejects_invalid_conditioned_prior(invalid_pair: np.ndarray) -> None:
    child = tuple(np.pad(invalid_pair, (0, 7)))
    with pytest.raises(ValueError, match="l1_priors"):
        build_hierarchical_world(l1_priors=(child, child))


def test_context_rejects_invalid_transition(invalid_pair: np.ndarray) -> None:
    with pytest.raises(ValueError, match="context_transition"):
        build_hierarchical_world(context_transition=(tuple(invalid_pair), tuple(invalid_pair)))


def test_three_level_rejects_invalid_default_prior(invalid_pair: np.ndarray) -> None:
    with pytest.raises(ValueError, match="l3_prior"):
        build_3level_world(l3_prior=tuple(invalid_pair))


def test_three_level_rejects_invalid_l2_prior(invalid_pair: np.ndarray) -> None:
    with pytest.raises(ValueError, match="l2_priors_given_l3"):
        build_3level_world(l2_priors_given_l3=(tuple(invalid_pair), (.5, .5)))


def test_three_level_rejects_invalid_l1_prior(invalid_pair: np.ndarray) -> None:
    child = tuple(np.pad(invalid_pair, (0, 7)))
    with pytest.raises(ValueError, match="l1_priors_given_l2"):
        build_3level_world(l1_priors_given_l2=(child, child))


@pytest.mark.parametrize("values", [[.3, .7], [0., 1.], [1., 0.]])
def test_nlevel_preserves_valid_supplied_priors(values: list[float]) -> None:
    prior = np.array(values)
    child = np.array([.2, .1, .1, .1, .1, .1, .1, .1, .1])
    layers = [LayerSpec(2, default_prior=prior, conditioned_priors=[child, child]),
              LayerSpec(9, default_prior=child)]
    world = build_nlevel_world(layers)
    np.testing.assert_array_equal(world["level_priors"][0], prior)
    np.testing.assert_array_equal(world["level_priors"][1], child)
    np.testing.assert_array_equal(world["conditioned_priors"][0][0], child)
