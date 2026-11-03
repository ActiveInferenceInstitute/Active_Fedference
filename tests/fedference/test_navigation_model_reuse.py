"""Real GNN admission counts and byte-bound navigation migration evidence."""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from fedference.experiments import run_efe_navigation_test
from fedference.gnn.structured import parameterize_moving
from fedference.pomdp import _moving_likelihood, build_moving_world

_REFERENCE_SHA256 = "092eae8bb700bf319d4455e25a97c2fbc92abd2f7eff2d435191cd5400aa889a"


def _original_navigation() -> Callable[..., dict[str, Any]]:
    """Load the exact f542f83 orchestration against this same numerical runtime."""
    path = Path(__file__).parents[1] / "fixtures/performance/navigation-f542f83.py.txt"
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == _REFERENCE_SHA256
    namespace: dict[str, Any] = {
        "__name__": "fedference.experiments._f542_navigation_reference",
        "__package__": "fedference.experiments",
    }
    exec(compile(data, str(path), "exec"), namespace)
    return cast(Callable[..., dict[str, Any]], namespace["run_efe_navigation_test"])


def test_navigation_admits_one_model_per_trial_and_rebuilds_on_next_call() -> None:
    admitted = 0

    def observe(frame: Any, event: str, argument: Any) -> None:
        nonlocal admitted
        if event == "call" and frame.f_code is parameterize_moving.__code__:
            admitted += 1

    prior = sys.getprofile()
    sys.setprofile(observe)
    try:
        first = run_efe_navigation_test(seed=41, n_agents=1, n_positions=2, n_steps=1)
        assert admitted == first["n_trials"] == 200
        second = run_efe_navigation_test(seed=42, n_agents=1, n_positions=2, n_steps=1)
        assert admitted == first["n_trials"] + second["n_trials"] == 400
    finally:
        sys.setprofile(prior)


@pytest.mark.parametrize("seed,agents,positions,steps", [(3, 2, 4, 3), (17, 3, 6, 2), (5, 1, 2, 1)])
def test_navigation_complete_report_matches_original_orchestration(
    seed: int, agents: int, positions: int, steps: int,
) -> None:
    options = {"seed": seed, "n_agents": agents, "n_positions": positions, "n_steps": steps}
    expected = _original_navigation()(**options)
    actual = run_efe_navigation_test(**options)
    assert actual == expected
    assert actual["n_trials"] == 200


@pytest.mark.parametrize("agents,positions", [(1, 2), (2, 4), (3, 6)])
def test_admitted_navigation_map_matches_each_original_position(agents: int, positions: int) -> None:
    world = build_moving_world(
        n_agents=agents, n_positions=positions, fov_width=positions // agents, seed=29,
    )
    observation_map = np.asarray(world["observation_map"], dtype=np.float64)
    for position in range(positions):
        original = _moving_likelihood(
            position, fov_width=int(world["fov_width"]), n_positions=positions, n_states=2,
        )
        np.testing.assert_array_equal(observation_map[:, :, position], original)
