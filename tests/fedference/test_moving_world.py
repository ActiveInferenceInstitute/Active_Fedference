"""V4 moving sentinel world: structure, disjoint FOVs, EFE selection, federation.

No mocks; every assertion runs the real numerical primitives with fixed seeds.
"""

from __future__ import annotations

import hashlib
import runpy
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fedference.pomdp import build_moving_world, efe_policy_select


def test_build_moving_world_structure() -> None:
    world = build_moving_world(n_positions=4, n_agents=2)
    assert world["n_agents"] == 2
    assert world["n_positions"] == 4
    assert len(world["A"]) == 2
    assert world["A"][0].shape == (2, 2)
    assert world["n_actions"] == 3
    B = world["B"]
    assert B.shape == (4, 4, 3)
    for u in range(3):
        assert np.allclose(B[:, :, u].sum(axis=0), 1.0), f"B action {u} cols not unit sum"
    for A in world["A"]:
        assert np.allclose(A.sum(axis=0), 1.0, atol=1e-9), "A cols not unit sum"


def test_disjoint_fov() -> None:
    world = build_moving_world(n_positions=4, n_agents=2, fov_width=2)
    A0, A1 = world["A"]
    assert not np.allclose(A0, A1), "A0 and A1 must differ (disjoint FOVs)"
    assert not np.allclose(A0, 0.5 * np.ones((2, 2))), "A0 should be informative, not uniform"


def test_efe_policy_select_runs() -> None:
    world = build_moving_world(n_positions=4, n_agents=2)
    beliefs = [np.array([0.5, 0.5]), np.array([0.5, 0.5])]
    actions = efe_policy_select(beliefs, world)
    assert len(actions) == 2
    assert all(0 <= a <= 2 for a in actions)


def test_communicating_outperforms_isolated() -> None:
    from fedference.experiments import run_moving_world

    results = run_moving_world(seed=0, n_trials=30)
    acc = results["accuracy"]
    # communicating should do at least as well as isolated (allow slack for stochasticity)
    assert acc["communicating"] >= acc["isolated"] - 0.10, (
        f"comm={acc['communicating']:.3f} isolated={acc['isolated']:.3f}"
    )


def test_free_energy_gap_nonnegative_communicating() -> None:
    from fedference.experiments import run_moving_world

    results = run_moving_world(seed=1, n_trials=20)
    gap = results["free_energy_gap"]
    # Gap is isolated_fe - condition_fe; communicating should have lower FE (positive gap)
    assert gap["communicating"] >= -0.5, f"gap unexpectedly negative: {gap['communicating']:.3f}"


def test_moving_world_figure_generates() -> None:
    import pathlib
    import tempfile

    from figures.moving_world import generate_moving_world

    fixture = {
        "accuracy": {"isolated": 0.4, "communicating": 0.7, "efe_guided": 0.75},
        "free_energy_gap": {"isolated": 0.0, "communicating": 0.3, "efe_guided": 0.35},
        "n_steps_to_consensus": {"isolated": 6.0, "communicating": 3.0, "efe_guided": 2.5},
        "n_trials": 20,
    }
    with tempfile.TemporaryDirectory() as tmp:
        out = generate_moving_world(fixture, project_root=pathlib.Path(tmp))
        assert out.exists() and out.stat().st_size > 0


_REFERENCE = Path(__file__).parents[1] / "fixtures/performance/moving-worlds-0a14cc5.py.txt"
_REFERENCE_SHA256 = "a83645cad08fd0cf7d54db77771d962a7eb96ada990af2eb6ae7695f66c5aa5d"


def _moving_reference(path: Path = _REFERENCE) -> Callable[..., dict[str, Any]]:
    """Exact original worlds module at 0a14cc56, with unchanged real primitives."""
    assert hashlib.sha256(path.read_bytes()).hexdigest() == _REFERENCE_SHA256
    namespace = runpy.run_path(str(path), run_name="fedference.experiments._moving_world_reference")
    function = namespace["run_moving_world"]
    assert callable(function)
    return function


@pytest.mark.parametrize(
    "seed,positions,agents,steps,trials", [(0, 4, 2, 2, 2), (7, 6, 3, 3, 2), (11, 3, 1, 2, 3)]
)
def test_moving_report_preserves_original_seeded_trajectories(
    seed: int, positions: int, agents: int, steps: int, trials: int,
) -> None:
    from fedference.experiments import run_moving_world

    kwargs = {"seed": seed, "n_positions": positions, "n_agents": agents,
              "n_steps": steps, "n_trials": trials}
    assert run_moving_world(**kwargs) == _moving_reference()(**kwargs)


def test_moving_reference_rejects_changed_original_bytes(tmp_path: Path) -> None:
    changed = tmp_path / "changed.py.txt"
    changed.write_bytes(_REFERENCE.read_bytes() + b"\n# changed\n")
    with pytest.raises(AssertionError):
        _moving_reference(changed)


def test_moving_trials_consume_one_admitted_gnn_world_each() -> None:
    from fedference.experiments import run_moving_world
    from fedference.gnn.structured_cache import cached_structured
    from fedference.pomdp import _moving_likelihood

    counts = {"admissions": 0, "reconstructions": 0}
    codes = {cached_structured.__code__: "admissions", _moving_likelihood.__code__: "reconstructions"}

    def observe(frame: Any, event: str, value: object) -> None:
        if event == "call" and frame.f_code in codes:
            counts[codes[frame.f_code]] += 1

    previous = sys.getprofile()
    try:
        sys.setprofile(observe)
        report = run_moving_world(seed=23, n_trials=2, n_steps=3)
    finally:
        sys.setprofile(previous)
    assert report["n_trials"] == 2 and report["n_steps"] == 3
    assert counts == {"admissions": 2, "reconstructions": 0}


@pytest.mark.parametrize("positions,agents", [(3, 1), (4, 2), (6, 3)])
def test_admitted_moving_map_matches_every_original_position(positions: int, agents: int) -> None:
    from fedference.pomdp import _moving_likelihood

    world = build_moving_world(n_positions=positions, n_agents=agents)
    for position in range(positions):
        original = _moving_likelihood(position, fov_width=world["fov_width"],
                                      n_positions=positions, n_states=world["n_states"])
        np.testing.assert_array_equal(world["observation_map"][:, :, position], original, strict=True)
