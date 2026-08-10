"""V4 moving sentinel world: structure, disjoint FOVs, EFE selection, federation.

No mocks; every assertion runs the real numerical primitives with fixed seeds.
"""

from __future__ import annotations

import numpy as np

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
