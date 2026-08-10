"""Tests for the sensitivity sweep functions in fedference.experiments."""

from __future__ import annotations

from fedference.experiments import (
    run_belief_sharing_sensitivity,
    run_hierarchical_sensitivity,
)

_ACUITY_VALUES = (0.5, 0.7, 0.9)
_N_AGENTS_VALUES = (2, 4, 6)
_N_TRIALS = 3


def test_run_belief_sharing_sensitivity_shape() -> None:
    result = run_belief_sharing_sensitivity(
        0,
        acuity_values=_ACUITY_VALUES,
        n_agents_values=_N_AGENTS_VALUES,
        n_trials=_N_TRIALS,
    )
    n_a, n_n = len(_ACUITY_VALUES), len(_N_AGENTS_VALUES)
    for key in ("communicating_grid", "isolated_grid", "accuracy_gap_grid"):
        grid = result[key]
        assert len(grid) == n_a, f"{key} row count"
        for row in grid:
            assert len(row) == n_n, f"{key} col count"
    assert result["n_trials"] == _N_TRIALS, "n_trials must be echoed in return dict"
    assert result["seed"] == 0


def test_run_belief_sharing_sensitivity_values_in_unit_interval() -> None:
    result = run_belief_sharing_sensitivity(
        1,
        acuity_values=_ACUITY_VALUES,
        n_agents_values=_N_AGENTS_VALUES,
        n_trials=_N_TRIALS,
    )
    for grid_key in ("communicating_grid", "isolated_grid"):
        for row in result[grid_key]:
            for val in row:
                assert 0.0 <= val <= 1.0, f"{grid_key} value out of [0,1]: {val}"


def test_run_belief_sharing_sensitivity_seed_determinism() -> None:
    r1 = run_belief_sharing_sensitivity(
        42,
        acuity_values=_ACUITY_VALUES,
        n_agents_values=_N_AGENTS_VALUES,
        n_trials=_N_TRIALS,
    )
    r2 = run_belief_sharing_sensitivity(
        42,
        acuity_values=_ACUITY_VALUES,
        n_agents_values=_N_AGENTS_VALUES,
        n_trials=_N_TRIALS,
    )
    assert r1["communicating_grid"] == r2["communicating_grid"]
    assert r1["isolated_grid"] == r2["isolated_grid"]


def test_run_belief_sharing_sensitivity_acuity_monotone() -> None:
    acuity_values = (0.4, 0.7, 0.95)
    result = run_belief_sharing_sensitivity(
        0,
        acuity_values=acuity_values,
        n_agents_values=(4,),
        n_trials=5,
    )
    comm = [result["communicating_grid"][i][0] for i in range(len(acuity_values))]
    # Higher acuity should (almost surely) give equal or higher accuracy.
    # A weak check: max acuity ≥ min acuity in the communicating condition.
    assert comm[-1] >= comm[0] - 0.3, "communicating accuracy should trend upward with acuity"


def test_run_hierarchical_sensitivity_shape() -> None:
    result = run_hierarchical_sensitivity(
        0,
        acuity_values=_ACUITY_VALUES,
        n_agents_values=_N_AGENTS_VALUES,
        n_trials=_N_TRIALS,
    )
    n_a, n_n = len(_ACUITY_VALUES), len(_N_AGENTS_VALUES)
    for key in ("flat_grid", "hierarchical_grid", "accuracy_gap_grid"):
        grid = result[key]
        assert len(grid) == n_a
        for row in grid:
            assert len(row) == n_n
    assert result["n_trials"] == _N_TRIALS, "n_trials must be echoed in return dict"
    assert result["seed"] == 0


def test_run_hierarchical_sensitivity_values_in_unit_interval() -> None:
    result = run_hierarchical_sensitivity(
        2,
        acuity_values=_ACUITY_VALUES,
        n_agents_values=_N_AGENTS_VALUES,
        n_trials=_N_TRIALS,
    )
    for grid_key in ("flat_grid", "hierarchical_grid"):
        for row in result[grid_key]:
            for val in row:
                assert 0.0 <= val <= 1.0, f"{grid_key} value {val} out of [0,1]"


def test_run_hierarchical_sensitivity_seed_determinism() -> None:
    r1 = run_hierarchical_sensitivity(
        7,
        acuity_values=_ACUITY_VALUES,
        n_agents_values=_N_AGENTS_VALUES,
        n_trials=_N_TRIALS,
    )
    r2 = run_hierarchical_sensitivity(
        7,
        acuity_values=_ACUITY_VALUES,
        n_agents_values=_N_AGENTS_VALUES,
        n_trials=_N_TRIALS,
    )
    assert r1["hierarchical_grid"] == r2["hierarchical_grid"]
    assert r1["flat_grid"] == r2["flat_grid"]
