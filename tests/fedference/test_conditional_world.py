"""Seed grouping, controls, and score-order tests for MED-1/MED-2 slices."""

from __future__ import annotations

import numpy as np
import pytest

from fedference.experiments.conditional_world import (
    conditional_scenario_grid,
    run_belief_quality_sensitivity,
    run_conditional_world_generalization,
)


def test_conditional_grid_varies_all_declared_coordinates():
    grid = conditional_scenario_grid()
    assert len(grid) == 40
    assert {row.true_state for row in grid} == {0, 1}
    assert {row.observability for row in grid} == {0.45, 0.70}
    assert {row.attack for row in grid} == {
        "clean", "confident_wrong", "permutation", "label_noise", "uniform"
    }
    assert {row.adversary_weight for row in grid} == {0.5, 1.0}


def test_conditional_report_is_deterministic_and_zero_control_passes():
    first = run_conditional_world_generalization(seed=7, n_seeds=3, n_trials=2)
    second = run_conditional_world_generalization(seed=7, n_seeds=3, n_trials=2)
    assert first == second
    assert first["controls"]["robustness_zero_recovers_log_pool"]
    assert first["controls"]["seed_is_independent_unit"]
    assert len(first["by_scenario"]) == len(first["grid"]) == 40
    assert all(len(cell["contrast_by_seed"]) == 3 for cell in first["by_scenario"].values())


def test_belief_quality_controls_have_expected_ordering():
    report = run_belief_quality_sensitivity(seed=5, n_seeds=3, n_trials=4)
    assert report["controls"]["oracle_best_log_score"]
    assert report["controls"]["confident_wrong_worse_than_uniform"]
    for cell in report["by_scenario"].values():
        assert len(cell["log_score_contrast_by_seed"]) == 3
        assert np.isfinite(cell["log_score_contrast_mean"])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_seeds": 1}, "n_seeds"),
        ({"n_trials": 0}, "n_trials"),
        ({"n_agents": 2}, "n_agents"),
        ({"robustness": -1.0}, "robustness"),
    ],
)
def test_both_extensions_fail_closed_on_invalid_budgets(kwargs, message):
    with pytest.raises(ValueError, match=message):
        run_conditional_world_generalization(**kwargs)
    with pytest.raises(ValueError, match=message):
        run_belief_quality_sensitivity(**kwargs)
