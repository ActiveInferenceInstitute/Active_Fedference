"""Tests for proper categorical scoring and calibration diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from fedference.scoring import (
    brier_score,
    categorical_log_score,
    deterministic_score_controls,
    expected_calibration_error,
    reliability_curve,
    summarize_scores,
)


def test_log_and_brier_scores_match_known_binary_values():
    probabilities = np.array([[0.75, 0.25], [0.10, 0.90]])
    states = np.array([0, 1])
    np.testing.assert_allclose(categorical_log_score(probabilities, states), np.log([0.75, 0.90]))
    np.testing.assert_allclose(brier_score(probabilities, states), [0.125, 0.02])


def test_zero_probability_is_clipped_and_controls_have_expected_ordering():
    probabilities = np.array([[0.0, 1.0]])
    score = categorical_log_score(probabilities, np.array([0]))
    assert np.isfinite(score[0])
    controls = deterministic_score_controls(4, 0)
    states = np.array([0])
    scores = {
        name: categorical_log_score(belief[None, :], states)[0]
        for name, belief in controls.items()
    }
    assert scores["oracle"] > scores["uniform"] > scores["confident_wrong"]


def test_reliability_curve_keeps_empty_bins_and_ece_is_zero_for_perfect_control():
    controls = deterministic_score_controls(4, 0)
    probabilities = np.vstack([controls["oracle"]] * 4)
    states = np.zeros(4, dtype=int)
    curve = reliability_curve(probabilities, states, n_bins=5)
    assert len(curve["bin_center"]) == 5
    assert sum(curve["count"]) == 4
    assert expected_calibration_error(probabilities, states, n_bins=5) == pytest.approx(0.0)


def test_summary_reports_declared_count_and_rejects_invalid_inputs():
    probabilities = np.array([[0.6, 0.4], [0.4, 0.6]])
    summary = summarize_scores(probabilities, np.array([0, 1]), n_bins=2)
    assert summary["n_observations"] == 2
    assert summary["n_bins"] == 2
    with pytest.raises(ValueError, match="sum to one"):
        categorical_log_score(np.array([[0.7, 0.4]]), np.array([0]))
    with pytest.raises(ValueError, match="n_bins"):
        reliability_curve(probabilities, np.array([0, 1]), n_bins=0)
