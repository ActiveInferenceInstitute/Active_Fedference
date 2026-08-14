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
    validate_score_summary,
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
    assert any(value is None for value in curve["accuracy"])
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


@pytest.mark.parametrize(
    ("probabilities", "states", "message"),
    (
        (np.asarray([]), np.asarray([], dtype=int), "shape"),
        (np.asarray([[1.0]]), np.asarray([0]), "shape"),
        (np.asarray([[np.nan, 0.0]]), np.asarray([0]), "finite"),
        (np.asarray([[-0.1, 1.1]]), np.asarray([0]), "non-negative"),
        (np.asarray([[0.6, 0.5]]), np.asarray([0]), "sum to one"),
        (np.asarray([[0.5, 0.5]]), np.asarray([[0]]), "one-dimensional"),
        (np.asarray([[0.5, 0.5]]), np.asarray([0.5]), "integer state"),
        (np.asarray([[0.5, 0.5]]), np.asarray([2]), "outside"),
    ),
)
def test_scoring_validates_shapes_values_and_labels(probabilities, states, message):
    with pytest.raises(ValueError, match=message):
        categorical_log_score(probabilities, states)


@pytest.mark.parametrize("clip", (0.0, -1.0, 1.1, float("nan")))
def test_log_score_rejects_invalid_clip(clip):
    with pytest.raises(ValueError, match="clip"):
        categorical_log_score(np.asarray([[0.5, 0.5]]), np.asarray([0]), clip=clip)


@pytest.mark.parametrize("n_bins", (True, 1.5, 0))
def test_reliability_curve_rejects_non_integer_bin_counts(n_bins):
    with pytest.raises(ValueError, match="n_bins"):
        reliability_curve(np.asarray([[0.5, 0.5]]), np.asarray([0]), n_bins=n_bins)


@pytest.mark.parametrize(
    ("n_states", "true_state", "message"),
    (
        (True, 0, "n_states"),
        (1, 0, "n_states"),
        (3, True, "true_state"),
        (3, 1.5, "true_state"),
        (3, 3, "within"),
    ),
)
def test_score_controls_reject_invalid_state_declarations(n_states, true_state, message):
    with pytest.raises(ValueError, match=message):
        deterministic_score_controls(n_states=n_states, true_state=true_state)


@pytest.mark.parametrize(
    ("summary", "message"),
    (
        ({}, "missing"),
        (
            {"mean_log_score": float("nan"), "mean_brier_score": 0.0, "ece": 0.0, "n_observations": 1},
            "finite",
        ),
        (
            {"mean_log_score": 0.0, "mean_brier_score": 0.0, "ece": 0.0, "n_observations": True},
            "positive integer",
        ),
        (
            {"mean_log_score": 0.0, "mean_brier_score": 0.0, "ece": 0.0, "n_observations": 0},
            "positive integer",
        ),
    ),
)
def test_serialized_score_summary_validation_is_fail_closed(summary, message):
    with pytest.raises(ValueError, match=message):
        validate_score_summary(summary)


def test_reliability_curve_populates_nonempty_accuracy_bins() -> None:
    curve = reliability_curve(
        np.asarray([[0.4, 0.35, 0.25], [0.8, 0.1, 0.1], [0.5, 0.3, 0.2]]),
        np.asarray([0, 1, 2]),
        n_bins=4,
    )

    assert curve["count"] == [0, 1, 1, 1]
    assert curve["accuracy"][1:] == [1.0, 0.0, 0.0]
