"""Proper scoring and calibration diagnostics for categorical beliefs.

The robustness experiments traditionally report the posterior mass assigned to
the true state (an accuracy-like quantity).  This module adds belief-quality
diagnostics that remain defined when two methods choose the same argmax but
express different uncertainty.  The categorical log score is the primary
estimand for the scoring extension: larger values are better.  The Brier score
and expected calibration error are secondary diagnostics; smaller values are
better for both.

Scores are evaluated on a declared independent unit by callers.  This module
does not treat agents or repeated episodes as independent observations and does
not make any claim about decision optimality, distribution-shift calibration,
or robustness outside the supplied data-generating process.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

ArrayF = np.ndarray
_DEFAULT_CLIP = 1e-12


def _validated_inputs(
    probabilities: ArrayF, true_states: ArrayF
) -> tuple[ArrayF, np.ndarray]:
    """Validate a row-wise categorical prediction matrix and labels."""
    probs = np.asarray(probabilities, dtype=np.float64)
    states_raw = np.asarray(true_states)
    if probs.ndim != 2 or probs.shape[0] == 0 or probs.shape[1] < 2:
        raise ValueError("probabilities must have shape (n_observations, n_states >= 2)")
    if not np.all(np.isfinite(probs)) or np.any(probs < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("each probability row must sum to one")
    if states_raw.ndim != 1 or states_raw.shape[0] != probs.shape[0]:
        raise ValueError("true_states must be a one-dimensional label vector matching probabilities")
    if np.issubdtype(states_raw.dtype, np.floating):
        if not np.all(np.isfinite(states_raw)) or not np.all(states_raw == np.floor(states_raw)):
            raise ValueError("true_states must contain integer state indices")
    states = states_raw.astype(np.int64, copy=False)
    if np.any(states < 0) or np.any(states >= probs.shape[1]):
        raise ValueError("true_states contain an index outside the probability columns")
    return probs, states


def categorical_log_score(
    probabilities: ArrayF,
    true_states: ArrayF,
    *,
    clip: float = _DEFAULT_CLIP,
) -> ArrayF:
    """Return per-observation categorical log scores in nats.

    The score is ``log(p_true)`` and therefore larger is better.  Exact zeros
    are clipped to ``clip`` so a finite score remains available for a declared
    deterministic negative control.  ``clip`` must be finite and strictly
    positive; it is a numerical convention, not an uncertainty correction.
    """
    if not np.isfinite(clip) or clip <= 0.0 or clip > 1.0:
        raise ValueError("clip must be finite and lie in (0, 1]")
    probs, states = _validated_inputs(probabilities, true_states)
    p_true = np.maximum(probs[np.arange(probs.shape[0]), states], clip)
    return np.log(p_true)


def brier_score(probabilities: ArrayF, true_states: ArrayF) -> ArrayF:
    """Return per-observation multiclass Brier scores (smaller is better)."""
    probs, states = _validated_inputs(probabilities, true_states)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(probs.shape[0]), states] = 1.0
    return np.sum((probs - one_hot) ** 2, axis=1)


def reliability_curve(
    probabilities: ArrayF,
    true_states: ArrayF,
    *,
    n_bins: int = 10,
) -> dict[str, list[float] | list[int]]:
    """Return equal-width confidence/reliability bins for multiclass beliefs.

    Each observation contributes its maximum predicted probability, whether
    that prediction is correct, and its count. Empty bins are retained with
    ``NaN`` summaries so the bin geometry remains explicit in reports. The
    result is a diagnostic curve, not a calibration guarantee.
    """
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    probs, states = _validated_inputs(probabilities, true_states)
    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    correct = (predicted == states).astype(np.float64)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    indices = np.minimum(np.searchsorted(edges, confidence, side="right") - 1, n_bins - 1)
    mean_confidence: list[float] = []
    accuracy: list[float] = []
    counts: list[int] = []
    centers: list[float] = []
    for index in range(int(n_bins)):
        mask = indices == index
        centers.append(float((edges[index] + edges[index + 1]) / 2.0))
        counts.append(int(mask.sum()))
        if not np.any(mask):
            mean_confidence.append(float("nan"))
            accuracy.append(float("nan"))
        else:
            mean_confidence.append(float(confidence[mask].mean()))
            accuracy.append(float(correct[mask].mean()))
    return {
        "bin_center": centers,
        "mean_confidence": mean_confidence,
        "accuracy": accuracy,
        "count": counts,
    }


def expected_calibration_error(
    probabilities: ArrayF,
    true_states: ArrayF,
    *,
    n_bins: int = 10,
) -> float:
    """Return equal-width expected calibration error.

    ECE is the count-weighted absolute gap between mean confidence and empirical
    accuracy over non-empty bins. It is a descriptive summary of the supplied
    finite sample; it does not establish calibration under distribution shift.
    """
    curve = reliability_curve(probabilities, true_states, n_bins=n_bins)
    counts = np.asarray(curve["count"], dtype=np.float64)
    confidence = np.asarray(curve["mean_confidence"], dtype=np.float64)
    accuracy = np.asarray(curve["accuracy"], dtype=np.float64)
    total = float(counts.sum())
    if total <= 0.0:
        raise ValueError("reliability curve contains no observations")
    nonempty = counts > 0.0
    return float(
        np.sum(
            (counts[nonempty] / total)
            * np.abs(confidence[nonempty] - accuracy[nonempty])
        )
    )


def summarize_scores(
    probabilities: ArrayF,
    true_states: ArrayF,
    *,
    n_bins: int = 10,
    clip: float = _DEFAULT_CLIP,
) -> dict[str, float | int]:
    """Return mean primary/secondary scores and their declared sample count."""
    log_scores = categorical_log_score(probabilities, true_states, clip=clip)
    brier = brier_score(probabilities, true_states)
    return {
        "mean_log_score": float(log_scores.mean()),
        "mean_brier_score": float(brier.mean()),
        "ece": expected_calibration_error(probabilities, true_states, n_bins=n_bins),
        "n_observations": int(log_scores.size),
        "clip": float(clip),
        "n_bins": int(n_bins),
    }


def deterministic_score_controls(n_states: int = 4, true_state: int = 0) -> dict[str, ArrayF]:
    """Return oracle, uniform, and confidently-wrong control beliefs.

    These controls are used by the seeded scoring experiment and are intentionally
    simple: the oracle must score best, the uniform control is finite and
    uninformative, and the confident-wrong control must score worse than uniform
    under the log score.
    """
    if isinstance(n_states, bool) or not isinstance(n_states, (int, np.integer)) or n_states < 2:
        raise ValueError("n_states must be an integer >= 2")
    if isinstance(true_state, bool) or not isinstance(true_state, (int, np.integer)):
        raise ValueError("true_state must be an integer index")
    if not 0 <= int(true_state) < int(n_states):
        raise ValueError("true_state must be within n_states")
    states = int(n_states)
    state = int(true_state)
    oracle = np.zeros(states, dtype=np.float64)
    oracle[state] = 1.0
    uniform = np.full(states, 1.0 / states, dtype=np.float64)
    wrong_state = (state + 1) % states
    wrong = np.full(states, 0.01 / (states - 1), dtype=np.float64)
    wrong[wrong_state] = 0.99
    return {
        "oracle": oracle,
        "uniform": uniform,
        "confident_wrong": wrong,
    }


def validate_score_summary(summary: Mapping[str, object]) -> None:
    """Fail closed if a serialized score summary loses its core fields."""
    required = ("mean_log_score", "mean_brier_score", "ece", "n_observations")
    missing = [key for key in required if key not in summary]
    if missing:
        raise ValueError(f"score summary missing required fields: {missing}")
    for key in required[:3]:
        value = summary[key]
        if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
            raise ValueError(f"score summary field {key!r} must be finite numeric")
    count = summary["n_observations"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("score summary n_observations must be a positive integer")


__all__ = [
    "brier_score",
    "categorical_log_score",
    "deterministic_score_controls",
    "expected_calibration_error",
    "reliability_curve",
    "summarize_scores",
    "validate_score_summary",
]
