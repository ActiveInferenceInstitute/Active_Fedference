"""Tests for run_parameter_recovery in the experiments harness (no mocks).

All computations use a small but real seeded run:
  acuity_grid=(0.65, 0.80), n_observations=10, n_trials=3, fit_resolution=8, seed=42
"""

from __future__ import annotations

import pytest

from fedference.experiments import run_parameter_recovery

# ---------------------------------------------------------------------------
# Shared fixture — small fast run used by every test
# ---------------------------------------------------------------------------

_ACUITY_GRID = (0.65, 0.80)
_COMMON_KWARGS = dict(
    acuity_grid=_ACUITY_GRID,
    n_observations=10,
    n_trials=3,
    fit_resolution=8,
)
_SEED = 42


@pytest.fixture(scope="module")
def result():
    return run_parameter_recovery(_SEED, **_COMMON_KWARGS)


# ---------------------------------------------------------------------------
# All report and provenance keys are present
# ---------------------------------------------------------------------------

_EXPECTED_KEYS = {
    "true_acuity",
    "recovered_acuity",
    "recovered_acuity_ci_lo",
    "recovered_acuity_ci_hi",
    "abs_error",
    "mean_abs_error",
    "r_squared",
    "n_trials",
    "n_observations",
    "acuity_grid",
    "interval_method",
    "interval_percent",
    "seed",
}


def test_run_parameter_recovery_returns_expected_keys(result):
    """All report and interval-provenance keys are present in the returned dict."""
    assert _EXPECTED_KEYS == set(result.keys())


# ---------------------------------------------------------------------------
# List lengths match the acuity_grid length
# ---------------------------------------------------------------------------

def test_run_parameter_recovery_shapes_match(result):
    """Per-level list lengths match the number of acuity grid points."""
    n = len(_ACUITY_GRID)
    for key in (
        "true_acuity",
        "recovered_acuity",
        "recovered_acuity_ci_lo",
        "recovered_acuity_ci_hi",
        "abs_error",
    ):
        assert len(result[key]) == n, f"key {key!r}: expected {n}, got {len(result[key])}"


# ---------------------------------------------------------------------------
# R-squared is bounded in [0, 1]
# ---------------------------------------------------------------------------

def test_run_parameter_recovery_r_squared_bounded(result):
    """r_squared is a float in [0.0, 1.0]."""
    r2 = result["r_squared"]
    assert isinstance(r2, float)
    assert 0.0 <= r2 <= 1.0


# ---------------------------------------------------------------------------
# Seed is echoed in the result
# ---------------------------------------------------------------------------

def test_run_parameter_recovery_seed_echoed(result):
    """result['seed'] matches the seed argument."""
    assert result["seed"] == _SEED


# ---------------------------------------------------------------------------
# CI bounds are valid: ci_lo <= recovered <= ci_hi for every level
# ---------------------------------------------------------------------------

def test_run_parameter_recovery_ci_bounds_valid(result):
    """Percentile-interval bounds are ordered: lo <= mean <= hi for all levels."""
    for i, (lo, rec, hi) in enumerate(
        zip(
            result["recovered_acuity_ci_lo"],
            result["recovered_acuity"],
            result["recovered_acuity_ci_hi"],
        )
    ):
        assert lo <= rec, f"Level {i}: ci_lo ({lo}) > recovered ({rec})"
        assert rec <= hi, f"Level {i}: recovered ({rec}) > ci_hi ({hi})"


def test_parameter_recovery_interval_provenance_is_explicit(result):
    """The uncertainty fields identify descriptive trial quantiles honestly."""
    assert result["interval_method"] == "empirical_percentile_across_independent_trials"
    assert result["interval_percent"] == 95


# ---------------------------------------------------------------------------
# ISC scientific claim: recovery must actually be calibrated
# (r² > 0.8, MAE < 0.15)
# ---------------------------------------------------------------------------

def test_run_parameter_recovery_recovery_quality_is_scientifically_defensible():
    """The grid-MLE recovers acuity well enough to publish.

    r² must be high (> 0.8) and MAE small (< 0.15) for the default grid at
    moderate n_observations. The existing r²∈[0,1] test is satisfied by any
    uncalibrated estimator; this test pins the actual identifiability claim.
    """
    result = run_parameter_recovery(
        0,
        acuity_grid=(0.60, 0.70, 0.80, 0.90),
        n_observations=30,
        n_trials=10,
        fit_resolution=20,
    )
    r2 = result["r_squared"]
    mae = result["mean_abs_error"]
    assert r2 > 0.8, f"r² = {r2:.3f}: recovery must be calibrated (need > 0.8)"
    assert mae < 0.15, f"MAE = {mae:.3f}: too large for identifiability claim (need < 0.15)"
    for i, (true_a, rec_a) in enumerate(
        zip(result["true_acuity"], result["recovered_acuity"])
    ):
        assert abs(rec_a - true_a) < 0.25, (
            f"grid point {i}: |{rec_a:.2f} - {true_a:.2f}| = {abs(rec_a - true_a):.2f} "
            f"exceeds per-point tolerance 0.25"
        )
