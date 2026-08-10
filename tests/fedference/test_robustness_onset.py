"""Tests for the per-mechanism robustness-onset sweep (no mocks).

Confirms the rate-resolved companion to the gallery: each directional mechanism
has a naive and best-robust accuracy curve over the rate grid, the naive curve is
non-increasing, and the byzantine onset is transient (a veto cliff) while the
additive attacks sustain robustness to the worst rate. Real seeded computation.
"""

from __future__ import annotations

import pytest

from fedference.experiments import run_robustness_onset


def test_onset_curves_have_expected_shape():
    r = run_robustness_onset(0, rates=(0.0, 0.6, 0.9), n_seeds=3, n_trials=8)
    assert set(r["by_kind"]) == {"confident_wrong", "byzantine", "drift"}
    for cell in r["by_kind"].values():
        n = len(cell["rates"])
        assert len(cell["naive_curve"]) == n
        assert len(cell["robust_curve"]) == n
        assert len(cell["win_curve"]) == n
        assert len(cell["naive_ci"]) == n
        assert len(cell["robust_ci"]) == n
        assert len(cell["best_robust_method_by_rate"]) == n
        assert all(
            lo <= mean <= hi
            for mean, (lo, hi) in zip(cell["naive_curve"], cell["naive_ci"])
        )
        assert all(
            lo <= mean <= hi
            for mean, (lo, hi) in zip(cell["robust_curve"], cell["robust_ci"])
        )
        # naive accuracy is non-increasing in contamination rate (allow tiny slack)
        nc = cell["naive_curve"]
        assert all(nc[i] >= nc[i + 1] - 0.05 for i in range(n - 1))


def test_additive_attacks_have_an_onset_and_sustain_robustness():
    r = run_robustness_onset(0, rates=(0.0, 0.6, 0.9), n_seeds=3, n_trials=8)
    for kind in ("confident_wrong", "drift"):
        cell = r["by_kind"][kind]
        assert cell["onset_rate"] is not None              # robust does overtake
        # past onset, robust stays above naive at the worst (highest) rate
        assert cell["robust_curve"][-1] >= cell["naive_curve"][-1]


def test_byzantine_onset_is_transient_veto_cliff():
    # byzantine overtakes at some rate but both collapse at the worst rate.
    r = run_robustness_onset(0, rates=(0.0, 0.6, 0.9), n_seeds=3, n_trials=8)
    byz = r["by_kind"]["byzantine"]
    assert byz["naive_curve"][-1] < 0.2   # naive vetoed at the worst rate
    assert byz["robust_curve"][-1] < 0.2  # robust also collapses (no rescue)


def test_onset_validation_raises():
    with pytest.raises(ValueError):
        run_robustness_onset(0, n_agents=2)
    with pytest.raises(ValueError):
        run_robustness_onset(0, n_seeds=1)
    with pytest.raises(ValueError, match="sorted"):
        run_robustness_onset(0, rates=(0.5, 0.1))
