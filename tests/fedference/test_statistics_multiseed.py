"""Tests for fedference.statistics.multiseed_summary."""

from __future__ import annotations

import numpy as np
import pytest

from fedference.statistics import multiseed_summary


def test_multiseed_summary_happy_path() -> None:
    values = [0.6, 0.65, 0.7, 0.75, 0.8]
    result = multiseed_summary(values)
    assert result["n"] == 5
    assert abs(result["mean"] - np.mean(values)) < 1e-9
    assert abs(result["std"] - np.std(values, ddof=1)) < 1e-9
    assert result["median"] == pytest.approx(0.7)
    assert result["mcse"] == pytest.approx(np.std(values, ddof=1) / np.sqrt(5))
    assert result["mde"] > 0.0
    assert result["min"] == pytest.approx(0.6)
    assert result["max"] == pytest.approx(0.8)
    assert result["ci_lo"] <= result["mean"] <= result["ci_hi"]


def test_multiseed_summary_constant() -> None:
    values = [0.5] * 8
    result = multiseed_summary(values, n_boot=500)
    assert result["std"] == pytest.approx(0.0, abs=1e-9)
    assert result["ci_lo"] == pytest.approx(result["mean"], abs=1e-9)
    assert result["ci_hi"] == pytest.approx(result["mean"], abs=1e-9)


def test_multiseed_summary_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        multiseed_summary([])


def test_multiseed_summary_is_deterministic() -> None:
    values = [0.1, 0.3, 0.5, 0.7, 0.9]
    r1 = multiseed_summary(values, rng_seed=7, n_boot=1000)
    r2 = multiseed_summary(values, rng_seed=7, n_boot=1000)
    assert r1["ci_lo"] == r2["ci_lo"]
    assert r1["ci_hi"] == r2["ci_hi"]


def test_multiseed_summary_different_seeds_differ() -> None:
    values = [0.1, 0.3, 0.5, 0.7, 0.9]
    r1 = multiseed_summary(values, rng_seed=0, n_boot=200)
    r2 = multiseed_summary(values, rng_seed=99, n_boot=200)
    # CIs from different seeds should not be identical (would be astronomically unlikely).
    assert (r1["ci_lo"] != r2["ci_lo"]) or (r1["ci_hi"] != r2["ci_hi"])


def test_multiseed_summary_single_value() -> None:
    result = multiseed_summary([0.42], n_boot=100)
    assert result["n"] == 1
    assert result["mean"] == pytest.approx(0.42)
    assert result["std"] == pytest.approx(0.0)
