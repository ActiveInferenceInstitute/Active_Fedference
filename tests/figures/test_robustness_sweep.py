"""Tests for the robustness-sweep generator (``figures.robustness_sweep``).

No mocks: the generator renders a real PNG to ``tmp_path`` (headless Agg) and
we assert the file exists, is a PNG, and that the documented error paths raise.

Split out of the former flat ``tests/test_figures.py`` to mirror
``src/figures/robustness_sweep.py`` under the three-tree discipline. Logic
unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from figures import generate_robustness_sweep

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_robustness_sweep_happy_path_with_threshold(tmp_path: Path) -> None:
    rates = [0.0, 0.5, 0.9]
    accuracy = {
        "KLD": {"0": 0.9, "0.5": 0.6, "0.9": 0.3},
        "RKL": {"0": 0.9, "0.5": 0.8, "0.9": 0.7},
    }
    path = generate_robustness_sweep(
        accuracy, rates, accuracy_threshold=0.5, project_root=tmp_path
    )
    assert path.exists()
    assert path.read_bytes()[:8] == _PNG_MAGIC


def test_robustness_sweep_happy_path_without_threshold(tmp_path: Path) -> None:
    rates = [0.0, 0.9]
    accuracy = {"KLD": {"0": 0.9, "0.9": 0.2}}
    path = generate_robustness_sweep(accuracy, rates, project_root=tmp_path)
    assert path.exists()


def test_robustness_sweep_profile_renders_trial_intervals(tmp_path: Path) -> None:
    rates = [0.0, 0.9]
    accuracy = {"KLD": {"0": 0.9, "0.9": 0.2}}
    summary = {
        key: {
            "n": 12,
            "methods": {
                "KLD": {
                    "mean": mean,
                    "ci_lo": mean - 0.02,
                    "ci_hi": mean + 0.02,
                }
            },
        }
        for key, mean in (("0", 0.88), ("0.9", 0.24))
    }
    path = generate_robustness_sweep(
        accuracy, rates, rate_summary=summary, project_root=tmp_path
    )
    assert path.exists()
    assert path.read_bytes()[:8] == _PNG_MAGIC


def test_robustness_sweep_rejects_empty_accuracy(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        generate_robustness_sweep({}, [0.0, 0.9], project_root=tmp_path)


def test_robustness_sweep_rejects_empty_rates(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        generate_robustness_sweep({"KLD": {}}, [], project_root=tmp_path)
