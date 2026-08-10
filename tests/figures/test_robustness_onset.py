"""Tests for the robustness-onset figure generator (no mocks)."""

from __future__ import annotations

from pathlib import Path

import pytest

from fedference.experiments import run_robustness_onset
from figures import generate_robustness_onset

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_onset_figure_happy_path(tmp_path: Path) -> None:
    by_kind = {
        "confident_wrong": {"rates": [0.0, 0.5, 0.9], "naive_curve": [1.0, 0.9, 0.7],
                            "robust_curve": [1.0, 0.92, 0.83], "win_curve": [0, 1, 1],
                            "naive_ci": [[0.99, 1.0], [0.88, 0.92], [0.68, 0.72]],
                            "robust_ci": [[0.99, 1.0], [0.90, 0.94], [0.81, 0.85]],
                            "best_robust_method_by_rate": ["AR", "AR", "AR"],
                            "onset_rate": 0.5},
        "byzantine": {"rates": [0.0, 0.5, 0.9], "naive_curve": [1.0, 0.6, 0.02],
                     "robust_curve": [1.0, 0.7, 0.0], "win_curve": [0, 1, 0],
                     "naive_ci": [[0.99, 1.0], [0.55, 0.65], [0.01, 0.03]],
                     "robust_ci": [[0.99, 1.0], [0.65, 0.75], [0.0, 0.01]],
                     "best_robust_method_by_rate": ["AR", "beta", "beta"],
                     "onset_rate": 0.5},
    }
    path = generate_robustness_onset(by_kind, project_root=tmp_path)
    assert path.exists()
    assert path.read_bytes()[:8] == _PNG_MAGIC


def test_onset_figure_from_real_report(tmp_path: Path) -> None:
    report = run_robustness_onset(0, n_seeds=3, n_trials=8)
    path = generate_robustness_onset(report["by_kind"], project_root=tmp_path)
    assert path.exists()


def test_onset_figure_rejects_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        generate_robustness_onset({}, project_root=tmp_path)
