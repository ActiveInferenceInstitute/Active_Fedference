"""Tests for the contamination-gallery figure generator (no mocks).

A real gallery dict (and one produced by
:func:`fedference.experiments.run_contamination_gallery`) is rendered to
``tmp_path``; we assert the PNG exists and the error path raises.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fedference.experiments import run_contamination_gallery
from figures import generate_contamination_gallery

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_gallery_figure_happy_path(tmp_path: Path) -> None:
    by_kind = {
        "confident_wrong": {"naive_mean": 0.984, "robust_mean": 0.988,
                            "naive_ci": [0.982, 0.986], "robust_ci": [0.987, 0.989],
                            "best_robust_method": "AR", "mean_diff": 0.004,
                            "diff_ci": [0.003, 0.005], "win_fraction": 1.0,
                            "reliably_beats": True, "directional": True},
        "byzantine": {"naive_mean": 0.641, "robust_mean": 0.669,
                     "naive_ci": [0.62, 0.66], "robust_ci": [0.64, 0.70],
                     "best_robust_method": "beta", "mean_diff": 0.028,
                     "diff_ci": [-0.017, 0.077], "win_fraction": 0.62,
                     "reliably_beats": False, "directional": True},
        "uniform": {"naive_mean": 0.999, "robust_mean": 0.995,
                   "naive_ci": [0.998, 1.0], "robust_ci": [0.994, 0.996],
                   "best_robust_method": "AR", "mean_diff": -0.004,
                   "diff_ci": [-0.004, -0.004], "win_fraction": 0.0,
                   "reliably_beats": False, "directional": False},
    }
    path = generate_contamination_gallery(by_kind, project_root=tmp_path)
    assert path.exists()
    assert path.read_bytes()[:8] == _PNG_MAGIC


def test_gallery_figure_from_real_report(tmp_path: Path) -> None:
    report = run_contamination_gallery(0, n_seeds=6, n_trials=12)
    path = generate_contamination_gallery(report["by_kind"], project_root=tmp_path)
    assert path.exists()


def test_gallery_figure_rejects_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        generate_contamination_gallery({}, project_root=tmp_path)
