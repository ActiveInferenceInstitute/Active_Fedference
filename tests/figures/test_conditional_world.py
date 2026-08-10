"""Figure-generator tests for the conditional and belief-quality diagnostics."""

from __future__ import annotations

from pathlib import Path

from fedference.experiments.conditional_world import (
    run_belief_quality_sensitivity,
    run_conditional_world_generalization,
)
from figures.belief_quality import generate_belief_quality
from figures.conditional_world import generate_conditional_world


def test_generate_conditional_world_writes_pair(tmp_path: Path) -> None:
    report = run_conditional_world_generalization(seed=3, n_seeds=3, n_trials=2)
    output = generate_conditional_world(report, project_root=tmp_path)
    assert output.exists()
    assert output.with_suffix(".pdf").exists()


def test_generate_belief_quality_writes_pair(tmp_path: Path) -> None:
    report = run_belief_quality_sensitivity(seed=3, n_seeds=3, n_trials=2)
    output = generate_belief_quality(report, project_root=tmp_path)
    assert output.exists()
    assert output.with_suffix(".pdf").exists()
