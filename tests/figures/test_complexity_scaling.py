"""Tests for the complexity-scaling figure generator."""

from __future__ import annotations

from pathlib import Path

import pytest

from fedference.complexity import ComplexityBenchmarkConfig
from fedference.experiments.complexity import run_complexity_scaling
from figures.complexity_scaling import generate_complexity_scaling


def _report():
    config = ComplexityBenchmarkConfig(
        agent_sizes=(2, 4, 8),
        state_sizes=(4, 8, 16),
        sharing_agent_sizes=(2, 4, 8),
        modality_sizes=(1, 2, 4),
        fixed_agent_count=4,
        fixed_state_count=8,
        inference_state_count=8,
        observation_count=3,
        repeats=1,
        warmups=0,
        max_iter=2,
        seed=21,
    )
    return run_complexity_scaling(config)


def test_generate_complexity_scaling_writes_png_and_pdf(tmp_path: Path) -> None:
    out = generate_complexity_scaling(_report(), project_root=tmp_path)
    assert out.name == "complexity_scaling.png"
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert out.with_suffix(".pdf").exists()


def test_generate_complexity_scaling_rejects_missing_measurement(tmp_path: Path) -> None:
    report = _report()
    report["measurements"] = [
        row
        for row in report["measurements"]
        if not (row["method"] == "infer_states" and row["axis"] == "modalities")
    ]
    with pytest.raises(ValueError, match="missing measurement"):
        generate_complexity_scaling(report, project_root=tmp_path)
