"""Figure-generator tests for the conditional and belief-quality diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import figures.conditional_world as conditional_world_module
from fedference.experiments.conditional_world import (
    run_belief_quality_sensitivity,
    run_conditional_world_generalization,
)
from figures._common import semantic_style
from figures.belief_quality import generate_belief_quality
from figures.conditional_world import generate_conditional_world


def test_generate_conditional_world_writes_pair(tmp_path: Path) -> None:
    report = run_conditional_world_generalization(seed=3, n_seeds=3, n_trials=2)
    output = generate_conditional_world(report, project_root=tmp_path)
    assert output.exists()
    assert output.with_suffix(".pdf").exists()


def test_conditional_world_uses_truthful_range_label_and_shared_zero_rule(
    tmp_path: Path,
    monkeypatch,
) -> None:
    report = run_conditional_world_generalization(seed=3, n_seeds=3, n_trials=2)
    captured: dict[str, object] = {}

    def capture_figure(fig, path: Path) -> Path:
        captured["figure"] = fig
        return path

    monkeypatch.setattr(conditional_world_module, "save_figure", capture_figure)
    generate_conditional_world(report, project_root=tmp_path)
    fig = captured["figure"]
    assert isinstance(fig, plt.Figure)
    try:
        summary_ax = fig.axes[1]
        range_container = next(
            container
            for container in summary_ax.containers
            if container.get_label() == "mean with asymmetric capped min–max range"
        )
        assert "±" not in range_container.get_label()

        zero_rule = next(
            line for line in summary_ax.lines if line.get_label() == "zero: no method contrast"
        )
        expected = semantic_style("reference_rule")
        assert np.allclose(zero_rule.get_xdata(), [0.0, 0.0])
        assert zero_rule.get_color() == expected.color
        assert zero_rule.get_linestyle() == expected.dash
        assert zero_rule.get_linewidth() == expected.linewidth
    finally:
        plt.close(fig)


def test_conditional_world_caption_declares_non_colour_and_range_semantics() -> None:
    caption = (
        Path(__file__).resolve().parents[2] / "manuscript/28_supplement_extended_methods.md"
    ).read_text(encoding="utf-8")

    assert "every heatmap cell prints its signed seed-level mean" in caption
    assert "asymmetric capped whiskers extend to the observed cell minimum and maximum" in caption
    assert "capped min/max spans are finite-grid ranges, not confidence intervals" in caption
    assert "not symmetric mean-plus-or-minus errors" in caption
    assert "dark-neutral dotted zero rule marks no method contrast" in caption


def test_generate_belief_quality_writes_pair(tmp_path: Path) -> None:
    report = run_belief_quality_sensitivity(seed=3, n_seeds=3, n_trials=2)
    output = generate_belief_quality(report, project_root=tmp_path)
    assert output.exists()
    assert output.with_suffix(".pdf").exists()
