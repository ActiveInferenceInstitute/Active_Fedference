"""Tests for figures.sensitivity_heatmap.generate_sensitivity_heatmap."""

from __future__ import annotations

from pathlib import Path

from figures.sensitivity_heatmap import generate_sensitivity_heatmap

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_sensitivity_heatmap_happy_path(tmp_path: Path) -> None:
    path = generate_sensitivity_heatmap(
        tmp_path,
        acuity_values=(0.5, 0.8),
        n_agents_values=(2, 4),
        n_trials=3,
    )
    assert path.exists(), "PNG file was not created"
    assert path.read_bytes()[:8] == _PNG_MAGIC, "file is not a valid PNG"


def test_sensitivity_heatmap_custom_filename(tmp_path: Path) -> None:
    path = generate_sensitivity_heatmap(
        tmp_path,
        acuity_values=(0.5, 0.8),
        n_agents_values=(2, 4),
        n_trials=2,
        filename="my_heatmap.png",
    )
    assert path.name == "my_heatmap.png"
    assert path.read_bytes()[:8] == _PNG_MAGIC


def test_sensitivity_heatmap_deterministic(tmp_path: Path) -> None:
    p1 = generate_sensitivity_heatmap(
        tmp_path,
        seed=0,
        acuity_values=(0.5, 0.8),
        n_agents_values=(2, 4),
        n_trials=2,
        filename="det1.png",
    )
    p2 = generate_sensitivity_heatmap(
        tmp_path,
        seed=0,
        acuity_values=(0.5, 0.8),
        n_agents_values=(2, 4),
        n_trials=2,
        filename="det2.png",
    )
    assert p1.read_bytes() == p2.read_bytes(), "same seed should produce identical PNG"
