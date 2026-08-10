"""Tests for figures.moving_world.generate_moving_world."""

from __future__ import annotations

from pathlib import Path

from figures.moving_world import generate_moving_world

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_PDF_MAGIC = b"%PDF"

_RESULTS_FIXTURE = {
    "accuracy": {
        "isolated": 0.50,
        "communicating": 0.75,
        "efe_guided": 0.85,
    },
    "free_energy_gap": {
        "isolated": 0.0,
        "communicating": 0.30,
        "efe_guided": 0.45,
    },
    "n_steps_to_consensus": {
        "isolated": 8.0,
        "communicating": 5.0,
        "efe_guided": 3.0,
    },
}


def test_moving_world_happy_path(tmp_path: Path) -> None:
    path = generate_moving_world(_RESULTS_FIXTURE, tmp_path)
    assert path.exists(), "output file was not created"
    assert path.name == "moving_world.png"
    assert path.read_bytes()[:8] == _PNG_MAGIC, "file is not a valid PNG"
    assert (tmp_path / "output" / "figures" / "moving_world.pdf").read_bytes()[:4] == _PDF_MAGIC


def test_moving_world_custom_filename(tmp_path: Path) -> None:
    path = generate_moving_world(_RESULTS_FIXTURE, tmp_path, filename="mw_test.pdf")
    assert path.name == "mw_test.pdf"
    assert path.read_bytes()[:4] == _PDF_MAGIC
    assert (tmp_path / "output" / "figures" / "mw_test.png").read_bytes()[:8] == _PNG_MAGIC


def test_moving_world_zero_values(tmp_path: Path) -> None:
    zero_results = {
        "accuracy": {"isolated": 0.0, "communicating": 0.0, "efe_guided": 0.0},
        "free_energy_gap": {"isolated": 0.0, "communicating": 0.0, "efe_guided": 0.0},
        "n_steps_to_consensus": {"isolated": 0.0, "communicating": 0.0, "efe_guided": 0.0},
    }
    path = generate_moving_world(zero_results, tmp_path, filename="zero_mw.pdf")
    assert path.exists()
    assert path.read_bytes()[:4] == _PDF_MAGIC
