"""Tests for figures.cross_study_summary.generate_cross_study_summary."""

from __future__ import annotations

from pathlib import Path

import pytest

from figures.cross_study_summary import generate_cross_study_summary

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _report() -> dict:
    """Return a small valid report so renderer tests do not rerun experiments."""
    return {
        "n_seeds": 2,
        "studies": [
            {
                "study": i,
                "label": f"Study {i}",
                "metric": "synthetic renderer input",
                "values": [float(i), float(i) + 0.1],
                "mean": float(i) + 0.05,
                "std": 0.1,
                "ci_lo": float(i),
                "ci_hi": float(i) + 0.1,
                "unit": "fraction" if i in {1, 4, 5, 6, 7, 8} else "nats" if i in {2, 3} else "R-sq",
            }
            for i in range(1, 10)
        ],
    }


def test_cross_study_summary_happy_path(tmp_path: Path) -> None:
    path = generate_cross_study_summary(_report(), project_root=tmp_path, n_seeds=2)
    assert path.exists(), "PNG file was not created"
    assert path.read_bytes()[:8] == _PNG_MAGIC, "file is not a valid PNG"


def test_cross_study_summary_custom_filename(tmp_path: Path) -> None:
    path = generate_cross_study_summary(
        _report(), project_root=tmp_path, n_seeds=2, filename="overview.png"
    )
    assert path.name == "overview.png"
    assert path.read_bytes()[:8] == _PNG_MAGIC


def test_cross_study_summary_deterministic(tmp_path: Path) -> None:
    p1 = generate_cross_study_summary(_report(), project_root=tmp_path, seed=0, n_seeds=2, filename="cs1.png")
    p2 = generate_cross_study_summary(_report(), project_root=tmp_path, seed=0, n_seeds=2, filename="cs2.png")
    assert p1.read_bytes() == p2.read_bytes(), "same seed should produce identical PNG"


def test_cross_study_summary_rejects_unknown_unit(tmp_path: Path) -> None:
    report = _report()
    report["studies"][0]["unit"] = "mixed"
    with pytest.raises(ValueError, match="known native unit"):
        generate_cross_study_summary(report, project_root=tmp_path)


def test_cross_study_summary_rejects_missing_native_facet(tmp_path: Path) -> None:
    report = _report()
    report["studies"] = [study for study in report["studies"] if study["unit"] != "R-sq"]
    with pytest.raises(ValueError, match="missing native-unit facet"):
        generate_cross_study_summary(report, project_root=tmp_path)
