"""Tests for the canonical Stage-02 artifact declaration."""

from __future__ import annotations

from pathlib import Path

from analysis.artifacts import (
    ANALYSIS_DATA_FILENAMES,
    ANALYSIS_FIGURE_FILENAMES,
    ANALYSIS_REPORT_FILENAMES,
    expected_artifacts,
)


def test_expected_artifacts_are_root_relative_and_complete() -> None:
    root = Path("/tmp/active-fedference-fixture").resolve()
    artifacts = expected_artifacts(root)

    assert len(artifacts) == (
        len(ANALYSIS_REPORT_FILENAMES)
        + len(ANALYSIS_FIGURE_FILENAMES)
        + len(ANALYSIS_DATA_FILENAMES)
        + 1
    )
    assert artifacts["report:belief_sharing.json"] == root / "output/reports/belief_sharing.json"
    assert artifacts["figure:graphical_abstract.png"] == root / "output/figures/graphical_abstract.png"
    assert artifacts["figure_registry"] == root / "output/figures/figure_registry.json"
    assert artifacts["data:analysis_execution.json"] == root / "output/data/analysis_execution.json"


def test_expected_artifacts_do_not_depend_on_present_directory_contents(tmp_path: Path) -> None:
    (tmp_path / "output" / "reports").mkdir(parents=True)
    (tmp_path / "output" / "reports" / "unrelated.json").write_text("{}\n", encoding="utf-8")

    artifacts = expected_artifacts(tmp_path)

    assert "report:unrelated.json" not in artifacts
    assert "report:belief_sharing.json" in artifacts
