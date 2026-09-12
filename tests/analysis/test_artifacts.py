"""Tests for the canonical Stage-02 artifact declaration."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from analysis.artifacts import (
    ANALYSIS_DATA_FILENAMES,
    ANALYSIS_FIGURE_FILENAMES,
    ANALYSIS_FIGURE_PDF_FILENAMES,
    ANALYSIS_FIGURE_SUPPORT_FILENAMES,
    ANALYSIS_REPORT_FILENAMES,
    expected_artifacts,
)


def test_expected_artifacts_are_root_relative_and_complete() -> None:
    root = Path("/tmp/active-fedference-fixture").resolve()
    artifacts = expected_artifacts(root)

    assert len(artifacts) == (
        len(ANALYSIS_REPORT_FILENAMES)
        + len(ANALYSIS_FIGURE_FILENAMES)
        + len(ANALYSIS_FIGURE_PDF_FILENAMES)
        + len(ANALYSIS_FIGURE_SUPPORT_FILENAMES)
        + len(ANALYSIS_DATA_FILENAMES)
        + 1
    )
    assert artifacts["report:belief_sharing.json"] == root / "output/reports/belief_sharing.json"
    assert artifacts["figure:graphical_abstract.png"] == root / "output/figures/graphical_abstract.png"
    assert artifacts["figure:system_overview.png"] == root / "output/figures/system_overview.png"
    assert artifacts["figure-vector:graphical_abstract.pdf"] == (
        root / "output/figures/graphical_abstract.pdf"
    )
    assert artifacts["figure_registry"] == root / "output/figures/figure_registry.json"
    assert artifacts["data:analysis_execution.json"] == root / "output/data/analysis_execution.json"


def test_figure_pair_inventory_is_sorted_unique_and_stem_aligned() -> None:
    assert list(ANALYSIS_FIGURE_FILENAMES) == sorted(set(ANALYSIS_FIGURE_FILENAMES))
    assert list(ANALYSIS_FIGURE_PDF_FILENAMES) == sorted(set(ANALYSIS_FIGURE_PDF_FILENAMES))
    assert [Path(name).stem for name in ANALYSIS_FIGURE_FILENAMES] == [
        Path(name).stem for name in ANALYSIS_FIGURE_PDF_FILENAMES
    ]


def test_expected_artifacts_do_not_depend_on_present_directory_contents(tmp_path: Path) -> None:
    (tmp_path / "output" / "reports").mkdir(parents=True)
    (tmp_path / "output" / "reports" / "unrelated.json").write_text("{}\n", encoding="utf-8")

    artifacts = expected_artifacts(tmp_path)

    assert "report:unrelated.json" not in artifacts
    assert "report:belief_sharing.json" in artifacts


def test_freshness_contract_can_import_before_workflow_barrel() -> None:
    """The small artifact contract must not recreate the workflow import cycle."""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from publication.pipeline_freshness import ANALYSIS_EXECUTION_PATH; "
                "from analysis import run_analysis_pipeline; "
                "assert ANALYSIS_EXECUTION_PATH.name == 'analysis_execution.json'; "
                "assert callable(run_analysis_pipeline)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_presentation_manifest_inventory_matches_every_authored_selector() -> None:
    import re

    from analysis.artifacts import ANALYSIS_FIGURE_SUPPORT_FILENAMES, PRESENTATION_FIGURE_STEMS

    root = Path(__file__).resolve().parents[2]
    selected = []
    for source in sorted((root / "manuscript").glob("*.md")):
        selected.extend(
            re.findall(r'data-slide-manifest="../output/figures/([a-z_]+)\.slides\.json"', source.read_text())
        )
    assert len(selected) == len(set(selected))
    assert set(selected) == set(PRESENTATION_FIGURE_STEMS)
    assert {name for name in ANALYSIS_FIGURE_SUPPORT_FILENAMES if name.endswith(".slides.json")} == {
        f"{stem}.slides.json" for stem in selected
    }
