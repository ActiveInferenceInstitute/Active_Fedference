"""Contract and rendering tests for the source-bound review-grid figure."""

from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import pytest

from analysis.report_schemas import ReportSchemaError
from fedference.experiments import run_review_grid
from figures import generate_robustness_review_grid

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_review_grid_figure_import_does_not_cycle_through_analysis_workflow() -> None:
    """Public figure import must validate schemas without eagerly importing workflow."""
    project_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "-c", "import figures.robustness_review_grid"],
        cwd=project_root,
        capture_output=True,
        text=True,
        # The full suite may already hold the workstation's Metal/Matplotlib
        # resources; retain a bounded import check without making that
        # unrelated contention look like an import-cycle failure.
        timeout=90,
    )
    assert completed.returncode == 0, completed.stderr


def test_review_grid_figure_cross_checks_a_real_report(tmp_path: Path) -> None:
    report = run_review_grid(
        seed=5,
        n_seeds=2,
        n_trials=2,
        n_agents=3,
        rates=(0.0, 0.5),
        divergences=("KLD", "RKL"),
        target_max_mcse=1.0,
    )
    path = generate_robustness_review_grid(report, project_root=tmp_path)
    assert path.exists()
    assert path.read_bytes()[:8] == _PNG_MAGIC


def test_review_grid_figure_rejects_missing_source_surfaces(tmp_path: Path) -> None:
    with pytest.raises(ReportSchemaError, match="missing required field"):
        generate_robustness_review_grid({}, project_root=tmp_path)
    report = run_review_grid(
        seed=6,
        n_seeds=2,
        n_trials=2,
        n_agents=3,
        rates=(0.0, 0.5),
        divergences=("KLD", "RKL"),
        target_max_mcse=1.0,
    )
    no_cells = copy.deepcopy(report)
    no_cells["conditional_world"]["by_scenario"] = {}
    with pytest.raises(ReportSchemaError, match="has no cells"):
        generate_robustness_review_grid(no_cells, project_root=tmp_path)
    no_profiles = copy.deepcopy(report)
    no_profiles["rate_profiles"]["by_kind"] = {}
    with pytest.raises(ReportSchemaError, match="rate profile mechanisms"):
        generate_robustness_review_grid(no_profiles, project_root=tmp_path)


def test_review_grid_figure_rejects_incomplete_ci_and_ignores_legacy_selection(
    tmp_path: Path,
) -> None:
    report = run_review_grid(
        seed=7,
        n_seeds=2,
        n_trials=2,
        n_agents=3,
        rates=(0.0, 0.5),
        divergences=("KLD", "RKL"),
        target_max_mcse=1.0,
    )
    incomplete_ci = copy.deepcopy(report)
    first_kind = next(iter(incomplete_ci["rate_profiles"]["by_kind"]))
    first_rate = next(iter(incomplete_ci["statistics"]["by_mechanism"][first_kind]["by_rate"]))
    incomplete_ci["statistics"]["by_mechanism"][first_kind]["by_rate"][first_rate]["methods"]["RKL"].pop(
        "contrast_ci"
    )
    with pytest.raises(ReportSchemaError, match="contrast_ci"):
        generate_robustness_review_grid(incomplete_ci, project_root=tmp_path, filename="no_ci.png")

    first = generate_robustness_review_grid(report, project_root=tmp_path, filename="selection_a.png")
    changed_selection = copy.deepcopy(report)
    changed_selection["rate_profiles"]["by_kind"][first_kind]["best_robust_method_by_rate"] = ["RKL", "RKL"]
    second = generate_robustness_review_grid(
        changed_selection, project_root=tmp_path, filename="selection_b.png"
    )
    assert first.read_bytes() == second.read_bytes()
