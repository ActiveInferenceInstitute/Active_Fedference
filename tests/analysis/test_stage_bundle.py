"""Real staged-analysis boundary regressions."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from analysis.stage_io import StageError, read_json
from tests.analysis.stage_fixtures import cli, run
from tests.analysis.stage_fixtures import prepared as prepared


def test_figures_require_all_completed_reports_before_any_render(prepared: tuple[Path, Path]) -> None:
    _, workspace = prepared
    result = cli(workspace, "run", "--stage", "figures", "--max-seconds", "10")
    assert result.returncode != 0
    assert not (workspace / "completed/figures").exists()
    assert not (workspace / "source/output").exists()


def test_publication_script_refuses_monolithic_or_canonical_checkpoint_compute(
    prepared: tuple[Path, Path],
) -> None:
    project, workspace = prepared
    script = project / "scripts/02_run_analysis.py"
    common = [sys.executable, "-IB", str(script), "--project-root", str(project), "--profile", "publication"]
    result = subprocess.run(common, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0 and "no monolithic publication rerun" in result.stderr
    result = subprocess.run(
        [*common, "--checkpoints", str(workspace)], capture_output=True, text=True, timeout=10
    )
    assert result.returncode != 0 and "frozen source" in result.stderr
    assert not (project / "output").exists()
    from analysis.workflow import run_analysis_pipeline

    with pytest.raises(ValueError, match="external report checkpoints"):
        run_analysis_pipeline(project_root=project, profile="publication")
    assert not (project / "output").exists()


@pytest.mark.slow
def test_explicit_smoke_report_set_and_figure_bundle(prepared: tuple[Path, Path]) -> None:
    """Small real stage budget, explicitly selected here; never publication scale."""
    from analysis.stage_bundle import import_bundle, validate_bundle
    from analysis.stage_catalog import STAGE_SCHEMAS

    project, workspace = prepared
    for name in STAGE_SCHEMAS:
        print(f"bounded smoke stage: {name}", flush=True)
        # The first retained instrumented run hit the genuine 20s Torch import
        # budget. This explicit smoke-test allowance is not a scientific budget.
        # The retained broad-instrumentation run also exhausted disjoint's
        # 20s allowance; the producer inputs and seeds stay unchanged.
        seconds = "90" if name in {"bnn_torch", "disjoint_fov_world"} else "20"
        # The Torch lane needs an explicit memory allowance for the same reason:
        # sampled peaks for this stage are 1084.4 MB and 1100.7 MB against the
        # generic 1024 MB default, while the same test has passed under that
        # default in shard 2. 1280 MB leaves about 16 % headroom over the observed
        # ceiling and, like the time allowance above, is not a scientific budget.
        rss = "1280" if name == "bnn_torch" else None
        run(workspace, name, seconds, max_rss_mb=rss)
    first = run(workspace, "figures", "120")
    assert first["status"] == "complete"
    bundle = read_json(workspace / "completed/figures/report.json")
    assert bundle["files"] and all(name.startswith("output/") for name in bundle["files"])
    validate_bundle(workspace, bundle)
    assert run(workspace, "figures", "120")["status"] == "reused"
    imported = import_bundle(workspace, project)
    assert imported["profile"] == "smoke" and imported["scientific_acceptance"] is False
    assert not (project / "output/data/pipeline_provenance.json").exists()
    assert (project / "output/figures/figure_registry.json").is_file()
    from analysis.workflow import run_analysis_pipeline

    peer_report = project / "output/reports/belief_sharing.json"
    peer_report.write_bytes(b"retained peer edit\n")
    with pytest.raises(StageError, match="frozen source"):
        run_analysis_pipeline(project_root=project, profile="smoke", checkpoints=workspace)
    assert peer_report.read_bytes() == b"retained peer edit\n"
    name = next(iter(bundle["files"]))
    output = workspace / "source" / name
    output.write_bytes(output.read_bytes() + b"corrupt")
    with pytest.raises(StageError, match="inventory/digest"):
        validate_bundle(workspace, bundle)


def test_library_checkpoint_assembly_refuses_a_destination_checkout(prepared: tuple[Path, Path]) -> None:
    from analysis.workflow import run_analysis_pipeline

    project, workspace = prepared
    with pytest.raises(StageError, match="frozen source"):
        run_analysis_pipeline(project_root=project, profile="smoke", checkpoints=workspace)
    assert not (project / "output").exists()
