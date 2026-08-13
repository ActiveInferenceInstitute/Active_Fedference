"""Smoke tests for the thin-orchestrator scripts.

No mocks: each script is executed as a real subprocess and asserted to exit 0
(or, for the invariants gate, exit 0 == all locked invariants hold). The
scripts own no research algorithm logic. They wire paths, call source
functions, and report status; specialized validators may parse their declared
format. A green exit code is the contract these tests pin.

All ``output/`` writes are redirected into a session-scoped temporary project
scaffold via ``ACTIVE_FEDFERENCE_PROJECT_ROOT`` (see ``src/project_paths.py``).
Before this override existed, these subprocess runs silently overwrote the real
committed ``output/reports/`` (n_seeds 240 -> 4), ``manuscript_variables.json``
and ``output/manuscript/`` with smoke-scale values on every full-suite run —
the tripwire in ``tests/test_report_scale_guard.py`` documents the incident
class. The real tree must never be a subprocess-test write target.

Scripts are still run with cwd at this standalone private repository root so
stale template-only path assumptions fail in the same way they would for users
(the cwd governs import wiring; the env override governs write targets).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.analysis.test_workflow import _make_project

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _PROJECT_ROOT / "scripts"


pytestmark = [pytest.mark.slow, pytest.mark.publication]


@pytest.fixture(scope="session")
def scaffold_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-scoped temporary project scaffold the scripts write into.

    Reuses the minimal-real-project builder from the workflow tests (smoke
    profile, ``manuscript/config.yaml`` present) so every script resolves its
    project root here instead of the real tree. Session scope lets the ordered
    scripts share state: ``02_run_analysis`` writes the reports that
    ``z_generate_manuscript_variables`` consumes.
    """
    root = tmp_path_factory.mktemp("scripts_smoke_scaffold")
    _make_project(root)
    # A real project root has ``src/``. Retaining that shape makes this
    # subprocess regression exercise the receipt branch, rather than passing
    # only because a minimal scaffold happens to omit the directory.
    (root / "src").mkdir()
    return root


def _run_script(name: str, scaffold_root: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    # Redirect every project-root-relative write (output/reports, output/data,
    # output/manuscript, output/docs) into the scaffold. Scripts validate this
    # override and fail loudly on an invalid value — no silent fallback.
    env["ACTIVE_FEDFERENCE_PROJECT_ROOT"] = str(scaffold_root)
    # The scaffold declares its smoke profile in config.yaml. Running the bare
    # command is intentional: it proves a config-selected smoke run cannot be
    # mistaken for publication simply because the CLI has no --profile value.
    args = [sys.executable, str(_SCRIPTS / name)]
    if name == "z_generate_manuscript_variables.py":
        # The session scaffold deliberately has no production analysis receipt.
        # Its hydration remains an explicitly non-release draft path; the final
        # CLI preflight is covered separately by test_validation_receipt.
        args.append("--allow-draft")
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(_PROJECT_ROOT),
        timeout=600,
    )


# Ordered so the report-producing analysis run precedes the manuscript-variable
# script that consumes those reports (the latter degrades to N/A without them,
# but ordering keeps the smoke run representative of a real pipeline pass).
_ORDERED_SCRIPTS = [
    "00_preflight.py",
    "generate_api_docs.py",
    "01_run_invariants.py",
    "02_run_analysis.py",
    "z_generate_manuscript_variables.py",
]

_THIN_ORCHESTRATOR_SCRIPTS = [
    *_ORDERED_SCRIPTS,
    "prepare_web_package.py",
    "record_pipeline_stage.py",
    "validate_all.py",
    "validate_test_coverage.py",
    "validate_clean_checkout.py",
    "validate_pipeline_freshness.py",
    "validate_rendered_surfaces.py",
    "validate_web_package.py",
    "validate_outputs.py",
    "summarize_tokens.py",
    "emit_metadata.py",
    "build_release.py",
    "zenodo_release.py",
    "validate_mermaid.py",
    "_generate_api_docs.py",
]

_PROJECT_ROOT_AWARE_SCRIPTS = [
    "00_preflight.py",
    "01_run_invariants.py",
    "02_run_analysis.py",
    "_generate_api_docs.py",
    "build_release.py",
    "emit_metadata.py",
    "generate_api_docs.py",
    "prepare_web_package.py",
    "record_pipeline_stage.py",
    "summarize_tokens.py",
    "validate_all.py",
    "validate_clean_checkout.py",
    "validate_mermaid.py",
    "validate_outputs.py",
    "validate_pipeline_freshness.py",
    "validate_rendered_surfaces.py",
    "validate_test_coverage.py",
    "validate_web_package.py",
    "z_generate_manuscript_variables.py",
    "zenodo_release.py",
]


@pytest.mark.parametrize("script_name", _ORDERED_SCRIPTS)
def test_script_exits_zero(script_name: str, scaffold_root: Path) -> None:
    result = _run_script(script_name, scaffold_root)
    assert result.returncode == 0, (
        f"{script_name} exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )


def test_scripts_never_write_into_the_real_project_tree(scaffold_root: Path) -> None:
    """The pipeline scripts must have written into the scaffold, not this repo.

    Regression pin for the smoke-clobbering incident class: asserts the
    scaffold actually received the analysis reports (proof the env override
    was honored, not silently ignored).
    """
    assert (scaffold_root / "output" / "reports" / "belief_sharing.json").exists(), (
        "02_run_analysis.py did not write into the scaffold — the "
        "ACTIVE_FEDFERENCE_PROJECT_ROOT override is not being honored"
    )
    execution = json.loads(
        (scaffold_root / "output" / "data" / "analysis_execution.json").read_text(encoding="utf-8")
    )
    assert execution["effective_profile"] == "smoke"
    assert not (scaffold_root / "output" / "data" / "pipeline_provenance.json").exists(), (
        "a config-selected smoke run must never mint an analysis receipt"
    )


@pytest.mark.parametrize("script_name", _PROJECT_ROOT_AWARE_SCRIPTS)
def test_project_root_aware_scripts_expose_explicit_root(script_name: str) -> None:
    """Every checkout-facing entry point advertises the shared root contract."""
    result = subprocess.run(
        [sys.executable, str(_SCRIPTS / script_name), "--help"],
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
        timeout=60,
    )
    assert result.returncode == 0, f"{script_name}: {result.stderr}"
    assert "--project-root" in result.stdout


def test_validate_outputs_does_not_promote_arbitrary_files(tmp_path: Path) -> None:
    """An unrelated output file cannot satisfy the registered artifact set."""
    for name in ("figures", "reports", "data"):
        directory = tmp_path / "output" / name
        directory.mkdir(parents=True)
        (directory / "unrelated.txt").write_text("not a registered artifact\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPTS / "validate_outputs.py"),
            "--project-root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
        timeout=60,
    )
    assert result.returncode in {1, 2}
    assert "validation_result: PASS" not in result.stdout


def test_invariants_script_writes_passing_report(scaffold_root: Path) -> None:
    """The invariants gate exits 0 only when every locked-core invariant holds."""
    result = _run_script("01_run_invariants.py", scaffold_root)
    assert result.returncode == 0, result.stderr
    report = result.stdout.strip().splitlines()[-1]
    report_path = Path(report)
    assert report_path.name == "invariants.json"
    assert report_path.exists()
    # The override must have redirected the write into the scaffold.
    assert report_path.resolve().is_relative_to(scaffold_root.resolve())


def test_each_script_is_logic_free() -> None:
    """Static guard: scripts import their work, they do not implement it.

    A thin orchestrator never reaches for numerics — flag any numpy/scipy/math
    import or array algebra that would mean algorithm logic crept into a script.
    """
    banned_imports = ("import numpy", "import scipy", "import math", "from numpy", "from scipy")
    for name in _THIN_ORCHESTRATOR_SCRIPTS:
        text = (_SCRIPTS / name).read_text(encoding="utf-8")
        for token in banned_imports:
            assert token not in text, f"{name} contains numeric import '{token}'"


def test_scripts_do_not_prepend_repository_ancestors_to_import_path() -> None:
    banned_path_patterns = ("parents[2]", "parents[3]", "parents[4]", "for _ancestor in")
    for path in sorted(_SCRIPTS.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for pattern in banned_path_patterns:
            assert pattern not in text, f"{path.name} prepends broad ancestor path via {pattern}"


def test_manual_stage_recorder_rejects_analysis_promotion(tmp_path: Path) -> None:
    """Only the source-bound analysis producer can mint its receipt."""
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPTS / "record_pipeline_stage.py"),
            "analysis",
            "--project-root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
        timeout=60,
    )
    assert result.returncode != 0
    assert "invalid choice" in result.stderr


def test_validate_all_dry_run_prints_profile_commands() -> None:
    result = subprocess.run(
        [sys.executable, str(_SCRIPTS / "validate_all.py"), "package", "--dry-run"],
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "scripts/prepare_web_package.py" in result.stdout
    assert "scripts/validate_web_package.py" in result.stdout


def test_validate_all_full_declares_source_and_release_gates() -> None:
    result = subprocess.run(
        [sys.executable, str(_SCRIPTS / "validate_all.py"), "full", "--dry-run"],
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "scripts/validate_all.py source" in result.stdout
    assert "scripts/validate_all.py rendered" in result.stdout
    assert "scripts/validate_all.py freshness" in result.stdout
    source = subprocess.run(
        [sys.executable, str(_SCRIPTS / "validate_all.py"), "source", "--dry-run"],
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
        timeout=60,
    )
    assert source.returncode == 0, source.stderr
    assert "ruff check src/ tests/ scripts/" in source.stdout
    assert "mypy src/" in source.stdout
    assert "scripts/build_release.py --verify" in source.stdout
    assert "grep" in source.stdout
    assert "import infrastructure" in source.stdout
    assert "src/fedference/" in source.stdout
