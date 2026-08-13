"""Checks for a genuinely clean, clone-correct project checkout."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REQUIRED_TRACKED_PATHS: tuple[str, ...] = (
    "AGENTS.md",
    "ISA.md",
    "README.md",
    "TODO.md",
    "LICENSE",
    "MANIFEST.in",
    "_fedference_build_backend.py",
    "pyproject.toml",
    "uv.lock",
    ".github/workflows/ci.yml",
    "src/analysis/report_schemas.py",
    "src/analysis/workflow.py",
    "src/experiment_config.py",
    "src/fedference/_validation.py",
    "src/fedference/aggregation.py",
    "src/fedference/aggregation_comparators.py",
    "src/fedference/bnn_fedgvi.py",
    "src/fedference/bnn_variational_torch.py",
    "src/fedference/benchmark.py",
    "src/fedference/calibration.py",
    "src/fedference/complexity.py",
    "src/fedference/data/README.md",
    "src/fedference/data/synthetic_tabular.csv",
    "src/fedference/evidence.py",
    "src/fedference/experiments/complexity.py",
    "src/fedference/experiments/conditional_world.py",
    "src/fedference/experiments/review_grid.py",
    "src/fedference/external_data.py",
    "src/fedference/hybrid.py",
    "src/fedference/hybrid_tracking.py",
    "src/fedference/hierarchy_tasks.py",
    "src/fedference/protocol_parity.py",
    "src/fedference/research_registry.py",
    "src/fedference/scoring.py",
    "src/fedference/single_machine.py",
    "src/fedference/server_theory.py",
    "src/fedference/torch_bnn.py",
    "src/fedference_cli/__init__.py",
    "src/fedference_cli/__main__.py",
    "src/project_paths.py",
    "src/publication/clean_checkout.py",
    "src/publication/identifiers.py",
    "src/publication/metadata.py",
    "src/publication/zenodo.py",
    "src/publication/pipeline_freshness.py",
    "src/publication/release_manifest.py",
    "src/publication/surface_validation.py",
    "src/publication/validation_receipt.py",
    "src/publication/web_package.py",
    "src/figures/_metadata.py",
    "src/figures/complexity_scaling.py",
    "src/figures/conditional_world.py",
    "src/figures/belief_quality.py",
    "src/figures/robustness_review_grid.py",
    "src/figures/generative_model_schema.py",
    "src/figures/message_passing.py",
    "src/figures/pomdp_loop.py",
    "output/data/pipeline_provenance.json",
    "output/data/analysis_execution.json",
    "output/data/test_coverage_receipt.json",
    "output/figures/conditional_world.png",
    "output/figures/conditional_world.pdf",
    "output/figures/belief_quality.png",
    "output/figures/belief_quality.pdf",
    "output/figures/robustness_review_grid.png",
    "output/figures/robustness_review_grid.pdf",
    "output/reports/conditional_world.json",
    "output/reports/robustness_review_grid.json",
    "output/reports/belief_quality.json",
    "scripts/record_pipeline_stage.py",
    "scripts/emit_metadata.py",
    "scripts/zenodo_release.py",
    "scripts/validate_clean_checkout.py",
    "scripts/validate_pipeline_freshness.py",
    "scripts/validate_rendered_surfaces.py",
    "scripts/validate_test_coverage.py",
    "scripts/validate_web_package.py",
    "tests/test_clean_checkout.py",
    "tests/test_build_backend.py",
    "tests/test_release_preflight.py",
    "tests/test_publication_metadata.py",
    "tests/test_publication_identifiers.py",
    "tests/test_zenodo.py",
    "tests/test_pipeline_freshness.py",
    "tests/test_validation_receipt.py",
    "tests/fedference/test_complexity.py",
    "tests/fedference/test_aggregation_comparators.py",
    "tests/fedference/test_aggregation_config.py",
    "tests/fedference/test_bnn_fedgvi.py",
    "tests/fedference/test_bnn_variational_torch.py",
    "tests/fedference/test_benchmark.py",
    "tests/fedference/test_calibration.py",
    "tests/fedference/test_evidence.py",
    "tests/fedference/test_experiments_complexity.py",
    "tests/fedference/test_conditional_world.py",
    "tests/fedference/test_review_hardening.py",
    "tests/fedference/test_external_data.py",
    "tests/fedference/test_hybrid.py",
    "tests/fedference/test_hybrid_tracking.py",
    "tests/fedference/test_hierarchy_tasks.py",
    "tests/fedference/test_protocol_parity.py",
    "tests/fedference/test_research_registry.py",
    "tests/fedference/test_single_machine.py",
    "tests/fedference/test_scoring.py",
    "tests/fedference/test_server_theory.py",
    "tests/fedference/test_torch_bnn.py",
    "tests/fedference/test_transport_envelope.py",
    "tests/test_fedference_cli.py",
    "tests/analysis/test_workflow.py",
    "tests/figures/test_complexity_scaling.py",
    "tests/figures/test_conditional_world.py",
    "tests/figures/test_robustness_review_grid.py",
    "tests/test_surface_validation.py",
    "tests/test_web_publication_contract.py",
    "tests/test_report_scale_guard.py",
    "docs/research/manuscript-claim-audit.md",
    "docs/research/visual-claim-audit.md",
    "docs/manuscript/accessibility.md",
    "docs/reference/api-stability.md",
    "docs/reference/zenodo-release.md",
    "Active_Fedference_Research_Manuscript_v1.0.1_Zenodo_10.5281-zenodo.21919307.pdf",
    "manuscript/30_supplement_notation.md",
    "docs/security/README.md",
    "docs/security/active_fedference-threat-model.md",
    "docs/todo/adaptive-robustness-calibration.md",
    "docs/todo/release-and-verification-ladder.md",
)


@dataclass(frozen=True)
class CleanCheckoutReport:
    """Results of the clean-checkout tracking and import probe."""

    findings: tuple[str, ...]
    tracked_files: int

    @property
    def ok(self) -> bool:
        """Whether the checkout is clean, clone-correct, and importable."""
        return not self.findings


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "git",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedcache=false",
            "-C",
            str(root),
            *arguments,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def _import_probe(root: Path) -> str | None:
    source_root = root / "src"
    script = (
        "import importlib\n"
        "import sys\n"
        "from pathlib import Path\n"
        f"source_root = Path({str(source_root)!r}).resolve()\n"
        "sys.path.insert(0, str(source_root))\n"
        "for name in ('analysis', 'fedference', 'figures', 'publication'):\n"
        "    module = importlib.import_module(name)\n"
        "    module_file = getattr(module, '__file__', None)\n"
        "    if not module_file:\n"
        "        raise RuntimeError(f'{name} has no concrete module file')\n"
        "    try:\n"
        "        Path(module_file).resolve().relative_to(source_root)\n"
        "    except ValueError as exc:\n"
        "        raise RuntimeError(\n"
        "            f'{name} resolved outside candidate source tree: {module_file}'\n"
        "        ) from exc\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            script,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return None
    detail = (result.stderr or result.stdout).strip().splitlines()
    return detail[-1] if detail else f"import probe exited {result.returncode}"


def inspect_clean_checkout(
    project_root: Path,
    *,
    check_imports: bool = True,
) -> CleanCheckoutReport:
    """Inspect Git cleanliness, required tracking, and package imports."""
    root = Path(project_root).resolve()
    findings: list[str] = []
    git_dir = root / ".git"
    if not git_dir.exists():
        return CleanCheckoutReport((f"missing Git metadata: {git_dir}",), 0)

    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status.returncode != 0:
        detail = (status.stderr or status.stdout).strip()
        findings.append(f"git status failed: {detail or status.returncode}")
    elif status.stdout.strip():
        paths = [line[3:] for line in status.stdout.splitlines() if len(line) >= 4]
        shown = ", ".join(paths[:12])
        suffix = "" if len(paths) <= 12 else f" (+{len(paths) - 12} more)"
        findings.append(f"worktree is dirty: {shown}{suffix}")

    tracked = _git(root, "ls-files")
    tracked_paths = set(tracked.stdout.splitlines()) if tracked.returncode == 0 else set()
    if tracked.returncode != 0:
        findings.append(f"git ls-files failed: {(tracked.stderr or '').strip()}")
    missing = sorted(set(REQUIRED_TRACKED_PATHS) - tracked_paths)
    if missing:
        findings.append("required files are not tracked: " + ", ".join(missing))

    if check_imports:
        import_failure = _import_probe(root)
        if import_failure is not None:
            findings.append(f"package import probe failed: {import_failure}")
    return CleanCheckoutReport(tuple(findings), len(tracked_paths))


__all__ = [
    "CleanCheckoutReport",
    "REQUIRED_TRACKED_PATHS",
    "inspect_clean_checkout",
]
