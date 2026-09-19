"""Checks for a genuinely clean, clone-correct project checkout."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from publication.metadata import validate_publication_lifecycle

IMMUTABLE_RELEASE_PDFS: tuple[str, ...] = (
    "Active_Fedference_Research_Manuscript_v0.1.0_Zenodo_10.5281-zenodo.21864004.pdf",
    "Active_Fedference_Research_Manuscript_v1.0.1_Zenodo_10.5281-zenodo.21919307.pdf",
    "Active_Fedference_Research_Manuscript_v1.0.2_Zenodo_10.5281-zenodo.21934992.pdf",
    "Active_Fedference_Research_Manuscript_v1.0.3_Zenodo_10.5281-zenodo.21969756.pdf",
    "Active_Fedference_Research_Manuscript_v1.0.4_Zenodo_10.5281-zenodo.21972644.pdf",
    "Active_Fedference_Research_Manuscript_v1.1.0_Zenodo_10.5281-zenodo.22149133.pdf",
)
HISTORICAL_RELEASE_PDF_LEDGER_PATH = "docs/reference/historical-release-pdfs.json"
TODO_TRACKED_PATHS: tuple[str, ...] = (
    "docs/todo/README.md",
    "docs/todo/adaptive-robustness-calibration.md",
    "docs/todo/beyond-discrete-categorical-state-spaces.md",
    "docs/todo/deeper-hierarchy-task-family.md",
    "docs/todo/external-benchmark-domain-pilot.md",
    "docs/todo/faithful-fedgvi-bnn-lane.md",
    "docs/todo/faithful-friston-protocol-replication.md",
    "docs/todo/independent-reproduction.md",
    "docs/todo/release-and-verification-ladder.md",
    "docs/todo/release-documentation-handoff.md",
    "docs/todo/scholarship-and-phase-plan.md",
    "docs/todo/true-multi-machine-federation.md",
    "docs/todo/v1-1-release-publication.md",
    "docs/todo/visual-scholarship-integration.md",
)

REQUIRED_TRACKED_PATHS: tuple[str, ...] = (
    *IMMUTABLE_RELEASE_PDFS,
    "AGENTS.md",
    "ISA.md",
    "README.md",
    "TODO.md",
    "LICENSE",
    "MANIFEST.in",
    "CITATION.cff",
    ".zenodo.json",
    "codemeta.json",
    "_fedference_build_backend.py",
    "pyproject.toml",
    "uv.lock",
    "manuscript/config.yaml",
    "manuscript/config.yaml.example",
    ".github/workflows/ci.yml",
    "src/analysis/artifacts.py",
    "src/analysis/figure_exact_values.py",
    "src/analysis/report_schemas.py",
    "src/analysis/visual_contracts.py",
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
    "src/fedference/py.typed",
    "src/fedference_cli/__init__.py",
    "src/fedference_cli/__main__.py",
    "src/fedference_cli/AGENTS.md",
    "src/fedference_cli/README.md",
    "src/fedference_cli/_commands.py",
    "src/fedference_cli/_parser.py",
    "src/fedference_cli/_support.py",
    "src/project_paths.py",
    "src/publication/clean_checkout.py",
    "src/publication/identifiers.py",
    "src/publication/metadata.py",
    "src/publication/zenodo.py",
    "src/publication/pipeline_freshness.py",
    "src/publication/release_assets.py",
    "src/publication/release_manifest.py",
    "src/publication/surface_validation.py",
    "src/publication/template_renderer.py",
    "src/publication/validation_receipt.py",
    "src/publication/web_package.py",
    "src/figures/_metadata.py",
    "src/figures/application_integrity_flow.py",
    "src/figures/complexity_scaling.py",
    "src/figures/conditional_world.py",
    "src/figures/belief_quality.py",
    "src/figures/evidence_replication_map.py",
    "src/figures/robustness_review_grid.py",
    "src/figures/generative_model_schema.py",
    "src/figures/message_passing.py",
    "src/figures/pomdp_loop.py",
    "src/figures/sensitivity_heatmap.py",
    "src/figures/source_render_provenance.py",
    "output/data/pipeline_provenance.json",
    "output/data/analysis_execution.json",
    "output/data/test_coverage_receipt.json",
    "output/figures/conditional_world.png",
    "output/figures/conditional_world.pdf",
    "output/figures/belief_quality.png",
    "output/figures/belief_quality.pdf",
    "output/figures/robustness_review_grid.png",
    "output/figures/robustness_review_grid.pdf",
    "output/figures/application_integrity_flow.png",
    "output/figures/application_integrity_flow.pdf",
    "output/figures/evidence_replication_map.png",
    "output/figures/evidence_replication_map.pdf",
    "output/figures/source_render_provenance.png",
    "output/figures/source_render_provenance.pdf",
    "output/figures/figure_exact_values.json",
    "output/figures/figure_exact_values.md",
    "output/figures/figure_registry.json",
    "output/reports/application_integrity_flow.json",
    "output/reports/evidence_replication_map.json",
    "output/reports/sensitivity.json",
    "output/reports/source_render_provenance.json",
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
    "docs/application-guide.md",
    "examples/05_labeled_application.py",
    "examples/data/labeled_aggregation_request.json",
    "tests/test_build_backend.py",
    "tests/test_release_preflight.py",
    "tests/test_release_assets.py",
    "tests/test_release_manifest.py",
    "tests/test_publication_metadata.py",
    "tests/test_publication_identifiers.py",
    "tests/test_zenodo.py",
    "tests/test_pipeline_freshness.py",
    "tests/test_template_renderer.py",
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
    "tests/analysis/test_artifacts.py",
    "tests/analysis/test_figure_exact_values.py",
    "tests/analysis/test_report_schemas.py",
    "tests/analysis/test_visual_contracts.py",
    "tests/analysis/test_workflow.py",
    "tests/figures/test_complexity_scaling.py",
    "tests/figures/test_conditional_world.py",
    "tests/figures/test_evidence_replication_map.py",
    "tests/figures/test_robustness_review_grid.py",
    "tests/figures/test_visual_flow_figures.py",
    "tests/test_surface_validation.py",
    "tests/test_web_publication_contract.py",
    "tests/test_report_scale_guard.py",
    "docs/research/manuscript-claim-audit.md",
    "docs/research/visual-claim-audit.md",
    "docs/manuscript/accessibility.md",
    "docs/reference/api-stability.md",
    HISTORICAL_RELEASE_PDF_LEDGER_PATH,
    "docs/reference/zenodo-release.md",
    "manuscript/30_supplement_notation.md",
    "docs/security/README.md",
    "docs/security/active_fedference-threat-model.md",
    *TODO_TRACKED_PATHS,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Construct a JSON object while rejecting ambiguous duplicate keys."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def historical_release_pdf_findings(project_root: Path) -> tuple[str, ...]:
    """Validate the checked-in ledger and bytes of every immutable release PDF."""
    root = Path(project_root).resolve()
    ledger_path = root / HISTORICAL_RELEASE_PDF_LEDGER_PATH
    if not ledger_path.is_file() or ledger_path.is_symlink():
        return (f"historical release PDF ledger is missing: {ledger_path}",)
    try:
        raw = json.loads(
            ledger_path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return (f"historical release PDF ledger is malformed: {exc}",)
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "sha256"}:
        return (
            "historical release PDF ledger must contain exactly schema_version and sha256",
        )
    if raw["schema_version"] != "1.0" or not isinstance(raw["sha256"], dict):
        return ("historical release PDF ledger schema is invalid",)

    expected_names = set(IMMUTABLE_RELEASE_PDFS)
    ledger = raw["sha256"]
    assert isinstance(ledger, dict)  # narrowed above
    ledger_names = set(ledger)
    findings: list[str] = []
    missing_names = sorted(expected_names - ledger_names)
    unknown_names = sorted(ledger_names - expected_names)
    if missing_names:
        findings.append("historical release PDF ledger is missing: " + ", ".join(missing_names))
    if unknown_names:
        findings.append(
            "historical release PDF ledger has unknown files: " + ", ".join(unknown_names)
        )
    for filename in IMMUTABLE_RELEASE_PDFS:
        expected_digest = ledger.get(filename)
        if not isinstance(expected_digest, str) or not _SHA256_RE.fullmatch(expected_digest):
            findings.append(f"historical release PDF digest is invalid: {filename}")
            continue
        artifact = root / filename
        if not artifact.is_file() or artifact.is_symlink():
            findings.append(f"historical release PDF is missing: {filename}")
            continue
        actual_digest = _sha256_file(artifact)
        if actual_digest != expected_digest:
            findings.append(
                "historical release PDF SHA-256 mismatch: "
                f"{filename} ({actual_digest} != {expected_digest})"
            )
    return tuple(findings)


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
        "for name in ('analysis', 'fedference', 'fedference_cli', 'figures', 'publication'):\n"
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


def _expected_manuscript_pdf(root: Path) -> str | None:
    """Return the release PDF name, or ``None`` for an unreleased revision."""
    try:
        lifecycle = validate_publication_lifecycle(
            root,
            require_generated_metadata=True,
            require_canonical_pdf=False,
        )
        return lifecycle.canonical_pdf
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"invalid publication lifecycle: {exc}") from exc


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

    try:
        expected_pdf = _expected_manuscript_pdf(root)
    except ValueError as exc:
        findings.append(str(exc))
    else:
        if expected_pdf is not None:
            expected_path = root / expected_pdf
            if expected_pdf not in tracked_paths:
                findings.append(f"configured manuscript PDF is not tracked: {expected_pdf}")
            elif not expected_path.is_file() or expected_path.is_symlink():
                findings.append(
                    f"configured manuscript PDF is missing or unsafe: {expected_pdf}"
                )

    findings.extend(historical_release_pdf_findings(root))

    if check_imports:
        import_failure = _import_probe(root)
        if import_failure is not None:
            findings.append(f"package import probe failed: {import_failure}")
    return CleanCheckoutReport(tuple(findings), len(tracked_paths))


__all__ = [
    "CleanCheckoutReport",
    "HISTORICAL_RELEASE_PDF_LEDGER_PATH",
    "IMMUTABLE_RELEASE_PDFS",
    "REQUIRED_TRACKED_PATHS",
    "TODO_TRACKED_PATHS",
    "historical_release_pdf_findings",
    "inspect_clean_checkout",
]
