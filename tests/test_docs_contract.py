from __future__ import annotations

import json
import os
import re
import urllib.parse
from pathlib import Path

import pytest
import yaml

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by the Python 3.10 lane
    import tomli as tomllib

from analysis.visual_contracts import SOURCE_RENDER_PRODUCER_EDGE_INVENTORY
from publication.metadata import validate_publication_lifecycle

ROOT = Path(__file__).resolve().parent.parent
GOVERNING_TODO_PATH = "docs/todo/scholarship-and-phase-plan.md"

DOC_CONTRACT_FILES = (
    "AGENTS.md",
    "README.md",
    "STANDALONE.md",
    "TODO.md",
    "examples/README.md",
    "docs/AGENTS.md",
    "docs/README.md",
    "docs/application-guide.md",
    "docs/core/architecture.md",
    "docs/core/experiments-and-artifacts.md",
    "docs/development/agent_instructions.md",
    "docs/development/quickstart.md",
    "docs/development/modularity.md",
    "docs/development/style_guide.md",
    "docs/development/testing_philosophy.md",
    "docs/manuscript/accessibility.md",
    "docs/manuscript/rendering_pipeline.md",
    "docs/manuscript/tokens-and-labels.md",
    "docs/operations/faq.md",
    "docs/operations/output-layout.md",
    "docs/operations/troubleshooting.md",
    "docs/reference/verification-commands.md",
    "docs/reference/zenodo-release.md",
    "docs/research/README.md",
    "docs/research/literature-audit.md",
    "docs/research/manuscript-claim-audit.md",
    "docs/research/computational-complexity-audit-2026-07-28.md",
    "docs/research/extended-statistical-audit-2026-07-14.md",
    "docs/research/first-principles-redteam-review-2026-07-16.md",
    "docs/research/cli-modularity-review-2026-08-15.md",
    "docs/research/runtime-surface-composability-review-2026-07-17.md",
    "docs/research/visual-claim-audit.md",
    "docs/security/README.md",
    "docs/security/active_fedference-threat-model.md",
    "data/AGENTS.md",
    "data/README.md",
    "manuscript/AGENTS.md",
    "manuscript/README.md",
    "manuscript/SYNTAX.md",
    "scripts/AGENTS.md",
    "scripts/CONVENTIONS.md",
    "scripts/README.md",
    "src/AGENTS.md",
    "src/README.md",
    "src/fedference_cli/README.md",
    "src/fedference_cli/AGENTS.md",
    "src/STYLE.md",
    "src/analysis/AGENTS.md",
    "src/analysis/README.md",
    "src/figures/AGENTS.md",
    "src/figures/README.md",
    "tests/AGENTS.md",
    "tests/PATTERNS.md",
    "tests/README.md",
)


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_application_guide_has_no_duplicate_table_separator_rows() -> None:
    guide = _read("docs/application-guide.md")
    assert "| --- | --- | --- |\n| --- | --- | --- |" not in guide


def test_open_scientific_waves_are_not_preassigned_obsolete_versions() -> None:
    plan = _read("docs/todo/scholarship-and-phase-plan.md")
    normalized_plan = re.sub(r"\s+", " ", plan)
    for stale_label in (
        "Calibration / v0.2",
        "Portable FedGVI / v0.3",
        "External data / v0.3",
        "Friston reconstruction / v0.4",
        "Hybrid tracking / v0.4",
        "Hierarchy tasks / v0.4",
        "Local emulator / v1.0",
    ):
        assert stale_label not in plan
    branch_names = (
        "codex/maj8-external-calibration",
        "codex/maj8-frozen-design",
        "codex/maj6-confirmatory",
    )
    offsets = [plan.index(branch_name) for branch_name in branch_names]
    assert offsets == sorted(offsets)
    assert "exact frozen maj-8 design" in plan.casefold()
    assert "chosen only after the observed result" in normalized_plan

    calibration = _read("docs/todo/adaptive-robustness-calibration.md")
    confirmation = _read("docs/todo/external-benchmark-domain-pilot.md")
    assert branch_names[0] in calibration
    assert branch_names[1] in calibration
    assert branch_names[1] in confirmation
    assert branch_names[2] in confirmation


def test_release_todo_pages_name_their_exact_public_branches() -> None:
    visual = _read("docs/todo/visual-scholarship-integration.md")
    release = _read("docs/todo/v1-1-release-publication.md")

    assert "codex/v1.1-visual-scholarship" in visual
    assert "codex/v1.1.0-release" in release


def test_web_publication_docs_state_the_fail_closed_resource_boundary() -> None:
    verification = _read("docs/reference/verification-commands.md")
    rendering = _read("docs/manuscript/rendering_pipeline.md")
    combined = f"{verification}\n{rendering}"
    normalized_verification = re.sub(r"\s+", " ", verification)

    for construct in ("`<base>`", "CDATA", "`<noscript>`"):
        assert construct in combined
    assert "no-clobber batch transaction" in rendering
    assert "browser-effective" in combined
    assert "does not fetch the remote response" in verification
    assert "prove that the declared digest matches its bytes" in normalized_verification


def test_release_reviewer_policy_distinguishes_eligibility_from_independence() -> None:
    ladder = _read("docs/todo/release-and-verification-ladder.md")
    release = _read("docs/todo/v1-1-release-publication.md")
    phase_plan = _read("docs/todo/scholarship-and-phase-plan.md")
    roadmap = _read("TODO.md")
    isa = _read("ISA.md")
    normalized_ladder = re.sub(r"\s+", " ", ladder)
    normalized_release = re.sub(r"\s+", " ", release)
    normalized_phase_plan = re.sub(r"\s+", " ", phase_plan)

    assert "either an identified human or a genuinely different-vendor" in normalized_ladder
    assert "A local subagent does not qualify" in normalized_ladder
    assert "identified owner-author human review" in ladder
    assert "not as independent external replication or cross-vendor review" in normalized_ladder
    assert "selected Daniel Ari Friedman structured verdict" in normalized_release
    assert "distinct from authorization to tag or publish" in normalized_release
    assert "technical eligibility rule is separate from publication authority" in normalized_phase_plan
    assert "fresh explicit approval" in roadmap
    assert "does not close ISC-89" in ladder
    assert "[DEFERRED-VERIFY] ISC-89" in isa
    assert "- [ ] ISC-242" in isa
    assert "does not close ISC-89 or the broader independent-reproduction lane" in isa


def test_public_release_policy_omits_personal_and_internal_workspace_paths() -> None:
    public_policy = "\n".join(
        _read(path)
        for path in (
            "ISA.md",
            "TODO.md",
            "docs/todo/release-and-verification-ladder.md",
            "docs/todo/scholarship-and-phase-plan.md",
        )
    )
    forbidden_paths = (
        f"{Path('/', 'Users')}/",
        f".{''.join(('clau', 'de'))}/",
        f"{''.join(('scratch', 'pad'))}/",
        f"{Path('/', 'private', 'tmp')}/",
        f"{Path('/', 'tmp')}/",
    )
    for forbidden in forbidden_paths:
        assert forbidden not in public_policy
    assert public_policy.count("/Volumes/blue/active_fedference-verification/") == 1
    assert "approved, non-confidential operational example" in re.sub(r"\s+", " ", public_policy)


def test_readme_distinguishes_completed_bnn_pilot_from_open_parity() -> None:
    readme = _read("README.md")
    assert "future cavity-conditioned client wiring" not in readme
    assert "completed synthetic-pilot cavity wiring" in readme
    assert "source-data/CUDA parity future" in readme


_MARKDOWN_LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def _local_markdown_targets(relative_path: str) -> set[str]:
    """Resolve existing local Markdown targets for one repository page."""
    source = ROOT / relative_path
    targets: set[str] = set()
    for match in _MARKDOWN_LINK_PATTERN.finditer(_read(relative_path)):
        target = match.group(1).strip()
        if not target or target.startswith(("#", "http://", "https://", "mailto:")):
            continue
        if target.startswith("<") and target.endswith(">"):
            target = target[1:-1]
        target_path = urllib.parse.unquote(target.split("#", 1)[0])
        if not target_path or Path(target_path).suffix.lower() != ".md":
            continue
        resolved = (source.parent / target_path).resolve()
        try:
            relative = resolved.relative_to(ROOT)
        except ValueError:
            continue
        if resolved.is_file():
            targets.add(relative.as_posix())
    return targets


def _todo_doc_files() -> tuple[str, ...]:
    todo_dir = ROOT / "docs" / "todo"
    if not todo_dir.exists():
        return ()
    return tuple(
        path.relative_to(ROOT).as_posix()
        for path in sorted(todo_dir.glob("*.md"))
    )


def _active_todo_doc_files() -> tuple[str, ...]:
    return tuple(
        path
        for path in _todo_doc_files()
        if path not in {"docs/todo/README.md", GOVERNING_TODO_PATH}
    )


def _repository_guide_files() -> tuple[str, ...]:
    """Discover every source-owned AGENTS/README contract, not a hand-picked subset."""
    guides: list[str] = []
    for path in ROOT.rglob("*.md"):
        relative = path.relative_to(ROOT)
        if path.name not in {"AGENTS.md", "README.md"}:
            continue
        if relative.parts[0] in {".pytest_cache", ".tmp", ".venv", "build", "node_modules", "output"}:
            continue
        guides.append(relative.as_posix())
    return tuple(sorted(guides))


def _source_owned_markdown_files() -> tuple[str, ...]:
    """Discover every source-owned page under ``docs/``.

    The curated list above documents the highest-risk contracts, but every
    source-owned page is part of the standalone reader surface. Keeping this
    discovery here prevents a new research note, section README, or roadmap
    page from escaping link and stale-claim checks merely because nobody added
    it to a test tuple.
    """
    docs_root = ROOT / "docs"
    return tuple(
        sorted(path.relative_to(ROOT).as_posix() for path in docs_root.rglob("*.md"))
    )


def _contract_files() -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                *DOC_CONTRACT_FILES,
                *_repository_guide_files(),
                *_source_owned_markdown_files(),
                *_todo_doc_files(),
            )
        )
    )


def test_docs_use_standalone_repo_root_commands() -> None:
    offenders = []
    stale_template_paths = []
    old_template_prefix = "/" + "/".join(("Users", "4d", "Documents", "GitHub", "template"))
    for path in _contract_files():
        for line_number, line in enumerate(_read(path).splitlines(), start=1):
            if old_template_prefix in line:
                stale_template_paths.append(f"{path}:{line_number}")
            if "projects/working/active_fedference" not in line:
                continue
            if "projects/working/active_fedference/manuscript" in line:
                continue
            if "working/active_fedference" in line:
                continue
            offenders.append(f"{path}:{line_number}")
    assert stale_template_paths == []
    assert offenders == []


def test_documented_local_commands_reference_existing_scripts() -> None:
    text = "\n".join(_read(path) for path in _contract_files())
    missing = []
    for line in text.splitlines():
        for ref in re.findall(r"\bscripts/[A-Za-z0-9_./-]+\.py", line):
            if ref in {
                "scripts/pipeline/stage_03_render.py",
                "scripts/pipeline/stage_04_validate.py",
                "scripts/pipeline/stage_05_copy.py",
                "scripts/maintenance/refresh_artifact_manifests.py",
                "scripts/runner/execute_pipeline.py",
            }:
                continue
            if not (ROOT / ref).exists():
                missing.append(ref)
    missing = sorted(set(missing))
    assert missing == []
    assert "execute_pipeline.py --project active_fedference" not in text
    assert "03_render_pdf.py --project active_fedference" not in text
    for obsolete in (
        "scripts/03_render_pdf.py",
        "scripts/04_validate_output.py",
        "scripts/05_copy_outputs.py",
        "scripts/execute_pipeline.py",
    ):
        assert obsolete not in text
    assert 'grep -r "{{"' not in text
    assert r"\{\{[A-Z][A-Z0-9_]*\}\}" in text


def test_script_quick_reference_covers_every_entrypoint() -> None:
    documented = _read("scripts/README.md")
    script_names = sorted(
        path.name for path in (ROOT / "scripts").glob("*.py") if path.name != "__init__.py"
    )
    missing = [name for name in script_names if f"`{name}" not in documented]
    assert missing == []


def test_documented_local_markdown_links_resolve_inside_standalone_repo() -> None:
    """Prevent docs from linking back to absent template-repo guides."""
    allowed_placeholder_targets = {
        "../output/figures/name.png",
        "../output/figures/NAME.png",
    }
    offenders: list[str] = []
    for path in _contract_files():
        source = ROOT / path
        for line_number, line in enumerate(_read(path).splitlines(), start=1):
            for match in _MARKDOWN_LINK_PATTERN.finditer(line):
                target = match.group(1).strip()
                if (
                    not target
                    or target.startswith(("#", "http://", "https://", "mailto:"))
                    or target in allowed_placeholder_targets
                ):
                    continue
                if target.startswith("<") and target.endswith(">"):
                    target = target[1:-1]
                target_path = urllib.parse.unquote(target.split("#", 1)[0])
                if not target_path:
                    continue
                resolved = (source.parent / target_path).resolve()
                try:
                    resolved.relative_to(ROOT)
                except ValueError:
                    offenders.append(f"{path}:{line_number}: {target} escapes repository")
                    continue
                if not resolved.exists():
                    offenders.append(f"{path}:{line_number}: {target} missing")
    assert offenders == []


def test_documentation_hub_and_section_indexes_reach_every_page() -> None:
    """A new docs page must be linked from its section index and the main hub."""
    docs_root = ROOT / "docs"
    hub_links = _local_markdown_targets("docs/README.md")

    top_level_pages = {
        path.relative_to(ROOT).as_posix()
        for path in docs_root.glob("*.md")
        if path.name != "README.md"
    }
    section_indexes = {
        path.relative_to(ROOT).as_posix()
        for path in docs_root.glob("*/README.md")
    }
    assert sorted(top_level_pages - hub_links) == []
    assert sorted(section_indexes - hub_links) == []

    missing_by_section: dict[str, list[str]] = {}
    for index in sorted(section_indexes):
        index_path = ROOT / index
        sibling_pages = {
            path.relative_to(ROOT).as_posix()
            for path in index_path.parent.glob("*.md")
            if path != index_path
        }
        missing = sorted(sibling_pages - _local_markdown_targets(index))
        if missing:
            missing_by_section[index] = missing
    assert missing_by_section == {}


def test_application_guide_is_the_lightweight_canonical_user_path() -> None:
    """Keep first use distinct from contributor and generated-output surfaces."""
    guide = _read("docs/application-guide.md")
    root_readme = _read("README.md")
    docs_hub = _read("docs/README.md")
    examples = _read("examples/README.md")
    verification = _read("docs/reference/verification-commands.md")
    examples_before_tests = examples.split("## Regression tests", maxsplit=1)[0]

    assert "docs/application-guide.md" in root_readme
    assert "application-guide.md" in docs_hub
    assert "uv sync --locked --extra dev" not in examples_before_tests
    assert "uv sync --locked" in examples_before_tests
    assert "examples/05_labeled_application.py" in guide
    assert "examples/data/labeled_aggregation_request.json" in guide
    assert "fedference aggregate" in guide
    assert "aggregate_labeled" in guide
    assert "LabeledAggregationRequest" in guide
    assert "from fedference import AggregationConfig, aggregate_result" in guide
    assert "run_multiprocess_round_result" in guide
    assert "solver_status" in guide
    assert "fallback_events" in guide
    assert "same ordered labels" in guide
    assert "one-dimensional" in guide
    assert "does **not** prove" in guide
    assert "scientific validity" in guide
    assert "downstream decision" in guide
    assert "py.typed" in guide
    assert "'torch' not in sys.modules" in guide
    assert "examples/05_labeled_application.py" in verification
    assert "src/fedference/py.typed" in verification
    diagram = guide.split("```mermaid", maxsplit=1)[1].split("```", maxsplit=1)[0]
    ordered_steps = (
        "Duplicate-key-safe JSON parsing",
        "Shared exact-field, identifier, finite-mass, and one-dimensional-shape validation",
        "Canonical semantic request with preserved order and masses",
        "Validate required provenance and destination safety",
        "Delegate exactly once to aggregate_result",
    )
    offsets = [diagram.index(step) for step in ordered_steps]
    assert offsets == sorted(offsets)
    assert "invalid labeled request" in diagram
    assert "unsafe destination or required provenance unavailable" in diagram
    assert "--require-nominal-solver" in verification
    assert "does not load Torch" in verification
    assert "../output/docs/" not in guide
    assert "predates" in root_readme
    assert "git checkout v1.0.4" not in f"{root_readme}\n{guide}"


def test_local_review_artifacts_are_ignored_and_test_profiles_are_declared() -> None:
    gitignore = _read(".gitignore")
    pyproject = _read("pyproject.toml")
    tests_readme = _read("tests/README.md")
    assert ".tmp/" in gitignore
    for marker in ("slow", "integration", "publication"):
        assert f'"{marker}:' in pyproject
    assert "-m \"not slow\"" in tests_readme


def test_public_mermaid_blocks_have_balanced_github_compatible_structure() -> None:
    from scripts.validate_mermaid import validate_mermaid_blocks

    blocks = validate_mermaid_blocks(ROOT)
    discovered_fences = sum(
        sum(
            line.strip().lower() == "```mermaid"
            for line in path.read_text(encoding="utf-8").splitlines()
        )
        for path in (ROOT / "README.md", *(ROOT / "docs").rglob("*.md"))
    )
    assert blocks
    assert len(blocks) == discovered_fences


def test_mermaid_accessibility_contract_fails_closed(tmp_path: Path) -> None:
    from scripts.validate_mermaid import validate_mermaid_blocks

    readme = tmp_path / "README.md"
    readme.write_text(
        "# Incomplete diagram\n\n"
        "```mermaid\n"
        "flowchart LR\n"
        "    A[unquoted/path] --> B[Destination]\n"
        "```\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as error:
        validate_mermaid_blocks(tmp_path)

    message = str(error.value)
    assert "accTitle" in message
    assert "accDescr" in message
    assert "quote node labels" in message
    assert "Text equivalent" in message


def test_required_accessible_workflow_diagrams_remain_source_visible() -> None:
    application = _read("docs/application-guide.md")
    provenance = _read("docs/core/experiments-and-artifacts.md")
    threat_model = _read("docs/security/active_fedference-threat-model.md")

    for marker in (
        "Labeled aggregation and application receipt flow",
        "Delegate exactly once to aggregate_result",
        "Retain complete artifacts for review; CLI exit 1",
        "No claim of calibration",
    ):
        assert marker in application
    for marker in (
        "Source-to-render production and stale-artifact invalidation",
        "exact clean Template renderer commit",
        'release_manifest -->|"review + authorize"| github',
        'github -->|"verify + fresh approval"| zenodo',
        "avoiding a circular dependency",
    ):
        assert marker in provenance
    for marker in (
        "sequenceDiagram",
        "participant G as Replay Guard",
        "unchanged protocol-v1 payload containing consensus and agent_weights",
        "Separate solver finding while integrity remains valid",
        "does **not** claim encryption",
        "Byzantine tolerance",
        "differential privacy",
    ):
        assert marker in threat_model


def test_source_to_render_mermaid_uses_the_exact_source_owned_branching_dag() -> None:
    provenance = _read("docs/core/experiments-and-artifacts.md")
    blocks = re.findall(r"```mermaid\n(.*?)\n```", provenance, flags=re.DOTALL)
    block = next(
        candidate
        for candidate in blocks
        if "accTitle: Source-to-render production and stale-artifact invalidation" in candidate
    )
    observed_edges = tuple(
        (match.group(1), match.group(2))
        for match in re.finditer(
            r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s+-->\s*"
            r"(?:\|[^|]*\|\s*)?([A-Za-z_][A-Za-z0-9_]*)",
            block,
            flags=re.MULTILINE,
        )
    )
    expected_edges = tuple(
        (source, target)
        for source, target, _disposition in SOURCE_RENDER_PRODUCER_EDGE_INVENTORY
    )

    assert observed_edges == expected_edges
    assert "template_renderer" in block
    assert "stale_target" in block and "changed_owner" in block
    assert '-. "reverse dependency: points back to changed owner" .->' in block
    assert "four coded inputs" not in provenance
    assert "nine dashed reverse" not in provenance
    assert "| `source` | `reports`, `figures`, `provisional`, `coverage`, `final_hydration` |" in provenance


def test_retired_platform_name_is_absent_from_textual_repository_surfaces() -> None:
    markers = ("hum" + "os", "hum" + " " + "os", "human operating" + " system")
    ignored_directories = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".playwright-cli",
        ".ruff_cache",
        ".tmp",
        ".venv",
        "build",
        "node_modules",
    }
    text_suffixes = {
        ".cff",
        ".html",
        ".json",
        ".md",
        ".py",
        ".tex",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
    offenders: list[str] = []
    for directory, directory_names, file_names in os.walk(ROOT):
        directory_names[:] = [
            name for name in directory_names if name not in ignored_directories
        ]
        for file_name in file_names:
            path = Path(directory) / file_name
            if path.suffix.lower() not in text_suffixes:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            if any(marker in text for marker in markers):
                offenders.append(path.relative_to(ROOT).as_posix())
    assert offenders == []


def test_publication_lifecycle_and_historical_release_identity_are_separated() -> None:
    config = yaml.safe_load(_read("manuscript/config.yaml"))
    expected_doi = config["publication"]["doi"]
    expected_doi_status = config["publication"].get("doi_status")
    expected_date = config["publication"]["date_released"]
    expected_version = config["paper"]["version"]
    lifecycle = validate_publication_lifecycle(ROOT, require_canonical_pdf=False)
    metadata = "\n".join(
        _read(path)
        for path in (
            "manuscript/config.yaml",
            ".zenodo.json",
            "codemeta.json",
            "CITATION.cff",
            "README.md",
            "STANDALONE.md",
        )
    )
    assert "Active Fedference" in metadata
    assert "ActiveInferenceInstitute/Active_Fedference" in metadata
    assert "template_code_project" not in metadata
    assert "Convergence Analysis of Gradient Descent Optimization" not in metadata
    assert "10.5281/zenodo.20417136" not in metadata
    assert "10.5281/zenodo.21972644" in metadata
    assert "https://doi.org/(forthcoming)" not in metadata

    assert ":" not in config["paper"]["title"]
    assert config["paper"]["subtitle"]
    assert ":" not in config["paper"]["subtitle"]
    assert config["publication"]["github_repository"] == "https://github.com/ActiveInferenceInstitute/Active_Fedference"
    assert config["metadata"]["license"] == "MIT"
    assert "active inference" in config["keywords"]
    assert "FedGVI" in config["keywords"]
    citation = yaml.safe_load(_read("CITATION.cff"))
    zenodo = json.loads(_read(".zenodo.json"))
    codemeta = json.loads(_read("codemeta.json"))
    package_metadata = tomllib.loads(_read("pyproject.toml"))
    assert citation["version"] == expected_version
    assert zenodo["version"] == expected_version
    assert codemeta["version"] == expected_version
    assert package_metadata["project"]["version"] == expected_version
    assert lifecycle.version == expected_version

    if lifecycle.state == "development":
        assert expected_version.endswith(".dev0")
        assert expected_doi == ""
        assert expected_doi_status == "(forthcoming)"
        assert expected_doi_status in metadata
        assert expected_date is None
        assert "identifiers" not in citation
        assert "date-released" not in citation
        assert "doi" not in zenodo
        assert "publication_date" not in zenodo
        assert "identifier" not in codemeta
        assert "dateModified" not in codemeta
        assert "DOI" not in package_metadata["project"]["urls"]
    else:
        assert re.fullmatch(r"[0-9]+(?:\.[0-9]+){2}", expected_version)
        assert lifecycle.doi == expected_doi
        assert lifecycle.date_released == str(expected_date)
        assert expected_doi_status is None
        assert citation["identifiers"] == [{"type": "doi", "value": expected_doi}]
        citation_date = citation["date-released"]
        assert (
            citation_date.isoformat()
            if hasattr(citation_date, "isoformat")
            else str(citation_date)
        ) == lifecycle.date_released
        assert zenodo["doi"] == expected_doi
        assert zenodo["publication_date"] == lifecycle.date_released
        assert codemeta["identifier"] == f"https://doi.org/{expected_doi}"
        assert codemeta["dateModified"] == lifecycle.date_released
        assert package_metadata["project"]["urls"]["DOI"] == (
            f"https://doi.org/{expected_doi}"
        )


def test_docs_do_not_reintroduce_stale_claim_language() -> None:
    stale_patterns = {
        "two-robustness": re.compile(r"two-robustness", re.IGNORECASE),
        "two axes": re.compile(r"\btwo[- ]axes\b", re.IGNORECASE),
        "old sidecar path": re.compile(r"projects/active_fedference"),
        "dead scope file": re.compile(r"manuscript/12_scope\.md"),
        "completion-log phrasing": re.compile(
            r"Completed iterations|Only MAJOR|Done and tested",
            re.IGNORECASE,
        ),
        "stale study count": re.compile(r"\bseven studies\b", re.IGNORECASE),
        "stale figure count": re.compile(r"\bfifteen (?:figures|PNG generators)\b", re.IGNORECASE),
        "ungated docs claim": re.compile(r"No CI gate parses these", re.IGNORECASE),
        "literal unresolved marker": re.compile(r"\?\?\?"),
    }
    offenders: list[str] = []
    for path in _contract_files():
        text = _read(path)
        for line_number, line in enumerate(text.splitlines(), start=1):
            for label, pattern in stale_patterns.items():
                if pattern.search(line):
                    offenders.append(f"{path}:{line_number}: {label}")
    assert offenders == []


def test_robustness_taxonomy_has_one_authoritative_definition() -> None:
    """Keep the abstract synthetic and limitations focused on non-transfer."""
    gap = _read("manuscript/02_gap.md")
    for heading in (
        "### Client-side: source-theorem-backed",
        "### Server-side: heuristic",
        "### Server-side: objective-backed and conservative",
    ):
        assert heading in gap

    abstract = _read("manuscript/00_abstract.md")
    synthesis = next(
        paragraph
        for paragraph in abstract.split("\n\n")
        if paragraph.startswith("Following the authoritative guarantee map")
    )
    assert synthesis.count(".") == 1
    assert "no robustness guarantee transferred between them" in synthesis

    limitations = _read("manuscript/23_discussion_limitations.md")
    section = limitations.split("## Three robustness axes", 1)[1].split(
        "## Scope boundaries",
        1,
    )[0]
    assert "this section records only what still cannot transfer" in section
    assert "The `robust_aggregate` axis is" not in section
    assert "The `variational_aggregate` axis is" not in section
    assert "The client-side bounded-loss axis is" not in section


def test_friston_eq7_bridge_language_is_scoped() -> None:
    """Keep the executable identity separate from source-protocol recovery."""
    canonical_sources = (
        "AGENTS.md",
        "README.md",
        "ISA.md",
        "STANDALONE.md",
        "docs/core/conceptual-foundations.md",
        "src/fedference/aggregation.py",
    )
    for path in canonical_sources:
        text = _read(path)
        assert "project-local" in text, path
        assert "posterior-log-potential" in text, path
        assert "reconstruction" in text, path


def test_canonical_two_pass_render_sequence_records_prepared_web_tree() -> None:
    """Keep the pre-test and final render passes ordered around their receipt."""
    text = _read("docs/manuscript/rendering_pipeline.md")
    heading = "## Source-current two-pass sequence"
    start = text.index(heading)
    end = text.index("## Phase 3", start)
    sequence = text[start:end]
    render = (
        "uv run --locked python scripts/pipeline/stage_03_render.py \\\n"
        "  --project working/active_fedference --skip-manuscript-hydration"
    )

    provisional = sequence.index(
        "uv run --locked python scripts/z_generate_manuscript_variables.py --provisional-validation"
    )
    first_render = sequence.index(render)
    first_preflight = sequence.index(
        'uv run --locked python scripts/00_preflight.py --template-root "$TEMPLATE_REPO"'
    )
    receipt = sequence.index("uv run --locked --extra dev python scripts/validate_test_coverage.py")
    final_hydration = sequence.index(
        "uv run --locked python scripts/z_generate_manuscript_variables.py",
        receipt,
    )
    second_render = sequence.index(render, final_hydration)
    second_preflight = sequence.index(
        'uv run --locked python scripts/00_preflight.py --template-root "$TEMPLATE_REPO"',
        first_preflight + 1,
    )
    package = sequence.index("uv run --locked python scripts/prepare_web_package.py")
    record = sequence.index("uv run --locked python scripts/record_pipeline_stage.py render")

    assert sequence.count(render) == 2
    assert first_preflight < provisional < first_render < receipt
    assert receipt < final_hydration < second_preflight < second_render
    assert second_render < package < record


def test_renderer_receipt_examples_use_source_locked_template_checkout() -> None:
    """Keep the public render recipe free of path-bearing ad hoc identities."""
    authoritative = (
        "docs/manuscript/rendering_pipeline.md",
        "docs/reference/verification-commands.md",
        "manuscript/README.md",
        "scripts/AGENTS.md",
        "scripts/README.md",
    )
    command = re.compile(
        r"record_pipeline_stage\.py render(?P<arguments>.{0,180})",
        flags=re.DOTALL,
    )
    examples: list[tuple[str, str]] = []
    for path in authoritative:
        text = _read(path)
        assert "TEMPLATE_DIFF_SHA256" not in text, path
        assert "TEMPLATE_COMMIT=" not in text, path
        for match in command.finditer(text):
            examples.append((path, match.group("arguments")))

    assert examples
    assert all("--template-root" in arguments for _, arguments in examples)
    assert all("--renderer" not in arguments for _, arguments in examples)


def test_copyable_config_documents_accessible_slides_and_renderer_lock() -> None:
    """Keep projection constraints and immutable renderer identity copyable."""
    example_path = ROOT / "manuscript" / "config.yaml.example"
    raw = example_path.read_text(encoding="utf-8")
    payload = yaml.safe_load(raw)

    slides = payload["render"]["slides"]
    assert slides == {
        "profile": "accessible",
        "max_prose_words": 80,
        "max_table_rows": 8,
        "min_figure_area_percent": 70,
        "title_font_pt": 28,
        "body_font_pt": 20,
        "figure_label_font_pt": 16,
        "reader_href": "../web/index.html",
    }
    renderer = payload["rendering"]["template_renderer"]
    assert renderer["schema_version"] == "template-renderer-lock-v1"
    assert renderer["repository"] == "docxology/template"
    assert re.fullmatch(r"[0-9a-f]{40}", renderer["git_commit"])
    assert renderer["git_tree_state"] == "clean"
    assert "never use a branch" in raw
    assert "runtime checkout" in raw


def test_source_current_render_examples_skip_implicit_hydration() -> None:
    """Do not let a documentation snippet trigger hydration ahead of its receipt."""
    command = re.compile(
        r"uv run --locked python scripts/pipeline/stage_03_render\.py"
        r"(?:[ \t]*\\\n\s*)?"
        r"--project working/active_fedference(?P<arguments>[^\n]*)"
    )
    canonical = _read("docs/manuscript/rendering_pipeline.md")

    assert 'git -C "$AF_REPO" log -1 --format=%ct' in canonical
    assert 'git -C "$TEMPLATE_REPO" log -1 --format=%ct' not in canonical

    examples = []
    for path in _contract_files():
        for match in command.finditer(_read(path)):
            examples.append((path, match.group("arguments")))
    assert examples
    assert all("--skip-manuscript-hydration" in arguments for _, arguments in examples)


def test_multi_machine_claims_stay_qualified() -> None:
    qualifying_words = (
        "future",
        "not",
        "scope",
        "extension",
        "caveat",
        "boundary",
        "remain",
        "later",
        "true",
    )
    offenders = []
    for path in _contract_files():
        lines = _read(path).splitlines()
        for index, line in enumerate(lines):
            if "multi-machine" not in line.lower():
                continue
            context = " ".join(lines[max(0, index - 1) : index + 2])
            lower = context.lower()
            if not any(word in lower for word in qualifying_words):
                offenders.append(f"{path}:{index + 1}")
    assert offenders == []


def test_security_and_accessibility_boundaries_are_explicit_and_gated() -> None:
    threat_model = _read("docs/security/active_fedference-threat-model.md")
    accessibility = _read("docs/manuscript/accessibility.md")
    web_validator = _read("src/publication/web_package.py")

    for heading in (
        "## Scope and assumptions",
        "## System model",
        "## Attacker model",
        "## Fault model for MAJ-4A",
        "## Threat model table",
        "## Focus paths for security review",
    ):
        assert heading in threat_model
    for threat_id in ("TM-001", "TM-002", "TM-003", "TM-009"):
        assert threat_id in threat_model
    for boundary in (
        "not a declaration of WCAG conformance",
        "Tagged: yes",
        "PDF/UA conformance not established",
        "StructTreeRoot",
    ):
        assert boundary in accessibility
    for enforced_marker in (
        "accessibility_issues",
        "main-content",
        "missing_image_alt",
        "figures_missing_captions",
        "duplicate_ids",
    ):
        assert enforced_marker in web_validator


def test_verification_docs_match_current_collection_count() -> None:
    text = _read("docs/reference/verification-commands.md")
    assert "all collected tests pass" in text
    assert "649 tests collected" not in text
    assert "568 tests" not in text


def test_todo_index_and_scoped_pages_are_bidirectionally_linked() -> None:
    todo_text = _read("TODO.md")
    active_tables = todo_text.split("## Minor", 1)[1].split("## Parked", 1)[0]
    linked_pages = set(
        re.findall(r"\]\((docs/todo/[a-z0-9-]+\.md)\)", active_tables)
    )
    scoped_pages = set(_active_todo_doc_files())
    assert linked_pages == scoped_pages
    assert "docs/todo/README.md" in todo_text
    assert GOVERNING_TODO_PATH in todo_text
    for page in scoped_pages:
        text = _read(page)
        assert "[Back to roadmap](../../TODO.md)" in text
        assert page in todo_text


def test_forward_todo_surfaces_do_not_retain_completed_items() -> None:
    offenders = []
    stale_patterns = (
        re.compile(r"SLICE LANDED", re.IGNORECASE),
        re.compile(r"State:\s*Done", re.IGNORECASE),
        re.compile(r"Recently completed", re.IGNORECASE),
        re.compile(r"Publication-polish closure", re.IGNORECASE),
        re.compile(r"\biteration \d+\b", re.IGNORECASE),
        re.compile(r"\b\d+\s+passed\b"),
    )
    for path in ("TODO.md", *_todo_doc_files()):
        text = _read(path)
        if "[x]" in text or any(pattern.search(text) for pattern in stale_patterns):
            offenders.append(path)
    assert offenders == []


def test_ci_workflow_runs_publication_package_and_release_round_trip() -> None:
    workflow = _read(".github/workflows/ci.yml")
    required_commands = (
        "uv run --locked python scripts/validate_all.py package",
        "sudo apt-get install --no-install-recommends -y poppler-utils qpdf",
        "tar -tzf \"$sdist\"",
        "uv run --locked python scripts/build_release.py",
        "uv run --locked python scripts/build_release.py --verify",
        "actions/upload-artifact@v4",
    )
    for command in required_commands:
        assert command in workflow


def test_scoped_todo_pages_are_complete() -> None:
    required_headings = (
        "## Status",
        "## Rationale",
        "## Scope",
        "## Implementation Notes",
        "## Acceptance Criteria",
        "## Verification Probes",
        "## Claim-Boundary Constraints",
        "## Dependencies",
    )
    replication_unit_pattern = re.compile(r"(independent|replication)[- ]unit", re.IGNORECASE)
    prohibited_claims_pattern = re.compile(r"(prohibited claims|no-claim)", re.IGNORECASE)
    for page in _active_todo_doc_files():
        text = _read(page)
        for heading in required_headings:
            assert heading in text, f"{page} missing {heading}"
        assert re.search(r"Priority class: (Minor|Medium|Major)", text), page
        assert re.search(r"State: Open", text), page
        assert re.search(r"estimand", text, re.IGNORECASE), f"{page} missing estimand"
        assert re.search(r"falsifier", text, re.IGNORECASE), f"{page} missing falsifier"
        assert (
            replication_unit_pattern.search(text) or "independent replication" in text.lower()
        ), f"{page} missing replication unit"
        assert prohibited_claims_pattern.search(text), f"{page} missing prohibited claims"


def test_todo_readme_links_every_scoped_page_bidirectionally() -> None:
    readme_text = _read("docs/todo/README.md")
    active_tables = readme_text.split("## Minor", 1)[1].split(
        "## Classification rule", 1
    )[0]
    linked_pages = {
        f"docs/todo/{name}"
        for name in re.findall(r"\]\(([a-z0-9-]+\.md)\)", active_tables)
    }
    scoped_pages = set(_active_todo_doc_files())
    missing = scoped_pages - linked_pages
    nonexistent = {
        page
        for page in linked_pages
        if not (ROOT / page).exists()
    }
    assert missing == set()
    assert nonexistent == set()
    assert "[Scholarship-indexed phase and claim governance]" in readme_text
    assert GOVERNING_TODO_PATH.endswith("scholarship-and-phase-plan.md")


def test_todo_priority_tables_match_each_page_and_open_state() -> None:
    index = _read("docs/todo/README.md")
    roadmap = _read("TODO.md")
    root_link_counts = {
        "docs/todo/faithful-fedgvi-bnn-lane.md": 2,
        "docs/todo/true-multi-machine-federation.md": 2,
    }
    classified_pages: set[str] = set()

    def mappings_for_priority(text: str, priority: str) -> set[tuple[str, str]]:
        match = re.search(
            rf"^## {priority}\b[^\n]*\n(?P<body>.*?)(?=^## |\Z)",
            text,
            re.MULTILINE | re.DOTALL,
        )
        assert match is not None, priority
        mappings: set[tuple[str, str]] = set()
        for identifier, target in re.findall(
            r"^\|\s*([^|]+?)\s*\|\s*\[[^]]+\]\(((?:docs/todo/)?[a-z0-9-]+\.md)\)",
            match.group("body"),
            re.MULTILINE,
        ):
            identifier = identifier.strip()
            split_identifier = re.fullmatch(
                r"((?:MIN|MED|MAJ)-\d+)([A-Z])/([A-Z])", identifier
            )
            identifiers = (
                (
                    f"{split_identifier.group(1)}{split_identifier.group(2)}",
                    f"{split_identifier.group(1)}{split_identifier.group(3)}",
                )
                if split_identifier
                else (identifier,)
            )
            page = f"docs/todo/{Path(target).name}"
            mappings.update((item_identifier, page) for item_identifier in identifiers)
        return mappings

    for priority in ("Minor", "Medium", "Major"):
        index_mappings = mappings_for_priority(index, priority)
        roadmap_mappings = mappings_for_priority(roadmap, priority)
        index_pages = {page for _, page in index_mappings}
        assert index_pages, priority
        assert roadmap_mappings == index_mappings, priority
        for page in index_pages:
            text = _read(page)
            assert f"Priority class: {priority}" in text, page
            assert len(re.findall(r"^\s*- State: Open\s*$", text, re.MULTILINE)) == 1, page
            assert roadmap.count(f"({page})") == root_link_counts.get(page, 1), page
            assert index.count(f"({Path(page).name})") == 1, page
        assert classified_pages.isdisjoint(index_pages)
        classified_pages.update(index_pages)

    assert classified_pages == set(_active_todo_doc_files())


def test_canonical_public_git_contract_is_consistent_across_active_release_pages() -> None:
    roadmap = re.sub(r"\s+", " ", _read("TODO.md"))
    index = re.sub(r"\s+", " ", _read("docs/todo/README.md"))
    visual = re.sub(
        r"\s+", " ", _read("docs/todo/visual-scholarship-integration.md")
    )
    release = re.sub(
        r"\s+", " ", _read("docs/todo/v1-1-release-publication.md")
    )
    handoff = re.sub(
        r"\s+", " ", _read("docs/todo/release-documentation-handoff.md")
    )

    public_url = "https://github.com/ActiveInferenceInstitute/Active_Fedference"
    assert public_url in roadmap
    assert public_url in index
    assert public_url in release
    assert public_url in handoff
    assert "`origin` remote, `docxology/active_fedference`" in roadmap
    assert "private/interim evidence remote" in release
    assert "private/interim evidence remote" in handoff
    assert "reviewed `codex/*` branches and normal merge commits" in index
    assert "merge only through a normal merge commit" in visual
    assert "direct, forced, squashed, rebased, stale-head, or check-bypassing" in roadmap
    assert "private synchronization is non-forced" in release
    assert "tree exactly equal to final public `main` after MIN-3 closes" in release


def test_release_dag_places_identity_merge_before_final_certification() -> None:
    roadmap = _read("TODO.md")
    index = re.sub(r"\s+", " ", _read("docs/todo/README.md"))
    release = _read("docs/todo/v1-1-release-publication.md")
    normalized_release = re.sub(r"\s+", " ", release)
    ladder = re.sub(
        r"\s+", " ", _read("docs/todo/release-and-verification-ladder.md")
    )
    phase_plan = re.sub(
        r"\s+", " ", _read("docs/todo/scholarship-and-phase-plan.md")
    )

    ordered_markers = (
        "complete the separate accessible Template renderer review",
        "candidate-specific portion of MIN-2",
        "merge it normally into public `main`",
        "run MED-5A",
        "certify that exact final public-main merge commit",
        "run MED-5B's publication phase",
        "complete and normally merge MIN-3",
        "finish MED-5B by synchronizing",
        "execute MAJ-8",
        "before MAJ-6 may inspect",
    )
    positions = [roadmap.index(marker) for marker in ordered_markers]
    assert positions == sorted(positions)
    for heading in (
        "### MED-5A — linked draft and release-identity integration",
        "### MIN-2 bridge — certify the exact final main commit",
        "### MED-5B — immutable publication and final private synchronization",
    ):
        assert heading in release
    assert "MIN-2 is not a prerequisite for creating or merging that PR" in ladder
    assert (
        "MIN-3 follows verified publication, precedes the single private synchronization"
        in normalized_release
    )
    assert "removal of its closed forward-backlog entry" in normalized_release
    assert "completed MIN-3 documentation handoff" in roadmap
    assert "completion of MIN-3's public documentation" in re.sub(
        r"\s+", " ", _read("docs/todo/adaptive-robustness-calibration.md")
    )
    assert "does not retrofit an already published release payload" in normalized_release
    handoff = re.sub(
        r"\s+", " ", _read("docs/todo/release-documentation-handoff.md")
    )
    assert "verification and discovery handoff" in handoff
    assert "Never patch the existing tag, GitHub asset, or published Zenodo bytes" in handoff
    assert "leave MIN-3 open and route the correction through a new reviewed" in handoff
    assert "before the single private-main synchronization" in handoff
    for surface in ("tagged source", "wheel", "sdist", "PDF", "HTML"):
        assert surface in release

    summary_sequences = (
        (
            index,
            (
                "MED-5B tag/assets/Zenodo",
                "MIN-3 closure",
                "before private sync",
            ),
        ),
        (
            phase_plan,
            (
                "verified GitHub/Zenodo publication",
                "MIN-3 documentation handoff",
                "one-time private synchronization",
                "MAJ-8 calibration freeze",
            ),
        ),
        (
            ladder,
            (
                "MED-5B owns tag, GitHub, and Zenodo publication",
                "MIN-3 then verifies and closes",
                "private synchronization",
                "MAJ-8 begins",
            ),
        ),
    )
    for surface, markers in summary_sequences:
        marker_positions = [surface.index(marker) for marker in markers]
        assert marker_positions == sorted(marker_positions)


def test_sanitized_history_policy_is_fixed_across_release_surfaces() -> None:
    ladder = re.sub(
        r"\s+", " ", _read("docs/todo/release-and-verification-ladder.md")
    )
    visual = re.sub(
        r"\s+", " ", _read("docs/todo/visual-scholarship-integration.md")
    )
    isa = re.sub(r"\s+", " ", _read("ISA.md"))

    for text in (ladder, visual, isa):
        assert "four-commit sanitized replay" in text
        assert "private evidence branch remains unchanged" in text
        assert "history-policy change requires new approval" in text
    assert "refreshed public `main`" in ladder
    assert "refreshed public `main`" in visual
    assert "refreshed `public/main`" in isa
    assert "cherry-picking of private commits" in visual


def test_isc89_requires_both_final_bundle_external_lanes() -> None:
    reproduction = re.sub(
        r"\s+", " ", _read("docs/todo/independent-reproduction.md")
    )
    roadmap = re.sub(r"\s+", " ", _read("TODO.md"))
    isa = re.sub(r"\s+", " ", _read("ISA.md"))

    assert "criterion is conjunctive" in reproduction
    assert "identified independent-human/Advisor review" in reproduction
    assert "genuinely different-vendor execution" in reproduction
    assert "Either lane alone" in reproduction
    assert "exact final SHA and artifact set" in reproduction
    assert "Both an identified independent-human/Advisor review" in roadmap
    assert "Advisor + cross-vendor" in isa
    assert "final artifact set" in isa


def test_governance_reference_is_not_an_active_backlog_item() -> None:
    plan = _read(GOVERNING_TODO_PATH)
    roadmap = _read("TODO.md")
    index = _read("docs/todo/README.md")

    assert "Document class: Governing reference" in plan
    assert "State: Active policy (not a backlog item)" in plan
    assert "| GOV-1 |" not in roadmap
    assert "| GOV-1 |" not in index
    assert "## Governing reference" in index
    assert (ROOT / GOVERNING_TODO_PATH).is_file()


def test_todo_index_is_derived_and_reconciles_legacy_ids() -> None:
    index = _read("docs/todo/README.md")
    roadmap = _read("TODO.md")

    assert "index currently contains thirteen" not in index.casefold()
    assert "active-page set is derived from the rows above" in index
    assert "## Identifier reconciliation" in index
    assert "`MIN-2` means the clone-correct v1.1" in index
    assert "`MAJ-7` means the faithful Friston" in index
    assert "identifier alone never implies status" in index
    assert "Minor item may require broad execution" in index
    assert "complete and normally merge MIN-3" in roadmap


def test_future_scientific_lanes_block_on_numeric_design_freezes() -> None:
    required_terms = {
        "docs/todo/faithful-fedgvi-bnn-lane.md": (
            "implementation remains blocked",
            "numeric smallest effect of interest",
            "seed-level MCSE",
            "multiplicity",
            "maximum rounds",
        ),
        "docs/todo/faithful-friston-protocol-replication.md": (
            "Execution remains blocked",
            "source-extraction artifact",
            "independent unit",
            "numerical tolerance",
            "maximum compute budget",
        ),
        "docs/todo/beyond-discrete-categorical-state-spaces.md": (
            "Confirmatory implementation remains blocked",
            "numerical held-out log-score SOEI",
            "world-level MCSE",
            "tracking horizon",
            "total compute budget",
        ),
        "docs/todo/deeper-hierarchy-task-family.md": (
            "confirmatory implementation remains blocked",
            "fixed intersection of exactly two named task units",
            "no population-level task inference",
            "numeric SOEI and MCSE",
            "maximum total budget",
        ),
        "docs/todo/true-multi-machine-federation.md": (
            "Execution of either tranche remains blocked",
            "exact round count",
            "comparison metric and tolerance",
            "clock-skew envelope",
            "maximum runtime/resource budget",
        ),
    }
    for path, terms in required_terms.items():
        text = re.sub(r"\s+", " ", _read(path))
        for term in terms:
            assert term.casefold() in text.casefold(), f"{path} missing {term!r}"


def test_maj8_freezes_dataset_first_bootstrap_and_cell_index_order() -> None:
    design = re.sub(
        r"\s+", " ", _read("docs/todo/adaptive-robustness-calibration.md")
    )

    assert "dataset-first hierarchical bootstrap" in design
    assert "draw three dataset strata with replacement" in design
    assert "draw that dataset's pilot seeds with replacement" in design
    assert "row count never changes the pack estimand" in design
    assert "cell_index = 4*d + 2*m + r" in design
    assert "WDBC/Dry Bean/Banknote" in design
    assert "m=0` for robust" in design
    assert "r=0` for rate 0.5" in design
    assert 'numpy.quantile(..., 0.05, method="linear")' in design
    assert "strictly greater than -0.010" in design
    assert "equality fails" in design
    assert "compute `E32`" in design
    assert "then compute `S32`" in design
    assert "`E64` supersedes rather than augments `E32`" in design
    assert "then compute `S64`" in design
    assert "Eligibility is never recomputed inside a stability resample" in design
    assert "Equality at the 80% threshold passes" in design
    assert "no third tranche is permitted" in design
    assert "retain it unchanged in every stability resample" in design
    assert "no eligible candidate" in design
    assert "must not reselect it" in design
    assert "numpy.std(..., ddof=1)" in design
    assert 'numpy.quantile(sd_draws, 0.95, method="linear")' in design
    assert "finite zero variance is permitted" in design
    assert "fewer than two rows or any nonfinite" in design
    assert "N_raw = ceil((max s_U / 0.005)^2)" in design
    assert "round upward to a multiple of 16" in design
    assert "floor of 32 and cap of 256" in design
    assert "upward-rounded uncapped requirement exceeds 256" in design
    assert "six stress contrasts" in design
    assert "fixed 32-seed schedule" in design
    assert "bootstrap_seed(cell) = 340_000_000 + 100*cell_index" in design
    for seed_rule in (
        "s_u_bootstrap_seed(c_primary) = 340_000_000 + 100*c_primary",
        "eligibility_bootstrap_seed(s,m,d) = 341_000_000 + 100_000*s + 10_000*m + 100*d",
        "stability_bootstrap_seed(s) = 342_000_000 + 100_000*s",
        "primary_interval_seed(c_primary) = 343_000_000 + 100*c_primary",
        "primary_null_seed(c_primary) = 343_000_000 + 100*c_primary + 1",
        "clean_interval_seed(c_clean) = 344_000_000 + 100*c_clean",
        "clean_null_seed(c_clean) = 344_000_000 + 100*c_clean + 1",
        "stress_interval_seed(c_stress) = 345_000_000 + 100*c_stress",
    ):
        assert seed_rule in design
    assert "shared by every candidate in that method/dataset/tranche" in design
    assert "shared by every robust and variational candidate" in design
    assert "schema validation proves pairwise non-overlap" in design

    seed_namespaces = {
        "s_u": {340_000_000 + 100 * cell for cell in range(12)},
        "eligibility": {
            341_000_000 + 100_000 * tranche + 10_000 * method + 100 * dataset
            for tranche in range(2)
            for method in range(2)
            for dataset in range(3)
        },
        "stability": {342_000_000 + 100_000 * tranche for tranche in range(2)},
        "primary_interval": {343_000_000 + 100 * cell for cell in range(12)},
        "primary_null": {343_000_000 + 100 * cell + 1 for cell in range(12)},
        "clean_interval": {344_000_000 + 100 * cell for cell in range(6)},
        "clean_null": {344_000_000 + 100 * cell + 1 for cell in range(6)},
        "stress_interval": {345_000_000 + 100 * cell for cell in range(6)},
    }
    legacy_bootstrap_seeds = {340_000_000 + 100 * cell for cell in range(12)}
    assert legacy_bootstrap_seeds == seed_namespaces["s_u"]
    all_seeds: set[int] = set()
    for namespace, seeds in seed_namespaces.items():
        assert len(seeds) > 1, namespace
        assert all_seeds.isdisjoint(seeds), namespace
        all_seeds.update(seeds)


def test_maj8_phase_plan_uses_fixed_pack_estimand_without_population_inference() -> None:
    plan = re.sub(
        r"\s+", " ", _read("docs/todo/scholarship-and-phase-plan.md")
    )

    assert "fixed three-dataset × two-contaminated-rate pack" in plan
    assert "dataset is the highest external unit" in plan
    assert "paired pilot seeds are nested" in plan
    assert "not dataset-population inference" in plan
    assert "Mean held-out log score per independent calibration world" not in plan


def test_maj6_declares_external_unit_and_exact_decision_algorithm() -> None:
    campaign = re.sub(
        r"\s+", " ", _read("docs/todo/external-benchmark-domain-pilot.md")
    )
    registry = _read("src/fedference/research_registry.py")

    assert "dataset is the external replication/generalization unit" in campaign
    assert "confirmatory seed is the paired Monte Carlo" in campaign
    assert "fixed intersection of the three named datasets" in campaign
    assert "Do not publish a pooled dataset-population p-value" in campaign
    assert "common `N` for the 12 primary contrasts and the six clean" in campaign
    assert "accuracy and expected calibration error from those same rows" in campaign
    assert "first 32 frozen confirmatory seeds per dataset" in campaign
    assert "3 datasets × 2 methods = 6 descriptive stress contrasts" in campaign
    assert "do not extend it to the common `N`" in campaign
    assert 'independent_unit="licensed external dataset, with seeds nested"' in registry
    for decision_rule in (
        r"$H_0:\Delta\leq0$",
        r"$H_0:\Delta\leq-0.010$",
        "5,000 bootstrap samples",
        "2.5th and 97.5th percentile",
        "delta_null = delta - mean(delta) + delta_0",
        "count(mean(delta_null_bootstrap) >= mean(delta))",
        "not the fraction of ordinary bootstrap means crossing the null",
        "equality remains in the null-compatible tail",
        "canonical dataset/method/rate cell key",
        "Stop rejection at the first failed Holm threshold",
        "mean improvement at least 0.010",
        "n = delta.size",
        "n == frozen_common_N",
        "numpy.std(delta, ddof=1) / numpy.sqrt(n)",
        "a row-count mismatch, or any nonfinite contrast or MCSE is a hard stop",
        "Every primary and clean cell must have finite MCSE at most 0.005",
        "equality passes",
        "do not apply the 0.005 ceiling",
        "singleton-MCSE convention remains a compatibility smoke/pilot behavior",
        "All 18 primary and clean cells must contain exactly the frozen common `N`",
    ):
        assert decision_rule in campaign


def test_maj4_local_and_physical_acceptance_are_separate() -> None:
    federation = _read("docs/todo/true-multi-machine-federation.md")

    for heading in (
        "### MAJ-4A — authenticated local-container emulator",
        "### MAJ-4B — physical distinct-host validation",
    ):
        assert heading in federation
    assert "Physical hosts are not required to close MAJ-4A" in federation
    assert "local-container transported consensus" in federation
    assert "physical-host transported consensus" in federation
    assert "host topology" in federation
    assert "clock-skew envelope" in federation
    assert "No cross-host execution is required for MAJ-4A acceptance" in federation


def test_parked_tracks_have_no_unconditional_version_or_priority() -> None:
    plan = _read(GOVERNING_TODO_PATH)
    roadmap = _read("TODO.md")

    assert "Later v1.x" not in plan
    assert "prioritize streaming" not in plan
    assert "no version, order, or execution authority is assigned" in plan
    assert "No parked track has a version, ordering priority" in roadmap


def test_future_source_links_use_current_primary_records() -> None:
    plan = _read(GOVERNING_TODO_PATH)
    audit = _read("docs/research/literature-audit.md")
    combined = plan + audit

    assert "https://proceedings.mlr.press/v84/futami18a.html" in plan
    assert "https://joss.theoj.org/papers/10.21105/joss.05161" in plan
    assert "*Variational Inference based on Robust Divergences*" in audit
    assert "proceedings.mlr.press/v80/futami18a.html" not in combined
    assert "baggepinnen.github.io/RxInfer.jl/stable/" not in combined


def test_todo_command_floor_uses_locked_environment() -> None:
    roadmap = _read("TODO.md")

    assert "uv run --locked ruff check" in roadmap
    assert "uv run --locked mypy src/" in roadmap
    assert "uv run ruff check" not in roadmap
    assert "uv run mypy src/" not in roadmap


def test_source_tree_has_no_unscoped_roadmap_markers() -> None:
    roadmap_pattern = re.compile(r"TODO:|\bFIXME\b|\bXXX\b")
    offenders: list[str] = []
    for base_dir in (ROOT / "src", ROOT / "scripts"):
        if not base_dir.exists():
            continue
        for path in sorted(base_dir.rglob("*.py")):
            relpath = path.relative_to(ROOT).as_posix()
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if roadmap_pattern.search(line):
                    offenders.append(f"{relpath}:{line_number}")
    assert offenders == []


def test_manuscript_syntax_registry_references_live_files() -> None:
    text = _read("manuscript/SYNTAX.md")
    owner_files = sorted(set(re.findall(r"`([^`]+\.md)`", text)))
    missing = [
        owner for owner in owner_files
        if not owner.startswith("../../")
        and not owner.startswith("docs/guides/")
        and not (ROOT / "manuscript" / owner).exists()
    ]
    assert missing == []
    assert "Ten figures" not in text
    assert "between 80–100%" in text
    assert "structured long description" in text
    assert "Always set `width=80%`" not in text
    assert "supply the long-form figure description" not in text


def test_figure_registry_matches_embeds_and_generator_outputs() -> None:
    """Reject a registry that names an owner but points at the wrong artifact.

    The older gate checked only that owner Markdown files existed. That could
    pass while the registry, manuscript embed, and generator default disagreed
    on an extension. This negative-control-resistant check compares all three
    surfaces for every live manuscript figure.
    """
    registry_path = ROOT / "output" / "figures" / "figure_registry.json"
    assert registry_path.exists(), "shipped figure registry is required"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registry.get("schema_version") == "1.2"
    entries = {entry["label"]: entry for entry in registry.get("figures", [])}
    embed_re = re.compile(
        r"!\[.*?\]\((?P<path>[^)]*output/figures/[^)]+)\)"
        r"\{#(?P<label>fig:[A-Za-z0-9_\-]+)",
        re.DOTALL,
    )
    embeds = {}
    for section in sorted((ROOT / "manuscript").glob("[0-9S]*.md")):
        if section.name in {"AGENTS.md", "README.md", "SYNTAX.md", "preamble.md"}:
            continue
        for match in embed_re.finditer(section.read_text(encoding="utf-8")):
            embeds[match.group("label")] = Path(match.group("path")).name
    assert set(embeds) == set(entries)
    assert len(entries) == len(embeds)
    from figures import figure_metadata

    for label, filename in embeds.items():
        entry = entries[label]
        assert entry["filename"] == filename, label
        assert Path(entry["path"]).name == filename, label
        generator = entry["generated_by"]
        source = ROOT / "src" / "figures" / f"{generator}.py"
        assert source.exists(), f"missing generator for {label}: {generator}"
        for field, value in figure_metadata(generator).items():
            assert entry.get(field) == value, f"registry metadata drift for {label}: {field}"
        for field in (
            "status",
            "source_relation",
            "estimand",
            "unit",
            "uncertainty",
            "replication_unit",
        ):
            assert entry.get(field), f"missing figure metadata {field}: {label}"
        if generator == "moving_world":
            source_text = source.read_text(encoding="utf-8")
            assert 'filename: str = "moving_world.png"' in source_text


def test_rendered_theorem_blocks_do_not_use_markdown_code_spans() -> None:
    offenders = []
    for path in sorted((ROOT / "manuscript").glob("*.md")):
        if path.name in {"AGENTS.md", "README.md", "SYNTAX.md", "preamble.md"}:
            continue
        text = path.read_text(encoding="utf-8")
        pattern = r"\\begin\{(definition|theorem|lemma|proposition|corollary)\}(.*?)\\end\{\1\}"
        for match in re.finditer(pattern, text, re.S):
            if "`" in match.group(2):
                line_number = text[: match.start()].count("\n") + 1
                offenders.append(f"manuscript/{path.name}:{line_number}")
    assert offenders == []


def test_display_math_does_not_use_code_spans_or_texttt() -> None:
    offenders = []
    for path in sorted((ROOT / "manuscript").glob("*.md")):
        if path.name in {"AGENTS.md", "README.md", "SYNTAX.md", "preamble.md"}:
            continue
        parts = path.read_text(encoding="utf-8").split("$$")
        for index, block in enumerate(parts[1::2], start=1):
            if "`" in block or "\\texttt" in block:
                offenders.append(f"manuscript/{path.name}:display_math_{index}")
    assert offenders == []


def test_theorem_blocks_do_not_contain_pandoc_equation_labels() -> None:
    offenders = []
    pattern = r"\\begin\{(definition|theorem|lemma|proposition|corollary)\}(.*?)\\end\{\1\}"
    for path in sorted((ROOT / "manuscript").glob("*.md")):
        if path.name in {"AGENTS.md", "README.md", "SYNTAX.md", "preamble.md"}:
            continue
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(pattern, text, re.S):
            if "$$ {#eq:" in match.group(2):
                line_number = text[: match.start()].count("\n") + 1
                offenders.append(f"manuscript/{path.name}:{line_number}")
    assert offenders == []


def test_proposition_sections_define_environment_for_slides() -> None:
    offenders = []
    for path in sorted((ROOT / "manuscript").glob("*.md")):
        if path.name in {"AGENTS.md", "README.md", "SYNTAX.md", "preamble.md"}:
            continue
        text = path.read_text(encoding="utf-8")
        if "\\begin{proposition}" in text and "\\ifcsname proposition\\endcsname" not in text:
            offenders.append(f"manuscript/{path.name}")
    assert offenders == []


def test_manuscript_latex_conditionals_do_not_look_like_citations() -> None:
    offenders = []
    for path in sorted((ROOT / "manuscript").glob("*.md")):
        if path.name in {"AGENTS.md", "README.md", "SYNTAX.md", "preamble.md"}:
            continue
        text = path.read_text(encoding="utf-8")
        if "\\@ifundefined" in text:
            offenders.append(f"manuscript/{path.name}")
    assert offenders == []


def test_claim_ledger_is_active_fedference_owned() -> None:
    text = _read("data/claim_ledger.yaml")
    assert "template_code_project" not in text
    assert "Convergence Analysis of Gradient Descent Optimization" not in text

    payload = yaml.safe_load(text)
    claims = payload["claims"]
    assert claims
    assert all(claim["freshness"] == "active" for claim in claims)
    assert all(claim["source_tier"] == "claim_ledger" for claim in claims)

    ledger_numbers = {float(claim["value"]) for claim in claims if claim["kind"] == "number"}
    required_numbers = {
        -1.0,
        2.0,
        6.0,
        7.0,
        8.0,
        9.0,
        11.0,
        12.0,
        13.0,
        16.0,
        18.0,
        0.40,
        0.69,
        0.75,
        0.85,
        0.95,
        0.99,
    }
    assert required_numbers <= ledger_numbers


def test_anchor_source_audit_table_is_not_interrupted_by_prose() -> None:
    text = _read("docs/research/literature-audit.md")
    section = text.split("## Anchor sources", 1)[1].split("## Sources governing the next phases", 1)[0]
    table_start = section.index("| Source |")
    table_end = section.index("\n\nThe pinned FashionMNIST shell")
    table_lines = section[table_start:table_end].splitlines()

    assert len(table_lines) >= 4
    assert all(line.startswith("|") for line in table_lines if line.strip())
    assert "Mildner, Giampouras & Damoulas" in section[table_start:table_end]
    assert "Efron & Tibshirani" in section[table_start:table_end]
