from __future__ import annotations

import json
import re
import urllib.parse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

DOC_CONTRACT_FILES = (
    "AGENTS.md",
    "README.md",
    "STANDALONE.md",
    "TODO.md",
    "docs/AGENTS.md",
    "docs/README.md",
    "docs/core/architecture.md",
    "docs/core/experiments-and-artifacts.md",
    "docs/development/agent_instructions.md",
    "docs/development/quickstart.md",
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
    "docs/research/literature-audit.md",
    "docs/research/manuscript-claim-audit.md",
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


def _todo_doc_files() -> tuple[str, ...]:
    todo_dir = ROOT / "docs" / "todo"
    if not todo_dir.exists():
        return ()
    return tuple(
        path.relative_to(ROOT).as_posix()
        for path in sorted(todo_dir.glob("*.md"))
    )


def _repository_guide_files() -> tuple[str, ...]:
    """Discover every source-owned AGENTS/README contract, not a hand-picked subset."""
    guides: list[str] = []
    for path in ROOT.rglob("*.md"):
        relative = path.relative_to(ROOT)
        if path.name not in {"AGENTS.md", "README.md"}:
            continue
        if relative.parts[0] in {".pytest_cache", ".venv", "output"}:
            continue
        guides.append(relative.as_posix())
    return tuple(sorted(guides))


def _contract_files() -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                *DOC_CONTRACT_FILES,
                *_repository_guide_files(),
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


def test_documented_local_markdown_links_resolve_inside_standalone_repo() -> None:
    """Prevent docs from linking back to absent template-repo guides."""
    link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    allowed_placeholder_targets = {
        "../output/figures/name.png",
        "../output/figures/NAME.png",
    }
    offenders: list[str] = []
    for path in _contract_files():
        source = ROOT / path
        for line_number, line in enumerate(_read(path).splitlines(), start=1):
            for match in link_pattern.finditer(line):
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
    assert len(blocks) >= 5


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
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        if ignored_directories.intersection(path.relative_to(ROOT).parts):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        if any(marker in text for marker in markers):
            offenders.append(path.relative_to(ROOT).as_posix())
    assert offenders == []


def test_release_metadata_matches_public_project_identity() -> None:
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
    assert "10.5281/zenodo.21864004" in metadata

    config = yaml.safe_load(_read("manuscript/config.yaml"))
    assert ":" not in config["paper"]["title"]
    assert config["paper"]["subtitle"]
    assert ":" not in config["paper"]["subtitle"]
    assert config["publication"]["github_repository"] == "https://github.com/ActiveInferenceInstitute/Active_Fedference"
    assert config["publication"]["doi"] == "10.5281/zenodo.21864004"
    assert config["publication"]["date_released"] == "2026-08-10"
    assert config["metadata"]["license"] == "MIT"
    assert "active inference" in config["keywords"]
    assert "FedGVI" in config["keywords"]
    assert yaml.safe_load(_read("CITATION.cff"))["identifiers"] == [
        {"type": "doi", "value": "10.5281/zenodo.21864004"}
    ]
    assert yaml.safe_load(_read("CITATION.cff"))["date-released"] == "2026-08-10"
    assert json.loads(_read(".zenodo.json"))["doi"] == "10.5281/zenodo.21864004"
    assert json.loads(_read(".zenodo.json"))["publication_date"] == "2026-08-10"
    assert json.loads(_read("codemeta.json"))["identifier"] == "https://doi.org/10.5281/zenodo.21864004"
    assert json.loads(_read("codemeta.json"))["dateModified"] == "2026-08-10"


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
    receipt = sequence.index("uv run --locked --extra dev python scripts/validate_test_coverage.py")
    final_hydration = sequence.index(
        "uv run --locked python scripts/z_generate_manuscript_variables.py",
        receipt,
    )
    second_render = sequence.index(render, final_hydration)
    package = sequence.index("uv run --locked python scripts/prepare_web_package.py")
    record = sequence.index("uv run --locked python scripts/record_pipeline_stage.py render")

    assert sequence.count(render) == 2
    assert provisional < first_render < receipt < final_hydration < second_render
    assert second_render < package < record


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
        "Tagged: no",
        "must not be described as tagged",
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
    linked_pages = set(re.findall(r"\]\((docs/todo/[a-z0-9-]+\.md)\)", todo_text))
    scoped_pages = {
        path
        for path in _todo_doc_files()
        if path != "docs/todo/README.md"
    }
    assert linked_pages == scoped_pages
    assert "docs/todo/README.md" in todo_text
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
    for page in _todo_doc_files():
        if page == "docs/todo/README.md":
            continue
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
    linked_pages = {
        f"docs/todo/{name}"
        for name in re.findall(r"\]\(([a-z0-9-]+\.md)\)", readme_text)
    }
    scoped_pages = {
        path
        for path in _todo_doc_files()
        if path != "docs/todo/README.md"
    }
    missing = scoped_pages - linked_pages
    nonexistent = {
        page
        for page in linked_pages
        if not (ROOT / page).exists()
    }
    assert missing == set()
    assert nonexistent == set()


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
    assert registry.get("schema_version") == "1.1"
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
