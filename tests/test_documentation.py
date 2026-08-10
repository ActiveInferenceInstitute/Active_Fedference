"""Tests for the API-documentation workflow module.

No mocks: ``run_api_doc_generation`` writes real markdown files into a throwaway
``tmp_path`` project root, and we assert on the files' presence and content. The
static API reference is always written; the infrastructure glossary is
best-effort (degrades to ``None`` rather than raising).
"""

from __future__ import annotations

from pathlib import Path

from documentation import (
    API_REFERENCE_TEMPLATE,
    build_api_reference_markdown,
    run_api_doc_generation,
)


def test_build_api_reference_markdown_is_the_template() -> None:
    assert build_api_reference_markdown() == API_REFERENCE_TEMPLATE
    assert "API Reference" in build_api_reference_markdown()


def test_run_api_doc_generation_writes_reference(tmp_path: Path) -> None:
    # A minimal src/ so the glossary index has something to walk.
    src = tmp_path / "src"
    src.mkdir()
    (src / "__init__.py").write_text("", encoding="utf-8")
    (src / "sample.py").write_text(
        '"""Sample module."""\n\n\ndef add(a: int, b: int) -> int:\n'
        '    """Return the sum of two integers."""\n    return a + b\n',
        encoding="utf-8",
    )

    result = run_api_doc_generation(tmp_path)

    api_ref = tmp_path / "output" / "docs" / "api_reference.md"
    assert api_ref.exists()
    api_reference = result["api_reference"]
    assert api_reference is not None
    assert api_reference == str(api_ref)
    assert api_ref.read_text(encoding="utf-8") == API_REFERENCE_TEMPLATE

    # Glossary is best-effort: either written or explicitly None — never raised.
    glossary = result["glossary"]
    if glossary is not None:
        assert Path(glossary).exists()
        assert Path(glossary).name == "api_glossary.md"


def test_run_api_doc_generation_degrades_without_src(tmp_path: Path) -> None:
    # No src/ dir: glossary build fails gracefully, reference still written.
    result = run_api_doc_generation(tmp_path)
    api_reference = result["api_reference"]
    assert api_reference is not None
    assert Path(api_reference).exists()
    # The static reference never depends on src/, so it always succeeds.
    assert api_reference.endswith("api_reference.md")


def test_run_api_doc_generation_glossary_failure_is_swallowed(tmp_path: Path) -> None:
    # Real (no-mock) failure path: a pre-existing directory occupying the
    # glossary's target filename makes the guarded write_text raise OSError.
    # The workflow must degrade glossary -> None and still write the reference.
    docs = tmp_path / "output" / "docs"
    docs.mkdir(parents=True)
    (docs / "api_glossary.md").mkdir()  # collide: target is now a directory

    result = run_api_doc_generation(tmp_path)

    assert result["glossary"] is None
    api_reference = result["api_reference"]
    assert api_reference is not None
    assert Path(api_reference).exists()
