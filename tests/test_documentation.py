"""Tests for source-owned API-reference and glossary generation."""

from __future__ import annotations

from pathlib import Path

from documentation import (
    API_REFERENCE_TEMPLATE,
    _build_glossary_markdown,
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

    glossary = result["glossary"]
    assert glossary is not None
    glossary_path = Path(glossary)
    assert glossary_path.exists()
    glossary_text = glossary_path.read_text(encoding="utf-8")
    assert "`sample`" in glossary_text
    assert "`add`" in glossary_text
    assert "Return the sum of two integers." in glossary_text


def test_run_api_doc_generation_degrades_without_src(tmp_path: Path) -> None:
    docs = tmp_path / "output" / "docs"
    docs.mkdir(parents=True)
    stale = docs / "api_glossary.md"
    stale.write_text("stale API claim\n", encoding="utf-8")

    result = run_api_doc_generation(tmp_path)
    api_reference = result["api_reference"]
    assert api_reference is not None
    assert Path(api_reference).exists()
    assert api_reference.endswith("api_reference.md")
    assert result["glossary"] is None
    assert not stale.exists()


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
    assert (docs / "api_glossary.md").is_dir()


def test_shipped_api_reference_matches_source_template() -> None:
    project_root = Path(__file__).resolve().parents[1]
    shipped = project_root / "output" / "docs" / "api_reference.md"
    assert shipped.read_text(encoding="utf-8") == API_REFERENCE_TEMPLATE


def test_shipped_glossary_matches_source_inventory() -> None:
    project_root = Path(__file__).resolve().parents[1]
    shipped = project_root / "output" / "docs" / "api_glossary.md"
    expected = _build_glossary_markdown(project_root / "src")
    actual = shipped.read_text(encoding="utf-8")
    assert actual == expected
    assert "Product-of-experts = Friston Eq. 7" not in actual
