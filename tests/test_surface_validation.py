"""Reviewer-surface validation tests."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from publication.surface_validation import (
    _log_findings,
    _pdf_structure_findings,
    _pdf_tagging_findings,
    _pdf_text_findings,
    _pdfinfo_tagging_status,
    _publication_text_findings,
    _qpdf_tagging_status,
    _slide_inventory_findings,
    validate_rendered_surfaces,
)


def test_publication_text_findings_rejects_unresolved_manuscript_tokens() -> None:
    findings = _publication_text_findings(
        Path("paper.pdf"),
        "Result {{TEST_COUNT}}; see [@sec:method] and ??\n",
    )

    assert any("unresolved manuscript token" in finding for finding in findings)
    assert any("raw Pandoc" in finding for finding in findings)
    assert any("unresolved reference" in finding for finding in findings)


def test_pdfinfo_tagging_status_parses_structural_fields() -> None:
    tagged, language = _pdfinfo_tagging_status("Tagged: yes\nLanguage: en\n")
    assert tagged
    assert language == "en"

    tagged, language = _pdfinfo_tagging_status("Tagged: no\n")
    assert not tagged
    assert language is None


def test_qpdf_tagging_status_reads_catalog_language_and_structure_tree() -> None:
    tagged_pdf_json = '{"objects":{"trailer":{"/Lang":"u:en","/StructTreeRoot":"2 0 R"}}}'
    assert _qpdf_tagging_status(tagged_pdf_json) == (True, True)
    assert _qpdf_tagging_status("not JSON") == (False, False)


def test_pdf_tagging_is_not_required_for_legacy_surface_profiles(tmp_path: Path) -> None:
    assert _pdf_tagging_findings(tmp_path / "paper.pdf", required=False) == []


def test_log_findings_accepts_small_hbox_but_blocks_material_layout_errors(tmp_path: Path) -> None:
    log = tmp_path / "deck.log"
    log.write_text(
        "Overfull \\hbox (0.4pt too wide) detected\n"
        "Overfull \\hbox (1.1pt too wide) detected\n"
        "Overfull \\vbox (4.0pt too high) detected\n"
        "LaTeX Warning: Reference `sec:x' undefined.\n",
        encoding="utf-8",
    )

    findings = _log_findings(log)
    assert len(findings) == 3
    assert all("0.4pt" not in finding for finding in findings)


def test_validate_rendered_surfaces_fails_when_outputs_are_absent(tmp_path: Path) -> None:
    result = validate_rendered_surfaces(tmp_path)

    assert not result.ok
    assert not result.manuscript_pdf
    assert result.slide_pdfs == 0
    assert any("missing combined manuscript PDF" in finding for finding in result.findings)
    assert any("missing generated slide PDFs" in finding for finding in result.findings)
    assert any("missing generated slide TeX" in finding for finding in result.findings)
    assert any("missing generated slide logs" in finding for finding in result.findings)


def test_validate_rendered_surfaces_propagates_web_accessibility_findings(
    tmp_path: Path,
) -> None:
    web = tmp_path / "output" / "web"
    web.mkdir(parents=True)
    (web / "index.html").write_text(
        "<html><head><title>Example</title></head><body>content</body></html>",
        encoding="utf-8",
    )

    result = validate_rendered_surfaces(tmp_path)

    assert not result.ok
    assert any("main element with id='main-content'" in finding for finding in result.findings)
    assert any(".skip-link" in finding for finding in result.findings)


def test_slide_inventory_requires_matching_pdf_tex_log_triplets(tmp_path: Path) -> None:
    pdf = tmp_path / "section_slides.pdf"
    orphan_log = tmp_path / "orphan_slides.log"

    findings = _slide_inventory_findings(tmp_path, [pdf], [], [orphan_log])

    assert findings == [
        f"missing slide TeX source: {tmp_path / 'section_slides.tex'}",
        f"missing slide LaTeX log: {tmp_path / 'section_slides.log'}",
        f"orphan slide LaTeX log without PDF: {orphan_log}",
    ]


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="pdftotext not installed")
def test_pdf_text_findings_accepts_current_clean_probe() -> None:
    project_root = Path(__file__).resolve().parent.parent
    probe = project_root / "output" / "slides" / "19_results_robustness_slides.pdf"
    if not probe.exists():
        pytest.skip("clean slide probe has not been rendered yet")

    assert _pdf_text_findings(probe) == []


@pytest.mark.skipif(shutil.which("qpdf") is None, reason="qpdf not installed")
def test_pdf_structure_findings_accepts_current_combined_manuscript() -> None:
    project_root = Path(__file__).resolve().parent.parent
    probe = project_root / "output" / "pdf" / "active_fedference_combined.pdf"
    if not probe.exists():
        pytest.skip("combined manuscript has not been rendered yet")

    assert _pdf_structure_findings(probe) == []
