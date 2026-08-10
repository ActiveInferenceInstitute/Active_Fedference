"""Reviewer-surface validation for generated PDF, slide, and web artifacts.

Source tests can prove that a renderer was called, but they cannot prove that
the shipped surface is readable. This module checks the combined manuscript,
the generated Beamer source/log/PDF triplets, and the web package. It
intentionally treats structural PDF failures, unresolved references, raw
Pandoc markers, missing characters, and material overfull boxes as release
findings. The web branch also enforces the deterministic HTML accessibility
contract; PDF structure/text checks do not imply tagged-PDF or PDF/UA status.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from publication.web_package import WebPackageValidation, validate_web_package

_RAW_PANDOC_RE = re.compile(r"\[@[A-Za-z0-9_:.#$%&+?<>~/-]+")
_UNRESOLVED_RE = re.compile(r"\?\?|\\(?:ref|eqref)\{[^}]+\}")
_UNRESOLVED_TOKEN_RE = re.compile(r"\{\{[A-Z][A-Z0-9_]*\}\}")
_MISSING_CHARACTER_RE = re.compile(r"Missing character", re.IGNORECASE)
_UNDEFINED_REFERENCE_RE = re.compile(
    r"(?:Reference|Citation).*undefined|There were undefined references",
    re.IGNORECASE,
)
_OVERFULL_VBOX_RE = re.compile(r"Overfull \\vbox")
_OVERFULL_HBOX_RE = re.compile(r"Overfull \\hbox \((?P<points>[0-9.]+)pt")


@dataclass(frozen=True)
class SurfaceValidation:
    """Aggregate result for generated reviewer-facing surfaces."""

    manuscript_pdf: bool
    manuscript_logs: int
    slide_pdfs: int
    slide_tex: int
    slide_logs: int
    findings: tuple[str, ...]
    web: WebPackageValidation

    @property
    def ok(self) -> bool:
        """Whether every checked surface passed its release-facing checks."""
        return (
            not self.findings
            and self.web.ok
            and self.manuscript_pdf
            and self.manuscript_logs > 0
            and self.slide_pdfs > 0
        )


def _log_findings(path: Path) -> list[str]:
    """Return material layout/reference findings from one LaTeX log."""
    findings: list[str] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        if _OVERFULL_VBOX_RE.search(line):
            findings.append(f"{path}:{line_number}: {line.strip()}")
            continue
        hbox = _OVERFULL_HBOX_RE.search(line)
        if hbox and float(hbox.group("points")) > 1.0:
            findings.append(f"{path}:{line_number}: {line.strip()}")
            continue
        if _MISSING_CHARACTER_RE.search(line) or _UNDEFINED_REFERENCE_RE.search(line):
            findings.append(f"{path}:{line_number}: {line.strip()}")
    return findings


def _publication_text_findings(path: Path, text: str) -> list[str]:
    """Reject raw citations, unresolved references, and token markers in text."""
    findings: list[str] = []
    if _RAW_PANDOC_RE.search(text):
        findings.append(f"{path}: raw Pandoc citation/reference marker in extracted text")
    if _UNRESOLVED_RE.search(text):
        findings.append(f"{path}: unresolved reference marker in extracted text")
    if _UNRESOLVED_TOKEN_RE.search(text):
        findings.append(f"{path}: unresolved manuscript token in extracted text")
    return findings


def _pdf_text_findings(path: Path) -> list[str]:
    """Extract one PDF's text and reject raw/unresolved publication markers."""
    if shutil.which("pdftotext") is None:
        return ["pdftotext is required for reviewer-surface PDF text validation"]
    result = subprocess.run(
        ["pdftotext", str(path), "-"],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if result.returncode != 0:
        return [f"{path}: pdftotext failed with exit code {result.returncode}"]
    return _publication_text_findings(path, result.stdout)


def _pdf_structure_findings(path: Path) -> list[str]:
    """Run qpdf's structural probe on one publication PDF."""
    if shutil.which("qpdf") is None:
        return ["qpdf is required for reviewer-surface PDF structural validation"]
    result = subprocess.run(
        ["qpdf", "--check", str(path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if result.returncode == 0:
        return []
    diagnostic = "\n".join(
        line.strip() for line in (*result.stdout.splitlines(), *result.stderr.splitlines()) if line.strip()
    )
    return [
        f"{path}: qpdf structural check failed with exit code {result.returncode}"
        + (f": {diagnostic}" if diagnostic else "")
    ]


def _slide_inventory_findings(
    slides_dir: Path,
    slide_pdfs: list[Path],
    slide_tex: list[Path],
    slide_logs: list[Path],
) -> list[str]:
    """Require one PDF/TeX/log triplet for every generated slide deck."""

    def stems(paths: list[Path]) -> set[str]:
        return {path.stem.removesuffix("_slides") for path in paths}

    pdf_stems = stems(slide_pdfs)
    findings: list[str] = []
    for label, suffix, artifact_stems in (
        ("TeX source", ".tex", stems(slide_tex)),
        ("LaTeX log", ".log", stems(slide_logs)),
    ):
        for stem in sorted(pdf_stems - artifact_stems):
            findings.append(f"missing slide {label}: {slides_dir / f'{stem}_slides{suffix}'}")
        for stem in sorted(artifact_stems - pdf_stems):
            findings.append(f"orphan slide {label} without PDF: {slides_dir / f'{stem}_slides{suffix}'}")
    return findings


def validate_rendered_surfaces(project_root: str | Path) -> SurfaceValidation:
    """Validate the combined manuscript, slide PDFs, logs, and HTML package."""
    root = Path(project_root)
    manuscript_dir = root / "output" / "pdf"
    manuscript_pdf = manuscript_dir / "active_fedference_combined.pdf"
    manuscript_logs = sorted(manuscript_dir.glob("*.log")) if manuscript_dir.exists() else []
    slides_dir = root / "output" / "slides"
    slide_pdfs = sorted(slides_dir.glob("*_slides.pdf")) if slides_dir.exists() else []
    slide_tex = sorted(slides_dir.glob("*_slides.tex")) if slides_dir.exists() else []
    slide_logs = sorted(slides_dir.glob("*_slides.log")) if slides_dir.exists() else []
    findings: list[str] = []

    if not manuscript_pdf.exists():
        findings.append(f"missing combined manuscript PDF: {manuscript_pdf}")
    else:
        if manuscript_pdf.stat().st_size < 5_000:
            findings.append(
                f"{manuscript_pdf}: suspiciously small PDF ({manuscript_pdf.stat().st_size} bytes)"
            )
        findings.extend(_pdf_structure_findings(manuscript_pdf))
        findings.extend(_pdf_text_findings(manuscript_pdf))
    if not manuscript_logs:
        findings.append(f"missing combined manuscript logs: {manuscript_dir}")
    for required_log in ("_combined_manuscript.log", "_latex_stdout.log"):
        required_path = manuscript_dir / required_log
        if not required_path.exists():
            findings.append(f"missing required manuscript log: {required_path}")
    for log in manuscript_logs:
        findings.extend(_log_findings(log))

    if not slide_pdfs:
        findings.append(f"missing generated slide PDFs: {slides_dir}")
    if not slide_tex:
        findings.append(f"missing generated slide TeX sources: {slides_dir}")
    if not slide_logs:
        findings.append(f"missing generated slide logs: {slides_dir}")
    findings.extend(_slide_inventory_findings(slides_dir, slide_pdfs, slide_tex, slide_logs))
    for pdf in slide_pdfs:
        if pdf.stat().st_size < 5_000:
            findings.append(f"{pdf}: suspiciously small PDF ({pdf.stat().st_size} bytes)")
        findings.extend(_pdf_structure_findings(pdf))
        findings.extend(_pdf_text_findings(pdf))
    for log in slide_logs:
        findings.extend(_log_findings(log))

    web = validate_web_package(root)
    if not web.ok:
        findings.extend(web.missing_assets)
        findings.extend(web.raw_xrefs)
        findings.extend(web.broken_xrefs)
        findings.extend(web.malformed_markup)
        findings.extend(web.accessibility_issues)

    return SurfaceValidation(
        manuscript_pdf=manuscript_pdf.exists(),
        manuscript_logs=len(manuscript_logs),
        slide_pdfs=len(slide_pdfs),
        slide_tex=len(slide_tex),
        slide_logs=len(slide_logs),
        findings=tuple(sorted(set(findings))),
        web=web,
    )


__all__ = ["SurfaceValidation", "validate_rendered_surfaces"]
