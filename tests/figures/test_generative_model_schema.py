"""Tests for the formal categorical generative-model schematic."""

from __future__ import annotations

from pathlib import Path

from figures import generate_generative_model_schema

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_PDF_MAGIC = b"%PDF"


def test_generative_model_schema_writes_png_and_pdf(tmp_path: Path) -> None:
    path = generate_generative_model_schema(project_root=tmp_path)
    assert path.read_bytes()[:8] == _PNG_MAGIC
    assert path.with_suffix(".pdf").read_bytes()[:4] == _PDF_MAGIC
    assert path.stat().st_size > 10_000


def test_generative_model_schema_is_deterministic(tmp_path: Path) -> None:
    first = generate_generative_model_schema(project_root=tmp_path)
    first_bytes = first.read_bytes()
    second = generate_generative_model_schema(project_root=tmp_path)
    assert second.read_bytes() == first_bytes


def test_presentation_equation_keeps_math_delimiters_on_one_line() -> None:
    from figures._presentation_flows import _text_panel

    equation = r"$q(s)=\mathrm{softmax}(\ln D_0+\ln A[o,\cdot])$"
    panel = _text_panel("local-equation", equation)
    artist = panel.figure.texts[0]
    assert artist.get_text() == equation
    panel.figure.canvas.draw()
    bounds = artist.get_window_extent(panel.figure.canvas.get_renderer())
    assert bounds.x0 >= 0 and bounds.x1 <= panel.figure.bbox.width
    assert bounds.y0 >= 0 and bounds.y1 <= panel.figure.bbox.height
    assert artist.get_fontsize() == 22
