"""Tests for the metadata-backed layered graphical abstract."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from figures import generate_graphical_abstract

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_PDF_MAGIC = b"%PDF"


def test_graphical_abstract_writes_figure_and_cover(tmp_path: Path) -> None:
    path = generate_graphical_abstract(project_root=tmp_path)
    assert path.read_bytes()[:8] == _PNG_MAGIC
    assert path.with_suffix(".pdf").read_bytes()[:4] == _PDF_MAGIC
    cover = tmp_path / "manuscript" / "cover_image.png"
    assert cover.read_bytes()[:8] == _PNG_MAGIC
    assert Image.open(path).mode == "RGB"
    assert Image.open(cover).mode == "RGB"


def test_graphical_abstract_is_deterministic(tmp_path: Path) -> None:
    first = generate_graphical_abstract(project_root=tmp_path)
    first_bytes = first.read_bytes()
    second = generate_graphical_abstract(project_root=tmp_path)
    assert second.read_bytes() == first_bytes
