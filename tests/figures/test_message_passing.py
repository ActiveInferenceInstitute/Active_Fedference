"""Tests for the belief-sharing message-passing schematic."""

from __future__ import annotations

from pathlib import Path

from figures import generate_message_passing

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_PDF_MAGIC = b"%PDF"


def test_message_passing_writes_png_and_pdf(tmp_path: Path) -> None:
    path = generate_message_passing(project_root=tmp_path)
    assert path.read_bytes()[:8] == _PNG_MAGIC
    assert path.with_suffix(".pdf").read_bytes()[:4] == _PDF_MAGIC
    assert path.stat().st_size > 10_000


def test_message_passing_is_deterministic(tmp_path: Path) -> None:
    first = generate_message_passing(project_root=tmp_path)
    first_bytes = first.read_bytes()
    second = generate_message_passing(project_root=tmp_path)
    assert second.read_bytes() == first_bytes

