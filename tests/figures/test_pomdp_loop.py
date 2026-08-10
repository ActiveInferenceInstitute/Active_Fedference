"""Tests for the hidden-state/action/federation loop schematic."""

from __future__ import annotations

from pathlib import Path

from figures import generate_pomdp_loop

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_PDF_MAGIC = b"%PDF"


def test_pomdp_loop_writes_png_and_pdf(tmp_path: Path) -> None:
    path = generate_pomdp_loop(project_root=tmp_path)
    assert path.read_bytes()[:8] == _PNG_MAGIC
    assert path.with_suffix(".pdf").read_bytes()[:4] == _PDF_MAGIC
    assert path.stat().st_size > 10_000


def test_pomdp_loop_is_deterministic(tmp_path: Path) -> None:
    first = generate_pomdp_loop(project_root=tmp_path)
    first_bytes = first.read_bytes()
    second = generate_pomdp_loop(project_root=tmp_path)
    assert second.read_bytes() == first_bytes

