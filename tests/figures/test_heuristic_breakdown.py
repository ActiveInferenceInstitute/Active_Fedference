"""Tests for the heuristic-characterization figure (MAJ-1) — no mocks."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from fedference.experiments import run_heuristic_characterization
from figures import generate_heuristic_breakdown


def test_heuristic_breakdown_writes_png(tmp_path: Path) -> None:
    report = run_heuristic_characterization(0)
    out = generate_heuristic_breakdown(report, project_root=tmp_path)
    assert out.exists()
    assert out.stat().st_size > 5_000
    with Image.open(out) as image:
        assert image.height >= 0.8 * image.width, "three diagnostics must retain the 2x2 reflow"
