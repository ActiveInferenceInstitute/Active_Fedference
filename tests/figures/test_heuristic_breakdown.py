"""Tests for the heuristic-characterization figure (MAJ-1) — no mocks."""

from __future__ import annotations

import json
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


def test_unobserved_capture_is_not_reported_as_zero(tmp_path: Path) -> None:
    report = run_heuristic_characterization(0)
    report["breakdown"]["robust_breakdown_k"] = None
    report["breakdown"]["variational_breakdown_k"] = None
    output = generate_heuristic_breakdown(report, project_root=tmp_path)
    manifest = json.loads(output.with_suffix(".slides.json").read_text())
    for method in ("heuristic", "variational"):
        row = next(
            panel for panel in manifest["panels"]
            if panel["src"].endswith(f".slide-{method}.png")
        )
        assert "no argmax capture observed" in row["alt"]
        assert "not a zero-adversary breakdown point" in row["alt"]
        assert (output.parent / Path(row["src"]).name).is_file()
        assert (output.parent / Path(row["src"]).name).with_suffix(".pdf").is_file()
