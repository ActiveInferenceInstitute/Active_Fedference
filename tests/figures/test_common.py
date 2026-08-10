"""Tests for the shared figure-styling helpers in ``figures._common``.

No mocks: ``apply_style`` mutates the real matplotlib rcParams (headless Agg)
and ``figures_dir`` creates a real directory under ``tmp_path``.

Split out of the former flat ``tests/test_figures.py`` to mirror
``src/figures/_common.py`` under the three-tree discipline. Logic unchanged.
"""

from __future__ import annotations

from pathlib import Path

from figures import apply_style, figures_dir


def test_apply_style_and_figures_dir(tmp_path: Path) -> None:
    apply_style()
    out = figures_dir(tmp_path)
    assert out.exists()
    assert out == tmp_path / "output" / "figures"
