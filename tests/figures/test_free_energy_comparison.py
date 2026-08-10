"""Tests for the free-energy comparison generator
(``figures.free_energy_comparison``).

No mocks: the generator renders a real PNG to ``tmp_path`` (headless Agg) and
we assert the file exists, is a PNG, and that the documented error paths raise.

Split out of the former flat ``tests/test_figures.py`` to mirror
``src/figures/free_energy_comparison.py`` under the three-tree discipline.
Logic unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from figures import generate_free_energy_comparison

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_free_energy_comparison_happy_path(tmp_path: Path) -> None:
    incom = [3.1, 3.0, 3.2, 2.9]
    comm = [2.1, 2.0, 2.2, 1.9]
    path = generate_free_energy_comparison(incom, comm, project_root=tmp_path)
    assert path.exists()
    assert path.read_bytes()[:8] == _PNG_MAGIC


def test_free_energy_comparison_rejects_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        generate_free_energy_comparison([], [], project_root=tmp_path)


def test_free_energy_comparison_rejects_length_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        generate_free_energy_comparison([1.0, 2.0], [1.0], project_root=tmp_path)
