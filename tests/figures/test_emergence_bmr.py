"""Tests for the Bayesian-model-reduction emergence figure generator.

No mocks: real free-energy differences (positive redundant, negative supported)
are rendered to ``tmp_path``; we assert the PNG exists and error paths raise.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from figures import generate_emergence_bmr

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_emergence_bmr_happy_path(tmp_path: Path) -> None:
    path = generate_emergence_bmr(
        delta_F_redundant=3.4, delta_F_supported=-5.1, convergence=True, project_root=tmp_path
    )
    assert path.exists()
    assert path.read_bytes()[:8] == _PNG_MAGIC
    assert path.stat().st_size > 0


def test_emergence_bmr_no_convergence_flag(tmp_path: Path) -> None:
    path = generate_emergence_bmr(1.0, 2.0, project_root=tmp_path)
    assert path.exists()
    assert path.read_bytes()[:8] == _PNG_MAGIC


def test_emergence_bmr_rejects_nonfinite(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        generate_emergence_bmr(math.inf, -1.0, project_root=tmp_path)
