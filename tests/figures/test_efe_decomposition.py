"""Tests for the EFE-decomposition identity figure generator.

No mocks: four EFE terms satisfying the Eq. 2 identity
``risk + ambiguity == -(pragmatic + epistemic)`` are rendered to ``tmp_path``;
we assert the PNG exists and the error path raises.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from figures import generate_efe_decomposition

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_efe_decomposition_happy_path(tmp_path: Path) -> None:
    # Identity: risk + ambiguity == -(pragmatic + epistemic).
    risk, ambiguity = 0.5, 2.0
    total = risk + ambiguity
    pragmatic, epistemic = -2.4, -0.1  # -(pragmatic + epistemic) == 2.5 == total
    assert abs((risk + ambiguity) + (pragmatic + epistemic) - 0.0) < 1e-9
    assert abs(-(pragmatic + epistemic) - total) < 1e-9
    path = generate_efe_decomposition(risk, ambiguity, pragmatic, epistemic, project_root=tmp_path)
    assert path.exists()
    assert path.read_bytes()[:8] == _PNG_MAGIC
    assert path.stat().st_size > 0


def test_efe_decomposition_rejects_nonfinite(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        generate_efe_decomposition(math.nan, 1.0, -1.0, -1.0, project_root=tmp_path)


def test_efe_decomposition_rejects_identity_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"violate risk \+ ambiguity"):
        generate_efe_decomposition(0.5, 2.0, -2.4, -0.2, project_root=tmp_path)


def test_efe_decomposition_preserves_public_filename_and_dimensions(tmp_path: Path) -> None:
    from PIL import Image

    path = generate_efe_decomposition(0.5, 2.0, -2.4, -0.1, project_root=tmp_path)
    with Image.open(path) as image:
        assert image.width >= 1200
        assert image.height >= 800
