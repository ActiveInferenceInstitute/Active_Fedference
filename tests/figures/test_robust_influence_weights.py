"""Tests for the robust-influence-weights figure generator.

No mocks: real per-agent weights with a down-weighted saboteur are rendered to
``tmp_path``; we assert the PNG exists and error paths raise.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from figures import generate_robust_influence_weights

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_robust_influence_weights_happy_path(tmp_path: Path) -> None:
    weights = [0.02, 0.20, 0.20, 0.20, 0.18, 0.20]
    path = generate_robust_influence_weights(weights, [0], project_root=tmp_path)
    assert path.exists()
    assert path.read_bytes()[:8] == _PNG_MAGIC
    assert path.stat().st_size > 0


def test_robust_influence_weights_no_contaminated(tmp_path: Path) -> None:
    path = generate_robust_influence_weights([0.25, 0.25, 0.25, 0.25], [], project_root=tmp_path)
    assert path.exists()
    assert path.read_bytes()[:8] == _PNG_MAGIC


def test_robust_influence_weights_rejects_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        generate_robust_influence_weights([], [], project_root=tmp_path)


def test_robust_influence_weights_rejects_bad_index(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        generate_robust_influence_weights([0.5, 0.5], [5], project_root=tmp_path)
