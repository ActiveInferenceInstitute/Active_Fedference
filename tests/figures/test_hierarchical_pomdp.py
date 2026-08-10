"""Tests for figures.hierarchical_pomdp.generate_hierarchical_pomdp."""

from __future__ import annotations

from pathlib import Path

import pytest

from figures.hierarchical_pomdp import generate_hierarchical_pomdp

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_hierarchical_pomdp_happy_path(tmp_path: Path) -> None:
    path = generate_hierarchical_pomdp(
        project_root=tmp_path,
        n_agents=2,
        n_trials=3,
        n_iters=2,
        allow_illustrative_fallback=True,
    )
    assert path.exists(), "PNG file was not created"
    assert path.read_bytes()[:8] == _PNG_MAGIC, "file is not a valid PNG"


def test_hierarchical_pomdp_custom_filename(tmp_path: Path) -> None:
    path = generate_hierarchical_pomdp(
        project_root=tmp_path,
        n_agents=2,
        n_trials=2,
        n_iters=2,
        filename="hier_test.png",
        allow_illustrative_fallback=True,
    )
    assert path.name == "hier_test.png"
    assert path.read_bytes()[:8] == _PNG_MAGIC


def test_hierarchical_pomdp_deterministic(tmp_path: Path) -> None:
    p1 = generate_hierarchical_pomdp(
        project_root=tmp_path,
        seed=0,
        n_agents=2,
        n_trials=2,
        n_iters=2,
        filename="det1.png",
        allow_illustrative_fallback=True,
    )
    p2 = generate_hierarchical_pomdp(
        project_root=tmp_path,
        seed=0,
        n_agents=2,
        n_trials=2,
        n_iters=2,
        filename="det2.png",
        allow_illustrative_fallback=True,
    )
    assert p1.read_bytes() == p2.read_bytes(), "same seed should produce identical PNG"


def test_hierarchical_pomdp_requires_executed_reports_by_default(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="executed hierarchical reports"):
        generate_hierarchical_pomdp(project_root=tmp_path, n_agents=2, n_iters=2)
