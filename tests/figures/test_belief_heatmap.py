"""Tests for the colony belief-heatmap generator (``figures.belief_heatmap``).

No mocks: the generator renders a real PNG to ``tmp_path`` (headless Agg) and
we assert the file exists, is a PNG, and that the documented error paths raise.

Split out of the former flat ``tests/test_figures.py`` to mirror
``src/figures/belief_heatmap.py`` under the three-tree discipline. Logic
unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from figures import generate_belief_heatmap

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _soft_colony(n_agents: int, n_states: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    colony = rng.dirichlet(np.ones(n_states), size=n_agents)
    return np.asarray(colony, dtype=np.float64)


def test_belief_heatmap_happy_path(tmp_path: Path) -> None:
    beliefs = _soft_colony(4, 9)
    consensus = beliefs.mean(axis=0)
    consensus = consensus / consensus.sum()
    path = generate_belief_heatmap(beliefs, consensus, project_root=tmp_path)
    assert path.exists()
    assert path.read_bytes()[:8] == _PNG_MAGIC


def test_belief_heatmap_rejects_1d_beliefs(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        generate_belief_heatmap(np.ones(9), np.ones(9), project_root=tmp_path)


def test_belief_heatmap_rejects_consensus_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        generate_belief_heatmap(_soft_colony(3, 9), np.ones(4), project_root=tmp_path)
