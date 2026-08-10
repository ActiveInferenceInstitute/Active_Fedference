"""Tests for the descent-comparison figure generator (no mocks).

Two real free-energy histories (and a pair produced by single- vs multi-start
:func:`fedference.aggregation.variational_aggregate` on a near-vertex colony) are
rendered to ``tmp_path``; we assert the PNG exists and the error paths raise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fedference.aggregation import variational_aggregate
from figures import generate_descent_comparison

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_comparison_happy_path(tmp_path: Path) -> None:
    single = [3.2, 1.9, 1.75, 1.72, 1.72]   # captured basin
    multi = [3.2, 1.4, 1.2, 1.18, 1.18]      # vetoing basin
    path = generate_descent_comparison(single, multi, project_root=tmp_path)
    assert path.exists()
    assert path.read_bytes()[:8] == _PNG_MAGIC


def test_comparison_from_real_aggregator(tmp_path: Path) -> None:
    honest = [np.array([0.4, 0.3, 0.3])] * 3
    liar = np.array([1e-4, 1e-4, 1.0 - 2e-4])
    colony = np.vstack([honest, [liar]])
    single = variational_aggregate(colony, robustness=1.5, multistart=False, max_iter=128)
    multi = variational_aggregate(colony, robustness=1.5, multistart=True, max_iter=128)
    path = generate_descent_comparison(
        single.free_energy_history, multi.free_energy_history, project_root=tmp_path
    )
    assert path.exists()
    # the single-start capture basin sits at higher (or equal) final F than multi.
    assert single.free_energy_history[-1] >= multi.free_energy_history[-1] - 1e-9


def test_comparison_rejects_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        generate_descent_comparison([], [1.0], project_root=tmp_path)
