"""Tests for the variational free-energy descent figure generator.

No mocks: a real descending free-energy history (and a real one produced by
:func:`fedference.aggregation.variational_aggregate`) is rendered to
``tmp_path``; we assert the PNG exists and the error paths raise.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from fedference.aggregation import variational_aggregate
from figures import generate_aggregation_descent

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_descent_happy_path(tmp_path: Path) -> None:
    history = [4.16, 1.10, 0.91, 0.84, 0.80, 0.79]
    path = generate_aggregation_descent(history, converged=True, project_root=tmp_path)
    assert path.exists()
    assert path.read_bytes()[:8] == _PNG_MAGIC
    assert path.stat().st_size > 0


def test_descent_from_real_aggregator(tmp_path: Path) -> None:
    beliefs = np.array([[0.7, 0.2, 0.1], [0.65, 0.25, 0.1],
                        [0.6, 0.25, 0.15], [0.05, 0.05, 0.9]])
    res = variational_aggregate(beliefs, robustness=2.0)
    path = generate_aggregation_descent(
        res.free_energy_history, converged=res.converged, project_root=tmp_path
    )
    assert path.exists()


def test_descent_rejects_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        generate_aggregation_descent([], project_root=tmp_path)


def test_descent_rejects_nonfinite(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="finite"):
        generate_aggregation_descent([1.0, math.inf], project_root=tmp_path)
