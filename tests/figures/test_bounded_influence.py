"""Tests for the legacy-named variational redescending-weight figure generator.

No mocks: a real influence-vs-drift curve (and one produced by
:func:`fedference.aggregation.variational_aggregate`) is rendered to
``tmp_path``; we assert the PNG exists and the error paths raise. The plotted
normalized-weight path is not an estimator-level B-robustness proof.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from fedference.aggregation import variational_aggregate
from figures import generate_bounded_influence

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_bounded_influence_happy_path(tmp_path: Path) -> None:
    drifts = [0.0, 0.3, 0.6, 0.9, 0.99]
    influence = [0.25, 0.24, 0.18, 0.05, 0.004]
    path = generate_bounded_influence(drifts, influence, 0.25, project_root=tmp_path)
    assert path.exists()
    assert path.read_bytes()[:8] == _PNG_MAGIC
    assert path.stat().st_size > 0


def test_bounded_influence_from_real_aggregator(tmp_path: Path) -> None:
    base = np.array([[0.5, 0.3, 0.2], [0.45, 0.35, 0.2], [0.5, 0.25, 0.25]])
    drifts = [0.0, 0.3, 0.6, 0.9, 0.99]
    influence = []
    for d in drifts:
        liar = (1 - d) * np.array([0.5, 0.3, 0.2]) + d * np.array([0.0, 0.0, 1.0])
        colony = np.vstack([base, liar])
        influence.append(float(variational_aggregate(colony, robustness=1.5).agent_weights[-1]))
    path = generate_bounded_influence(drifts, influence, 0.25, project_root=tmp_path)
    assert path.exists()


def test_bounded_influence_length_mismatch_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="equal length"):
        generate_bounded_influence([0.0, 0.5], [0.25], 0.25, project_root=tmp_path)


def test_bounded_influence_empty_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        generate_bounded_influence([], [], 0.25, project_root=tmp_path)


def test_bounded_influence_nonfinite_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="finite"):
        generate_bounded_influence([0.0, 0.5], [0.25, math.nan], 0.25, project_root=tmp_path)
