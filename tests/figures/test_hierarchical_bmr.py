"""Tests for the hierarchical-BMR figure generator (MAJ-7).

No mocks: real ``hierarchical_reduce`` outputs on the degenerate/informative
3-level worlds are rendered to ``tmp_path``; we assert the PNG exists and the
error path raises.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from fedference.bayesian_model_reduction import hierarchical_reduce
from fedference.pomdp import (
    N_LOCATIONS,
    LayerSpec,
    build_nlevel_world,
    build_sentinel_world,
)
from figures import generate_hierarchical_bmr


def _world(cond: list[np.ndarray]) -> dict:
    loc_a = np.full(N_LOCATIONS, 0.02)
    loc_a[4] = 1.0 - 0.02 * (N_LOCATIONS - 1)
    loc_b = np.full(N_LOCATIONS, 0.02)
    loc_b[0] = 1.0 - 0.02 * (N_LOCATIONS - 1)
    l3 = LayerSpec(n_states=2, labels=("low_threat", "high_threat"),
                   default_prior=np.array([0.5, 0.5]), conditioned_priors=cond)
    l2 = LayerSpec(n_states=2, labels=("quiet", "alert"),
                   default_prior=np.array([0.5, 0.5]), conditioned_priors=[loc_a, loc_b])
    leaf = LayerSpec(n_states=N_LOCATIONS, labels=tuple(str(i) for i in range(N_LOCATIONS)))
    return build_nlevel_world([l3, l2, leaf], acuity=0.85)


def _A() -> np.ndarray:
    return np.asarray(build_sentinel_world(np.random.default_rng(0), acuity=0.85)["A"][0], float)


def test_hierarchical_bmr_writes_png(tmp_path: Path) -> None:
    deg = hierarchical_reduce(_world([np.array([0.5, 0.5]), np.array([0.5, 0.5])]), _A(), obs=4)
    inf = hierarchical_reduce(_world([np.array([0.9, 0.1]), np.array([0.1, 0.9])]), _A(), obs=4)
    out = generate_hierarchical_bmr(deg, inf, project_root=tmp_path)
    assert out.exists()
    assert out.stat().st_size > 5_000


def test_hierarchical_bmr_rejects_empty_levels(tmp_path: Path) -> None:
    empty: dict[str, Any] = {
        "levels": [],
        "recommended_prune": None,
        "n_levels": 1,
    }
    try:
        generate_hierarchical_bmr(empty, empty, project_root=tmp_path)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "non-leaf level" in str(exc)
