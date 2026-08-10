"""Arbitrary-depth N-level POMDP identities (MAJ-5) — no mocks, real inference.

Two things are pinned here, both extending the depth-generic architecture past
the former hardcoded ``depth in {2, 3}`` cap:

1. ``run_nlevel_world`` builds and federates a valid stack at any ``depth >= 2``.
2. Hierarchical model reduction (MAJ-7's ``hierarchical_reduce``) keeps working
   at depth 4: a non-gating level collapses (Bayesian surprise ~0, prunable)
   while a genuinely informative level is kept — the SAME structural claim as
   the 3-level case, verified one level deeper, with a positive control so the
   "collapse" is not vacuous.
"""

from __future__ import annotations

import numpy as np
import pytest

from fedference.bayesian_model_reduction import hierarchical_reduce
from fedference.experiments.worlds import run_nlevel_world
from fedference.pomdp import (
    N_LOCATIONS,
    LayerSpec,
    build_nlevel_world,
    build_sentinel_world,
)


@pytest.mark.parametrize("depth", [2, 3, 4, 5])
def test_run_nlevel_world_builds_any_depth(depth: int) -> None:
    result = run_nlevel_world(0, depth=depth, n_trials=8)
    assert result["n_levels"] == depth
    assert result["depth"] == depth
    # Federation runs end-to-end: accuracies are valid fractions.
    for key in ("flat", "nlevel"):
        assert 0.0 <= result["location_accuracy"][key] <= 1.0
    assert 0.0 <= result["top_level_accuracy"] <= 1.0


def test_run_nlevel_world_rejects_depth_below_two() -> None:
    with pytest.raises(ValueError, match="depth must be >= 2"):
        run_nlevel_world(0, depth=1)


def _leaf_A() -> np.ndarray:
    return np.asarray(build_sentinel_world(np.random.default_rng(0), acuity=0.85)["A"][0], float)


def _depth4_world(top_conditioned: list[np.ndarray]) -> dict:
    """A depth-4 world (L0 meta2 -> L1 meta -> L2 context -> L3 leaf) whose ONLY
    varying knob is the topmost level's conditioned priors."""
    loc_a = np.full(N_LOCATIONS, 0.02)
    loc_a[4] = 1.0 - 0.02 * (N_LOCATIONS - 1)
    loc_b = np.full(N_LOCATIONS, 0.02)
    loc_b[0] = 1.0 - 0.02 * (N_LOCATIONS - 1)
    top = LayerSpec(n_states=2, labels=("top_low", "top_high"),
                    default_prior=np.array([0.5, 0.5]), conditioned_priors=top_conditioned)
    meta = LayerSpec(n_states=2, labels=("meta_low", "meta_high"),
                     default_prior=np.array([0.5, 0.5]),
                     conditioned_priors=[np.array([0.9, 0.1]), np.array([0.1, 0.9])])
    ctx = LayerSpec(n_states=2, labels=("quiet", "alert"),
                    default_prior=np.array([0.5, 0.5]), conditioned_priors=[loc_a, loc_b])
    leaf = LayerSpec(n_states=N_LOCATIONS, labels=tuple(str(i) for i in range(N_LOCATIONS)))
    return build_nlevel_world([top, meta, ctx, leaf], acuity=0.85)


def test_depth4_nongating_top_collapses() -> None:
    """A depth-4 top level that predicts identical children (non-gating) earns
    ~zero Bayesian surprise and is flagged prunable — MAJ-7's structure learning
    holds one level deeper."""
    world = _depth4_world([np.array([0.5, 0.5]), np.array([0.5, 0.5])])
    out = hierarchical_reduce(world, _leaf_A(), obs=4)
    assert out["n_levels"] == 4
    top = next(lv for lv in out["levels"] if lv["level"] == 0)
    assert top["bayesian_surprise"] < 1e-6
    assert top["prunable"] is True
    assert out["recommended_prune"] == 0


def test_depth4_informative_top_is_kept_positive_control() -> None:
    """Positive control: a genuinely informative depth-4 top level moves under
    the evidence (strictly positive surprise) and is kept — so the collapse
    above is a real discrimination, not a vacuous zero."""
    world = _depth4_world([np.array([0.9, 0.1]), np.array([0.1, 0.9])])
    out = hierarchical_reduce(world, _leaf_A(), obs=4)
    top = next(lv for lv in out["levels"] if lv["level"] == 0)
    assert top["bayesian_surprise"] > 1e-2
    assert top["prunable"] is False
    assert out["recommended_prune"] != 0


def test_depth4_reduce_reports_all_three_non_leaf_levels() -> None:
    world = _depth4_world([np.array([0.9, 0.1]), np.array([0.1, 0.9])])
    out = hierarchical_reduce(world, _leaf_A(), obs=4)
    assert [lv["level"] for lv in out["levels"]] == [0, 1, 2]  # leaf L3 never a target
