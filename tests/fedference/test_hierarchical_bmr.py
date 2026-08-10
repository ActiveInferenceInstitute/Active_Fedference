"""Hierarchical Bayesian model reduction (MAJ-7) — no mocks, real inference.

Structure learning at the LEVEL granularity: on a known-degenerate N=3 world
whose top meta-context is non-gating (its states predict identical children),
``hierarchical_reduce`` must flag that level prunable and recommend removing it,
recovering the 2-level structure; on an informative world it must keep every
level. Both directions are pinned (proof-of-detection: the degenerate and
informative worlds differ only in the top level's conditioned priors).
"""

from __future__ import annotations

import numpy as np

from fedference.bayesian_model_reduction import hierarchical_reduce
from fedference.pomdp import (
    N_LOCATIONS,
    LayerSpec,
    build_nlevel_world,
    build_sentinel_world,
)


def _leaf_A() -> np.ndarray:
    world = build_sentinel_world(np.random.default_rng(0), acuity=0.85)
    return np.asarray(world["A"][0], dtype=np.float64)


def _three_level_world(l3_conditioned: list[np.ndarray]) -> dict:
    """A 3-level world whose ONLY varying knob is the L3 conditioned priors."""
    loc_a = np.full(N_LOCATIONS, 0.02)
    loc_a[4] = 1.0 - 0.02 * (N_LOCATIONS - 1)
    loc_b = np.full(N_LOCATIONS, 0.02)
    loc_b[0] = 1.0 - 0.02 * (N_LOCATIONS - 1)
    l3 = LayerSpec(
        n_states=2,
        labels=("low_threat", "high_threat"),
        default_prior=np.array([0.5, 0.5]),
        conditioned_priors=l3_conditioned,
    )
    l2 = LayerSpec(
        n_states=2,
        labels=("quiet", "alert"),
        default_prior=np.array([0.5, 0.5]),
        conditioned_priors=[loc_a, loc_b],
    )
    leaf = LayerSpec(n_states=N_LOCATIONS, labels=tuple(str(i) for i in range(N_LOCATIONS)))
    return build_nlevel_world([l3, l2, leaf], acuity=0.85)


#: The degenerate top level: both meta-contexts predict the SAME L2 distribution,
#: so L3 gates nothing and is structurally redundant.
_DEGENERATE_L3 = [np.array([0.5, 0.5]), np.array([0.5, 0.5])]
#: The informative top level: the two meta-contexts predict DIFFERENT L2 mixes.
_INFORMATIVE_L3 = [np.array([0.9, 0.1]), np.array([0.1, 0.9])]


def test_degenerate_meta_context_level_is_prunable() -> None:
    world = _three_level_world(_DEGENERATE_L3)
    out = hierarchical_reduce(world, _leaf_A(), obs=4)
    top = next(lv for lv in out["levels"] if lv["level"] == 0)
    # The non-gating meta-context earns ~zero Bayesian surprise and is flagged.
    assert top["bayesian_surprise"] < 1e-6
    assert top["prunable"] is True
    # Pruning the topmost redundant level recovers the 2-level structure.
    assert out["recommended_prune"] == 0


def test_informative_meta_context_level_is_kept() -> None:
    world = _three_level_world(_INFORMATIVE_L3)
    out = hierarchical_reduce(world, _leaf_A(), obs=4)
    top = next(lv for lv in out["levels"] if lv["level"] == 0)
    # An informative meta-context moves under the evidence: strictly positive
    # surprise, not prunable.
    assert top["bayesian_surprise"] > 1e-2
    assert top["prunable"] is False
    assert out["recommended_prune"] != 0


def test_prune_flag_separates_the_two_worlds() -> None:
    """The SAME machinery, differing only in L3's conditioned priors, must give
    opposite prune verdicts — the discriminating power, not a constant."""
    degen = hierarchical_reduce(_three_level_world(_DEGENERATE_L3), _leaf_A(), obs=4)
    inform = hierarchical_reduce(_three_level_world(_INFORMATIVE_L3), _leaf_A(), obs=4)
    degen_top = next(lv for lv in degen["levels"] if lv["level"] == 0)
    inform_top = next(lv for lv in inform["levels"] if lv["level"] == 0)
    assert degen_top["prunable"] and not inform_top["prunable"]
    assert inform_top["bayesian_surprise"] > degen_top["bayesian_surprise"]


def test_reports_every_non_leaf_level_and_never_the_leaf() -> None:
    world = _three_level_world(_INFORMATIVE_L3)
    out = hierarchical_reduce(world, _leaf_A(), obs=4)
    assert out["n_levels"] == 3
    reported = [lv["level"] for lv in out["levels"]]
    assert reported == [0, 1]  # L1 leaf (index 2) is never a reduction target


def test_deterministic_under_repeat() -> None:
    world = _three_level_world(_INFORMATIVE_L3)
    a = hierarchical_reduce(world, _leaf_A(), obs=4)
    b = hierarchical_reduce(world, _leaf_A(), obs=4)
    assert [lv["bayesian_surprise"] for lv in a["levels"]] == [
        lv["bayesian_surprise"] for lv in b["levels"]
    ]


def test_rejects_degenerate_world_shape() -> None:
    try:
        hierarchical_reduce({"n_levels": 1}, _leaf_A(), obs=0)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "n_levels" in str(exc)
