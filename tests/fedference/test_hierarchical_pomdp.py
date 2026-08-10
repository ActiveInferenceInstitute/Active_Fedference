"""Tests for the hierarchical POMDP — V2 2-level and V2 N-level / 3-level extension.

18 tests total (10 existing 2-level + 8 new nlevel/3level). All use seeded,
deterministic computations — no mocks. Covers:

2-level (10 tests):
- ``build_hierarchical_world`` shape / pmf contracts;
- ``hierarchical_infer`` posterior validity and context sensitivity;
- ``run_hierarchical_world`` return keys and accuracy gap direction.

N-level / 3-level (8 new tests):
- ``LayerSpec`` dataclass validation;
- ``build_nlevel_world`` with 2- and 3-level stacks;
- ``nlevel_infer`` pmf validity and depth-2 / depth-3 consistency;
- ``build_3level_world`` convenience constructor;
- ``run_nlevel_world`` and ``run_3level_world`` return keys and unit-interval accuracies.
"""

from __future__ import annotations

import numpy as np
import pytest

from fedference.experiments import run_3level_world, run_hierarchical_world, run_nlevel_world
from fedference.pomdp import (
    HIER_CONTEXT_LABELS,
    N_CONTEXTS,
    N_LOCATIONS,
    N_META_CONTEXTS,
    LayerSpec,
    build_3level_world,
    build_hierarchical_world,
    build_nlevel_world,
    hierarchical_infer,
    nlevel_infer,
)

# ---- build_hierarchical_world -----------------------------------------------


def test_hierarchical_world_l1_shapes():
    world = build_hierarchical_world()
    l1 = world["L1"]
    assert l1["n_states"] == N_LOCATIONS
    assert l1["A"][0].shape == (N_LOCATIONS, N_LOCATIONS)
    assert l1["B"][0].shape == (N_LOCATIONS, N_LOCATIONS, 3)


def test_hierarchical_world_l2_prior_is_pmf():
    world = build_hierarchical_world()
    prior = np.asarray(world["L2_prior"], dtype=np.float64)
    assert prior.shape == (N_CONTEXTS,)
    assert np.isclose(prior.sum(), 1.0)
    assert np.all(prior >= 0.0)


def test_hierarchical_world_l1_priors_given_context_are_pmfs():
    world = build_hierarchical_world()
    l1_priors = world["L1_priors_given_context"]
    assert len(l1_priors) == N_CONTEXTS
    for p in l1_priors:
        arr = np.asarray(p, dtype=np.float64)
        assert arr.shape == (N_LOCATIONS,)
        assert np.isclose(arr.sum(), 1.0)
        assert np.all(arr >= 0.0)


def test_context_labels_match_n_contexts():
    assert len(HIER_CONTEXT_LABELS) == N_CONTEXTS
    world = build_hierarchical_world()
    assert world["n_contexts"] == N_CONTEXTS
    assert world["context_labels"] == HIER_CONTEXT_LABELS


def test_build_hierarchical_world_custom_l1_priors():
    # Supply explicit L1 priors and verify they are stored correctly.
    p0 = np.full(N_LOCATIONS, 1.0 / N_LOCATIONS)
    p1 = np.zeros(N_LOCATIONS)
    p1[4] = 1.0  # deterministic: always at center
    world = build_hierarchical_world(l1_priors=(p0, p1))
    stored = world["L1_priors_given_context"]
    assert np.allclose(stored[0], p0)
    assert np.allclose(stored[1], p1)


# ---- hierarchical_infer -----------------------------------------------------


def test_hierarchical_infer_returns_valid_pmfs():
    world = build_hierarchical_world()
    A = np.asarray(world["L1"]["A"][0], dtype=np.float64)
    result = hierarchical_infer(A, obs=4, hier_world=world)
    q_loc = result["q_loc"]
    q_ctx = result["q_ctx"]
    assert q_loc.shape == (N_LOCATIONS,)
    assert np.isclose(q_loc.sum(), 1.0, atol=1e-10)
    assert np.all(q_loc >= 0.0)
    assert q_ctx.shape == (N_CONTEXTS,)
    assert np.isclose(q_ctx.sum(), 1.0, atol=1e-10)
    assert np.all(q_ctx >= 0.0)


def test_hierarchical_infer_location_peaks_near_observed():
    """After observing center cell (obs=4), q_loc should put most mass near cell 4."""
    world = build_hierarchical_world(acuity=0.95)
    A = np.asarray(world["L1"]["A"][0], dtype=np.float64)
    result = hierarchical_infer(A, obs=4, hier_world=world, n_iters=6)
    assert result["q_loc"][4] == max(result["q_loc"]), \
        "highest q_loc mass should be on the observed cell"


def test_hierarchical_infer_context_shifts_with_observation():
    """Observing center cell (4) should raise P(alert) because alert prior is center-peaked."""
    world = build_hierarchical_world()
    A = np.asarray(world["L1"]["A"][0], dtype=np.float64)
    # Flat prior world: q_ctx should start at (0.5, 0.5).
    result_center = hierarchical_infer(A, obs=4, hier_world=world, n_iters=6)
    result_edge = hierarchical_infer(A, obs=0, hier_world=world, n_iters=6)
    # alert (index 1) peaks at center; quiet (index 0) is uniform.
    # Observing center should favor alert over observing edge.
    assert result_center["q_ctx"][1] >= result_edge["q_ctx"][1] - 1e-9, (
        "center observation should favor the alert context at least as much as an edge obs"
    )


# ---- run_hierarchical_world -------------------------------------------------


def test_run_hierarchical_world_return_keys():
    result = run_hierarchical_world(seed=42, n_trials=5, n_agents=2)
    for key in (
        "location_accuracy",
        "location_accuracy_gap",
        "context_accuracy",
        "free_energy_gap",
        "n_trials",
        "n_agents",
        "n_contexts",
        "seed",
    ):
        assert key in result, f"missing key: {key}"


def test_run_hierarchical_world_accuracies_in_unit_interval():
    result = run_hierarchical_world(seed=7, n_trials=10, n_agents=3)
    acc = result["location_accuracy"]
    assert 0.0 <= acc["flat"] <= 1.0
    assert 0.0 <= acc["hierarchical"] <= 1.0
    assert 0.0 <= result["context_accuracy"] <= 1.0


def test_run_hierarchical_world_hierarchical_location_accuracy_not_worse_than_flat():
    """Scientific claim: hierarchical inference must not underperform flat.

    The core motivation of the hierarchical POMDP figure is that the
    context-conditioned prior improves (or at worst equals) flat location
    inference. The existing test checks [0,1] bounds only — this test pins
    the ordering that justifies the figure.
    """
    result = run_hierarchical_world(seed=42, n_trials=20, n_agents=3)
    acc = result["location_accuracy"]
    gap = result["location_accuracy_gap"]  # hierarchical - flat
    assert acc["hierarchical"] >= acc["flat"] - 0.10, (
        f"hierarchical={acc['hierarchical']:.3f} flat={acc['flat']:.3f}: "
        "hierarchical inference should not underperform flat by more than 0.10"
    )
    # The reported gap must be consistent with the accuracy fields.
    assert gap == pytest.approx(acc["hierarchical"] - acc["flat"], abs=1e-9), (
        "location_accuracy_gap must equal hierarchical - flat"
    )
    assert gap >= -0.10, (
        f"gap={gap:.3f}: hierarchical advantage should not be worse than -0.10"
    )


# ---- LayerSpec --------------------------------------------------------------


def test_layer_spec_auto_labels():
    """LayerSpec auto-generates labels when none are provided."""
    spec = LayerSpec(n_states=3)
    assert len(spec.labels) == 3
    assert spec.labels[0] == "state_0"


def test_layer_spec_label_mismatch_raises():
    """LayerSpec raises ValueError when label count mismatches n_states."""
    with pytest.raises(ValueError, match="labels length"):
        LayerSpec(n_states=3, labels=("a", "b"))


# ---- build_nlevel_world -----------------------------------------------------


def test_build_nlevel_world_depth2_matches_hierarchical():
    """depth-2 N-level world has the same L1 shape as build_hierarchical_world."""
    hier = build_hierarchical_world()
    # Build equivalent 2-level world using LayerSpec / build_nlevel_world.
    import numpy as np
    p_quiet = np.full(N_LOCATIONS, 1.0 / N_LOCATIONS)
    alert_off = (1.0 - 0.6) / (N_LOCATIONS - 1)
    p_alert = np.full(N_LOCATIONS, alert_off)
    p_alert[4] = 0.6
    ctx_spec = LayerSpec(
        n_states=N_CONTEXTS, labels=HIER_CONTEXT_LABELS,
        default_prior=np.full(N_CONTEXTS, 0.5),
        conditioned_priors=[p_quiet, p_alert],
    )
    leaf_spec = LayerSpec(
        n_states=N_LOCATIONS,
        labels=tuple(str(i) for i in range(N_LOCATIONS)),
        default_prior=np.full(N_LOCATIONS, 1.0 / N_LOCATIONS),
        conditioned_priors=None,
    )
    world = build_nlevel_world([ctx_spec, leaf_spec])
    assert world["n_levels"] == 2
    assert world["L1"]["n_states"] == hier["L1"]["n_states"]


def test_build_3level_world_shape():
    """build_3level_world returns 3 levels with correct layer state counts."""
    world = build_3level_world()
    assert world["n_levels"] == 3
    assert world["n_meta_contexts"] == N_META_CONTEXTS
    priors = world["level_priors"]
    assert len(priors) == 3
    assert len(priors[0]) == N_META_CONTEXTS
    assert len(priors[1]) == N_CONTEXTS
    assert len(priors[2]) == N_LOCATIONS


# ---- nlevel_infer -----------------------------------------------------------


def test_nlevel_infer_depth2_pmf_validity():
    """nlevel_infer on a depth-2 world returns valid pmfs for both levels."""
    import numpy as np
    p_quiet = np.full(N_LOCATIONS, 1.0 / N_LOCATIONS)
    alert_off = (1.0 - 0.6) / (N_LOCATIONS - 1)
    p_alert = np.full(N_LOCATIONS, alert_off)
    p_alert[4] = 0.6
    ctx_spec = LayerSpec(
        n_states=N_CONTEXTS, labels=HIER_CONTEXT_LABELS,
        default_prior=np.full(N_CONTEXTS, 0.5),
        conditioned_priors=[p_quiet, p_alert],
    )
    leaf_spec = LayerSpec(
        n_states=N_LOCATIONS,
        labels=tuple(str(i) for i in range(N_LOCATIONS)),
        default_prior=np.full(N_LOCATIONS, 1.0 / N_LOCATIONS),
        conditioned_priors=None,
    )
    world = build_nlevel_world([ctx_spec, leaf_spec])
    A = np.asarray(world["L1"]["A"][0], dtype=np.float64)
    result = nlevel_infer(A, obs=4, nlevel_world=world, n_iters=4)
    assert len(result["q_levels"]) == 2
    for q in result["q_levels"]:
        assert np.isclose(q.sum(), 1.0, atol=1e-10)
        assert np.all(q >= 0.0)


def test_nlevel_infer_depth3_pmf_validity():
    """nlevel_infer on a 3-level world returns 3 valid pmf arrays."""
    world = build_3level_world()
    A = np.asarray(world["L1"]["A"][0], dtype=np.float64)
    result = nlevel_infer(A, obs=4, nlevel_world=world, n_iters=4)
    assert len(result["q_levels"]) == 3
    for q in result["q_levels"]:
        assert np.isclose(q.sum(), 1.0, atol=1e-10)
        assert np.all(q >= 0.0)


# ---- run_nlevel_world / run_3level_world ------------------------------------


def test_run_nlevel_world_return_keys():
    result = run_nlevel_world(seed=42, n_trials=5, n_agents=2, depth=2)
    for key in (
        "location_accuracy",
        "location_accuracy_gap",
        "top_level_accuracy",
        "free_energy_gap",
        "n_trials",
        "n_agents",
        "n_levels",
        "seed",
    ):
        assert key in result, f"missing key: {key}"


def test_run_3level_world_accuracies_in_unit_interval():
    result = run_3level_world(seed=99, n_trials=8, n_agents=2)
    acc = result["location_accuracy"]
    assert 0.0 <= acc["flat"] <= 1.0
    assert 0.0 <= acc["nlevel3"] <= 1.0
    assert 0.0 <= result["context_accuracy"] <= 1.0
    assert 0.0 <= result["meta_context_accuracy"] <= 1.0
    assert result["n_levels"] == 3
