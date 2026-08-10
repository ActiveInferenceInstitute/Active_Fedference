"""Tests for the Markov-blanket-separated sentinel ensemble (Friston 2024, §3.1).

ISC-22: the private **gaze** factor stays inside each agent's Markov blanket —
it never appears in the broadcast vector and is never written by an assimilated
consensus — while the **shared** location/proximity/pose factors are the only
thing exchanged across agents. Every assertion is a real seeded computation with
explicit numeric expectations; there are no mocks.
"""

from __future__ import annotations

import numpy as np
import pytest

from fedference.agents import (
    N_GAZE,
    N_LOCATIONS,
    N_POSE,
    N_PROXIMITY,
    SHARED_FACTORS,
    Observation,
    Sentinel,
    SentinelEnsemble,
    _gaze_likelihood,
    _pose_likelihood,
    _proximity_likelihood,
    _uniform,
)
from fedference.aggregation import log_linear_pool
from fedference.belief_updating import infer_states
from fedference.generalized_bayes import softmax
from fedference.pomdp import build_sentinel_world

# ---- construction ---------------------------------------------------------

def test_from_world_builds_requested_number_of_agents():
    ens = SentinelEnsemble.from_world(4, seed=7)
    assert ens.n_agents == 4
    assert len(ens.agents) == 4
    assert ens.n_shared == N_LOCATIONS + N_PROXIMITY + N_POSE


def test_world_prior_retained_running_belief_starts_uniform():
    ens = SentinelEnsemble.from_world(2, seed=1)
    world = build_sentinel_world()  # canonical prior D: all mass at the center.
    d = np.asarray(world["D"][0]).ravel()
    d = d / d.sum()
    for agent in ens.agents:
        # The structural prior D is retained, all mass at center cell 4.
        assert agent.D_location == pytest.approx(d, abs=1e-12)
        assert agent.D_location[4] == pytest.approx(1.0, abs=1e-12)
        # The RUNNING belief, however, starts uniform (revisable by data).
        assert agent.qs_location == pytest.approx(_uniform(N_LOCATIONS), abs=1e-12)


def test_initial_factors_are_uniform():
    ens = SentinelEnsemble.from_world(1, seed=3)
    agent = ens.agents[0]
    assert agent.qs_location == pytest.approx(_uniform(N_LOCATIONS), abs=1e-12)
    assert agent.qs_proximity == pytest.approx(_uniform(N_PROXIMITY), abs=1e-12)
    assert agent.qs_pose == pytest.approx(_uniform(N_POSE), abs=1e-12)
    assert agent.qs_gaze == pytest.approx(_uniform(N_GAZE), abs=1e-12)


def test_from_world_is_deterministic_for_a_seed():
    a = SentinelEnsemble.from_world(3, seed=42)
    b = SentinelEnsemble.from_world(3, seed=42)
    np.testing.assert_allclose(a.broadcast(), b.broadcast())
    np.testing.assert_allclose(a.gaze_beliefs(), b.gaze_beliefs())


def test_from_world_rejects_non_positive_agents():
    with pytest.raises(ValueError, match="positive integer"):
        SentinelEnsemble.from_world(0)


def test_ensemble_rejects_empty_agent_list():
    with pytest.raises(ValueError, match="at least one sentinel"):
        SentinelEnsemble([])


# ---- likelihood builders are proper column-pmfs --------------------------

def test_shared_likelihoods_have_unit_sum_columns():
    for a in (_proximity_likelihood(), _pose_likelihood(None), _gaze_likelihood(None)):
        np.testing.assert_allclose(a.sum(axis=0), 1.0, atol=1e-12)
    assert _proximity_likelihood().shape == (N_PROXIMITY, N_LOCATIONS)
    assert _pose_likelihood(None).shape == (N_POSE, N_LOCATIONS)
    assert _gaze_likelihood(None).shape == (N_GAZE, N_GAZE)


def test_proximity_centre_cell_reads_near():
    # Center cell (flat index 4) is ring 0 -> proximity band 0 is dominant.
    a = _proximity_likelihood()
    assert np.argmax(a[:, 4]) == 0
    # A corner (flat index 0) is ring 1 -> band 1.
    assert np.argmax(a[:, 0]) == 1


def test_pose_and_gaze_jitter_is_seeded():
    rng_a = np.random.default_rng(11)
    rng_b = np.random.default_rng(11)
    np.testing.assert_allclose(_pose_likelihood(rng_a), _pose_likelihood(rng_b))
    rng_c = np.random.default_rng(11)
    rng_d = np.random.default_rng(11)
    np.testing.assert_allclose(_gaze_likelihood(rng_c), _gaze_likelihood(rng_d))


# ---- ISC-22: gaze is PRIVATE, not in the broadcast vector ----------------

def test_broadcast_vector_length_excludes_gaze():
    ens = SentinelEnsemble.from_world(3, seed=5)
    bcast = ens.broadcast()
    # Shape is (n_agents, location+proximity+pose) — gaze (N_GAZE) is absent.
    assert bcast.shape == (3, N_LOCATIONS + N_PROXIMITY + N_POSE)
    assert bcast.shape[1] == ens.n_shared
    # The gaze dimension is NOT folded into the shared width.
    assert bcast.shape[1] != N_LOCATIONS + N_PROXIMITY + N_POSE + N_GAZE


def test_broadcast_rows_are_three_concatenated_pmfs():
    ens = SentinelEnsemble.from_world(2, seed=8)
    bcast = ens.broadcast()
    for i, agent in enumerate(ens.agents):
        loc = bcast[i, :N_LOCATIONS]
        prox = bcast[i, N_LOCATIONS:N_LOCATIONS + N_PROXIMITY]
        pose = bcast[i, N_LOCATIONS + N_PROXIMITY:]
        assert loc == pytest.approx(agent.qs_location, abs=1e-12)
        assert prox == pytest.approx(agent.qs_proximity, abs=1e-12)
        assert pose == pytest.approx(agent.qs_pose, abs=1e-12)
        # Each of the three shared factors is itself a normalized pmf.
        assert loc.sum() == pytest.approx(1.0, abs=1e-12)
        assert prox.sum() == pytest.approx(1.0, abs=1e-12)
        assert pose.sum() == pytest.approx(1.0, abs=1e-12)


def test_gaze_never_appears_in_any_broadcast_row():
    # Drive gaze to a sharp, distinctive belief; prove that exact vector is not
    # recoverable anywhere in the broadcast matrix (ISC-22).
    ens = SentinelEnsemble.from_world(2, seed=2)
    obs = [Observation(location=0, proximity=2, pose=1, gaze=7) for _ in ens.agents]
    ens.perceive(obs)
    ens.perceive(obs)  # twice, to sharpen the private gaze posterior.
    bcast = ens.broadcast()
    for agent in ens.agents:
        gaze = agent.qs_gaze
        # gaze is concentrated on cell 7 (its observed value) — distinctive.
        assert np.argmax(gaze) == 7
        # No length-9 window of any broadcast row equals the gaze pmf.
        for row in bcast:
            for start in range(row.shape[0] - N_GAZE + 1):
                window = row[start:start + N_GAZE]
                if window.shape[0] == N_GAZE:
                    assert not np.allclose(window, gaze, atol=1e-9)


def test_gaze_beliefs_accessor_returns_private_factor():
    ens = SentinelEnsemble.from_world(3, seed=9)
    gazes = ens.gaze_beliefs()
    assert gazes.shape == (3, N_GAZE)
    np.testing.assert_allclose(gazes.sum(axis=1), 1.0, atol=1e-12)


# ---- perceive: both private and shared factors update --------------------

def test_perceive_updates_shared_location_via_locked_infer_states():
    ens = SentinelEnsemble.from_world(1, seed=4)
    agent = ens.agents[0]
    prior_loc = agent.qs_location.copy()
    obs = [Observation(location=2, proximity=0, pose=3, gaze=5)]
    ens.perceive(obs)
    # The new location belief must equal the locked one-step posterior exactly.
    expected = infer_states(agent.A_location, 2, np.log(np.clip(prior_loc, 1e-12, None)))
    assert agent.qs_location == pytest.approx(expected, abs=1e-12)


def test_perceive_updates_private_gaze():
    ens = SentinelEnsemble.from_world(1, seed=6)
    agent = ens.agents[0]
    prior_gaze = agent.qs_gaze.copy()
    ens.perceive([Observation(location=0, proximity=0, pose=0, gaze=3)])
    expected = infer_states(agent.A_gaze, 3, np.log(np.clip(prior_gaze, 1e-12, None)))
    assert agent.qs_gaze == pytest.approx(expected, abs=1e-12)
    # gaze moved away from uniform toward the observed cell.
    assert np.argmax(agent.qs_gaze) == 3


def test_perceive_rejects_wrong_observation_count():
    ens = SentinelEnsemble.from_world(2, seed=1)
    with pytest.raises(ValueError, match="one observation per agent"):
        ens.perceive([Observation(0, 0, 0, 0)])


# ---- assimilate: shared factors fuse, gaze untouched ---------------------

def test_assimilate_leaves_private_gaze_unchanged():
    ens = SentinelEnsemble.from_world(3, seed=10)
    ens.perceive([Observation(1, 1, 1, 4) for _ in ens.agents])
    gaze_before = ens.gaze_beliefs().copy()
    consensus = log_linear_pool(ens.broadcast())
    ens.assimilate(consensus)
    gaze_after = ens.gaze_beliefs()
    # ISC-22: assimilation must not touch the private gaze factor.
    np.testing.assert_allclose(gaze_before, gaze_after, atol=1e-12)


def test_assimilate_fuses_shared_location_as_log_product():
    ens = SentinelEnsemble.from_world(1, seed=12)
    agent = ens.agents[0]
    ens.perceive([Observation(3, 2, 1, 0)])
    own_loc = agent.qs_location.copy()
    consensus = log_linear_pool(ens.broadcast())
    c_loc = consensus[:N_LOCATIONS]
    ens.assimilate(consensus, learning_rate=1.0)
    expected = softmax(np.log(np.clip(own_loc, 1e-12, None))
                       + np.log(np.clip(c_loc, 1e-12, None)))
    assert agent.qs_location == pytest.approx(expected, abs=1e-12)


def test_assimilate_zero_learning_rate_is_identity_on_shared():
    ens = SentinelEnsemble.from_world(2, seed=13)
    ens.perceive([Observation(2, 0, 2, 1) for _ in ens.agents])
    loc_before = [a.qs_location.copy() for a in ens.agents]
    consensus = log_linear_pool(ens.broadcast())
    ens.assimilate(consensus, learning_rate=0.0)
    for a, before in zip(ens.agents, loc_before):
        # learning_rate 0 -> softmax(log own + 0) == own (renormalized).
        assert a.qs_location == pytest.approx(before, abs=1e-12)


def test_assimilate_rejects_wrong_consensus_length():
    ens = SentinelEnsemble.from_world(1, seed=1)
    with pytest.raises(ValueError, match="must have length"):
        ens.assimilate(np.ones(3))


def test_assimilate_rejects_negative_learning_rate():
    ens = SentinelEnsemble.from_world(1, seed=1)
    consensus = log_linear_pool(ens.broadcast())
    with pytest.raises(ValueError, match="non-negative"):
        ens.assimilate(consensus, learning_rate=-0.5)


# ---- full round: perceive -> broadcast -> assimilate ---------------------

def test_full_round_drives_colony_toward_a_shared_cell():
    # All sentinels see the creature at the same cell; after one share round the
    # colony's mean location belief on that cell strictly rises.
    ens = SentinelEnsemble.from_world(5, seed=21)
    target = 6
    obs = [Observation(location=target, proximity=1, pose=0, gaze=i % N_GAZE)
           for i in range(ens.n_agents)]
    ens.perceive(obs)
    loc_pre = ens.broadcast_location()[:, target].mean()
    consensus = log_linear_pool(ens.broadcast())
    ens.assimilate(consensus)
    loc_post = ens.broadcast_location()[:, target].mean()
    assert loc_post > loc_pre
    # And the colony agrees: post-assimilation argmax location is the target.
    for agent in ens.agents:
        assert np.argmax(agent.qs_location) == target


def test_shared_factor_names_and_broadcast_layout_agree():
    assert SHARED_FACTORS == ("location", "proximity", "pose")
    ens = SentinelEnsemble.from_world(1, seed=1)
    sizes = {"location": N_LOCATIONS, "proximity": N_PROXIMITY, "pose": N_POSE}
    assert ens.n_shared == sum(sizes[name] for name in SHARED_FACTORS)
    factors = ens.agents[0].shared_factors
    for name in SHARED_FACTORS:
        assert factors[name].shape[0] == sizes[name]


def test_broadcast_location_matches_broadcast_slice():
    ens = SentinelEnsemble.from_world(3, seed=14)
    ens.perceive([Observation(i % N_LOCATIONS, 0, 0, 0) for i in range(3)])
    full = ens.broadcast()
    np.testing.assert_allclose(ens.broadcast_location(), full[:, :N_LOCATIONS])


def test_sentinel_shared_vector_round_trips():
    ens = SentinelEnsemble.from_world(1, seed=1)
    agent = ens.agents[0]
    vec = agent.shared_vector()
    assert vec.shape[0] == N_LOCATIONS + N_PROXIMITY + N_POSE
    np.testing.assert_allclose(vec[:N_LOCATIONS], agent.qs_location, atol=1e-12)


def test_sentinel_dataclass_is_constructible_directly():
    # Cover the bare-dataclass path independent of from_world.
    s = Sentinel(
        A_location=np.eye(N_LOCATIONS),
        A_proximity=_proximity_likelihood(),
        A_pose=_pose_likelihood(None),
        A_gaze=_gaze_likelihood(None),
        qs_location=_uniform(N_LOCATIONS),
        qs_proximity=_uniform(N_PROXIMITY),
        qs_pose=_uniform(N_POSE),
        qs_gaze=_uniform(N_GAZE),
        D_location=_uniform(N_LOCATIONS),
    )
    ens = SentinelEnsemble([s])
    assert ens.n_agents == 1
    assert s.shared_vector().shape[0] == N_LOCATIONS + N_PROXIMITY + N_POSE
