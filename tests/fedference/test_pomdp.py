"""Tests for Friston's sentinel POMDP generative model (no mocks).

Every number is a real, seeded computation. The headline contract is **ISC-15**:
every column of the likelihood ``A`` and of each transition slice ``B[:, :, u]``
is a proper categorical pmf summing to 1 (Friston et al. 2024, Fig. 1/4 — a 3x3
9-location shared world). Error paths of every public function are exercised so
the module is fully covered.
"""

from __future__ import annotations

import numpy as np
import pytest

from fedference.pomdp import (
    CONTROL_LABELS,
    GRID_SIDE,
    N_LOCATIONS,
    build_sentinel_world,
    normalise_columns,
)

# ---- normalise_columns ---------------------------------------------------

def test_normalise_columns_makes_unit_sum_columns():
    m = np.array([[1.0, 2.0], [3.0, 2.0]])
    out = normalise_columns(m)
    assert np.allclose(out.sum(axis=0), 1.0)
    # column 0: [1,3] -> [0.25, 0.75]
    assert out[0, 0] == pytest.approx(0.25)
    assert out[1, 0] == pytest.approx(0.75)


def test_normalise_columns_does_not_mutate_input():
    m = np.array([[1.0, 1.0], [1.0, 1.0]])
    _ = normalise_columns(m)
    assert m[0, 0] == 1.0  # original untouched


def test_normalise_columns_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D"):
        normalise_columns(np.array([1.0, 2.0, 3.0]))


def test_normalise_columns_rejects_empty():
    with pytest.raises(ValueError, match="2-D"):
        normalise_columns(np.empty((0, 0)))


def test_normalise_columns_rejects_negative():
    with pytest.raises(ValueError, match="negative"):
        normalise_columns(np.array([[-1.0, 2.0], [1.0, 1.0]]))


def test_normalise_columns_rejects_zero_column():
    with pytest.raises(ValueError, match="positive mass"):
        normalise_columns(np.array([[0.0, 1.0], [0.0, 1.0]]))


# ---- build_sentinel_world shapes -----------------------------------------

def test_world_has_nine_locations():
    assert N_LOCATIONS == 9
    assert GRID_SIDE == 3
    assert CONTROL_LABELS == ("still", "left", "right")


def test_world_shapes_and_metadata():
    world = build_sentinel_world()
    assert world["n_states"] == 9
    assert world["n_obs"] == 9
    assert world["n_controls"] == 3
    assert world["control_labels"] == ("still", "left", "right")
    assert world["A"][0].shape == (9, 9)
    assert world["B"][0].shape == (9, 9, 3)
    assert world["C"][0].shape == (9, 1)
    assert world["D"][0].shape == (9, 1)


# ---- ISC-15: every A and B column sums to 1 ------------------------------

def test_isc15_A_columns_sum_to_one():
    a = build_sentinel_world()["A"][0]
    assert np.allclose(a.sum(axis=0), np.ones(9), atol=1e-12)


def test_isc15_B_columns_sum_to_one_for_every_control():
    b = build_sentinel_world()["B"][0]
    for u in range(b.shape[2]):
        assert np.allclose(b[:, :, u].sum(axis=0), np.ones(9), atol=1e-12)


def test_isc15_holds_under_seeded_rng():
    rng = np.random.default_rng(7)
    world = build_sentinel_world(rng=rng)
    a = world["A"][0]
    b = world["B"][0]
    assert np.allclose(a.sum(axis=0), np.ones(9), atol=1e-12)
    for u in range(b.shape[2]):
        assert np.allclose(b[:, :, u].sum(axis=0), np.ones(9), atol=1e-12)


def test_D_is_a_pmf_at_centre():
    d = build_sentinel_world()["D"][0]
    assert d.sum() == pytest.approx(1.0)
    # center of a 3x3 grid is flat index 4
    assert d[4, 0] == pytest.approx(1.0)


# ---- likelihood structure ------------------------------------------------

def test_likelihood_diagonal_is_acuity():
    a = build_sentinel_world(acuity=0.9)["A"][0]
    for s in range(9):
        assert a[s, s] == pytest.approx(0.9)
        off = a[(s + 1) % 9, s]
        assert off == pytest.approx(0.1 / 8)


def test_acuity_one_gives_identity_likelihood():
    a = build_sentinel_world(acuity=1.0)["A"][0]
    assert np.allclose(a, np.eye(9))


def test_invalid_acuity_raises():
    with pytest.raises(ValueError, match="acuity"):
        build_sentinel_world(acuity=0.0)
    with pytest.raises(ValueError, match="acuity"):
        build_sentinel_world(acuity=1.5)


# ---- transition structure (still / left / right) -------------------------

def test_still_is_identity():
    b = build_sentinel_world()["B"][0]
    assert np.allclose(b[:, :, 0], np.eye(9))


def test_right_increments_column_and_reflects_at_wall():
    b = build_sentinel_world()["B"][0]
    # location 0 is (row0,col0) -> right -> (row0,col1) = flat 1
    assert b[1, 0, 2] == pytest.approx(1.0)
    # location 2 is (row0,col2) at the right wall -> stays at flat 2
    assert b[2, 2, 2] == pytest.approx(1.0)


def test_left_decrements_column_and_reflects_at_wall():
    b = build_sentinel_world()["B"][0]
    # location 1 -> left -> flat 0
    assert b[0, 1, 1] == pytest.approx(1.0)
    # location 0 at the left wall -> stays
    assert b[0, 0, 1] == pytest.approx(1.0)


def test_right_does_not_cross_grid_rows():
    b = build_sentinel_world()["B"][0]
    # location 5 is (row1,col2) right wall -> stays at 5, never wraps to row2
    assert b[5, 5, 2] == pytest.approx(1.0)


# ---- preferences ---------------------------------------------------------

def test_log_preference_bumps_centre_only():
    c = build_sentinel_world(goal_bonus=2.0)["C"][0]
    assert c[4, 0] == pytest.approx(2.0)
    assert np.count_nonzero(c) == 1


# ---- determinism ---------------------------------------------------------

def test_seeded_rng_is_reproducible():
    a1 = build_sentinel_world(rng=np.random.default_rng(3))["A"][0]
    a2 = build_sentinel_world(rng=np.random.default_rng(3))["A"][0]
    assert np.array_equal(a1, a2)


def test_seeded_jitter_increases_acuity():
    # With base acuity 0.9 the jitter is non-negative, so the seeded diagonal
    # never drops below the base acuity.
    a = build_sentinel_world(rng=np.random.default_rng(0), acuity=0.9)["A"][0]
    assert a[0, 0] >= 0.9 - 1e-12
