"""Tests for adversarial / miscalibrated sentinel contamination (no mocks).

Every number is a real seeded computation on small categorical pmfs. The
load-bearing anchor is **ISC-26**: ``rate=0`` returns the belief unchanged for
every kind, and ``rate=1`` ``confident_wrong`` concentrates on ``wrong_state``.
The contaminated sentinels are then fed to :func:`robust_aggregate` to confirm
they are genuinely down-weighted — the whole point of manufacturing them.
"""

from __future__ import annotations

import numpy as np
import pytest

from fedference.aggregation import robust_aggregate
from fedference.contamination import contaminate


def _rng() -> np.random.Generator:
    return np.random.default_rng(20260624)


# ---- ISC-26: the identity anchor -----------------------------------------

@pytest.mark.parametrize(
    "kind", ["confident_wrong", "label_noise", "uniform", "byzantine", "drift"]
)
def test_rate_zero_is_identity(kind):
    belief = np.array([0.6, 0.3, 0.1])
    out = contaminate(belief, kind=kind, rate=0.0, rng=_rng(), n_rounds=4, round_index=2)
    # input is already normalized, so the pmf must come back bit-for-bit.
    np.testing.assert_allclose(out, belief, atol=1e-12)


def test_rate_zero_identity_renormalises_unnormalised_input():
    belief = np.array([6.0, 3.0, 1.0])  # not summing to 1
    out = contaminate(belief, kind="uniform", rate=0.0, rng=_rng())
    np.testing.assert_allclose(out, np.array([0.6, 0.3, 0.1]), atol=1e-12)
    assert out.sum() == pytest.approx(1.0)


# ---- ISC-26: the rate=1 confident_wrong limit ----------------------------

def test_confident_wrong_rate_one_concentrates_on_wrong_state():
    belief = np.array([0.7, 0.2, 0.1])
    out = contaminate(belief, kind="confident_wrong", rate=1.0, rng=_rng(),
                      wrong_state=2)
    assert np.argmax(out) == 2
    assert out[2] == pytest.approx(1.0, abs=1e-9)
    assert out[0] == pytest.approx(0.0, abs=1e-9)
    assert out[1] == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(out, np.array([0.0, 0.0, 1.0]), atol=1e-9)


def test_confident_wrong_default_targets_least_mass_state():
    belief = np.array([0.7, 0.05, 0.25])  # least mass is index 1
    out = contaminate(belief, kind="confident_wrong", rate=1.0, rng=_rng())
    assert np.argmax(out) == 1


def test_confident_wrong_partial_mix_is_convex():
    belief = np.array([0.7, 0.2, 0.1])
    rate = 0.4
    out = contaminate(belief, kind="confident_wrong", rate=rate, rng=_rng(),
                      wrong_state=2)
    onehot = np.array([0.0, 0.0, 1.0])
    expected = (1 - rate) * belief + rate * onehot
    np.testing.assert_allclose(out, expected, atol=1e-9)
    assert out.sum() == pytest.approx(1.0)


def test_confident_wrong_out_of_range_raises():
    belief = np.array([0.5, 0.5])
    with pytest.raises(ValueError, match="wrong_state out of range"):
        contaminate(belief, kind="confident_wrong", rate=0.5, rng=_rng(),
                    wrong_state=5)


# ---- label_noise ---------------------------------------------------------

def test_label_noise_is_deterministic_given_seed():
    belief = np.array([0.5, 0.3, 0.2])
    a = contaminate(belief, kind="label_noise", rate=0.5, rng=np.random.default_rng(7))
    b = contaminate(belief, kind="label_noise", rate=0.5, rng=np.random.default_rng(7))
    np.testing.assert_allclose(a, b, atol=1e-15)


def test_label_noise_changes_belief_and_stays_a_pmf():
    belief = np.array([0.5, 0.3, 0.2])
    out = contaminate(belief, kind="label_noise", rate=0.5, rng=_rng())
    assert out.sum() == pytest.approx(1.0)
    assert np.all(out > 0.0)
    assert not np.allclose(out, belief)


# ---- uniform -------------------------------------------------------------

def test_uniform_rate_one_is_uniform():
    belief = np.array([0.7, 0.2, 0.1])
    out = contaminate(belief, kind="uniform", rate=1.0, rng=_rng())
    np.testing.assert_allclose(out, np.full(3, 1.0 / 3.0), atol=1e-9)


def test_uniform_partial_increases_entropy():
    belief = np.array([0.8, 0.15, 0.05])
    out = contaminate(belief, kind="uniform", rate=0.5, rng=_rng())
    h_in = -np.sum(belief * np.log(belief))
    h_out = -np.sum(out * np.log(out))
    assert h_out > h_in


# ---- byzantine: multiplicative targeted boost ----------------------------

def test_byzantine_tilts_toward_target_state():
    belief = np.array([0.6, 0.3, 0.1])
    out = contaminate(belief, kind="byzantine", rate=1.0, rng=_rng(), target_state=2)
    # the targeted state's log-odds are boosted, so it becomes the mode...
    assert np.argmax(out) == 2
    # ...but the attack is multiplicative, so the *other* states keep their
    # relative order (unlike a pure one-hot spike): state 0 still beats state 1.
    assert out[0] > out[1]
    assert out.sum() == pytest.approx(1.0)


def test_byzantine_strength_is_monotone_in_rate():
    belief = np.array([0.6, 0.3, 0.1])
    masses = [
        contaminate(belief, kind="byzantine", rate=r, rng=_rng(), target_state=2)[2]
        for r in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    # heavier rate piles more mass on the target, strictly.
    assert all(masses[i] < masses[i + 1] for i in range(len(masses) - 1))


def test_byzantine_default_targets_least_mass_state():
    belief = np.array([0.7, 0.05, 0.25])  # least mass at index 1
    out = contaminate(belief, kind="byzantine", rate=1.0, rng=_rng())
    assert np.argmax(out) == 1


# ---- drift: slowly-moving bias across rounds ------------------------------

def test_drift_first_round_is_clean():
    belief = np.array([0.6, 0.3, 0.1])
    out = contaminate(belief, kind="drift", rate=0.9, rng=_rng(),
                      target_state=2, round_index=0, n_rounds=5)
    np.testing.assert_allclose(out, belief, atol=1e-12)


def test_drift_bias_grows_across_rounds():
    belief = np.array([0.6, 0.3, 0.1])
    masses = [
        contaminate(belief, kind="drift", rate=0.9, rng=_rng(),
                    target_state=2, round_index=r, n_rounds=5)[2]
        for r in range(5)
    ]
    # the target-state mass rises monotonically as the bias creeps in.
    assert all(masses[i] <= masses[i + 1] + 1e-12 for i in range(len(masses) - 1))
    assert masses[-1] > masses[0]


def test_drift_round_index_out_of_range_raises():
    with pytest.raises(ValueError, match="round_index"):
        contaminate(np.array([0.5, 0.5]), kind="drift", rate=0.5, rng=_rng(),
                    round_index=5, n_rounds=3)


def test_zero_rounds_raises():
    with pytest.raises(ValueError, match="n_rounds must be"):
        contaminate(np.array([0.5, 0.5]), kind="drift", rate=0.5, rng=_rng(),
                    n_rounds=0)


# ---- error paths ---------------------------------------------------------

def test_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown kind"):
        contaminate(np.array([0.5, 0.5]), kind="bogus", rate=0.5, rng=_rng())


@pytest.mark.parametrize("rate", [-0.1, 1.5])
def test_rate_out_of_bounds_raises(rate):
    with pytest.raises(ValueError, match="rate must lie"):
        contaminate(np.array([0.5, 0.5]), kind="uniform", rate=rate, rng=_rng())


def test_missing_rng_raises():
    with pytest.raises(ValueError, match="rng is required"):
        contaminate(np.array([0.5, 0.5]), kind="uniform", rate=0.5, rng=None)


def test_empty_belief_raises():
    with pytest.raises(ValueError, match="empty"):
        contaminate(np.array([]), kind="uniform", rate=0.5, rng=_rng())


def test_negative_belief_raises():
    with pytest.raises(ValueError, match="negative"):
        contaminate(np.array([0.5, -0.5, 1.0]), kind="uniform", rate=0.5, rng=_rng())


# ---- integration: robust aggregation suppresses the contaminated agent ---

def test_confident_wrong_agent_is_downweighted_by_robust_pool():
    honest = np.array([0.8, 0.15, 0.05])
    # three honest sentinels agree; one is confidently wrong on state 2.
    # rate=0.9 makes the adversary's argmax state 2 while keeping a non-zero
    # floor on the true state — a pure rate=1 delta is a measure-zero veto that
    # captures the product-of-experts consensus before reweighting can act.
    adversary = contaminate(honest, kind="confident_wrong", rate=0.9, rng=_rng(),
                            wrong_state=2)
    assert np.argmax(adversary) == 2  # the adversary is indeed confidently wrong
    beliefs = [honest, honest, honest, adversary]
    res = robust_aggregate(beliefs, robustness=5.0)
    # the adversary (index 3) must hold strictly less influence than each honest
    assert res.agent_weights[3] < res.agent_weights[0]
    assert res.agent_weights[3] < 0.25  # below its equal share
    # consensus stays close to the honest belief, not the adversarial spike
    assert np.argmax(res.consensus) == 0
