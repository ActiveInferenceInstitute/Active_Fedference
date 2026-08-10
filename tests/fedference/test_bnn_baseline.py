"""Tests for the FedGVI logistic-regression baseline (no mocks, real seeded runs).

Pins the two anchoring claims:

* **ISC-31** — :func:`fedference.bnn_baseline.fed_gvi_logreg` is deterministic
  under a fixed seed and returns a ``test_accuracy`` in ``[0, 1]``.
* **ISC-32** — at ``contamination = 0.3`` the robust RCCE client
  (``loss_param ~ 0.7``) achieves strictly higher test accuracy than the
  NLL / KLD baseline — the federated robustness property FedGVI predicts.

Every number is a real computation on synthetic Gaussian-blob data with an
explicit ``np.random.default_rng`` seed; there are no mocks.
"""

from __future__ import annotations

import numpy as np
import pytest

from fedference.bnn_baseline import (
    _accuracy,
    _client_update,
    _loss_grad_scale,
    _sigmoid,
    contaminate,
    fed_gvi_logreg,
    make_blobs,
)

# ---- ISC-31: determinism and bounded accuracy ----------------------------

def test_returns_accuracy_in_unit_interval():
    out = fed_gvi_logreg(seed=0)
    assert set(out) == {"test_accuracy", "weights"}
    assert 0.0 <= out["test_accuracy"] <= 1.0


def test_deterministic_under_fixed_seed():
    a = fed_gvi_logreg(n_clients=4, n_per=50, contamination=0.2, seed=7)
    b = fed_gvi_logreg(n_clients=4, n_per=50, contamination=0.2, seed=7)
    assert a["test_accuracy"] == b["test_accuracy"]
    assert np.array_equal(a["weights"], b["weights"])


def test_weights_shape_is_features_plus_bias():
    out = fed_gvi_logreg(seed=1)
    assert out["weights"].shape == (3,)  # 2 features + bias


def test_different_seeds_give_different_runs():
    a = fed_gvi_logreg(seed=0)
    b = fed_gvi_logreg(seed=1)
    assert not np.array_equal(a["weights"], b["weights"])


def test_clean_baseline_learns_a_useful_boundary():
    # With no contamination the federated estimator must beat chance comfortably.
    out = fed_gvi_logreg(contamination=0.0, loss="nll", seed=0)
    assert out["test_accuracy"] > 0.75


# ---- ISC-32: robustness of RCCE under contamination ----------------------

def test_rcce_beats_nll_under_contamination():
    nll = fed_gvi_logreg(contamination=0.3, loss="nll", divergence="KLD", seed=0)
    rcce = fed_gvi_logreg(
        contamination=0.3, loss="rcce", loss_param=0.7, divergence="KLD", seed=0
    )
    assert rcce["test_accuracy"] > nll["test_accuracy"]


def test_rcce_advantage_holds_across_seeds():
    # Not a fluke of one seed: the robust client wins on every seed in a sweep.
    for seed in range(5):
        nll = fed_gvi_logreg(contamination=0.3, loss="nll", seed=seed)
        rcce = fed_gvi_logreg(
            contamination=0.3, loss="rcce", loss_param=0.7, seed=seed
        )
        assert rcce["test_accuracy"] > nll["test_accuracy"], f"seed {seed}"


def test_rcce_with_zero_param_recovers_nll():
    # loss_param q -> 0 makes the RCCE gradient scale p^q -> 1, i.e. exact NLL.
    nll = fed_gvi_logreg(contamination=0.3, loss="nll", seed=2)
    rcce0 = fed_gvi_logreg(contamination=0.3, loss="rcce", loss_param=0.0, seed=2)
    assert rcce0["test_accuracy"] == pytest.approx(nll["test_accuracy"])
    assert np.allclose(rcce0["weights"], nll["weights"])


def test_ar_divergence_path_runs():
    out = fed_gvi_logreg(divergence="AR", seed=0)
    assert 0.0 <= out["test_accuracy"] <= 1.0


def test_rcce_separation_is_not_a_knife_edge_in_loss_param():
    # Advisor + Forge cross-vendor review of the bnn_robustness figure
    # operating point (loss_param q=1.0) flagged two risks: (1) separation
    # appears only at that exact value — figure-tuning, not a genuine effect;
    # (2) a sign-only check (robust > standard) at a single contamination
    # level would pass on a noise-level 0.0001 margin. This requires a real
    # minimum margin (0.005 — well above the ~0.002 noise floor seen at low
    # contamination) across a neighborhood of q (0.6-0.95) AND at two
    # contamination levels (0.3, 0.35) inside the genuine-separation range,
    # at the n_per=200 / 20-seed operating point the figure actually uses.
    n_per = 200
    seeds = range(20)
    min_margin = 0.005
    for contamination in (0.3, 0.35):
        standard_mean = float(np.mean([
            fed_gvi_logreg(
                n_per=n_per, contamination=contamination, loss="nll",
                divergence="KLD", seed=s,
            )["test_accuracy"]
            for s in seeds
        ]))
        for q in (0.6, 0.7, 0.8, 0.9, 1.0):
            robust_mean = float(np.mean([
                fed_gvi_logreg(
                    n_per=n_per, contamination=contamination, loss="rcce",
                    loss_param=q, divergence="AR", seed=s,
                )["test_accuracy"]
                for s in seeds
            ]))
            gap = robust_mean - standard_mean
            assert gap > min_margin, (
                f"contamination={contamination}, loss_param={q}: "
                f"gap {gap:.4f} does not clear the {min_margin} margin floor"
            )


# ---- component-level checks (full branch coverage) -----------------------

def test_sigmoid_is_stable_both_branches():
    s = _sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    assert s[0] == pytest.approx(0.0, abs=1e-9)
    assert s[1] == pytest.approx(0.5)
    assert s[2] == pytest.approx(1.0)


def test_make_blobs_shapes_and_labels():
    rng = np.random.default_rng(3)
    x, y = make_blobs(40, rng=rng)
    assert x.shape == (80, 2)
    assert set(np.unique(y)) == {0, 1}
    assert int((y == 0).sum()) == 40 and int((y == 1).sum()) == 40


def test_make_blobs_rejects_nonpositive():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        make_blobs(0, rng=rng)


def test_contaminate_zero_fraction_is_identity():
    rng = np.random.default_rng(0)
    x, y = make_blobs(20, rng=rng)
    assert np.array_equal(contaminate(x, y, 0.0, rng=rng), y)


def test_contaminate_flips_expected_count():
    rng = np.random.default_rng(1)
    x, y = make_blobs(20, rng=rng)
    flipped = contaminate(x, y, 0.25, rng=np.random.default_rng(5))
    assert int((flipped != y).sum()) == 10  # round(0.25 * 40)


def test_contaminate_rejects_out_of_range_fraction():
    rng = np.random.default_rng(0)
    x, y = make_blobs(10, rng=rng)
    with pytest.raises(ValueError):
        contaminate(x, y, 1.5, rng=rng)


def test_contaminate_targets_outliers():
    # Flips should land on high-|x0| (high-leverage) points, not uniformly.
    rng = np.random.default_rng(0)
    x, y = make_blobs(60, rng=rng)
    flipped = contaminate(x, y, 0.2, rng=np.random.default_rng(0))
    changed = flipped != y
    assert np.mean(np.abs(x[changed, 0])) > np.mean(np.abs(x[~changed, 0]))


def test_loss_grad_scale_branches():
    p = np.array([0.2, 0.9])
    assert np.array_equal(_loss_grad_scale(p, "nll", 0.0), np.ones(2))
    rcce_scale = _loss_grad_scale(p, "rcce", 0.7)
    assert np.allclose(rcce_scale, p**0.7)
    # the confidently-wrong (low-p) point is down-weighted more than the right one.
    assert rcce_scale[0] < rcce_scale[1]


def test_loss_grad_scale_rejects_bad_inputs():
    with pytest.raises(ValueError):
        _loss_grad_scale(np.array([0.5]), "rcce", 1.5)
    with pytest.raises(ValueError):
        _loss_grad_scale(np.array([0.5]), "bogus", 0.0)


def test_client_update_reduces_to_useful_fit():
    rng = np.random.default_rng(0)
    x, y = make_blobs(60, rng=rng)
    w0 = np.zeros(3)
    w = _client_update(x, y, w0, loss="nll", loss_param=0.0, steps=100, lr=0.8, l2=0.0)
    assert _accuracy(x, y, w) > 0.8
    assert not np.allclose(w, w0)


def test_fed_gvi_rejects_zero_clients():
    with pytest.raises(ValueError):
        fed_gvi_logreg(n_clients=0)


def test_fed_gvi_rejects_unknown_loss_and_divergence():
    with pytest.raises(ValueError):
        fed_gvi_logreg(loss="bogus")
    with pytest.raises(ValueError):
        fed_gvi_logreg(divergence="ZZ")


def test_fed_gvi_rejects_rcce_param_out_of_range():
    with pytest.raises(ValueError):
        fed_gvi_logreg(loss="rcce", loss_param=1.5)
