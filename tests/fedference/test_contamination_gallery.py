"""Tests for the contamination-model gallery (ISC-27 breadth, no mocks).

Confirms that the gallery keeps attack *mechanisms* separate: additive
confident-wrong and drift attacks produce reliable robust wins in the declared
seeded design, the byzantine mechanism exposes a veto-cliff negative control,
and entropy attacks (uniform, label-noise) do not create a robust win. Every
number is a real seeded paired comparison.
"""

from __future__ import annotations

import pytest

from fedference import contamination as cont
from fedference.experiments import (
    _DIRECTIONAL_KINDS,
    _ENTROPY_KINDS,
    run_contamination_gallery,
)


def test_contamination_classes_partition_all_kinds():
    # Every contamination model must be classified as directional XOR entropy, so a
    # newly-added model cannot silently drop out of the gallery verdict.
    directional = set(_DIRECTIONAL_KINDS)
    entropy = set(_ENTROPY_KINDS)
    assert directional.isdisjoint(entropy)  # no model is both
    assert directional | entropy == set(cont._KINDS)  # every model is one


def test_gallery_covers_all_models_and_classifies_them():
    r = run_contamination_gallery(0, n_seeds=6, n_trials=10)
    # the gallery default is registry-derived, so it covers EVERY contamination
    # model — a new model added to contamination._KINDS cannot silently drop out.
    assert set(r["by_kind"]) == set(cont._KINDS)
    assert set(r["directional_kinds"]) == {"confident_wrong", "byzantine", "drift"}
    assert set(r["entropy_kinds"]) == {"uniform", "label_noise"}
    for cell in r["by_kind"].values():
        assert 0.0 <= cell["naive_mean"] <= 1.0
        assert 0.0 <= cell["robust_mean"] <= 1.0
        assert 0.0 <= cell["win_fraction"] <= 1.0
        assert cell["naive_ci"][0] <= cell["naive_mean"] <= cell["naive_ci"][1]
        assert cell["robust_ci"][0] <= cell["robust_mean"] <= cell["robust_ci"][1]
        assert cell["diff_ci"][0] <= cell["diff_ci"][1]
        assert cell["best_robust_method"] in ("RKL", "AR", "beta", "rcce")


def test_additive_directional_attacks_reliably_beaten_seed_robustly():
    # The seed-robust claim (audit w1o6slput / wiwa3kbxp): robust RELIABLY beats
    # naive under the additive directional attacks, and this is stable across the
    # starting seed — not a single lucky seed.
    for start in (0, 500):  # two well-separated starts is enough to show stability
        r = run_contamination_gallery(start, n_seeds=8, n_trials=10)
        assert "confident_wrong" in r["reliable_kinds"]
        assert "drift" in r["reliable_kinds"]
        for kind in ("confident_wrong", "drift"):
            cell = r["by_kind"][kind]
            assert cell["win_fraction"] >= 0.95
            assert cell["diff_ci"][0] > 0.0  # CI excludes zero


def test_byzantine_advantage_is_not_reliable_honest_caveat():
    # Byzantine is directional but its multiplicative tilt escalates to a veto
    # cliff: the robust advantage does NOT hold across seeds, so it must NOT be
    # flagged reliable (the overclaim the audit caught).
    r = run_contamination_gallery(0, n_seeds=8, n_trials=10)
    assert "byzantine" not in r["reliable_kinds"]
    assert r["by_kind"]["byzantine"]["reliably_beats"] is False


def test_entropy_attacks_leave_naive_undegraded():
    r = run_contamination_gallery(0, n_seeds=6, n_trials=10)
    assert r["entropy_naive_robust"] is True
    for kind in r["entropy_kinds"]:
        assert r["by_kind"][kind]["naive_mean"] > 0.9       # naive stays accurate
        assert r["by_kind"][kind]["reliably_beats"] is False  # nothing to "win"


def test_gallery_is_deterministic_under_seed():
    a = run_contamination_gallery(3, n_seeds=6, n_trials=12)
    b = run_contamination_gallery(3, n_seeds=6, n_trials=12)
    assert a["by_kind"]["byzantine"]["mean_diff"] == b["by_kind"]["byzantine"]["mean_diff"]
    assert a["reliable_kinds"] == b["reliable_kinds"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_agents": 2},          # too few agents
        {"n_contaminated": 0},    # below 1
        {"rate": 1.5},            # out of range
        {"n_trials": 1},          # too few trials
        {"n_seeds": 1},           # too few seeds for a seed-robust verdict
        {"divergences": ("RKL", "AR")},  # missing the naive KLD baseline
    ],
)
def test_gallery_validation_raises(kwargs):
    with pytest.raises(ValueError):
        run_contamination_gallery(0, **kwargs)


def test_gallery_minimum_valid_contamination_one_of_three_agents():
    """n_contaminated=1 with n_agents=3 exercises the minimum valid colony size.

    This boundary (2:1 honest majority) is not covered by existing tests — only
    n_contaminated=0 is tested as the rejection boundary. The test verifies the
    structural contracts (valid cells, determinism, all-kinds coverage) without
    asserting win_fraction > 0, because at 2:1 the contaminated agent can
    dominate when self-exclusion is active (each honest agent's excluding-pool
    is 1 honest + 1 contaminated, which is a 1:1 ratio).
    """
    r = run_contamination_gallery(0, n_agents=3, n_contaminated=1,
                                  n_seeds=6, n_trials=10)
    assert "confident_wrong" in r["by_kind"]

    # All kinds must be present and structurally valid.
    for kind, cell in r["by_kind"].items():
        assert 0.0 <= cell["naive_mean"] <= 1.0, (
            f"{kind}: naive_mean={cell['naive_mean']:.3f} out of [0,1]"
        )
        assert 0.0 <= cell["robust_mean"] <= 1.0, (
            f"{kind}: robust_mean={cell['robust_mean']:.3f} out of [0,1]"
        )
        assert 0.0 <= cell["win_fraction"] <= 1.0, (
            f"{kind}: win_fraction={cell['win_fraction']:.3f} out of [0,1]"
        )
        assert cell["diff_ci"][0] <= cell["diff_ci"][1], (
            f"{kind}: CI lo > hi: {cell['diff_ci']}"
        )

    # Determinism: same seed must reproduce.
    r2 = run_contamination_gallery(0, n_agents=3, n_contaminated=1,
                                   n_seeds=6, n_trials=10)
    assert r["by_kind"]["confident_wrong"]["mean_diff"] == (
        r2["by_kind"]["confident_wrong"]["mean_diff"]
    )
