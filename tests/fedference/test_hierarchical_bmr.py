"""Hierarchical surprise-threshold control (legacy BMR stem), with real inference.

On a configured degenerate N=3 world whose top meta-context is non-gating,
``hierarchical_reduce`` must flag that level below the declared surprise
threshold; on its configured informative control it must keep every level.
Both directions are pinned as implementation controls because the worlds differ
only in the top-level conditioned priors. They are not a universal structure-
emergence or model-selection-consistency result.
"""

from __future__ import annotations

import numpy as np
import pytest

from fedference.bayesian_model_reduction import hierarchical_reduce
from fedference.experiments import run_hierarchical_bmr
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
    # The configured rule recommends the topmost redundant level.
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
    opposite threshold verdicts, showing this configured output is not constant."""
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
    assert [lv["bayesian_surprise"] for lv in a["levels"]] == [lv["bayesian_surprise"] for lv in b["levels"]]


def test_rejects_degenerate_world_shape() -> None:
    try:
        hierarchical_reduce({"n_levels": 1}, _leaf_A(), obs=0)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "n_levels" in str(exc)


def test_report_identifies_surprise_threshold_as_primary_control() -> None:
    report = run_hierarchical_bmr(surprise_tol=1e-3)
    assert report["schema_version"] == "1.0"
    assert report["study_status"] == "configured_deterministic_control"
    assert report["configured_control_passed"] is True
    assert report["surprise_tol"] == 1e-3
    assert report["degenerate_top_surprise"] < report["surprise_tol"]
    assert report["informative_top_surprise"] >= report["surprise_tol"]
    assert "bayesian_surprise < surprise_tol" in report["decision_rule"]
    assert "does not determine" in report["secondary_diagnostic"]
    assert "no universal structure-emergence" in report["claim_boundary"]


def test_report_threshold_controls_nested_prunable_flags() -> None:
    report = run_hierarchical_bmr(surprise_tol=5e-2)
    for world_name in ("degenerate", "informative"):
        for level in report[world_name]["levels"]:
            assert level["prunable"] is (level["bayesian_surprise"] < report["surprise_tol"])


def test_report_seed_is_compatibility_provenance_not_replication() -> None:
    first = run_hierarchical_bmr(seed=1)
    second = run_hierarchical_bmr(seed=99)
    assert first["degenerate"] == second["degenerate"]
    assert first["informative"] == second["informative"]
    assert first["seed"] == 1
    assert second["seed"] == 99
    assert "no RNG draw" in first["seed_role"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"surprise_tol": 0.0}, "surprise_tol"),
        ({"surprise_tol": float("nan")}, "surprise_tol"),
        ({"n_iters": 0}, "n_iters"),
        ({"obs": -1}, "obs"),
    ),
)
def test_report_rejects_invalid_control_configuration(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        run_hierarchical_bmr(**kwargs)
