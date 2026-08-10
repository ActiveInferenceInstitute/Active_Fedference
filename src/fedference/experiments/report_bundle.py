"""Workflow report bundles: single-seed point estimates plus multi-seed statistics."""

from __future__ import annotations

from typing import Any

from ..statistics import interpret_effect_size, multiseed_summary, paired_test
from .navigation import run_disjoint_fov_world, run_efe_navigation_test
from .worlds import run_3level_world, run_hierarchical_world, run_moving_world

_DEFAULT_HIER_SEED = 42
_DEFAULT_MULTISEED = 10


def _paired_test_or_tied(
    primary_vals: list[float], baseline_vals: list[float]
) -> dict[str, float]:
    """Wilcoxon paired test; identical series yield a documented null result."""
    try:
        pt = paired_test(primary_vals, baseline_vals)
    except ValueError:
        pt = {"statistic": 0.0, "pvalue": 1.0, "effect_size": 0.0}
    return {
        "pvalue": float(pt["pvalue"]),
        "effect_size": float(pt["effect_size"]),
    }


def _attach_multiseed_paired(
    report: dict[str, Any],
    *,
    n_multiseed: int,
    primary_vals: list[float],
    baseline_vals: list[float],
    gap_vals: list[float] | None = None,
) -> dict[str, Any]:
    primary_ms = multiseed_summary(primary_vals)
    baseline_ms = multiseed_summary(baseline_vals)
    gap_ms = multiseed_summary(gap_vals) if gap_vals is not None else None
    pt = _paired_test_or_tied(primary_vals, baseline_vals)
    enriched = {
        **report,
        "multiseed": {
            "n_seeds": int(n_multiseed),
            "primary": primary_ms,
            "baseline": baseline_ms,
            "paired_test": pt,
            "effect_label": interpret_effect_size(abs(float(pt["effect_size"]))),
        },
    }
    if gap_ms is not None:
        enriched["multiseed"]["gap"] = gap_ms
    return enriched


def hierarchical_world_report(
    seed: int = _DEFAULT_HIER_SEED,
    *,
    n_multiseed: int = _DEFAULT_MULTISEED,
    n_agents: int = 4,
    n_trials: int = 20,
    acuity: float = 0.85,
    n_iters: int = 4,
) -> dict[str, Any]:
    """Point estimate at ``seed`` plus multi-seed location-accuracy statistics."""
    run_kwargs: dict[str, Any] = {
        "n_agents": n_agents,
        "n_trials": n_trials,
        "acuity": acuity,
        "n_iters": n_iters,
    }
    point = dict(run_hierarchical_world(seed=seed, **run_kwargs))
    hier_vals: list[float] = []
    flat_vals: list[float] = []
    gap_vals: list[float] = []
    for s in range(n_multiseed):
        r = run_hierarchical_world(seed=s, **run_kwargs)
        hier_vals.append(float(r["location_accuracy"]["hierarchical"]))
        flat_vals.append(float(r["location_accuracy"]["flat"]))
        gap_vals.append(float(r["location_accuracy_gap"]))
    return _attach_multiseed_paired(
        point,
        n_multiseed=n_multiseed,
        primary_vals=hier_vals,
        baseline_vals=flat_vals,
        gap_vals=gap_vals,
    )


def nlevel3_world_report(
    seed: int = _DEFAULT_HIER_SEED,
    *,
    n_multiseed: int = _DEFAULT_MULTISEED,
    n_agents: int = 4,
    n_trials: int = 20,
    acuity: float = 0.85,
    n_iters: int = 4,
) -> dict[str, Any]:
    """Point estimate at ``seed`` plus multi-seed 3-level location-accuracy statistics."""
    run_kwargs: dict[str, Any] = {
        "n_agents": n_agents,
        "n_trials": n_trials,
        "acuity": acuity,
        "n_iters": n_iters,
    }
    point = dict(run_3level_world(seed=seed, **run_kwargs))
    nlevel_vals: list[float] = []
    flat_vals: list[float] = []
    gap_vals: list[float] = []
    for s in range(n_multiseed):
        r = run_3level_world(seed=s, **run_kwargs)
        nlevel_vals.append(float(r["location_accuracy"]["nlevel3"]))
        flat_vals.append(float(r["location_accuracy"]["flat"]))
        gap_vals.append(float(r["location_accuracy_gap"]))
    return _attach_multiseed_paired(
        point,
        n_multiseed=n_multiseed,
        primary_vals=nlevel_vals,
        baseline_vals=flat_vals,
        gap_vals=gap_vals,
    )


def moving_world_report(
    seed: int,
    *,
    n_multiseed: int = _DEFAULT_MULTISEED,
    n_trials: int = 20,
    **run_kwargs: Any,
) -> dict[str, Any]:
    """Moving-world report with multi-seed accuracy and EFE-vs-isolated statistics."""
    point = dict(run_moving_world(seed, n_trials=n_trials, **run_kwargs))
    iso_vals: list[float] = []
    comm_vals: list[float] = []
    efe_vals: list[float] = []
    fe_gap_vals: list[float] = []
    for s in range(n_multiseed):
        r = run_moving_world(s, n_trials=n_trials, **run_kwargs)
        iso_vals.append(float(r["accuracy"]["isolated"]))
        comm_vals.append(float(r["accuracy"]["communicating"]))
        efe_vals.append(float(r["accuracy"]["efe_guided"]))
        fe_gap_vals.append(float(r["free_energy_gap"]["efe_guided"]))
    iso_ms = multiseed_summary(iso_vals)
    comm_ms = multiseed_summary(comm_vals)
    efe_ms = multiseed_summary(efe_vals)
    fe_gap_ms = multiseed_summary(fe_gap_vals)
    pt_efe = _paired_test_or_tied(efe_vals, iso_vals)
    return {
        **point,
        "multiseed": {
            "n_seeds": int(n_multiseed),
            "isolated": iso_ms,
            "communicating": comm_ms,
            "efe_guided": efe_ms,
            "efe_free_energy_gap": fe_gap_ms,
            "efe_vs_isolated": {
                "paired_test": pt_efe,
                "effect_label": interpret_effect_size(abs(float(pt_efe["effect_size"]))),
            },
        },
    }


def disjoint_fov_report(
    seed: int = 0,
    *,
    n_multiseed: int = _DEFAULT_MULTISEED,
    **run_kwargs: Any,
) -> dict[str, Any]:
    """Disjoint-FOV necessity contrast plus EFE-navigation multi-seed statistics."""
    point = dict(run_disjoint_fov_world(seed=seed, **run_kwargs))
    iso_vals: list[float] = []
    comm_vals: list[float] = []
    efe_vals: list[float] = []
    rnd_vals: list[float] = []
    for s in range(n_multiseed):
        fov = run_disjoint_fov_world(seed=s, **run_kwargs)
        iso_vals.append(float(fov["isolated_accuracy"]))
        comm_vals.append(float(fov["communicating_accuracy"]))
        nav = run_efe_navigation_test(seed=s)
        efe_vals.append(float(nav["efe_accuracy"]))
        rnd_vals.append(float(nav["random_accuracy"]))
    iso_ms = multiseed_summary(iso_vals)
    comm_ms = multiseed_summary(comm_vals)
    pt_comm = _paired_test_or_tied(comm_vals, iso_vals)
    efe_ms = multiseed_summary(efe_vals)
    rnd_ms = multiseed_summary(rnd_vals)
    pt_efe = _paired_test_or_tied(efe_vals, rnd_vals)
    return {
        **point,
        "multiseed": {
            "n_seeds": int(n_multiseed),
            "isolated": iso_ms,
            "communicating": comm_ms,
            "communicating_vs_isolated": {
                "paired_test": pt_comm,
                "effect_label": interpret_effect_size(abs(float(pt_comm["effect_size"]))),
            },
            "efe_guided": efe_ms,
            "random": rnd_ms,
            "efe_vs_random": {
                "paired_test": pt_efe,
                "effect_label": interpret_effect_size(abs(float(pt_efe["effect_size"]))),
            },
        },
        "efe_navigation": dict(run_efe_navigation_test(seed=seed)),
    }


__all__ = [
    "disjoint_fov_report",
    "hierarchical_world_report",
    "moving_world_report",
    "nlevel3_world_report",
]
