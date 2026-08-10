"""Experiment harness submodule — see :mod:`fedference.experiments`."""

from __future__ import annotations

from typing import Any

from experiment_config import DEFAULT_SENSITIVITY_ACUITY, DEFAULT_SENSITIVITY_COLONY_SIZES

from .belief_sharing import run_belief_sharing
from .worlds import run_hierarchical_world

#: Default trials-per-cell of the sensitivity sweep. Single definition consumed
#: by :func:`run_belief_sharing_sensitivity` below AND the SENS_N_TRIALS
#: manuscript token (src/manuscript_vars/loaders.py) so prose cannot drift from
#: the executed default.
DEFAULT_SENSITIVITY_N_TRIALS: int = 20


def run_belief_sharing_sensitivity(
    seed: int = 0,
    *,
    acuity_values: tuple[float, ...] = DEFAULT_SENSITIVITY_ACUITY,
    n_agents_values: tuple[int, ...] = DEFAULT_SENSITIVITY_COLONY_SIZES,
    n_trials: int = DEFAULT_SENSITIVITY_N_TRIALS,
) -> dict[str, Any]:
    """2-D location-accuracy sweep over sensor acuity x colony size.

    For each (acuity, n_agents) pair, runs ``n_trials`` independent instances
    of :func:`run_belief_sharing` (each with a unique seed offset) under both
    the communicating and the incommunicado condition and records the mean
    ``mean_accuracy`` across trials for each condition and the gap between them.

    Args:
        seed: Base RNG seed; individual cells use ``seed + trial_offset``.
        acuity_values: Sensor acuity levels to sweep (outer axis — rows).
        n_agents_values: Colony sizes to sweep (inner axis — columns).
        n_trials: Number of independent trials per cell (different seeds).

    Returns:
        Dict with:

        * ``acuity_values`` — list of acuity levels used (rows).
        * ``n_agents_values`` — list of colony sizes used (columns).
        * ``communicating_grid`` — 2-D list ``[n_acuity][n_agents]`` of mean
          accuracy under the communicating condition.
        * ``isolated_grid`` — same shape, incommunicado condition.
        * ``accuracy_gap_grid`` — communicating minus isolated (the federation
          benefit); positive entries mean sharing helps.
        * ``seed`` — base seed supplied.
    """
    acuity_list = list(acuity_values)
    n_agents_list = [int(n) for n in n_agents_values]
    n_trials = int(n_trials)
    if n_trials < 1:
        raise ValueError("n_trials must be a positive integer")

    comm_grid: list[list[float]] = []
    isol_grid: list[list[float]] = []
    gap_grid: list[list[float]] = []

    for ai, acuity in enumerate(acuity_list):
        row_comm: list[float] = []
        row_isol: list[float] = []
        row_gap: list[float] = []
        for ni, n_agents in enumerate(n_agents_list):
            comm_acc = 0.0
            isol_acc = 0.0
            for t in range(n_trials):
                # Deterministic per-cell seed: spread across a large range to
                # avoid collisions across the 2-D grid.
                cell_seed = seed + ai * 100_000 + ni * 1_000 + t
                r_comm = run_belief_sharing(
                    cell_seed, communicate=True, n_agents=n_agents, acuity=acuity
                )
                r_isol = run_belief_sharing(
                    cell_seed, communicate=False, n_agents=n_agents, acuity=acuity
                )
                comm_acc += r_comm["mean_accuracy"] / n_trials
                isol_acc += r_isol["mean_accuracy"] / n_trials
            row_comm.append(float(comm_acc))
            row_isol.append(float(isol_acc))
            row_gap.append(float(comm_acc - isol_acc))
        comm_grid.append(row_comm)
        isol_grid.append(row_isol)
        gap_grid.append(row_gap)

    return {
        "acuity_values": acuity_list,
        "n_agents_values": n_agents_list,
        "communicating_grid": comm_grid,
        "isolated_grid": isol_grid,
        "accuracy_gap_grid": gap_grid,
        "seed": int(seed),
        "n_trials": int(n_trials),
    }


def run_hierarchical_sensitivity(
    seed: int = 0,
    *,
    acuity_values: tuple[float, ...] = DEFAULT_SENSITIVITY_ACUITY,
    n_agents_values: tuple[int, ...] = DEFAULT_SENSITIVITY_COLONY_SIZES,
    n_trials: int = 20,
    n_iters: int = 4,
) -> dict[str, Any]:
    """2-D location-accuracy sweep for the hierarchical POMDP.

    For each (acuity, n_agents) pair, runs :func:`run_hierarchical_world` with
    ``n_trials`` trials and records the flat and hierarchical location accuracy
    and their gap.

    Args:
        seed: RNG seed (same seed per cell so the sweep is fully reproducible).
        acuity_values: Sensor acuity levels to sweep (outer axis — rows).
        n_agents_values: Colony sizes to sweep (inner axis — columns).
        n_trials: Trials per cell inside each :func:`run_hierarchical_world` call.
        n_iters: Alternating-minimization iterations per inference call.

    Returns:
        Dict with:

        * ``acuity_values``, ``n_agents_values`` — parameter axes.
        * ``flat_grid`` — 2-D list ``[n_acuity][n_agents]`` of flat location accuracy.
        * ``hierarchical_grid`` — same shape, hierarchical condition.
        * ``accuracy_gap_grid`` — hierarchical minus flat.
        * ``seed`` — seed supplied.
    """
    acuity_list = list(acuity_values)
    n_agents_list = [int(n) for n in n_agents_values]

    flat_grid: list[list[float]] = []
    hier_grid: list[list[float]] = []
    gap_grid: list[list[float]] = []

    for acuity in acuity_list:
        row_flat: list[float] = []
        row_hier: list[float] = []
        row_gap: list[float] = []
        for n_agents in n_agents_list:
            r = run_hierarchical_world(
                seed,
                n_agents=n_agents,
                n_trials=n_trials,
                acuity=acuity,
                n_iters=n_iters,
            )
            flat_acc = float(r["location_accuracy"]["flat"])
            hier_acc = float(r["location_accuracy"]["hierarchical"])
            row_flat.append(flat_acc)
            row_hier.append(hier_acc)
            row_gap.append(float(r["location_accuracy_gap"]))
        flat_grid.append(row_flat)
        hier_grid.append(row_hier)
        gap_grid.append(row_gap)

    return {
        "acuity_values": acuity_list,
        "n_agents_values": n_agents_list,
        "flat_grid": flat_grid,
        "hierarchical_grid": hier_grid,
        "accuracy_gap_grid": gap_grid,
        "seed": int(seed),
        "n_trials": int(n_trials),
    }

