"""Experiment harness submodule — see :mod:`fedference.experiments`."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..aggregation import log_linear_pool


def run_disjoint_fov_world(
    seed: int = 0,
    *,
    n_agents: int = 3,
    n_positions: int = 6,
    fov_width: int = 2,
    n_steps: int = 4,
) -> dict[str, Any]:
    """Multi-agent moving world: each agent sees only ``fov_width`` consecutive positions.

    The hidden state is the *exact position* of the threat in a linear grid of
    ``n_positions`` cells (so ``n_states == n_positions``). Agent ``i`` starts
    at position ``i * (n_positions // n_agents)`` and observes only the
    ``fov_width`` cells beginning there — making the FOVs **disjoint** by
    construction when ``n_agents * fov_width == n_positions``.

    Two conditions are evaluated over many seeded trials:

    * **Isolated** — each agent updates its belief from its own noisy observation
      only (no communication). Because each agent's FOV covers exactly
      ``fov_width / n_positions`` of the state space, its posterior over states
      outside its window stays near the prior, and its accuracy is bounded.
    * **Communicating** — agents additionally fuse beliefs via
      :func:`fedference.aggregation.log_linear_pool` after every observation,
      so complementary partial evidence is combined into a full-coverage
      consensus.

    Args:
        seed: RNG seed for reproducible trials.
        n_agents: Number of sentinels. Must evenly divide ``n_positions``.
        n_positions: Grid cells and hidden states (``n_states == n_positions``).
        fov_width: Cells observed per agent per step (``fov_width * n_agents == n_positions``).
        n_steps: Observation steps per trial; more steps let beliefs converge.

    Returns:
        Dict with ``isolated_accuracy``, ``communicating_accuracy``, ``gap``
        (communicating minus isolated), ``n_agents``, and ``fov_width``.
    """
    _EPS = 1e-12
    rng = np.random.default_rng(seed)
    n_states = n_positions  # threat position = hidden state

    # Number of trials large enough for stable estimates.
    n_trials = 200

    # Agent starting positions: evenly tiled, non-overlapping when
    # n_agents * fov_width == n_positions.
    spacing = n_positions // n_agents
    agent_positions = [i * spacing for i in range(n_agents)]

    # Build per-agent likelihood matrices: shape (2, n_states).
    # Observation 0 = "detected"  (threat is inside this agent's FOV).
    # Observation 1 = "not_detected".
    # Acuity 0.90: high probability of correct reading.
    acuity = 0.90
    A_agents = []
    for i, p in enumerate(agent_positions):
        lo = p
        hi = min(p + fov_width, n_positions)
        a = np.full((2, n_states), (1.0 - acuity) / max(1, n_states - (hi - lo)), dtype=np.float64)
        # Columns for states in the FOV window.
        for s in range(lo, hi):
            a[0, s] = acuity        # P(detect | threat at s, s in FOV)
            a[1, s] = 1.0 - acuity  # P(not_detect | threat at s, s in FOV)
        # Columns for states outside the FOV: very low detection probability.
        out_states = [s for s in range(n_states) if s < lo or s >= hi]
        n_out = len(out_states)
        if n_out > 0:
            for s in out_states:
                a[0, s] = (1.0 - acuity) / n_out  # rare false alarm
                a[1, s] = 1.0 - a[0, s]
        # Normalise columns to valid pmfs.
        for s in range(n_states):
            col_sum = a[:, s].sum()
            if col_sum > 0:
                a[:, s] /= col_sum
        A_agents.append(a)

    iso_correct = 0
    comm_correct = 0

    for trial in range(n_trials):
        true_state = int(rng.integers(0, n_states))

        # Initialise flat beliefs for both conditions.
        iso_local_posteriors = [
            np.full(n_states, 1.0 / n_states) for _ in range(n_agents)
        ]
        comm_local_posteriors = [
            np.full(n_states, 1.0 / n_states) for _ in range(n_agents)
        ]

        for _step in range(n_steps):
            # Each agent observes and updates.
            for i in range(n_agents):
                A_i = A_agents[i]
                # Sample observation from the true state column.
                p_obs = A_i[:, true_state]
                o = int(rng.choice(A_i.shape[0], p=p_obs / p_obs.sum()))

                # --- isolated update ---
                prior_iso = np.clip(iso_local_posteriors[i], _EPS, None)
                posterior_iso = A_i[o, :] * prior_iso
                posterior_iso = np.clip(posterior_iso, _EPS, None)
                iso_local_posteriors[i] = posterior_iso / posterior_iso.sum()

                # --- communicating update (same Bayesian step, pool comes after) ---
                prior_comm = np.clip(comm_local_posteriors[i], _EPS, None)
                posterior_comm = A_i[o, :] * prior_comm
                posterior_comm = np.clip(posterior_comm, _EPS, None)
                comm_local_posteriors[i] = posterior_comm / posterior_comm.sum()

            # Communicating condition: fuse after each step.
            consensus = log_linear_pool(comm_local_posteriors)
            comm_local_posteriors = [consensus.copy() for _ in range(n_agents)]

        # Score isolated: each agent votes; majority decides.
        iso_votes = [int(np.argmax(posterior)) for posterior in iso_local_posteriors]
        iso_consensus_state = int(np.bincount(iso_votes, minlength=n_states).argmax())
        iso_correct += int(iso_consensus_state == true_state)

        # Score communicating: use the shared consensus.
        comm_consensus_state = int(np.argmax(comm_local_posteriors[0]))
        comm_correct += int(comm_consensus_state == true_state)

    isolated_accuracy = float(iso_correct) / n_trials
    communicating_accuracy = float(comm_correct) / n_trials

    return {
        "isolated_accuracy": isolated_accuracy,
        "communicating_accuracy": communicating_accuracy,
        "gap": communicating_accuracy - isolated_accuracy,
        "n_agents": int(n_agents),
        "fov_width": int(fov_width),
        "n_positions": int(n_positions),
        "n_steps": int(n_steps),
        "n_trials": n_trials,
        "seed": int(seed),
    }


def run_efe_navigation_test(
    seed: int = 0,
    *,
    n_agents: int = 2,
    n_positions: int = 4,
    n_steps: int = 3,
) -> dict[str, Any]:
    """Compare EFE-guided vs. random movement combined with belief sharing.

    Both conditions share beliefs each step via
    :func:`fedference.aggregation.log_linear_pool`. The only difference is how
    agents choose their next position:

    * **EFE-guided** — each agent moves to the position that *minimises expected
      posterior entropy* after one observation (i.e. the most information-seeking
      move). Implemented by delegating to
      :func:`fedference.pomdp.efe_policy_select` which scores all three actions
      (stay / left / right) for every agent.
    * **Random** — each agent picks a uniformly random action each step.

    The hidden state is binary (left half vs. right half of the grid) and agents
    begin with disjoint FOVs as in :func:`build_moving_world`.

    Args:
        seed: RNG seed for reproducibility.
        n_agents: Number of sentinels.
        n_positions: Grid size (must be ``>= n_agents``).
        n_steps: Steps of observation per trial.

    Returns:
        Dict with ``efe_accuracy``, ``random_accuracy``, ``efe_gap``
        (efe minus random), and run parameters.
    """
    from ..pomdp import _moving_likelihood, build_moving_world, efe_policy_select

    _EPS = 1e-12
    rng = np.random.default_rng(seed)
    n_trials = 200
    n_states = 2  # binary hidden state (left-half / right-half threat)

    efe_correct = 0
    random_correct = 0

    for trial in range(n_trials):
        world = build_moving_world(
            n_positions=n_positions,
            n_agents=n_agents,
            n_states=n_states,
            fov_width=n_positions // n_agents,
            seed=int(rng.integers(0, 100_000)),
        )
        B = np.asarray(world["B"], dtype=np.float64)
        true_state = int(rng.integers(0, n_states))

        # Initialise per-condition beliefs and positions.
        efe_local_posteriors = [
            np.full(n_states, 1.0 / n_states) for _ in range(n_agents)
        ]
        rnd_local_posteriors = [
            np.full(n_states, 1.0 / n_states) for _ in range(n_agents)
        ]
        efe_positions = list(world["agent_positions"])
        rnd_positions = list(world["agent_positions"])

        for _step in range(n_steps):
            # --- action selection ---
            w_tmp = dict(world)
            w_tmp["agent_positions"] = efe_positions
            efe_actions = efe_policy_select(efe_local_posteriors, w_tmp)
            rnd_actions = [int(rng.integers(0, world["n_actions"])) for _ in range(n_agents)]

            # --- move ---
            for i in range(n_agents):
                efe_positions[i] = int(np.argmax(B[:, efe_positions[i], efe_actions[i]]))
                rnd_positions[i] = int(np.argmax(B[:, rnd_positions[i], rnd_actions[i]]))

            # --- observe and update beliefs ---
            for i in range(n_agents):
                for local_posteriors, positions in (
                    (efe_local_posteriors, efe_positions),
                    (rnd_local_posteriors, rnd_positions),
                ):
                    A_i = _moving_likelihood(
                        positions[i],
                        fov_width=int(world["fov_width"]),
                        n_positions=n_positions,
                        n_states=n_states,
                    )
                    p_obs = A_i[:, true_state]
                    p_obs = np.clip(p_obs, 0.0, None)
                    p_obs = p_obs / p_obs.sum()
                    o = int(rng.choice(A_i.shape[0], p=p_obs))
                    prior = np.clip(local_posteriors[i], _EPS, None)
                    posterior = A_i[o, :] * prior
                    posterior = np.clip(posterior, _EPS, None)
                    local_posteriors[i] = posterior / posterior.sum()

            # --- communicate (both conditions) ---
            for local_posteriors in (
                efe_local_posteriors,
                rnd_local_posteriors,
            ):
                consensus = log_linear_pool(local_posteriors)
                for i in range(n_agents):
                    local_posteriors[i] = consensus.copy()

        # --- score ---
        efe_consensus = log_linear_pool(efe_local_posteriors)
        rnd_consensus = log_linear_pool(rnd_local_posteriors)
        efe_correct += int(int(np.argmax(efe_consensus)) == true_state)
        random_correct += int(int(np.argmax(rnd_consensus)) == true_state)

    efe_accuracy = float(efe_correct) / n_trials
    random_accuracy = float(random_correct) / n_trials

    return {
        "efe_accuracy": efe_accuracy,
        "random_accuracy": random_accuracy,
        "efe_gap": efe_accuracy - random_accuracy,
        "n_agents": int(n_agents),
        "n_positions": int(n_positions),
        "n_steps": int(n_steps),
        "n_trials": n_trials,
        "seed": int(seed),
    }
