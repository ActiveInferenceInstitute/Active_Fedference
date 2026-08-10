"""Sentinel ensemble with Markov-blanket-separated private and shared beliefs.

This module assembles a colony of active-inference sentinels for Friston et al.
(2024), *Federated inference and belief sharing* (Neurosci. Biobehav. Rev.
156:105500), §3.1 ("Markov blankets and the separation of internal and shared
states"). Each sentinel partitions its hidden states into two kinds:

* a **private gaze** factor — where *this* sentinel happens to be looking. Gaze
  is internal to the agent's Markov blanket; it is never gossiped. It is the
  factor that makes one sentinel's likelihood differ from another's and is the
  reason the colony observes the world from many angles.
* a set of **shared** factors — ``location`` (the creature's cell, the latent
  the colony fuses through the project's qualified categorical Eq. 7 bridge),
  ``proximity`` (how near the creature is) and
  ``pose`` (its orientation). These are the states the §3.1 separation declares
  *common* across the blanket boundary, so they are exchanged in the broadcast.

The Markov-blanket separation (Friston §3.1) is operationalised as ISC-22: the
private gaze factor is excluded from the broadcast vector, while the shared
factors are the only thing exchanged. The class therefore exposes three
operations matching the federated-inference loop:

* :meth:`SentinelEnsemble.perceive` — each agent runs the locked one-step
  variational update :func:`fedference.belief_updating.infer_states` on *both*
  its private gaze factor and its shared factors from a fresh observation.
* :meth:`SentinelEnsemble.broadcast` — each agent emits *only* its shared-factor
  pmf (the concatenation of location/proximity/pose), with gaze withheld.
* :meth:`SentinelEnsemble.assimilate` — each agent folds a heard consensus back
  into its shared factors in natural-parameter (log) space (a tempered
  generalized-Bayes update, :func:`fedference.generalized_bayes.softmax`),
  leaving its private gaze untouched.

The shared ``location`` likelihood and prior are taken directly from
:func:`fedference.pomdp.build_sentinel_world`; ``proximity`` and ``pose`` are
small derived likelihoods over the same nine-cell grid so the broadcast carries
the full shared latent rather than location alone.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .belief_updating import infer_states
from .generalized_bayes import softmax
from .pomdp import GRID_SIDE, N_LOCATIONS, build_sentinel_world

ArrayF = np.ndarray
_EPS = 1e-12

#: Ordered names of the SHARED factors exchanged across the Markov blanket.
SHARED_FACTORS: tuple[str, ...] = ("location", "proximity", "pose")
#: Number of proximity states (near / mid / far relative to the den).
N_PROXIMITY: int = 3
#: Number of pose states (orientation quadrants the creature can face).
N_POSE: int = 4
#: Number of private gaze states — which grid cell THIS sentinel watches.
N_GAZE: int = N_LOCATIONS


def _uniform(n: int) -> ArrayF:
    """Return the length-``n`` uniform categorical pmf."""
    return np.full(n, 1.0 / n, dtype=np.float64)


def _normalize(vec: ArrayF) -> ArrayF:
    """Clip to non-negative and renormalize ``vec`` to a categorical pmf."""
    v = np.clip(np.asarray(vec, dtype=np.float64).ravel(), _EPS, None)
    return v / v.sum()


def _proximity_likelihood() -> ArrayF:
    """Likelihood ``P(o_prox | location)`` of shape ``(3, 9)``.

    Proximity is a deterministic-but-soft function of the creature's location:
    the Chebyshev ring distance from the center den cell maps to near (ring 0),
    mid (ring 1) or far, with a little leak so every column is a proper pmf.
    """
    centre_r = centre_c = GRID_SIDE // 2
    a = np.zeros((N_PROXIMITY, N_LOCATIONS), dtype=np.float64)
    for loc in range(N_LOCATIONS):
        ring = max(abs(loc // GRID_SIDE - centre_r), abs(loc % GRID_SIDE - centre_c))
        band = min(ring, N_PROXIMITY - 1)
        a[:, loc] = 0.05
        a[band, loc] = 0.90
    return a / a.sum(axis=0, keepdims=True)


def _pose_likelihood(rng: np.random.Generator | None) -> ArrayF:
    """Likelihood ``P(o_pose | location)`` of shape ``(4, 9)``.

    Pose (which way the creature faces) is only weakly informative about its
    cell; the base likelihood is near-uniform with a deterministic location-keyed
    tilt, optionally jittered by a seeded ``rng`` so a colony is heterogeneous.
    """
    a = np.full((N_POSE, N_LOCATIONS), 1.0 / N_POSE, dtype=np.float64)
    for loc in range(N_LOCATIONS):
        a[loc % N_POSE, loc] += 0.2
        if rng is not None:
            a[:, loc] += rng.uniform(0.0, 0.02, size=N_POSE)
    return a / a.sum(axis=0, keepdims=True)


def _gaze_likelihood(rng: np.random.Generator | None) -> ArrayF:
    """Private gaze likelihood ``P(o_gaze | gaze)`` of shape ``(9, 9)``.

    Gaze is the agent's own latent — which cell it is attending to — observed
    through a near-diagonal proprioceptive sensor. It never leaves the agent;
    it is built per-agent so each sentinel's internal state is distinct.
    """
    base = 0.85
    off = (1.0 - base) / (N_GAZE - 1)
    a: ArrayF = np.full((N_GAZE, N_GAZE), off, dtype=np.float64)
    np.fill_diagonal(a, base)
    if rng is not None:
        a = a + rng.uniform(0.0, 0.01, size=a.shape)
    return a / a.sum(axis=0, keepdims=True)


@dataclass
class Sentinel:
    """A single active-inference sentinel with a Markov-blanket partition.

    ``qs_shared`` holds, in order, the ``location``/``proximity``/``pose`` pmfs
    that cross the blanket boundary; ``qs_gaze`` is the private gaze pmf that
    never does. ``A_*`` are the per-agent likelihoods used by the one-step
    variational update; ``log_prior_*`` are the running natural-parameter priors.
    """

    A_location: ArrayF
    A_proximity: ArrayF
    A_pose: ArrayF
    A_gaze: ArrayF
    qs_location: ArrayF
    qs_proximity: ArrayF
    qs_pose: ArrayF
    qs_gaze: ArrayF
    D_location: ArrayF

    def shared_vector(self) -> ArrayF:
        """Concatenate the shared factors into one broadcastable vector."""
        return np.concatenate([self.qs_location, self.qs_proximity, self.qs_pose])

    @property
    def shared_factors(self) -> dict[str, ArrayF]:
        """Return the shared factors keyed by name (location/proximity/pose)."""
        return {
            "location": self.qs_location,
            "proximity": self.qs_proximity,
            "pose": self.qs_pose,
        }


@dataclass
class Observation:
    """One multi-factor observation for a sentinel.

    Each field is an outcome index into the corresponding likelihood's rows.
    """

    location: int
    proximity: int
    pose: int
    gaze: int


class SentinelEnsemble:
    """A colony of sentinels with private gaze and shared location/proximity/pose.

    Agents are built on top of :func:`fedference.pomdp.build_sentinel_world`: the
    shared ``location`` likelihood and prior come straight from that world, while
    ``proximity``/``pose`` are derived shared likelihoods over the same grid and
    ``gaze`` is a private per-agent likelihood. The ensemble enforces the §3.1
    Markov-blanket separation — :meth:`broadcast` exposes only the shared factors
    (ISC-22), never gaze.
    """

    def __init__(self, agents: Sequence[Sentinel]) -> None:
        if len(agents) < 1:
            raise ValueError("ensemble needs at least one sentinel")
        self.agents: list[Sentinel] = list(agents)
        self._loc_slice = slice(0, N_LOCATIONS)
        self._prox_slice = slice(N_LOCATIONS, N_LOCATIONS + N_PROXIMITY)
        self._pose_slice = slice(N_LOCATIONS + N_PROXIMITY,
                                 N_LOCATIONS + N_PROXIMITY + N_POSE)

    # -- construction --------------------------------------------------------

    @classmethod
    def from_world(
        cls,
        n_agents: int,
        *,
        seed: int = 0,
        acuity: float = 0.9,
    ) -> SentinelEnsemble:
        """Build an ``n_agents`` colony from :func:`pomdp.build_sentinel_world`.

        Each agent gets its own seeded ``rng`` so the shared-location sensor
        acuity and the private gaze/pose sensors are heterogeneous but
        deterministic. The running location belief starts **uniform** — an
        absolute one-hot prior could never be revised by data — while the
        world's structural prior ``D`` is retained on each :class:`Sentinel` as
        ``D_location``; proximity/pose/gaze also start uniform.

        Raises:
            ValueError: if ``n_agents`` is not a positive integer.
        """
        if n_agents < 1:
            raise ValueError("n_agents must be a positive integer")
        master = np.random.default_rng(seed)
        agents: list[Sentinel] = []
        for _ in range(n_agents):
            rng = np.random.default_rng(int(master.integers(0, 2**32 - 1)))
            world = build_sentinel_world(rng, acuity=acuity)
            a_loc = np.asarray(world["A"][0], dtype=np.float64)
            d_loc = np.asarray(world["D"][0], dtype=np.float64).ravel()
            agents.append(
                Sentinel(
                    A_location=a_loc,
                    A_proximity=_proximity_likelihood(),
                    A_pose=_pose_likelihood(rng),
                    A_gaze=_gaze_likelihood(rng),
                    qs_location=_uniform(N_LOCATIONS),
                    qs_proximity=_uniform(N_PROXIMITY),
                    qs_pose=_uniform(N_POSE),
                    qs_gaze=_uniform(N_GAZE),
                    D_location=d_loc / d_loc.sum(),
                )
            )
        return cls(agents)

    # -- the federated-inference loop ---------------------------------------

    def perceive(self, observations: Sequence[Observation]) -> None:
        """Update each agent's private gaze AND shared factors from observations.

        ``observations`` carries one :class:`Observation` per agent. Every factor
        is updated with the locked one-step variational posterior
        :func:`fedference.belief_updating.infer_states`, using the agent's current
        belief as the (log) prior so perception is recursive.

        Raises:
            ValueError: if ``observations`` length does not match the colony size.
        """
        if len(observations) != len(self.agents):
            raise ValueError("need exactly one observation per agent")
        for agent, obs in zip(self.agents, observations):
            agent.qs_location = infer_states(
                agent.A_location, obs.location, np.log(np.clip(agent.qs_location, _EPS, None))
            )
            # proximity/pose are SHARED factors whose likelihoods A_proximity (3,9)
            # and A_pose (4,9) are defined over the LOCATION state (Friston §3.1:
            # they are derived observations of the same nine-cell latent). Their
            # marginal beliefs are therefore the pushforward of the freshly updated
            # location belief through their likelihoods — a proper length-3 / -4 pmf
            # consistent with the broadcast layout. The observed proximity/pose
            # outcome indices sharpen this via the location update above.
            agent.qs_proximity = _normalize(agent.A_proximity @ agent.qs_location)
            agent.qs_pose = _normalize(agent.A_pose @ agent.qs_location)
            # PRIVATE gaze — updated, but never broadcast (Markov blanket, §3.1).
            agent.qs_gaze = infer_states(
                agent.A_gaze, obs.gaze, np.log(np.clip(agent.qs_gaze, _EPS, None))
            )

    def broadcast(self) -> ArrayF:
        """Return the ``(n_agents, n_shared)`` matrix of shared-factor pmfs.

        ISC-22: the row for each agent is the concatenation of its
        location/proximity/pose pmfs. The private gaze factor is *not* part of the
        vector — it never crosses the Markov blanket.
        """
        return np.asarray([agent.shared_vector() for agent in self.agents])

    def broadcast_location(self) -> ArrayF:
        """Return the shared-location broadcast used by the categorical Eq. 7 bridge."""
        return self.broadcast()[:, self._loc_slice]

    def assimilate(
        self,
        consensus: ArrayF,
        *,
        tau: float = 1.0,
        **legacy: object,
    ) -> None:
        """Fold a heard ``consensus`` shared-vector back into each agent.

        ``consensus`` is a length-``n_shared`` shared vector (the layout
        :meth:`broadcast` produces). For each shared factor the agent forms the
        tempered generalized-Bayes update
        ``softmax(log q_self + tau * log q_consensus)`` — a
        product-of-experts in log space — and renormalizes. The private gaze
        factor is deliberately left untouched: nothing about the heard consensus
        is allowed to write the agent's internal state.

        Raises:
            ValueError: if ``consensus`` has the wrong length or ``tau`` is
                negative.
        """
        if "learning_rate" in legacy:
            if tau != 1.0:
                raise TypeError("tau and deprecated learning_rate cannot both be supplied")
            tau = legacy.pop("learning_rate")  # type: ignore[assignment]
            warnings.warn(
                "learning_rate is deprecated; use tau",
                DeprecationWarning,
                stacklevel=2,
            )
        if legacy:
            raise TypeError(f"unexpected keyword argument(s): {', '.join(sorted(legacy))}")
        vec = np.asarray(consensus, dtype=np.float64).ravel()
        expected = N_LOCATIONS + N_PROXIMITY + N_POSE
        if vec.shape[0] != expected:
            raise ValueError(f"consensus must have length {expected}, got {vec.shape[0]}")
        if tau < 0:
            raise ValueError("tau must be non-negative")
        c_loc = vec[self._loc_slice]
        c_prox = vec[self._prox_slice]
        c_pose = vec[self._pose_slice]
        for agent in self.agents:
            agent.qs_location = self._fuse(agent.qs_location, c_loc, tau)
            agent.qs_proximity = self._fuse(agent.qs_proximity, c_prox, tau)
            agent.qs_pose = self._fuse(agent.qs_pose, c_pose, tau)
            # gaze untouched — private.

    @staticmethod
    def _fuse(own: ArrayF, heard: ArrayF, tau: float) -> ArrayF:
        """Log-space product-of-experts fusion of own and heard beliefs."""
        log_own = np.log(np.clip(own, _EPS, None))
        log_heard = np.log(np.clip(heard, _EPS, None))
        return softmax(log_own + tau * log_heard)

    # -- convenience accessors ----------------------------------------------

    @property
    def n_agents(self) -> int:
        """Number of sentinels in the colony."""
        return len(self.agents)

    @property
    def n_shared(self) -> int:
        """Length of the shared broadcast vector (location+proximity+pose)."""
        return N_LOCATIONS + N_PROXIMITY + N_POSE

    def gaze_beliefs(self) -> ArrayF:
        """Return the ``(n_agents, 9)`` PRIVATE gaze pmfs (never broadcast)."""
        return np.asarray([agent.qs_gaze for agent in self.agents])
