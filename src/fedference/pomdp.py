"""Categorical sentinel POMDP specialization for federated belief-sharing.

This is a reduced discrete generative model using the predator/sentinel world
structure illustrated by Friston et al. (2024), *Federated inference and belief
sharing* (Neurosci. Biobehav. Rev. 156:105500), Figures 1 and 4. A colony of
sentinels each watch a
shared world that contains a single hidden creature (a "predator") whose
location is one of nine cells laid out in a 3x3 grid. The single hidden factor
is therefore the creature's **location** with ``n_s = 9`` states; this is the
factor sentinels gossip about (the shared latent of
:mod:`fedference.belief_sharing` / :func:`fedference.aggregation.log_linear_pool`,
the project's categorical Eq. 7 specialization rather than a full source
protocol reconstruction).

Generative model (POMDP) following the active-inference convention used by
``template_active_inference`` — a categorical pmf is a 1-D array summing to 1 and
a likelihood ``A`` is shape ``(n_o, n_s)`` whose **columns** (indexed by hidden
state) are categorical:

* ``A`` — observation likelihood ``P(o | s)``. Each sentinel observes the
  creature's location through a noisy sensor: with probability ``acuity`` it
  reports the true cell, the residual mass leaks to the other cells. A single
  shared-``location`` modality is returned as one ``(n_o, n_s)`` matrix with
  ``n_o = n_s = 9``.
* ``B`` — transition ``P(s' | s, u)`` of shape ``(n_s, n_s, n_u)``. The creature
  moves on the grid under three control paths: ``still`` (stay put), ``left``
  (decrement column, reflecting at the wall) and ``right`` (increment column).
* ``C`` — log-preference over outcomes, shape ``(n_o, 1)``. The sentinel
  prefers to *see* the creature near the den (center cell), encoded as a
  log-preference bump.
* ``D`` — initial prior over location, shape ``(n_s, 1)``; the creature is
  believed to start at the grid center.

ISC-15 (verified in ``tests/test_pomdp.py``): every column of ``A`` and every
column of each ``B[..., u]`` is a proper categorical pmf summing to 1.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from dataclasses import field as _field
from typing import Any, cast
from typing import Sequence as _Seq

import numpy as np

from .generalized_bayes import softmax as _softmax_1d  # single implementation, no duplicate

ArrayF = np.ndarray

#: Side length of the square sentinel grid (3x3 -> 9 locations, source-inspired).
GRID_SIDE: int = 3
#: Number of hidden location states (creature cells).
N_LOCATIONS: int = GRID_SIDE * GRID_SIDE
#: Control paths of the transition tensor, in order: still / left / right.
CONTROL_LABELS: tuple[str, ...] = ("still", "left", "right")

_EPS = 1e-12


def normalise_columns(matrix: ArrayF) -> ArrayF:
    """Return a copy of ``matrix`` with every column renormalized to sum to 1.

    Columns are indexed by the trailing axis pair's first index in the
    active-inference convention: for a 2-D ``(n_o, n_s)`` likelihood this divides
    each ``column s`` by its mass so ``out[:, s]`` is a categorical pmf.

    Args:
        matrix: A non-empty 2-D array with non-negative entries. Each column
            must carry strictly positive mass.

    Returns:
        A new ``float64`` array of the same shape with unit-sum columns.

    Raises:
        ValueError: if ``matrix`` is not 2-D, is empty, has negative entries,
            or contains an all-zero column.
    """
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("matrix must be a non-empty 2-D array")
    if np.any(arr < -1e-9):
        raise ValueError("matrix has negative entries")
    arr = np.clip(arr, 0.0, None)
    col_sums = arr.sum(axis=0)
    if np.any(col_sums <= _EPS):
        raise ValueError("every column must carry positive mass")
    return arr / col_sums


def _location_likelihood(acuity: float) -> ArrayF:
    """Build the noisy location sensor ``P(o | s)`` of shape ``(9, 9)``.

    With probability ``acuity`` the sentinel reports the true cell; the residual
    ``1 - acuity`` is spread uniformly over the other eight cells.
    """
    if not 0.0 < acuity <= 1.0:
        raise ValueError("acuity must lie in (0, 1]")
    off = (1.0 - acuity) / (N_LOCATIONS - 1)
    a = np.full((N_LOCATIONS, N_LOCATIONS), off, dtype=np.float64)
    np.fill_diagonal(a, acuity)
    return normalise_columns(a)


def _column(row: int, side: int = GRID_SIDE) -> int:
    """Grid column index of flat location ``row`` (row-major 3x3)."""
    return row % side


def _row(loc: int, side: int = GRID_SIDE) -> int:
    """Grid row index of flat location ``loc`` (row-major 3x3)."""
    return loc // side


def _shift_column(loc: int, delta: int, side: int = GRID_SIDE) -> int:
    """Move ``loc`` by ``delta`` columns, reflecting at the grid walls."""
    r, c = _row(loc, side), _column(loc, side)
    c_new = c + delta
    if c_new < 0:
        c_new = 0
    elif c_new > side - 1:
        c_new = side - 1
    return r * side + c_new


def _transition_tensor() -> ArrayF:
    """Build ``P(s' | s, u)`` of shape ``(9, 9, 3)`` for still/left/right."""
    n = N_LOCATIONS
    b = np.zeros((n, n, len(CONTROL_LABELS)), dtype=np.float64)
    for s in range(n):
        # still: deterministic self-loop
        b[s, s, 0] = 1.0
        # left: decrement column (reflect)
        b[_shift_column(s, -1), s, 1] = 1.0
        # right: increment column (reflect)
        b[_shift_column(s, +1), s, 2] = 1.0
    # Each B[:, :, u] now has unit-sum columns by construction; normalize to be
    # robust to any future stochastic edits and to satisfy ISC-15 explicitly.
    for u in range(b.shape[2]):
        b[:, :, u] = normalise_columns(b[:, :, u])
    return b


def _log_preference(goal_bonus: float) -> ArrayF:
    """Log-preference ``C`` over outcomes, shape ``(9, 1)``.

    Center cell (the den, flat index 4 on a 3x3 grid) carries a positive
    log-preference bump; all other outcomes are neutral (zero log-preference).
    """
    c = np.zeros((N_LOCATIONS, 1), dtype=np.float64)
    center = (GRID_SIDE // 2) * GRID_SIDE + (GRID_SIDE // 2)
    c[center, 0] = float(goal_bonus)
    return c


def _initial_prior() -> ArrayF:
    """Initial prior ``D`` over location, shape ``(9, 1)`` — start at center."""
    d = np.zeros((N_LOCATIONS, 1), dtype=np.float64)
    center = (GRID_SIDE // 2) * GRID_SIDE + (GRID_SIDE // 2)
    d[center, 0] = 1.0
    return normalise_columns(d)


def build_sentinel_world(
    rng: np.random.Generator | None = None,
    *,
    acuity: float = 0.9,
    goal_bonus: float = 2.0,
) -> dict[str, Any]:
    """Construct the categorical sentinel POMDP specialization (Friston et al.
    2024, Figs. 1/4).

    The world has a single hidden factor — the creature's **location** over a
    3x3 grid (``n_s = 9``) — which is the shared latent that sentinels gossip
    about in :func:`fedference.belief_sharing.share_round`.

    Args:
        rng: Optional seeded generator. When provided, each sentinel's sensor
            acuity is jittered by a small deterministic perturbation drawn from
            ``rng`` so that a colony of sentinels has slightly heterogeneous
            likelihoods (still column-normalized). When ``None`` the canonical
            noiseless-construction model is returned.
        acuity: Base probability the location sensor reports the true cell.
        goal_bonus: Log-preference bump for observing the creature at the den.

    Returns:
        A dict with keys ``A`` (list with one ``(9, 9)`` location likelihood),
        ``B`` (list with one ``(9, 9, 3)`` transition tensor), ``C`` (list with
        one ``(9, 1)`` log-preference), ``D`` (list with one ``(9, 1)`` prior),
        plus metadata ``n_states``, ``n_obs``, ``n_controls`` and
        ``control_labels``.

    Raises:
        ValueError: if ``acuity`` is outside ``(0, 1]``.
    """
    if rng is None:
        a = _location_likelihood(acuity)
    else:
        # Deterministic jitter: tighten acuity by a small non-negative amount so
        # the column stays a valid pmf for any acuity in (0, 1].
        jitter = float(rng.uniform(0.0, min(0.05, 1.0 - acuity + _EPS)))
        a = _location_likelihood(min(1.0, acuity + jitter))

    b = _transition_tensor()
    c = _log_preference(goal_bonus)
    d = _initial_prior()

    return {
        "A": [a],
        "B": [b],
        "C": [c],
        "D": [d],
        "n_states": N_LOCATIONS,
        "n_obs": N_LOCATIONS,
        "n_controls": len(CONTROL_LABELS),
        "control_labels": CONTROL_LABELS,
    }


# --- V4: moving sentinel world ------------------------------------------------
#: Control paths of the 1-D moving world, in order: stay / left / right.
MOVING_ACTIONS: tuple[str, ...] = ("stay", "left", "right")


def _moving_fov(position: int, fov_width: int, n_positions: int) -> tuple[int, int]:
    """Half-open cell window ``[lo, hi)`` an agent at ``position`` can see."""
    lo = max(0, min(position, n_positions - 1))
    hi = min(n_positions, lo + max(1, int(fov_width)))
    return lo, hi


def _moving_likelihood(
    position: int, *, fov_width: int, n_positions: int, n_states: int
) -> ArrayF:
    """Build agent likelihood ``P(o | s)`` of shape ``(2, n_states)``.

    The hidden binary state is the *half* of the linear grid that holds the
    threat: ``state 0`` = threat in the left half, ``state 1`` = threat in the
    right half. An agent is *informative* about the state whose half its
    field-of-view covers — a confident, signed reading ("detected" when the
    threat is in its half, "not_detected" when it is in the other half). When the
    FOV does not overlap the grid (a degenerate position) the sensor is uniform.

    Outcomes: row 0 = "detected", row 1 = "not_detected". Columns (indexed by
    hidden state) are categorical pmfs summing to 1.

    Honesty note (binary-complement): with a binary state and two disjoint
    halves, ruling out one's own half logically implies the complement, so a
    single agent's "not_detected" still carries information about the global
    state. The intended "neither agent alone can be certain" only holds in the
    high-noise / few-step regime where a single sensor's evidence is too weak to
    overcome the flat prior; the communicating colony fuses the two complementary
    views. The reported moving-world numbers are measured, not assumed (see the
    results section), and the colony does not claim a strictly-necessary-comms
    result the binary world cannot deliver.
    """
    lo, hi = _moving_fov(position, fov_width, n_positions)
    half = n_positions // 2
    # The half (0 = left, 1 = right) that the FOV's leftmost cell lies in.
    fov_half = 0 if lo < half else 1
    a = np.full((2, n_states), 0.5, dtype=np.float64)
    # Informative whenever the FOV actually overlaps the grid.
    overlaps = hi > lo and 0 <= lo < n_positions
    if overlaps and n_states == 2:
        s = fov_half  # the state this viewpoint can detect
        other = 1 - s
        a[0, s] = 0.85   # P(detect | threat in this agent's half)
        a[1, s] = 0.15
        a[0, other] = 0.10  # P(detect | threat in the other half)
        a[1, other] = 0.90
    return normalise_columns(a)


def _moving_transition(n_positions: int) -> ArrayF:
    """Build ``P(p' | p, u)`` of shape ``(n_positions, n_positions, 3)``.

    Three deterministic control paths on a 1-D linear grid: ``stay`` (self-loop),
    ``left`` (decrement position, reflecting at cell 0) and ``right`` (increment,
    reflecting at the last cell). Columns are unit-sum by construction; we
    normalise to satisfy the column-pmf invariant explicitly.
    """
    n = int(n_positions)
    b = np.zeros((n, n, len(MOVING_ACTIONS)), dtype=np.float64)
    for p in range(n):
        b[p, p, 0] = 1.0                       # stay
        b[max(0, p - 1), p, 1] = 1.0           # left (reflect at 0)
        b[min(n - 1, p + 1), p, 2] = 1.0       # right (reflect at wall)
    for u in range(b.shape[2]):
        b[:, :, u] = normalise_columns(b[:, :, u])
    return b


def build_moving_world(
    *,
    n_positions: int = 4,
    n_agents: int = 2,
    n_states: int = 2,
    fov_width: int = 2,
    seed: int = 0,
) -> dict[str, Any]:
    """Construct a moving sentinel POMDP with disjoint fields-of-view (V4).

    A linear grid of ``n_positions`` cells holds a single binary hidden state —
    *which half of the grid the threat occupies* (``state 0`` = left half,
    ``state 1`` = right half). Agent ``i`` starts at position
    ``i * (n_positions // n_agents)`` and sees a half-open window of ``fov_width``
    cells from there. With the default 2-agent, 4-position, ``fov_width=2`` setup
    agent 0 sees cells ``[0, 2)`` (the left half — informative about state 0) and
    agent 1 sees cells ``[2, 4)`` (the right half — informative about state 1):
    **disjoint** FOVs, so neither agent alone can be certain of the hidden state
    and the two must communicate to reach a confident consensus.

    Args:
        n_positions: Number of cells on the linear grid.
        n_agents: Number of sentinels (positions tiled evenly across the grid).
        n_states: Hidden-state cardinality (binary by design — 2).
        fov_width: Number of cells each agent observes from its position.
        seed: Accepted for signature stability / determinism (the construction is
            deterministic; no stochastic jitter is applied).

    Returns:
        A dict with ``A`` (list of ``n_agents`` likelihoods, each ``(2, n_states)``),
        ``B`` (one ``(n_positions, n_positions, 3)`` transition tensor),
        ``D`` (uniform ``(n_states,)`` prior), ``agent_positions``, ``n_actions``,
        ``actions``, ``n_states``, ``n_positions`` and ``n_agents``.
    """
    if n_agents < 1:
        raise ValueError("need at least one agent")
    if n_positions < n_agents:
        raise ValueError("need at least one position per agent")
    if n_states != 2:
        raise ValueError("moving world is binary (n_states must be 2)")

    spacing = n_positions // n_agents
    agent_positions = [int(i * spacing) for i in range(n_agents)]
    a_list = [
        _moving_likelihood(
            p, fov_width=fov_width, n_positions=n_positions, n_states=n_states
        )
        for p in agent_positions
    ]
    b = _moving_transition(n_positions)
    d = np.full(n_states, 1.0 / n_states, dtype=np.float64)

    return {
        "A": a_list,
        "B": b,
        "D": d,
        "agent_positions": agent_positions,
        "n_actions": len(MOVING_ACTIONS),
        "actions": MOVING_ACTIONS,
        "n_states": int(n_states),
        "n_positions": int(n_positions),
        "n_agents": int(n_agents),
        "fov_width": int(fov_width),
        "seed": int(seed),
    }


# --- V2: hierarchical POMDP -------------------------------------------------
#: Context states at Level 2 (L2): which global regime is active.
#: E.g. ``quiet`` (creature rarely present) vs ``alert`` (creature frequently present).
HIER_CONTEXT_LABELS: tuple[str, ...] = ("quiet", "alert")
#: Number of Level-2 context states.
N_CONTEXTS: int = len(HIER_CONTEXT_LABELS)
#: Prior mass placed on the grid-center ``den`` cell when the context is
#: ``alert``. Single definition shared by the 2-level and N-level default
#: priors below and surfaced as the HIER_ALERT_CENTER_MASS /
#: NLEVEL3_ALERT_CENTER_MASS manuscript tokens (never re-typed).
ALERT_CENTER_MASS: float = 0.6
#: Diagonal persistence of the default symmetric L2 context transition
#: matrix (probability the context stays in its current state per step).
#: Surfaced as the HIER_CTX_PERSIST manuscript token.
CONTEXT_PERSISTENCE: float = 0.9
#: Exact literal off-diagonal (switch) probability of that default transition.
#: Kept as a literal, not ``1.0 - CONTEXT_PERSISTENCE``, so the refactor is
#: bit-identical to the original ``((0.9, 0.1), (0.1, 0.9))`` default
#: (IEEE-754 ``1.0 - 0.9 != 0.1``).
CONTEXT_SWITCH_PROB: float = 0.1


def build_hierarchical_world(
    *,
    acuity: float = 0.9,
    goal_bonus: float = 2.0,
    context_prior: tuple[float, ...] = (0.5, 0.5),
    context_transition: tuple[tuple[float, float], ...] = (
        (CONTEXT_PERSISTENCE, CONTEXT_SWITCH_PROB),
        (CONTEXT_SWITCH_PROB, CONTEXT_PERSISTENCE),
    ),
    l1_priors: tuple[tuple[float, ...], ...] | None = None,
) -> dict[str, Any]:
    """Construct a 2-level hierarchical POMDP (V2).

    Level 1 (L1) is the standard sentinel 9-location POMDP from
    :func:`build_sentinel_world`. Level 2 (L2) is a 2-state ``context`` factor
    (``quiet`` / ``alert``) whose current value gates the **L1 prior over
    creature location** via context-conditioned priors — when the context is
    ``alert`` the prior mass shifts toward the grid center (the ``den``) because
    the creature is expected to be closer to the sentinels.

    Args:
        acuity: Sensor acuity for the L1 location likelihood.
        goal_bonus: Log-preference bump for the center cell in L1.
        context_prior: Initial categorical pmf over L2 context states.
        context_transition: Row-major 2x2 L2 transition matrix; row i is
            ``P(context' | context = i)`` (each row sums to 1).
        l1_priors: Two length-9 pmf vectors (one per context) for the
            context-conditioned L1 prior over creature location. When
            ``None`` the defaults are: ``quiet`` = uniform;
            ``alert`` = peaked at the grid center (mass
            :data:`ALERT_CENTER_MASS` on the center, rest uniform).

    Returns:
        Dict with:
        - ``L1``: the sentinel-world dict for each agent's local POMDP;
        - ``L2_prior``: ``(2,)`` float64 array, the initial context prior;
        - ``L2_transition``: ``(2, 2)`` float64 row-stochastic matrix;
        - ``L1_priors_given_context``: list of two ``(9,)`` float64 arrays
          (L1 prior conditioned on context 0 and context 1 respectively);
        - ``n_contexts``, ``context_labels``.
    """
    # ---- L2 prior and transition ----
    ctx_prior = np.asarray(context_prior, dtype=np.float64)
    if ctx_prior.shape != (N_CONTEXTS,) or not np.isclose(ctx_prior.sum(), 1.0):
        raise ValueError(f"context_prior must be a pmf of length {N_CONTEXTS}")
    ctx_trans = np.asarray(context_transition, dtype=np.float64)
    if ctx_trans.shape != (N_CONTEXTS, N_CONTEXTS):
        raise ValueError(
            f"context_transition must be shape ({N_CONTEXTS}, {N_CONTEXTS})"
        )
    for row_i, row in enumerate(ctx_trans):
        if not np.isclose(row.sum(), 1.0, atol=1e-9):
            raise ValueError(
                f"context_transition row {row_i} must sum to 1 (got {row.sum()})"
            )

    # ---- Context-conditioned L1 priors ----
    center = (GRID_SIDE // 2) * GRID_SIDE + (GRID_SIDE // 2)  # flat index 4
    if l1_priors is None:
        # quiet: uniform over all locations
        p_quiet = np.full(N_LOCATIONS, 1.0 / N_LOCATIONS, dtype=np.float64)
        # alert: peaked at center (the den)
        alert_off = (1.0 - ALERT_CENTER_MASS) / (N_LOCATIONS - 1)
        p_alert = np.full(N_LOCATIONS, alert_off, dtype=np.float64)
        p_alert[center] = ALERT_CENTER_MASS
        l1_priors_arr: list[np.ndarray] = [p_quiet, p_alert]
    else:
        l1_priors_arr = [np.asarray(p, dtype=np.float64) for p in l1_priors]
        if len(l1_priors_arr) != N_CONTEXTS:
            raise ValueError(f"l1_priors must supply {N_CONTEXTS} priors (one per context)")
        for ci, p in enumerate(l1_priors_arr):
            if p.shape != (N_LOCATIONS,) or not np.isclose(p.sum(), 1.0, atol=1e-9):
                raise ValueError(
                    f"l1_priors[{ci}] must be a pmf of length {N_LOCATIONS}"
                )

    # ---- L1 sentinel world (shared likelihood and transition) ----
    l1 = build_sentinel_world(acuity=acuity, goal_bonus=goal_bonus)

    return {
        "L1": l1,
        "L2_prior": ctx_prior,
        "L2_transition": ctx_trans,
        "L1_priors_given_context": l1_priors_arr,
        "n_contexts": int(N_CONTEXTS),
        "context_labels": HIER_CONTEXT_LABELS,
    }


def hierarchical_infer(
    A: "np.ndarray",
    obs: int,
    hier_world: dict[str, Any],
    *,
    n_iters: int = 4,
) -> dict[str, "np.ndarray"]:
    """Alternating minimization for the 2-level hierarchical POMDP (V2).

    Performs ``n_iters`` passes of **empirical-prior coupling** between L1
    (location) and L2 (context):

    1. **L2 → L1 message**: given the current L2 context belief ``q_ctx``, the
       L1 prior is the expectation ``sum_c q_ctx[c] * p_L1[c]`` — a soft mixture
       of the context-conditioned location priors.
    2. **L1 update**: one-step variational posterior
       ``q_loc = softmax(ln prior_l1 + ln A[obs, :])``.
    3. **L1 → L2 message**: the likelihood that the current observation was
       generated under each context, i.e.
       ``log_lik_ctx[c] = sum_s p_L1[c][s] * A[obs, s]`` (marginal evidence
       under context ``c``).
    4. **L2 update**: ``q_ctx = softmax(ln L2_prior + log_lik_ctx)``.

    Args:
        A: L1 observation likelihood ``(n_o, n_s)`` array (columns indexed by
            hidden location state).
        obs: Observed outcome index in ``[0, n_o)``.
        hier_world: Dict returned by :func:`build_hierarchical_world`.
        n_iters: Number of alternating-minimization sweeps (default 4).

    Returns:
        Dict with ``q_loc`` (length-9 L1 location posterior) and ``q_ctx``
        (length-2 L2 context posterior) after the final iteration.
    """
    a = np.asarray(A, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] < 1 or a.shape[1] != N_LOCATIONS:
        raise ValueError(
            f"A must be shape (n_o, {N_LOCATIONS}); got {a.shape}"
        )
    if not 0 <= obs < a.shape[0]:
        raise ValueError(f"obs {obs} out of range [0, {a.shape[0]})")
    if n_iters < 1:
        raise ValueError("n_iters must be >= 1")

    l2_prior: np.ndarray = np.asarray(hier_world["L2_prior"], dtype=np.float64)
    l1_priors: list[np.ndarray] = [
        np.asarray(p, dtype=np.float64)
        for p in cast(_Seq[Any], hier_world["L1_priors_given_context"])
    ]

    # Initialise context belief at the L2 prior.
    q_ctx = l2_prior.copy()
    q_loc = np.full(N_LOCATIONS, 1.0 / N_LOCATIONS, dtype=np.float64)

    for _ in range(n_iters):
        # --- Step 1: L2 -> L1 empirical prior ---
        prior_l1 = sum(q_ctx[c] * l1_priors[c] for c in range(N_CONTEXTS))
        prior_l1 = np.asarray(prior_l1, dtype=np.float64)
        prior_l1 = np.clip(prior_l1, _EPS, None)
        prior_l1 = prior_l1 / prior_l1.sum()

        # --- Step 2: L1 update (one-step variational posterior) ---
        log_prior_l1 = np.log(prior_l1)
        log_lik_row = np.log(np.clip(a[obs, :], _EPS, None))
        q_loc = _softmax_1d(log_prior_l1 + log_lik_row)

        # --- Step 3: L1 -> L2 marginal evidence ---
        log_lik_ctx = np.array(
            [float(np.log(np.clip(l1_priors[c] @ a[obs, :], _EPS, None)))
             for c in range(N_CONTEXTS)],
            dtype=np.float64,
        )

        # --- Step 4: L2 update ---
        log_ctx_prior = np.log(np.clip(l2_prior, _EPS, None))
        q_ctx = _softmax_1d(log_ctx_prior + log_lik_ctx)

    return {"q_loc": q_loc, "q_ctx": q_ctx}


# --- V2 extension: generic N-level hierarchical POMDP ----------------------


@dataclass
class LayerSpec:
    """Specification for one level of an N-level hierarchical POMDP.

    Args:
        n_states: Number of hidden states at this level.
        labels: Human-readable label for each state (length must equal ``n_states``).
        default_prior: Default pmf over this level's states (length ``n_states``).
            If ``None`` a uniform prior is used.
        conditioned_priors: For non-top levels, one prior over the *child* level's
            states per state of *this* level.  Shape ``(n_states, n_child_states)``.
            ``None`` means the child prior is uniform (no top-down gating from this
            level).  Only leaf levels may omit this (they have no child).
    """

    n_states: int
    labels: tuple[str, ...] = _field(default_factory=tuple)
    default_prior: np.ndarray | None = None
    conditioned_priors: list[np.ndarray] | None = None

    def __post_init__(self) -> None:
        if not self.labels:
            object.__setattr__(
                self, "labels", tuple(f"state_{i}" for i in range(self.n_states))
            )
        if len(self.labels) != self.n_states:
            raise ValueError(
                f"LayerSpec.labels length {len(self.labels)} != n_states {self.n_states}"
            )


def build_nlevel_world(
    layers: _Seq[LayerSpec],
    *,
    acuity: float = 0.9,
    goal_bonus: float = 2.0,
) -> dict[str, Any]:
    """Construct a generic N-level hierarchical POMDP.

    The lowest layer (``layers[-1]``) is always a sentinel 9-location POMDP.
    Higher layers provide context-conditioned priors for the layer below.

    Args:
        layers: Ordered list of :class:`LayerSpec` from highest (top) to lowest.
            The lowest layer must have ``n_states == N_LOCATIONS``.
        acuity: Sensor acuity for the leaf-level location likelihood.
        goal_bonus: Log-preference bump for the center cell.

    Returns:
        A dict with keys:
        - ``"L1"``: The leaf sentinel-world dict (bottom level).
        - ``"layers"``: The list of :class:`LayerSpec` objects (top → bottom).
        - ``"n_levels"``: Number of levels.
        - ``"level_priors"``: list of pmf arrays, one per level (top → bottom),
          length ``n_levels``.
        - ``"conditioned_priors"``: nested list; ``conditioned_priors[i][j]`` is the
          prior for level ``i+1`` given state ``j`` at level ``i``.

    Raises:
        ValueError: If fewer than 2 layers, or the leaf level has the wrong size.
    """
    layers = list(layers)
    if len(layers) < 2:
        raise ValueError("build_nlevel_world requires at least 2 layers")

    leaf = layers[-1]
    if leaf.n_states != N_LOCATIONS:
        raise ValueError(
            f"leaf layer must have n_states == N_LOCATIONS ({N_LOCATIONS}); "
            f"got {leaf.n_states}"
        )

    # Build uniform defaults and validate pmfs.
    level_priors: list[np.ndarray] = []
    for depth, spec in enumerate(layers):
        if spec.default_prior is None:
            prior = np.full(spec.n_states, 1.0 / spec.n_states, dtype=np.float64)
        else:
            prior = np.asarray(spec.default_prior, dtype=np.float64)
        if prior.shape != (spec.n_states,) or not np.isclose(prior.sum(), 1.0, atol=1e-9):
            raise ValueError(f"layer {depth} default_prior is not a valid pmf")
        level_priors.append(prior)

    # Validate / auto-build conditioned_priors for non-leaf levels.
    n_levels = len(layers)
    conditioned: list[list[np.ndarray]] = []
    for depth in range(n_levels - 1):
        parent = layers[depth]
        child = layers[depth + 1]
        if parent.conditioned_priors is None:
            # Uniform child prior for each parent state.
            cp: list[np.ndarray] = [
                np.full(child.n_states, 1.0 / child.n_states, dtype=np.float64)
                for _ in range(parent.n_states)
            ]
        else:
            cp = [np.asarray(p, dtype=np.float64) for p in parent.conditioned_priors]
        if len(cp) != parent.n_states:
            raise ValueError(
                f"layer {depth} conditioned_priors must supply {parent.n_states} priors"
            )
        for si, p in enumerate(cp):
            if p.shape != (child.n_states,) or not np.isclose(p.sum(), 1.0, atol=1e-9):
                raise ValueError(
                    f"layer {depth} conditioned_prior[{si}] is not a valid pmf of "
                    f"length {child.n_states}"
                )
        conditioned.append(cp)

    l1 = build_sentinel_world(acuity=acuity, goal_bonus=goal_bonus)

    return {
        "L1": l1,
        "layers": layers,
        "n_levels": n_levels,
        "level_priors": level_priors,
        "conditioned_priors": conditioned,
    }


def nlevel_infer(
    A: "np.ndarray",
    obs: int,
    nlevel_world: dict[str, Any],
    *,
    n_iters: int = 4,
) -> dict[str, list[np.ndarray]]:
    """Alternating minimization for a generic N-level hierarchical POMDP.

    Performs ``n_iters`` passes of top-down / bottom-up message passing:

    1. **Top-down pass** (level N → N-1 → … → L1): compute the empirical prior
       for each level as the expectation of the conditioned priors from the level
       above.
    2. **L1 update**: one-step variational posterior on the observation.
    3. **Bottom-up pass** (L1 → … → level N): update each level's belief from the
       marginal evidence contributed by the level below.

    Args:
        A: L1 observation likelihood ``(n_o, N_LOCATIONS)``.
        obs: Observed outcome index.
        nlevel_world: Dict returned by :func:`build_nlevel_world`.
        n_iters: Number of alternating-minimization sweeps.

    Returns:
        Dict with keys ``"q_levels"`` — a list of ``n_levels`` posterior pmf arrays
        (top → bottom), where ``q_levels[-1]`` is the L1 (location) posterior.
    """
    a = np.asarray(A, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != N_LOCATIONS:
        raise ValueError(f"A must be shape (n_o, {N_LOCATIONS}); got {a.shape}")
    if not 0 <= obs < a.shape[0]:
        raise ValueError(f"obs {obs} out of range [0, {a.shape[0]})")
    if n_iters < 1:
        raise ValueError("n_iters must be >= 1")

    layers: list[LayerSpec] = list(cast(_Seq[LayerSpec], nlevel_world["layers"]))
    level_priors: list[np.ndarray] = [
        np.asarray(p, dtype=np.float64)
        for p in cast(_Seq[Any], nlevel_world["level_priors"])
    ]
    conditioned: list[list[np.ndarray]] = [
        [np.asarray(p, dtype=np.float64) for p in cp]
        for cp in cast(_Seq[_Seq[Any]], nlevel_world["conditioned_priors"])
    ]
    n_levels: int = int(cast(int, nlevel_world["n_levels"]))

    # Initialise each level's belief at its default prior.
    q_levels: list[np.ndarray] = [p.copy() for p in level_priors]

    for _ in range(n_iters):
        # --- Top-down pass: build empirical prior for each level ---------------
        empirical: list[np.ndarray] = [q_levels[0].copy()]  # top level unchanged
        for depth in range(n_levels - 1):
            cp = conditioned[depth]  # shape: (parent_states, child_states)
            parent_q = q_levels[depth]
            child_prior = sum(
                parent_q[j] * cp[j] for j in range(len(parent_q))
            )
            child_prior = np.asarray(child_prior, dtype=np.float64)
            child_prior = np.clip(child_prior, _EPS, None)
            child_prior = child_prior / child_prior.sum()
            empirical.append(child_prior)

        # --- L1 update (bottom level) -----------------------------------------
        log_prior_l1 = np.log(empirical[-1])
        log_lik_row = np.log(np.clip(a[obs, :], _EPS, None))
        q_levels[-1] = _softmax_1d(log_prior_l1 + log_lik_row)

        # --- Bottom-up pass: update each non-leaf level -------------------------
        # For each level depth, the marginal evidence for parent state j given the
        # observation is computed by marginalising over the child level's posterior:
        #   ell_j = sum_{child state c} conditioned_prior[depth][j][c] * q_child[c]
        # where q_child is the posterior at depth+1 from the current iteration.
        # At the leaf (depth == n_levels - 2) the child is the L1 location level and
        # the leaf likelihood A[obs, s] provides the direct evidence; for higher
        # levels the child's posterior already incorporates all lower-level evidence.
        for depth in range(n_levels - 2, -1, -1):
            cp = conditioned[depth]  # list of (child_n_states,) priors per parent state
            child_q = q_levels[depth + 1]  # current child posterior (updated above)
            log_lik_parent = np.array(
                [float(np.log(np.clip(float(np.asarray(cp[j], dtype=np.float64) @ child_q), _EPS, None)))
                 for j in range(layers[depth].n_states)],
                dtype=np.float64,
            )
            log_lev_prior = np.log(np.clip(level_priors[depth], _EPS, None))
            q_levels[depth] = _softmax_1d(log_lev_prior + log_lik_parent)

    return {"q_levels": q_levels}


# --- 3-level example world --------------------------------------------------
#: Meta-context states at Level 3 (L3) — the slowest-changing contextual factor.
L3_META_LABELS: tuple[str, ...] = ("low_threat", "high_threat")
#: Number of Level-3 meta-context states.
N_META_CONTEXTS: int = len(L3_META_LABELS)
#: Default L2 (context) prior given L3 = ``high_threat``: peaked at ``alert``.
#: Single definition consumed by :func:`build_3level_world` and surfaced as the
#: NLEVEL3_HIGH_THREAT_QUIET_PRIOR / NLEVEL3_HIGH_THREAT_ALERT_PRIOR tokens.
L2_HIGH_THREAT_ALERT_PRIOR: float = 0.8
#: Exact literal complement of :data:`L2_HIGH_THREAT_ALERT_PRIOR` (kept as a
#: literal, not ``1.0 - x``, so the refactor is bit-identical to the original
#: ``np.array([0.2, 0.8])`` — IEEE-754 ``1.0 - 0.8 != 0.2``).
L2_HIGH_THREAT_QUIET_PRIOR: float = 0.2


def build_3level_world(
    *,
    acuity: float = 0.9,
    goal_bonus: float = 2.0,
    l3_prior: tuple[float, ...] = (0.5, 0.5),
    l2_priors_given_l3: tuple[tuple[float, ...], ...] | None = None,
    l1_priors_given_l2: tuple[tuple[float, ...], ...] | None = None,
) -> dict[str, Any]:
    """Construct a 3-level hierarchical POMDP (L3=meta-context → L2=context → L1=location).

    Three levels:
    - L3 (meta-context): 2 states (``low_threat`` / ``high_threat``), the
      slowest-changing contextual factor that gates the L2 context prior.
    - L2 (context): 2 states (``quiet`` / ``alert``), gating the L1 location prior.
    - L1 (location): 9 states (the 3x3 grid), the directly observed level.

    Args:
        acuity: L1 sensor acuity.
        goal_bonus: Log-preference bump at the center cell.
        l3_prior: Initial pmf over L3 states.
        l2_priors_given_l3: Two length-2 pmf vectors (L2 prior per L3 state).
            Defaults: ``low_threat`` → uniform L2; ``high_threat`` → peaked at alert.
        l1_priors_given_l2: Two length-9 pmf vectors (L1 prior per L2 state).
            Defaults: ``quiet`` → uniform L1; ``alert`` → peaked at center.

    Returns:
        The N-level world dict plus convenience keys ``"L3_prior"``,
        ``"L2_priors_given_l3"``, ``"L1_priors_given_l2"``, and
        ``"n_meta_contexts"``.
    """
    # ---- L3 prior ----
    l3_arr = np.asarray(l3_prior, dtype=np.float64)
    if l3_arr.shape != (N_META_CONTEXTS,) or not np.isclose(l3_arr.sum(), 1.0, atol=1e-9):
        raise ValueError(f"l3_prior must be a pmf of length {N_META_CONTEXTS}")

    # ---- L2 priors conditioned on L3 ----
    if l2_priors_given_l3 is None:
        # low_threat: uniform L2 (equal chance quiet/alert)
        l2_low = np.full(N_CONTEXTS, 1.0 / N_CONTEXTS, dtype=np.float64)
        # high_threat: peaked at alert
        l2_high = np.array(
            [L2_HIGH_THREAT_QUIET_PRIOR, L2_HIGH_THREAT_ALERT_PRIOR],
            dtype=np.float64,
        )
        l2_given_l3: list[np.ndarray] = [l2_low, l2_high]
    else:
        l2_given_l3 = [np.asarray(p, dtype=np.float64) for p in l2_priors_given_l3]
        if len(l2_given_l3) != N_META_CONTEXTS:
            raise ValueError(f"l2_priors_given_l3 must supply {N_META_CONTEXTS} priors")
        for ci, p in enumerate(l2_given_l3):
            if p.shape != (N_CONTEXTS,) or not np.isclose(p.sum(), 1.0, atol=1e-9):
                raise ValueError(
                    f"l2_priors_given_l3[{ci}] must be a pmf of length {N_CONTEXTS}"
                )

    # ---- L1 priors conditioned on L2 ----
    center = (GRID_SIDE // 2) * GRID_SIDE + (GRID_SIDE // 2)
    if l1_priors_given_l2 is None:
        p_quiet = np.full(N_LOCATIONS, 1.0 / N_LOCATIONS, dtype=np.float64)
        alert_off = (1.0 - ALERT_CENTER_MASS) / (N_LOCATIONS - 1)
        p_alert = np.full(N_LOCATIONS, alert_off, dtype=np.float64)
        p_alert[center] = ALERT_CENTER_MASS
        l1_given_l2: list[np.ndarray] = [p_quiet, p_alert]
    else:
        l1_given_l2 = [np.asarray(p, dtype=np.float64) for p in l1_priors_given_l2]
        if len(l1_given_l2) != N_CONTEXTS:
            raise ValueError(f"l1_priors_given_l2 must supply {N_CONTEXTS} priors")
        for ci, p in enumerate(l1_given_l2):
            if p.shape != (N_LOCATIONS,) or not np.isclose(p.sum(), 1.0, atol=1e-9):
                raise ValueError(
                    f"l1_priors_given_l2[{ci}] must be a pmf of length {N_LOCATIONS}"
                )

    # ---- Assemble LayerSpec objects ----
    l3_spec = LayerSpec(
        n_states=N_META_CONTEXTS,
        labels=L3_META_LABELS,
        default_prior=l3_arr,
        conditioned_priors=l2_given_l3,
    )
    l2_default = np.full(N_CONTEXTS, 1.0 / N_CONTEXTS, dtype=np.float64)
    l2_spec = LayerSpec(
        n_states=N_CONTEXTS,
        labels=HIER_CONTEXT_LABELS,
        default_prior=l2_default,
        conditioned_priors=l1_given_l2,
    )
    l1_spec = LayerSpec(
        n_states=N_LOCATIONS,
        labels=tuple(str(i) for i in range(N_LOCATIONS)),
        default_prior=np.full(N_LOCATIONS, 1.0 / N_LOCATIONS, dtype=np.float64),
        conditioned_priors=None,
    )

    world = build_nlevel_world([l3_spec, l2_spec, l1_spec], acuity=acuity, goal_bonus=goal_bonus)
    world["L3_prior"] = l3_arr
    world["L2_priors_given_l3"] = l2_given_l3
    world["L1_priors_given_l2"] = l1_given_l2
    world["n_meta_contexts"] = int(N_META_CONTEXTS)
    world["meta_context_labels"] = L3_META_LABELS
    return world


def efe_policy_select(
    local_posteriors=None,
    world_dict=None,
    **legacy: object,
) -> list[int]:
    """Select one information-seeking action per agent by expected free energy.

    For each agent ``i`` with current belief ``q_i`` and position ``p_i``, every
    candidate action ``a in (stay, left, right)`` lands the agent at the
    deterministic next position ``argmax_p B[p, p_i, a]``. From that viewpoint we
    reconstruct the agent's likelihood and score the action by the *expected
    posterior entropy* after one observation — the epistemic (information-seeking)
    component of the expected free energy:

    ``H_expected(a) = sum_o P(o) * H( P(s | o) )``

    where ``P(o) = A_new[o, :] @ q_i`` and ``P(s | o)`` is the Bayesian posterior
    under the reconstructed likelihood from the prior ``q_i``. The chosen action
    minimises this expected entropy (most informative move). Ties break to the
    lowest action index.

    Args:
        local_posteriors: Sequence of per-agent posterior pmfs (each length
            ``n_states``).
        world_dict: A moving-world dict (as returned by :func:`build_moving_world`)
            whose ``agent_positions`` reflect the agents' current positions.

    Returns:
        A list of ``n_agents`` integer action indices in ``[0, n_actions)``.
    """
    if "beliefs" in legacy:
        if local_posteriors is not None:
            raise TypeError(
                "local_posteriors and deprecated beliefs cannot both be supplied"
            )
        local_posteriors = legacy.pop("beliefs")
        warnings.warn(
            "beliefs is deprecated; use local_posteriors",
            DeprecationWarning,
            stacklevel=2,
        )
    if legacy:
        raise TypeError(f"unexpected keyword argument(s): {', '.join(sorted(legacy))}")
    if local_posteriors is None or world_dict is None:
        raise TypeError("local_posteriors and world_dict are required")
    b = np.asarray(world_dict["B"], dtype=np.float64)
    positions = list(world_dict["agent_positions"])
    n_positions = int(world_dict["n_positions"])
    n_states = int(world_dict["n_states"])
    fov_width = int(world_dict.get("fov_width", 2))
    n_actions = int(world_dict["n_actions"])

    actions: list[int] = []
    for i, q in enumerate(local_posteriors):
        q_arr = np.clip(np.asarray(q, dtype=np.float64), 0.0, None)
        q_arr = q_arr / q_arr.sum()
        p_i = int(positions[i])
        efe = np.empty(n_actions, dtype=np.float64)
        for a in range(n_actions):
            new_pos = int(np.argmax(b[:, p_i, a]))
            a_new = _moving_likelihood(
                new_pos, fov_width=fov_width, n_positions=n_positions, n_states=n_states
            )
            h_expected = 0.0
            for o in range(a_new.shape[0]):
                p_o = float(a_new[o, :] @ q_arr)
                if p_o > 1e-10:
                    posterior = a_new[o, :] * q_arr
                    posterior = posterior / posterior.sum()
                    ent = float(
                        -np.sum(posterior * np.log(posterior + _EPS))
                    )
                    h_expected += p_o * ent
            efe[a] = h_expected
        actions.append(int(np.argmin(efe)))
    return actions
