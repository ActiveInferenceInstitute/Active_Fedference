"""Closed-form Expected Free Energy (EFE) decomposition over a categorical POMDP.

Active Inference scores a policy ``pi`` by its Expected Free Energy. Following
Friston et al. (2024), *Federated inference and belief sharing*
(Neurosci. Biobehav. Rev. 156:105500), Eq. 2, the EFE admits two exactly equal
decompositions of the same scalar ``G(pi)``:

* **risk + ambiguity** — ``risk`` is ``KL( q(o|pi) || p(o) )`` (the deviation of
  the policy-predicted outcomes from the preferred outcomes ``p(o) = softmax(C)``)
  and ``ambiguity`` is the expected likelihood entropy ``E_q(s)[ H[p(o|s)] ]``
  (outcome uncertainty given the state).
* **pragmatic + epistemic value** — ``pragmatic_value = E_q(o)[ ln p(o) ]`` is the
  expected log-preference (utility / exploitation), and ``epistemic_value`` is the
  state-outcome mutual information ``I(s;o|pi) = H[q(o)] - E_q(s)[H[p(o|s)]]``
  (expected information gain / salience driving exploration).

These are tied by the algebraic identity (per visited timestep, hence summed over
the policy horizon):

    risk + ambiguity == -(pragmatic_value + epistemic_value)

which follows from ``risk = -H[q(o)] - pragmatic_value`` (KL split into a
cross-entropy minus an entropy) and ``epistemic_value = H[q(o)] - ambiguity``.
Equivalently ``G(pi) = -(pragmatic_value + epistemic_value)``: minimizing EFE
maximizes the sum of utility and information gain. Every term is computed in
closed form from the categorical generative model ``(A, B, C, prior)`` — no
sampling — so the identity is machine-checkable to floating-point tolerance
(project gate ISC-19, ``1e-9``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .divergences import kl_divergence
from .generalized_bayes import softmax

ArrayF = np.ndarray
_EPS = 1e-12

# Absolute tolerance for the EFE decomposition identity
# risk + ambiguity == -(pragmatic + epistemic); pinned by ISC-19, consumed by
# the invariants verifier and the ISC_EFE_TOLERANCE manuscript token.
EFE_IDENTITY_ATOL: float = 1e-9


@dataclass(frozen=True)
class EFETerms:
    """The four EFE terms for one policy, summed over its horizon.

    ``total`` (a property) is the Expected Free Energy ``G(pi) = risk + ambiguity``.
    ``identity_residual`` is ``total + pragmatic_value + epistemic_value`` and is
    zero (to floating-point tolerance) whenever the closed form is correct.
    """

    risk: float
    ambiguity: float
    pragmatic_value: float
    epistemic_value: float

    @property
    def total(self) -> float:
        """Expected Free Energy ``G(pi) = risk + ambiguity``."""
        return self.risk + self.ambiguity

    @property
    def identity_residual(self) -> float:
        """``(risk + ambiguity) + pragmatic_value + epistemic_value`` (== 0 when valid)."""
        return self.total + self.pragmatic_value + self.epistemic_value


def _as_2d(matrix: ArrayF) -> ArrayF:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("expected a 2-D array")
    return arr


def _column_pmf(a_matrix: ArrayF) -> ArrayF:
    """Return ``A`` with each column clipped and renormalized to a pmf over outcomes."""
    clipped = np.clip(a_matrix, _EPS, None)
    return clipped / clipped.sum(axis=0, keepdims=True)


def _raw_column_pmf(a_matrix: ArrayF) -> ArrayF:
    """Column-normalize ``A`` WITHOUT an ``_EPS`` floor, preserving exact zeros.

    Used only for the likelihood-entropy term ``H[p(o|s)]``: flooring true zeros
    to ``_EPS`` would leak a spurious ``-2 * _EPS * ln(_EPS)`` (~5e-11) into the
    ambiguity of a deterministic likelihood, breaking the ``H = 0`` semantics.
    :func:`_entropy` already treats ``0 ln 0 = 0`` exactly.
    """
    arr = np.clip(np.asarray(a_matrix, dtype=np.float64), 0.0, None)
    col_sums = arr.sum(axis=0, keepdims=True)
    col_sums = np.where(col_sums > 0.0, col_sums, 1.0)
    return arr / col_sums


def _entropy(distribution: ArrayF) -> float:
    """Shannon entropy in nats, dropping zero-probability outcomes (``0 ln 0 := 0``)."""
    p = np.asarray(distribution, dtype=np.float64).ravel()
    nonzero = p[p > 0.0]
    return float(-np.sum(nonzero * np.log(nonzero)))


def preferred_outcomes(c_vector: ArrayF) -> ArrayF:
    """Preferred-outcome distribution ``p(o) = softmax(C)`` from log-preferences ``C``."""
    return softmax(c_vector)


def decompose(
    A: ArrayF,
    B: ArrayF,
    C: ArrayF,
    prior: ArrayF,
    policy: Sequence[int],
) -> EFETerms:
    """Closed-form EFE decomposition for one policy over a categorical POMDP.

    Args:
        A: Likelihood tensor ``(n_o, n_s)``; column ``s`` is ``p(o | s)``.
        B: Transition tensor ``(n_s, n_s, n_a)``; ``B[:, :, a] @ s`` propagates the
           state belief one step under action ``a``.
        C: Log-preference vector ``(n_o,)``; ``p(o) = softmax(C)``.
        prior: Initial state belief ``(n_s,)`` (a pmf; clipped + renormalized).
        policy: Sequence of action indices (the horizon is its length).

    Returns:
        :class:`EFETerms` with ``risk``, ``ambiguity``, ``pragmatic_value`` and
        ``epistemic_value`` summed over the visited timesteps. ``total`` is the EFE
        and ``identity_residual`` is zero to floating-point tolerance.

    Raises:
        ValueError: If the array shapes are inconsistent or an action index is out
            of range for ``B``.
    """
    a_matrix = _column_pmf(_as_2d(A))
    b_tensor = np.asarray(B, dtype=np.float64)
    if b_tensor.ndim != 3:
        raise ValueError("B must be a 3-D (n_s, n_s, n_a) transition tensor")
    n_o, n_s = a_matrix.shape
    if b_tensor.shape[0] != n_s or b_tensor.shape[1] != n_s:
        raise ValueError("B leading dimensions must match the hidden-state count of A")

    preference = preferred_outcomes(C)
    if preference.shape[0] != n_o:
        raise ValueError("C length must match the outcome count of A")
    log_pref = np.log(np.clip(preference, _EPS, None))

    # Per-state likelihood entropy H[p(o | s)] for every hidden state. Computed
    # from the un-floored columns so a deterministic likelihood has H == 0
    # exactly (the _EPS floor used elsewhere would leak ~5e-11 of spurious mass).
    entropy_matrix = _raw_column_pmf(_as_2d(A))
    state_entropy = np.array([_entropy(entropy_matrix[:, s]) for s in range(n_s)], dtype=np.float64)

    state_belief = np.clip(np.asarray(prior, dtype=np.float64).ravel(), _EPS, None)
    if state_belief.shape[0] != n_s:
        raise ValueError("prior length must match the hidden-state count of A")
    state_belief = state_belief / state_belief.sum()

    n_a = b_tensor.shape[2]
    risk = ambiguity = pragmatic = epistemic = 0.0

    for action in policy:
        a = int(action)
        if not 0 <= a < n_a:
            raise ValueError(f"action index {a} out of range for B with {n_a} actions")
        predicted_obs = a_matrix @ state_belief  # q(o | pi) at this step
        ambiguity_step = float(state_belief @ state_entropy)  # E_q(s)[ H[p(o|s)] ]

        risk += kl_divergence(predicted_obs, preference)
        ambiguity += ambiguity_step
        pragmatic += float(np.sum(predicted_obs * log_pref))  # E_q(o)[ ln p(o) ]
        epistemic += _entropy(predicted_obs) - ambiguity_step  # I(s;o) = H[q(o)] - ambiguity

        state_belief = b_tensor[:, :, a] @ state_belief

    return EFETerms(
        risk=risk,
        ambiguity=ambiguity,
        pragmatic_value=pragmatic,
        epistemic_value=epistemic,
    )
