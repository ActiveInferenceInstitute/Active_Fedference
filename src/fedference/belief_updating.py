"""Single-agent variational state inference (Friston et al., 2024, Eq. 4).

Discrete active inference infers a posterior over hidden states ``s`` from an
observation ``o`` by a single softmax step. With a categorical prior ``D`` and a
likelihood (``A``) tensor of shape ``(n_o, n_s)`` whose column ``s`` is the pmf
``p(o | s)``, Friston et al. (2024) Eq. 4 gives the variational posterior

    q(s)  =  softmax( ln D  +  ln A[o, :] )

i.e. the prior in log-space plus the log-likelihood *message* ``ln A[o, :]``
(the row of ``A`` selected by the observed outcome). For several conditionally
independent observation modalities ``m`` the messages simply add,

    q(s)  =  softmax( ln D  +  sum_m ln A_m[o_m, :] )

which is the categorical product-of-experts that, summed over modalities, makes
each modality an additive evidence term.

The associated **variational free energy** (the quantity the softmax minimizes,
Friston et al. 2024) is

    F[q]  =  E_q[ ln q(s) - ln D(s) - sum_m ln A_m[o_m, s] ]
          =  KL( q || D )  -  E_q[ sum_m ln A_m[o_m, s] ]

and is minimized exactly by the one-step posterior above; at the minimizer ``F``
equals ``-ln sum_s D(s) prod_m A_m[o_m, s]`` (the negative log model evidence).

This module reimplements the math in pure numpy (the reference pymdp/JAX code is
read for the equations only, never imported). It reuses the locked, tested
:func:`fedference.generalized_bayes.softmax`, so the one-step posterior is the
``loss = NLL``, ``learning_rate = 1`` special case of generalized Bayes — the
hinge identity that lets federated robust inference degrade to Friston (2024).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .generalized_bayes import softmax

ArrayF = np.ndarray
_EPS = 1e-12


def _likelihoods(A: ArrayF | Sequence[ArrayF]) -> list[ArrayF]:
    """Normalize the ``A`` argument to a list of 2-D ``(n_o, n_s)`` arrays.

    Accepts either a single likelihood tensor or a sequence of per-modality
    tensors. Every tensor must be 2-D and share the same number of hidden
    states (``n_s``, the column count).
    """
    if isinstance(A, np.ndarray):
        mats: list[ArrayF] = [A]
    else:
        mats = [np.asarray(a, dtype=np.float64) for a in A]
        if not mats:
            raise ValueError("at least one likelihood modality is required")
    out: list[ArrayF] = []
    n_s: int | None = None
    for a in mats:
        arr = np.asarray(a, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError("each likelihood A must be a 2-D (n_o, n_s) array")
        if n_s is None:
            n_s = arr.shape[1]
        elif arr.shape[1] != n_s:
            raise ValueError("all modalities must share the same n_s (column count)")
        out.append(arr)
    return out


def _observations(obs: int | Sequence[int], n_modalities: int) -> list[int]:
    """Normalize ``obs`` to a list of one outcome index per modality."""
    if isinstance(obs, (int, np.integer)):
        idx = [int(obs)]
    else:
        idx = [int(o) for o in obs]
    if len(idx) != n_modalities:
        raise ValueError(
            f"expected {n_modalities} observation(s), got {len(idx)}"
        )
    return idx


def _log_likelihood_message(mats: list[ArrayF], idx: list[int]) -> ArrayF:
    """Return ``sum_m ln A_m[o_m, :]`` — the additive log-likelihood over states."""
    n_s = mats[0].shape[1]
    message = np.zeros(n_s, dtype=np.float64)
    for a, o in zip(mats, idx):
        if not 0 <= o < a.shape[0]:
            raise ValueError(f"observation index {o} out of range for n_o={a.shape[0]}")
        row = np.clip(a[o, :], _EPS, None)
        message += np.log(row)
    return message


def infer_states(
    A: ArrayF | Sequence[ArrayF],
    obs: int | Sequence[int],
    log_prior: ArrayF,
) -> ArrayF:
    """One-step variational posterior over hidden states (Friston Eq. 4).

    ``A``        : likelihood ``(n_o, n_s)`` tensor, or a sequence of such
                   tensors (one per observation modality), columns indexed by
                   hidden state.
    ``obs``      : observed outcome index (single modality) or a sequence of
                   per-modality outcome indices.
    ``log_prior``: length-``n_s`` log prior ``ln D`` (need not be normalized).

    Returns the categorical posterior pmf
    ``softmax(log_prior + sum_m ln A_m[o_m, :])``.
    """
    mats = _likelihoods(A)
    idx = _observations(obs, len(mats))
    lp = np.asarray(log_prior, dtype=np.float64).ravel()
    if lp.shape[0] != mats[0].shape[1]:
        raise ValueError("log_prior length must equal n_s (likelihood columns)")
    message = _log_likelihood_message(mats, idx)
    return softmax(lp + message)


def vfe(
    qs: ArrayF,
    A: ArrayF | Sequence[ArrayF],
    obs: int | Sequence[int],
    log_prior: ArrayF,
) -> float:
    """Variational free energy ``F[q]`` for a belief ``qs`` (Friston 2024).

    ``F = E_q[ln q - ln D - sum_m ln A_m[o_m, s]]``, in nats. The locked
    one-step posterior from :func:`infer_states` is the unique minimizer; at the
    minimizer ``F`` equals the negative log model evidence.
    """
    mats = _likelihoods(A)
    idx = _observations(obs, len(mats))
    q = np.clip(np.asarray(qs, dtype=np.float64).ravel(), _EPS, None)
    q = q / q.sum()
    lp = np.asarray(log_prior, dtype=np.float64).ravel()
    if lp.shape[0] != mats[0].shape[1] or q.shape[0] != mats[0].shape[1]:
        raise ValueError("qs and log_prior length must equal n_s")
    # Normalize the prior so the cross-entropy term uses a genuine ln D.
    prior = softmax(lp)
    log_prior_norm = np.log(np.clip(prior, _EPS, None))
    message = _log_likelihood_message(mats, idx)
    energy = float(np.sum(q * (np.log(q) - log_prior_norm - message)))
    return energy
