"""Robust loss functions for generalized Bayes (the data-fidelity term).

In standard Bayes the data term is the negative log-likelihood (NLL). FedGVI
(Mildner et al., 2025) replaces it with a *robust* loss so that a single
mislabelled / contaminated observation cannot dominate the posterior. We
implement the categorical analogues used in the paper plus the density-power
(beta) score.

Each loss takes a categorical likelihood row ``p = p(o | s)`` (a pmf over the
``n_o`` outcomes) and an observed outcome index ``o``, and returns a scalar
*loss* ``L(p, o)`` (smaller = better fit). The defining property tested by the
suite is the **limit to NLL**:

* ``rcce(p, o, q_loss) -> nll(p, o)`` as ``q_loss -> 0`` (generalized cross-entropy)
* ``beta_loss(p, o, beta) -> nll``    as ``beta -> 0`` (density power)

so that ``loss_param = 0`` exactly reproduces standard Bayes — the regime in
which federated active inference reduces to Friston et al. (2024).
"""

from __future__ import annotations

import warnings

import numpy as np

from ._validation import as_pmf

ArrayF = np.ndarray


def _row(p: ArrayF) -> ArrayF:
    return as_pmf(p, name="likelihood row")


def _outcome_index(o: int, n_outcomes: int) -> int:
    """Validate an observed categorical outcome and return its array index."""
    if isinstance(o, (bool, np.bool_)):
        raise ValueError("outcome index must be an integer")
    try:
        raw = float(o)
    except (TypeError, ValueError) as exc:
        raise ValueError("outcome index must be an integer") from exc
    if not np.isfinite(raw) or raw != np.floor(raw):
        raise ValueError("outcome index must be an integer")
    index = int(raw)
    if not 0 <= index < n_outcomes:
        raise ValueError(f"outcome index must lie in [0, {n_outcomes})")
    return index


def nll(p: ArrayF, o: int) -> float:
    """Standard negative log-likelihood loss ``-log p(o | s)``."""
    row = _row(p)
    return float(-np.log(row[_outcome_index(o, row.size)]))


def rcce(
    p: ArrayF,
    o: int,
    q_loss: float | None = None,
    *,
    q: float | None = None,
) -> float:
    """Robust categorical cross-entropy (generalized cross-entropy, GCE).

    ``L_{q_loss}(p, o) = (1 - p(o)^q_loss) / q_loss`` for
    ``q_loss in (0, 1]`` (Zhang & Sabuncu, 2018). This is FedGVI's ``rcce``
    loss with ``loss_param = q_loss``. As ``q_loss -> 0`` it
    recovers :func:`nll` (l'Hopital: ``(1 - p^q_loss)/q_loss -> -log p``); at
    ``q_loss = 1`` it
    is the bounded mean-absolute-error loss ``1 - p(o)``, fully robust to
    outliers.

    ``q`` remains a deprecated keyword-only compatibility alias. The explicit
    ``q_loss`` name prevents a collision with the posterior distribution
    conventionally denoted ``q`` elsewhere in the model.
    """
    if q_loss is not None and q is not None:
        raise TypeError("pass q_loss or deprecated q, not both")
    if q_loss is None:
        if q is None:
            raise TypeError("rcce requires q_loss")
        warnings.warn(
            "rcce(q=...) is deprecated; use rcce(q_loss=...) instead",
            DeprecationWarning,
            stacklevel=2,
        )
        q_loss = q
    if not np.isfinite(q_loss) or not 0.0 <= q_loss <= 1.0:
        raise ValueError("rcce parameter q_loss must lie in [0, 1]")
    row = _row(p)
    po = row[_outcome_index(o, row.size)]
    if q_loss < 1e-9:
        return float(-np.log(po))
    return float((1.0 - po**q_loss) / q_loss)


def beta_loss(p: ArrayF, o: int, beta: float) -> float:
    """Density-power (beta) loss for a categorical likelihood.

    ``L_beta(p, o) = -(1/beta) p(o)^beta + (1/(beta+1)) sum_k p_k^(beta+1)``
    up to an additive constant (Basu et al., 1998). As ``beta -> 0`` it
    recovers :func:`nll`. Larger ``beta`` discounts low-probability
    observations, giving robustness to contamination.
    """
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be non-negative")
    row = _row(p)
    outcome = _outcome_index(o, row.size)
    if beta < 1e-9:
        return float(-np.log(row[outcome]))
    po = row[outcome]
    # Recentred density-power score: each term -> the NLL contribution as
    # beta -> 0. -(p^beta - 1)/beta -> -log p; (sum p^(beta+1) - 1)/(beta+1) -> 0.
    data_term = -(po**beta - 1.0) / beta
    norm_term = (np.sum(row ** (beta + 1.0)) - 1.0) / (beta + 1.0)
    return float(data_term + norm_term)


def loss_vector(likelihood: ArrayF, o: int, *, loss: str = "nll", param: float = 0.0) -> ArrayF:
    """Return the per-state loss vector ``L(p(o|s), o)`` over hidden states ``s``.

    ``likelihood`` is the ``(n_o, n_s)`` ``A`` tensor (column = state). For each
    state column it evaluates the chosen loss at the observed outcome ``o`` and
    returns the length-``n_s`` vector consumed by
    :func:`fedference.generalized_bayes.generalized_posterior`.
    """
    a = np.asarray(likelihood, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("likelihood must be a 2-D (n_o, n_s) array")
    if a.shape[0] == 0 or a.shape[1] == 0:
        raise ValueError("likelihood must have at least one outcome and state")
    _outcome_index(o, a.shape[0])
    n_s = a.shape[1]
    fn = {
        "nll": lambda col: nll(col, o),
        "rcce": lambda col: rcce(col, o, q_loss=param),
        "beta": lambda col: beta_loss(col, o, param),
    }.get(loss)
    if fn is None:
        raise ValueError(f"unknown loss {loss!r}; choose from nll, rcce, beta")
    return np.array([fn(a[:, s]) for s in range(n_s)], dtype=np.float64)
