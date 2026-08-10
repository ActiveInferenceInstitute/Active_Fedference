"""Shared experiment harness helpers."""

from __future__ import annotations

import warnings

import numpy as np

from ..aggregation import log_linear_pool
from ..belief_sharing import share_round
from ..statistics import d_equivalent_from_rank_biserial

ArrayF = np.ndarray
_EPS = 1e-12
_N_BOOT = 5000
_D_EQUIVALENT_CAP = 1e6

#: Configured sweep label -> distinct server-side robust-aggregation operating
#: point (``KLD`` = naive). These values are *not* implementations of the
#: corresponding FedGVI client divergences; the report records this mapping so
#: the server-heuristic results cannot be misread as client-side evidence.
_DIVERGENCE_ROBUSTNESS: dict[str, float] = {
    "KLD": 0.0,
    "RKL": 1.5,
    "AR": 1.3,
    "beta": 1.7,
    "rcce": 1.6,
}


def _finite_d_equivalent(r: float) -> float:
    """Finite JSON-safe rank-biserial-derived ``d``-equivalent."""
    d = d_equivalent_from_rank_biserial(r)
    if not np.isfinite(d):
        return float(np.sign(d) * _D_EQUIVALENT_CAP)
    return float(d)


def _finite_cohens_d(r: float) -> float:
    """Deprecated compatibility alias for :func:`_finite_d_equivalent`."""
    warnings.warn(
        "_finite_cohens_d is deprecated; use _finite_d_equivalent",
        DeprecationWarning,
        stacklevel=2,
    )
    return _finite_d_equivalent(r)


def _sample_observation(
    likelihood: ArrayF, true_state: int, rng: np.random.Generator
) -> int:
    """Draw an outcome ``o ~ P(o | s = true_state)`` from a likelihood column."""
    column = likelihood[:, int(true_state)]
    column = np.clip(column, 0.0, None)
    column = column / column.sum()
    return int(rng.choice(column.shape[0], p=column))


def _divergence_to_robustness(divergence: str) -> float:
    if divergence not in _DIVERGENCE_ROBUSTNESS:
        raise ValueError(
            f"unknown divergence {divergence!r}; choose from {sorted(_DIVERGENCE_ROBUSTNESS)}"
        )
    return _DIVERGENCE_ROBUSTNESS[divergence]


def _consensus_accuracy(
    local_posteriors: ArrayF, divergence: str, true_state: int
) -> float:
    """Fuse local posteriors under ``divergence`` and return true-state mass."""
    robustness = _divergence_to_robustness(divergence)
    if robustness == 0.0:
        consensus = log_linear_pool(local_posteriors=local_posteriors)
    else:
        consensus = share_round(
            local_posteriors,
            method="robust",
            robustness=robustness,
            exclude_self=False,
            true_state=true_state,
        ).consensus
    return float(consensus[int(true_state)])


__all__ = [
    "ArrayF",
    "_EPS",
    "_N_BOOT",
    "_DIVERGENCE_ROBUSTNESS",
    "_finite_d_equivalent",
    "_finite_cohens_d",
    "_sample_observation",
    "_divergence_to_robustness",
    "_consensus_accuracy",
]
