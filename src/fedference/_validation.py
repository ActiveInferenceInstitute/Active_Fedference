"""Shared finite-simplex validation for the categorical FedGVI core.

The aggregation, divergence, and belief-sharing methods all operate on the same
mathematical object: a finite categorical probability mass function.  Keeping
the boundary checks in one small module prevents one method from silently
accepting NaNs, negative mass, or mismatched rows while another rejects them.
The positive floor is retained for valid zero-probability entries so the
log-domain formulas remain finite; invalid negative and non-finite inputs are
never repaired by clipping.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

ArrayF = np.ndarray
_EPS = 1e-12


def as_pmf(values: ArrayF, *, name: str = "probability vector") -> ArrayF:
    """Return a finite, non-negative, one-dimensional normalized pmf.

    Exact zeros are floored at ``1e-12`` for the log-domain categorical
    formulas.  Negative values are not numerical zeros: they violate the
    simplex and raise ``ValueError`` instead of being silently clipped.
    """
    arr: ArrayF = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be non-negative")
    if float(arr.sum()) <= 0.0:
        raise ValueError(f"{name} must have a positive finite sum")
    arr = np.clip(arr, _EPS, None)
    total = float(arr.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"{name} must have a positive finite sum")
    return arr / total


def as_pmf_matrix(
    values: Iterable[ArrayF], *, name: str = "local_posteriors"
) -> ArrayF:
    """Return a non-empty ``(n_items, n_states)`` matrix of local posteriors."""
    raw_rows = list(values)
    if any(np.asarray(value).ndim == 0 for value in raw_rows):
        raise ValueError(f"{name} must contain vector rows")
    rows = [
        as_pmf(value, name=f"{name}[{index}]")
        for index, value in enumerate(raw_rows)
    ]
    if not rows:
        raise ValueError(f"{name} must be a non-empty iterable")
    n_states = rows[0].size
    if any(row.size != n_states for row in rows[1:]):
        raise ValueError(f"{name} must contain equal-length pmfs")
    return np.vstack(rows)


def as_nonnegative_weights(
    values: Iterable[float] | None,
    n_items: int,
    *,
    name: str = "base_weights",
) -> ArrayF:
    """Return finite non-negative weights with at least one positive entry."""
    if values is None:
        return np.ones(n_items, dtype=np.float64)
    raw = values if isinstance(values, np.ndarray) else list(values)
    weights = np.asarray(raw, dtype=np.float64).ravel()
    if weights.size != n_items:
        raise ValueError(f"{name} length must match number of agents")
    if not np.all(np.isfinite(weights)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(weights < 0.0):
        raise ValueError(f"{name} must be non-negative")
    if float(weights.sum()) <= _EPS:
        raise ValueError(f"{name} must contain at least one positive value")
    return weights


__all__ = ["as_nonnegative_weights", "as_pmf", "as_pmf_matrix"]
