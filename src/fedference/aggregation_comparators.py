"""Experimental belief-space comparators for the MAJ-1 evidence ladder.

These methods are intentionally outside the stable ``AggregationConfig``
dispatch. They provide geometry-aware controls for experiments; they do not
inherit parameter-space robust-federated-learning guarantees.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from ._validation import as_nonnegative_weights, as_pmf_matrix
from .generalized_bayes import softmax

ArrayF = np.ndarray
_EPS = 1e-12


@dataclass(frozen=True)
class ComparatorResult:
    """Consensus and convergence diagnostics for an experimental comparator."""

    consensus: ArrayF
    iterations: int
    converged: bool


def linear_opinion_pool(
    local_posteriors: Iterable[ArrayF] | None = None,
    base_weights: Iterable[float] | None = None,
    **legacy: object,
) -> ArrayF:
    """Weighted arithmetic mixture of categorical beliefs."""
    if "beliefs" in legacy:
        if local_posteriors is not None:
            raise TypeError(
                "local_posteriors and deprecated beliefs cannot both be supplied"
            )
        local_posteriors = legacy.pop("beliefs")  # type: ignore[assignment]
        warnings.warn(
            "beliefs is deprecated; use local_posteriors",
            DeprecationWarning,
            stacklevel=2,
        )
    if "weights" in legacy:
        if base_weights is not None:
            raise TypeError("base_weights and deprecated weights cannot both be supplied")
        base_weights = legacy.pop("weights")  # type: ignore[assignment]
        warnings.warn(
            "weights is deprecated; use base_weights",
            DeprecationWarning,
            stacklevel=2,
        )
    if legacy:
        raise TypeError(f"unexpected keyword argument(s): {', '.join(sorted(legacy))}")
    if local_posteriors is None:
        raise TypeError("local_posteriors is required")
    matrix = as_pmf_matrix(local_posteriors)
    base = as_nonnegative_weights(base_weights, matrix.shape[0])
    consensus = base @ matrix
    return consensus / consensus.sum()


def _centered_log_ratio(matrix: ArrayF) -> ArrayF:
    logged = np.log(matrix)
    return logged - logged.mean(axis=1, keepdims=True)


def _observed_point_minimizer(
    points: ArrayF,
    weights: ArrayF,
    *,
    tol: float,
) -> ArrayF | None:
    """Return an observed weighted-median point when its subgradient contains zero."""
    for index, point in enumerate(points):
        distances = np.linalg.norm(points - point, axis=1)
        coincident = distances <= _EPS
        noncoincident = ~coincident
        residual = np.sum(
            weights[noncoincident, None]
            * (points[noncoincident] - point)
            / distances[noncoincident, None],
            axis=0,
        )
        if np.linalg.norm(residual) <= float(weights[coincident].sum()) + tol:
            return point
    return None


def clr_geometric_median_pool(
    local_posteriors: Iterable[ArrayF] | None = None,
    *,
    base_weights: Iterable[float] | None = None,
    max_iter: int = 256,
    tol: float = 1e-10,
    **legacy: object,
) -> ComparatorResult:
    """Pool beliefs by a weighted geometric median in CLR coordinates.

    The centered-log-ratio map respects the simplex's relative geometry. The
    weighted geometric median is solved by a guarded Weiszfeld iteration and
    mapped back with softmax. This is a comparator, not an objective or
    robustness result for :func:`fedference.aggregation.robust_aggregate`.
    """
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("tol must be a finite non-negative value")
    if "beliefs" in legacy:
        if local_posteriors is not None:
            raise TypeError(
                "local_posteriors and deprecated beliefs cannot both be supplied"
            )
        local_posteriors = legacy.pop("beliefs")  # type: ignore[assignment]
        warnings.warn(
            "beliefs is deprecated; use local_posteriors",
            DeprecationWarning,
            stacklevel=2,
        )
    if "weights" in legacy:
        if base_weights is not None:
            raise TypeError("base_weights and deprecated weights cannot both be supplied")
        base_weights = legacy.pop("weights")  # type: ignore[assignment]
        warnings.warn(
            "weights is deprecated; use base_weights",
            DeprecationWarning,
            stacklevel=2,
        )
    if legacy:
        raise TypeError(f"unexpected keyword argument(s): {', '.join(sorted(legacy))}")
    if local_posteriors is None:
        raise TypeError("local_posteriors is required")
    matrix = as_pmf_matrix(local_posteriors)
    base = as_nonnegative_weights(base_weights, matrix.shape[0])
    base = base / base.sum()
    points = _centered_log_ratio(matrix)
    observed_minimizer = _observed_point_minimizer(points, base, tol=tol)
    if observed_minimizer is not None:
        return ComparatorResult(
            consensus=softmax(observed_minimizer),
            iterations=0,
            converged=True,
        )
    estimate = base @ points
    converged = False
    iteration = 0
    for iteration in range(1, max_iter + 1):
        distances = np.linalg.norm(points - estimate, axis=1)
        coincident = np.flatnonzero(distances <= _EPS)
        if coincident.size:
            # A data point is a valid minimizer when its subgradient norm is
            # within the weight at that point. This deterministic branch also
            # avoids division by zero in Weiszfeld's update.
            index = int(coincident[0])
            noncoincident = distances > _EPS
            residual = np.sum(
                base[noncoincident, None]
                * (points[noncoincident] - points[index])
                / np.maximum(
                    distances[noncoincident, None],
                    _EPS,
                ),
                axis=0,
            )
            if np.linalg.norm(residual) <= float(base[coincident].sum()) + tol:
                estimate = points[index]
                converged = True
                break
        inverse = base / np.maximum(distances, _EPS)
        updated = (inverse @ points) / inverse.sum()
        if np.linalg.norm(updated - estimate) <= tol:
            estimate = updated
            converged = True
            break
        estimate = updated
    return ComparatorResult(
        consensus=softmax(estimate),
        iterations=iteration,
        converged=converged,
    )


__all__ = [
    "ComparatorResult",
    "clr_geometric_median_pool",
    "linear_opinion_pool",
]
