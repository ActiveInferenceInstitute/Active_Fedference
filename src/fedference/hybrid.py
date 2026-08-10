"""Minimal mixed discrete/Gaussian belief representation (MAJ-3 slice).

The categorical project can be extended without pretending that a one-dimensional
Gaussian example is already a continuous active-inference system.  A
:class:`HybridBelief` stores a categorical mixture weight and one Gaussian
conditional for each discrete component.  The log-linear pool combines the
categorical weights and precision-combines the matching Gaussian conditionals.
The robust path iteratively applies the same declared reverse-KL reweighting
heuristic to a hybrid divergence.

The only release-level identity supplied here is the recovery corner:
``hybrid_aggregate(..., robustness=0)`` is exactly the hybrid log-linear pool,
and a one-component hybrid reduces to the existing Gaussian product rule.  No
objective or bounded-influence claim is transferred to the robust hybrid path.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ._validation import as_nonnegative_weights, as_pmf
from .divergences import gaussian_kl, kl_divergence

ArrayF = np.ndarray
_EPS = 1e-12


@dataclass(frozen=True)
class HybridBelief:
    """A categorical mixture with a Gaussian conditional per component."""

    discrete: ArrayF
    gaussian_mean: ArrayF
    gaussian_var: ArrayF

    def __post_init__(self) -> None:
        discrete = as_pmf(self.discrete, name="discrete")
        mean = np.asarray(self.gaussian_mean, dtype=np.float64).ravel()
        var = np.asarray(self.gaussian_var, dtype=np.float64).ravel()
        if mean.shape != discrete.shape or var.shape != discrete.shape:
            raise ValueError("Gaussian mean/variance arrays must match discrete shape")
        if not np.all(np.isfinite(mean)):
            raise ValueError("gaussian_mean must be finite")
        if not np.all(np.isfinite(var)) or np.any(var <= 0.0):
            raise ValueError("gaussian_var must be finite and strictly positive")
        discrete = discrete.copy()
        mean = mean.copy()
        var = var.copy()
        discrete.setflags(write=False)
        mean.setflags(write=False)
        var.setflags(write=False)
        object.__setattr__(self, "discrete", discrete)
        object.__setattr__(self, "gaussian_mean", mean)
        object.__setattr__(self, "gaussian_var", var)

    @property
    def n_components(self) -> int:
        """Number of discrete mixture components."""
        return int(self.discrete.size)


@dataclass(frozen=True)
class HybridAggregationResult:
    """Consensus and reweighting diagnostics for a hybrid aggregation round."""

    consensus: HybridBelief
    normalized_effective_weights: ArrayF
    iterations: int
    converged: bool

    def __post_init__(self) -> None:
        if not isinstance(self.consensus, HybridBelief):
            raise ValueError("consensus must be a HybridBelief")
        weights = np.array(self.normalized_effective_weights, dtype=np.float64, copy=True).ravel()
        if (
            weights.size == 0
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0.0)
            or not np.isclose(float(weights.sum()), 1.0, rtol=0.0, atol=1e-12)
        ):
            raise ValueError("normalized_effective_weights must be a non-negative probability vector")
        if (
            isinstance(self.iterations, (bool, np.bool_))
            or not isinstance(self.iterations, (int, np.integer))
            or self.iterations < 0
        ):
            raise ValueError("iterations must be a non-negative integer")
        if not isinstance(self.converged, bool):
            raise ValueError("converged must be a boolean")
        weights.setflags(write=False)
        object.__setattr__(self, "normalized_effective_weights", weights)
        object.__setattr__(self, "iterations", int(self.iterations))

    @property
    def agent_weights(self) -> ArrayF:
        """Deprecated alias for normalized effective influence weights."""
        warnings.warn(
            "HybridAggregationResult.agent_weights is deprecated; use "
            "normalized_effective_weights",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.normalized_effective_weights


def _stack(local_posteriors: Iterable[HybridBelief]) -> list[HybridBelief]:
    rows = list(local_posteriors)
    if not rows:
        raise ValueError("local_posteriors must be non-empty")
    if any(not isinstance(row, HybridBelief) for row in rows):
        raise TypeError("local_posteriors must contain HybridBelief instances")
    components = rows[0].n_components
    if any(row.n_components != components for row in rows):
        raise ValueError("all hybrid beliefs must have the same component count")
    return rows


def _normalise_weights(
    base_weights: Iterable[float] | None,
    n_agents: int,
) -> ArrayF:
    if base_weights is None:
        return np.full(n_agents, 1.0 / n_agents, dtype=np.float64)
    raw = as_nonnegative_weights(base_weights, n_agents, name="base_weights")
    total = float(raw.sum())
    if total <= _EPS:
        raise ValueError("base_weights must contain positive total mass")
    return raw / total


def _pool(rows: list[HybridBelief], weights: ArrayF) -> HybridBelief:
    discrete = np.exp(np.sum(weights[:, None] * np.log(np.vstack([row.discrete for row in rows])), axis=0))
    discrete /= discrete.sum()
    means = np.vstack([row.gaussian_mean for row in rows])
    variances = np.vstack([row.gaussian_var for row in rows])
    precision = np.sum(weights[:, None] / variances, axis=0)
    mean = np.sum(weights[:, None] * means / variances, axis=0) / precision
    var = 1.0 / precision
    return HybridBelief(discrete, mean, var)

def hybrid_log_linear_pool(
    local_posteriors: Iterable[HybridBelief] | None = None,
    *,
    base_weights: Iterable[float] | None = None,
    **legacy: object,
) -> HybridBelief:
    """Combine hybrid beliefs by a categorical log pool and Gaussian precision pool."""
    if "beliefs" in legacy:
        if local_posteriors is not None:
            raise TypeError("local_posteriors and deprecated beliefs cannot both be supplied")
        local_posteriors = legacy.pop("beliefs")  # type: ignore[assignment]
        warnings.warn("beliefs is deprecated; use local_posteriors", DeprecationWarning, stacklevel=2)
    if "weights" in legacy:
        if base_weights is not None:
            raise TypeError("base_weights and deprecated weights cannot both be supplied")
        base_weights = legacy.pop("weights")  # type: ignore[assignment]
        warnings.warn("weights is deprecated; use base_weights", DeprecationWarning, stacklevel=2)
    if legacy:
        names = ", ".join(sorted(legacy))
        raise TypeError(f"unexpected keyword argument(s): {names}")
    if local_posteriors is None:
        raise TypeError("local_posteriors is required")
    rows = _stack(local_posteriors)
    return _pool(rows, _normalise_weights(base_weights, len(rows)))


def _hybrid_divergence(row: HybridBelief, reference: HybridBelief) -> float:
    """Aligned-component joint ``KL(row || reference)`` diagnostic."""
    categorical = kl_divergence(row.discrete, reference.discrete)
    gaussian = sum(
        float(row.discrete[k])
        * gaussian_kl(
            row.gaussian_mean[k],
            row.gaussian_var[k],
            reference.gaussian_mean[k],
            reference.gaussian_var[k],
        )
        for k in range(row.n_components)
    )
    return float(categorical + gaussian)


def hybrid_aggregate(
    local_posteriors: Iterable[HybridBelief] | None = None,
    *,
    base_weights: Iterable[float] | None = None,
    robustness: float = 0.0,
    max_iter: int = 128,
    tol: float = 1e-12,
    **legacy: object,
) -> HybridAggregationResult:
    """Aggregate hybrid beliefs with exact zero-robustness recovery.

    For ``robustness == 0`` the returned consensus is the hybrid log-linear
    pool.  For positive robustness this is a finite fixed-point diagnostic, not
    a variational objective; its status is intentionally marked in module and
    manuscript documentation.
    """
    if (
        isinstance(robustness, (bool, np.bool_))
        or not isinstance(
            robustness,
            (int, float, np.integer, np.floating),
        )
        or not np.isfinite(robustness)
        or robustness < 0.0
    ):
        raise ValueError("robustness must be finite and non-negative")
    robustness = float(robustness)
    if (
        isinstance(max_iter, (bool, np.bool_))
        or not isinstance(max_iter, (int, np.integer))
        or max_iter < 1
    ):
        raise ValueError("max_iter must be a positive integer")
    if (
        isinstance(tol, (bool, np.bool_))
        or not isinstance(tol, (int, float, np.integer, np.floating))
        or not np.isfinite(tol)
        or tol <= 0.0
    ):
        raise ValueError("tol must be finite and positive")
    if "beliefs" in legacy:
        if local_posteriors is not None:
            raise TypeError("local_posteriors and deprecated beliefs cannot both be supplied")
        local_posteriors = legacy.pop("beliefs")  # type: ignore[assignment]
        warnings.warn("beliefs is deprecated; use local_posteriors", DeprecationWarning, stacklevel=2)
    if "weights" in legacy:
        if base_weights is not None:
            raise TypeError("base_weights and deprecated weights cannot both be supplied")
        base_weights = legacy.pop("weights")  # type: ignore[assignment]
        warnings.warn("weights is deprecated; use base_weights", DeprecationWarning, stacklevel=2)
    if legacy:
        names = ", ".join(sorted(legacy))
        raise TypeError(f"unexpected keyword argument(s): {names}")
    if local_posteriors is None:
        raise TypeError("local_posteriors is required")
    rows = _stack(local_posteriors)
    base = _normalise_weights(base_weights, len(rows))
    if robustness == 0.0:
        return HybridAggregationResult(hybrid_log_linear_pool(rows, base_weights=base), base, 0, True)

    consensus = _pool(rows, base)
    converged = False
    effective = base.copy()
    iterations = 0
    for iterations in range(1, max_iter + 1):
        distances = np.asarray([_hybrid_divergence(row, consensus) for row in rows])
        effective = base * np.exp(-robustness * distances)
        if float(effective.sum()) <= _EPS:
            effective = base.copy()
        effective /= effective.sum()
        updated = _pool(rows, effective)
        if (
            np.max(np.abs(updated.discrete - consensus.discrete)) < tol
            and np.max(np.abs(updated.gaussian_mean - consensus.gaussian_mean)) < tol
            and np.max(np.abs(updated.gaussian_var - consensus.gaussian_var)) < tol
        ):
            consensus = updated
            converged = True
            break
        consensus = updated
    final_distances = np.asarray(
        [_hybrid_divergence(row, consensus) for row in rows]
    )
    effective = base * np.exp(-robustness * final_distances)
    if float(effective.sum()) <= _EPS:
        effective = base.copy()
    effective /= effective.sum()
    return HybridAggregationResult(consensus, effective, iterations, converged)


__all__ = ["HybridAggregationResult", "HybridBelief", "hybrid_aggregate", "hybrid_log_linear_pool"]
