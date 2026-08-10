"""Executable witnesses for the scoped MAJ-1 server-objective result.

The sharp :func:`fedference.aggregation.robust_aggregate` iteration uses raw
weights ``a_n = w_n exp(-c KL(q_n || q))`` and the unnormalised log pool
``softmax(sum_n a_n log q_n)``.  This module records two distinct facts:

* the earlier orientation witness rejects one direct forward-KL / generalized-
  KL block pairing; and
* an exact interior-simplex witness proves that no ``C1`` objective in the
  declared separable class ``sum_n a_n KL(q || q_n) + R(a, w) + G(q)`` can
  have that raw log-pool as its ``q``-coordinate minimizer for every interior
  input.

The result is deliberately scoped.  It does not exclude nonseparable
``q``--``a`` couplings, source-dependent terms, non-differentiable objectives,
or constructions that encode selected fixed points without reproducing the
heuristic's block maps.  It therefore preserves the heuristic label and does
not transfer any FedGVI client-side guarantee.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from ._validation import as_nonnegative_weights, as_pmf, as_pmf_matrix
from .divergences import kl_divergence
from .generalized_bayes import softmax

ArrayF = np.ndarray


@dataclass(frozen=True)
class ObjectiveOrientationWitness:
    """Forward-objective and reverse-heuristic weight blocks at one consensus."""

    consensus: ArrayF
    objective_weights: ArrayF
    heuristic_weights: ArrayF
    max_absolute_gap: float
    result_scope: str = (
        "orientation mismatch for the declared generalized-KL-regularized "
        "block objective; not a universal no-go theorem"
    )


@dataclass(frozen=True)
class RawLogPoolNoGoWitness:
    """Exact witness for the declared raw-log-pool ``q``-block no-go.

    For a non-uniform interior consensus ``q``, each stored source satisfies
    ``softmax(alpha * log(source_alpha)) == q``.  A shared differentiable
    ``G(q)`` would then need two incompatible tangential gradients at ``q``.
    ``tangential_contradiction_norm`` is the non-zero norm of that difference.
    """

    consensus: ArrayF
    scales: tuple[float, float]
    sources: tuple[ArrayF, ArrayF]
    max_q_block_error: float
    tangential_contradiction_norm: float
    result_scope: str = (
        "no C1 separable block-coordinate objective of the declared form has "
        "the raw log-pool q block for every interior input; not a universal "
        "no-objective theorem"
    )


@dataclass(frozen=True)
class NormalizedWeightNoGoWitness:
    """Companion witness for a normalized-weight reparameterization.

    The implementation uses raw effective weights.  This witness separately
    shows that merely constraining the natural forward-KL class to normalized
    weights cannot reproduce the reverse-KL weight block at both declared
    interior consensuses.
    """

    local_posteriors: ArrayF
    consensus_a: ArrayF
    consensus_b: ArrayF
    robustness: float
    normalized_weights_a: ArrayF
    normalized_weights_b: ArrayF
    reverse_kl_difference_a: float
    reverse_kl_difference_b: float
    forward_kl_difference_a: float
    forward_kl_difference_b: float
    normalized_weight_max_absolute_gap: float
    forward_difference_gap: float
    result_scope: str = (
        "normalized-weight companion for the declared forward-KL data term; "
        "not the raw-weight implementation and not a universal no-go theorem"
    )


def objective_weight_block(
    consensus: ArrayF,
    local_posteriors: Iterable[ArrayF] | None = None,
    *,
    base_weights: Iterable[float] | None = None,
    robustness: float,
    **legacy: object,
) -> ArrayF:
    """Exact ``a`` update for the declared forward-KL block objective."""
    if not np.isfinite(robustness) or robustness <= 0.0:
        raise ValueError("robustness must be finite and positive")
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
    if legacy:
        raise TypeError(f"unexpected keyword argument(s): {', '.join(sorted(legacy))}")
    if local_posteriors is None:
        raise TypeError("local_posteriors is required")
    q = as_pmf(consensus, name="global_posterior")
    matrix = as_pmf_matrix(local_posteriors)
    if matrix.shape[1] != q.size:
        raise ValueError("belief state dimension must match consensus")
    base = as_nonnegative_weights(base_weights, matrix.shape[0])
    divergences = np.asarray(
        [kl_divergence(q, belief) for belief in matrix],
        dtype=np.float64,
    )
    return base * np.exp(-robustness * divergences)


def heuristic_weight_block(
    consensus: ArrayF,
    local_posteriors: Iterable[ArrayF] | None = None,
    *,
    base_weights: Iterable[float] | None = None,
    robustness: float,
    **legacy: object,
) -> ArrayF:
    """Reverse-KL weight update used by ``robust_aggregate``."""
    if not np.isfinite(robustness) or robustness <= 0.0:
        raise ValueError("robustness must be finite and positive")
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
    if legacy:
        raise TypeError(f"unexpected keyword argument(s): {', '.join(sorted(legacy))}")
    if local_posteriors is None:
        raise TypeError("local_posteriors is required")
    q = as_pmf(consensus, name="global_posterior")
    matrix = as_pmf_matrix(local_posteriors)
    if matrix.shape[1] != q.size:
        raise ValueError("belief state dimension must match consensus")
    base = as_nonnegative_weights(base_weights, matrix.shape[0])
    divergences = np.asarray(
        [kl_divergence(belief, q) for belief in matrix],
        dtype=np.float64,
    )
    return base * np.exp(-robustness * divergences)


def construct_orientation_witness(
    consensus: ArrayF,
    local_posteriors: Iterable[ArrayF] | None = None,
    *,
    base_weights: Iterable[float] | None = None,
    robustness: float = 1.0,
    **legacy: object,
) -> ObjectiveOrientationWitness:
    """Construct the finite-simplex witness used by theory tests and reports."""
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
    if legacy:
        raise TypeError(f"unexpected keyword argument(s): {', '.join(sorted(legacy))}")
    if local_posteriors is None:
        raise TypeError("local_posteriors is required")
    posterior_tuple = tuple(local_posteriors)
    objective = objective_weight_block(
        consensus,
        posterior_tuple,
        base_weights=base_weights,
        robustness=robustness,
    )
    heuristic = heuristic_weight_block(
        consensus,
        posterior_tuple,
        base_weights=base_weights,
        robustness=robustness,
    )
    return ObjectiveOrientationWitness(
        consensus=as_pmf(consensus, name="global_posterior"),
        objective_weights=objective,
        heuristic_weights=heuristic,
        max_absolute_gap=float(np.max(np.abs(objective - heuristic))),
    )


def _source_for_raw_log_pool(consensus: ArrayF, scale: float) -> ArrayF:
    """Return an interior source whose raw log-pool at ``scale`` is ``consensus``."""
    source = consensus ** (1.0 / scale)
    return source / source.sum()


def _raw_log_pool(scale: float, source: ArrayF) -> ArrayF:
    """Independent one-agent form of the heuristic's raw weighted log pool."""
    return softmax(scale * np.log(source))


def construct_raw_log_pool_no_go_witness(
    consensus: ArrayF | None = None,
) -> RawLogPoolNoGoWitness:
    """Construct the exact witness for the scoped separable-class no-go.

    Let ``F(q, a; s, w) = sum_n a_n KL(q || s_n) + R(a, w) + G(q)`` with
    ``G`` continuously differentiable and independent of ``a`` and ``s``.
    If the raw log pool were its ``q`` block for every interior ``a, s``, a
    one-agent construction at two positive scales would require incompatible
    tangential values of ``grad G`` at a non-uniform interior ``q``.
    """
    default_consensus = np.asarray([0.5, 0.3, 0.2], dtype=np.float64)
    q = as_pmf(
        default_consensus if consensus is None else consensus,
        name="consensus",
    )
    if q.size < 2:
        raise ValueError("consensus must have at least two states")
    centered_log_q = np.log(q) - float(np.mean(np.log(q)))
    if float(np.linalg.norm(centered_log_q)) <= 1e-12:
        raise ValueError("consensus must be non-uniform for the no-go witness")

    scales = (1.0, 2.0)
    sources = tuple(_source_for_raw_log_pool(q, scale) for scale in scales)
    q_blocks = tuple(_raw_log_pool(scale, source) for scale, source in zip(scales, sources))
    max_error = max(float(np.max(np.abs(q_block - q))) for q_block in q_blocks)
    contradiction_norm = float(abs(scales[1] - scales[0]) * np.linalg.norm(centered_log_q))
    return RawLogPoolNoGoWitness(
        consensus=q,
        scales=scales,
        sources=(sources[0], sources[1]),
        max_q_block_error=max_error,
        tangential_contradiction_norm=contradiction_norm,
    )


def construct_normalized_weight_no_go_witness(
    *,
    robustness: float = 1.0,
) -> NormalizedWeightNoGoWitness:
    """Construct the normalized-weight companion to the raw-block no-go.

    Equal base weights and the two fixed interior consensuses give identical
    normalized reverse-KL weights, while their forward-KL data-term
    differences disagree.  A ``C1`` regularizer ``R(a, w)`` independent of
    ``q`` and the local posteriors cannot satisfy both simplex-stationarity
    equations at that same normalized weight vector.
    """
    if not np.isfinite(robustness) or robustness <= 0.0:
        raise ValueError("robustness must be finite and positive")
    local_posteriors = np.asarray(
        [[3.0, 1.0, 1.0], [1.0, 3.0, 1.0]],
        dtype=np.float64,
    ) / 5.0
    consensus_a = np.asarray([2.0, 1.0, 2.0], dtype=np.float64) / 5.0
    consensus_b = np.asarray([2.0, 1.0, 7.0], dtype=np.float64) / 10.0
    base_weights = np.asarray([0.5, 0.5], dtype=np.float64)

    raw_a = heuristic_weight_block(
        consensus_a,
        local_posteriors,
        base_weights=base_weights,
        robustness=robustness,
    )
    raw_b = heuristic_weight_block(
        consensus_b,
        local_posteriors,
        base_weights=base_weights,
        robustness=robustness,
    )
    normalized_a = raw_a / raw_a.sum()
    normalized_b = raw_b / raw_b.sum()
    reverse_a = kl_divergence(local_posteriors[0], consensus_a) - kl_divergence(
        local_posteriors[1], consensus_a
    )
    reverse_b = kl_divergence(local_posteriors[0], consensus_b) - kl_divergence(
        local_posteriors[1], consensus_b
    )
    forward_a = kl_divergence(consensus_a, local_posteriors[0]) - kl_divergence(
        consensus_a, local_posteriors[1]
    )
    forward_b = kl_divergence(consensus_b, local_posteriors[0]) - kl_divergence(
        consensus_b, local_posteriors[1]
    )
    return NormalizedWeightNoGoWitness(
        local_posteriors=local_posteriors,
        consensus_a=consensus_a,
        consensus_b=consensus_b,
        robustness=float(robustness),
        normalized_weights_a=normalized_a,
        normalized_weights_b=normalized_b,
        reverse_kl_difference_a=reverse_a,
        reverse_kl_difference_b=reverse_b,
        forward_kl_difference_a=forward_a,
        forward_kl_difference_b=forward_b,
        normalized_weight_max_absolute_gap=float(np.max(np.abs(normalized_a - normalized_b))),
        forward_difference_gap=float(abs(forward_a - forward_b)),
    )


__all__ = [
    "NormalizedWeightNoGoWitness",
    "ObjectiveOrientationWitness",
    "RawLogPoolNoGoWitness",
    "construct_normalized_weight_no_go_witness",
    "construct_orientation_witness",
    "construct_raw_log_pool_no_go_witness",
    "heuristic_weight_block",
    "objective_weight_block",
]
