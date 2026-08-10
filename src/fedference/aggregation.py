"""Federated aggregation of categorical agent beliefs.

This module is where Friston et al. (2024) and FedGVI (Mildner et al., 2025)
meet. Each agent ``n`` broadcasts a categorical local posterior ``q_n`` over a *shared*
latent factor (e.g. a predator's location), optionally with a scalar
confidence ``w_n``. The server fuses them into a consensus.

Three fusion rules:

* :func:`log_linear_pool` — ``softmax(sum_n w_n log q_n)`` — the product-of-experts
  / log-linear pool. It is the project's categorical posterior-log-potential
  specialization of Friston et al. (2024) Eq. 7 when local posteriors share
  support, ``q_n = softmax(m_n)`` for admitted message potentials, and the
  project weight mapping is fixed. It is not a reconstruction of the source
  factor graph, cavity structure, or message schedule.

* :func:`robust_aggregate` — an independently motivated, iteratively-reweighted
  server heuristic that discounts each
  agent by ``exp(-c * KL(q_n || q))``. A confidently-wrong (contaminated)
  agent can sit far from the emerging consensus and earn a small weight in the
  declared diagnostic regimes. No FedGVI client-side guarantee transfers to
  this server rule.

* :func:`variational_aggregate` — a conservative, objective-backed server rule
  with exact block updates and explicit convergence diagnostics.

The defining identity (tested) is **project-local**:
``robust_aggregate(..., robustness=0.0)`` is **bit-identical** to
:func:`log_linear_pool`. The heuristic is therefore a recovery-anchored
extension of the project pool, not a replacement for or reconstruction of the
full Friston message-passing protocol and not a server theorem.
"""

from __future__ import annotations

import json
import logging
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Literal, Protocol

import numpy as np

from ._validation import as_nonnegative_weights, as_pmf_matrix
from .divergences import kl_divergence
from .generalized_bayes import softmax

ArrayF = np.ndarray
_EPS = 1e-12
_BASE_WEIGHT_FALLBACK = "all effective weights collapsed; base weights substituted"
TEMPERED_ENTROPY_WEIGHT_DEFAULT: float = 1.0
AggregationMethod = Literal["naive", "robust", "variational"]

_log = logging.getLogger(__name__)


def _resolve_compat_alias(
    value: Any,
    legacy: dict[str, Any],
    *,
    canonical: str,
    aliases: tuple[str, ...],
) -> Any:
    """Resolve additive keyword aliases without silently accepting typos."""
    for alias in aliases:
        if alias not in legacy:
            continue
        if value is not None:
            raise TypeError(f"{canonical} and deprecated {alias} cannot both be supplied")
        value = legacy.pop(alias)
        warnings.warn(
            f"{alias} is deprecated; use {canonical}",
            DeprecationWarning,
            stacklevel=3,
        )
    return value


def _reject_unexpected_keywords(legacy: dict[str, Any]) -> None:
    """Fail closed after all compatibility aliases for one call are resolved."""
    if legacy:
        names = ", ".join(sorted(legacy))
        raise TypeError(f"unexpected keyword argument(s): {names}")


def _stack(local_posteriors: Iterable[ArrayF]) -> ArrayF:
    return as_pmf_matrix(local_posteriors, name="local_posteriors")


def _weights(base_weights: Iterable[float] | None, n: int) -> ArrayF:
    return as_nonnegative_weights(base_weights, n, name="base_weights")


def _validate_solver_controls(max_iter: int, tol: float) -> None:
    if (
        isinstance(max_iter, (bool, np.bool_))
        or not isinstance(max_iter, (int, np.integer))
        or max_iter < 0
    ):
        raise ValueError("max_iter must be a non-negative integer")
    if (
        isinstance(tol, (bool, np.bool_))
        or not isinstance(tol, (int, float, np.integer, np.floating))
        or not np.isfinite(tol)
        or tol < 0.0
    ):
        raise ValueError("tol must be a finite non-negative value")


def _nonnegative_scalar(value: object, *, name: str) -> float:
    """Return one finite non-negative real while rejecting booleans/coercion."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        or value < 0.0
    ):
        raise ValueError(f"{name} must be non-negative")
    return float(value)


def _closed_simplex(values: ArrayF, *, n_states: int, name: str) -> ArrayF:
    """Validate a closed-simplex vector without flooring exact boundary zeros."""
    result = np.asarray(values, dtype=np.float64).ravel()
    if result.size != n_states:
        raise ValueError(f"{name} length must match the belief state dimension")
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")
    total = float(result.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"{name} must have a positive finite sum")
    return result / total


def log_linear_pool(
    local_posteriors: Iterable[ArrayF] | None = None,
    base_weights: Iterable[float] | None = None,
    **legacy: Any,
) -> ArrayF:
    """Return the project's product-of-experts consensus.

    Under the documented shared-support, posterior-log-potential, and fixed
    weight-mapping assumptions, this is a categorical specialization of Friston
    et al. (2024) Eq. 7. It does not reconstruct that paper's complete
    message-passing protocol.

    ``local_posteriors`` and ``base_weights`` are the canonical names. The old
    ``beliefs``/``weights`` keywords remain additive compatibility aliases.
    """
    local_posteriors = _resolve_compat_alias(
        local_posteriors,
        legacy,
        canonical="local_posteriors",
        aliases=("beliefs",),
    )
    base_weights = _resolve_compat_alias(
        base_weights,
        legacy,
        canonical="base_weights",
        aliases=("weights",),
    )
    _reject_unexpected_keywords(legacy)
    if local_posteriors is None:
        raise TypeError("local_posteriors is required")
    mat = _stack(local_posteriors)
    w = _weights(base_weights, mat.shape[0])
    return softmax(w @ np.log(mat))


# Historical convenience alias for the project pool; not a full-protocol claim.
friston_belief_share = log_linear_pool


@dataclass
class AggregationResult:
    """Consensus plus per-agent influence diagnostics.

    ``free_energy_history`` is populated only by :func:`variational_aggregate`
    (the finite-simplex objective-backed aggregator): it records the variational
    free energy after each nominal block-coordinate-descent iteration, so a
    caller can show the descent is monotone when ``fallback_events`` is empty.
    :func:`robust_aggregate` (the heuristic) leaves it empty — the scoped
    MAJ-1 no-go rejects the declared separable block-objective class for its
    raw log-pool update, but does not provide a broader objective certificate.
    ``fallback_events`` is empty on the nominal numerical path. A non-empty
    tuple records any stability fallback observed during the operation so that
    benchmark reports and evidence receipts do not silently promote a
    base-weight substitution to solver convergence.
    """

    consensus: ArrayF
    normalized_effective_weights: ArrayF
    iterations: int
    converged: bool
    raw_effective_weights: ArrayF | None = None
    history: list[ArrayF] = field(default_factory=list)
    free_energy_history: list[float] = field(default_factory=list)
    fallback_events: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Normalize the diagnostic arrays while retaining both weight scales."""
        normalized = np.asarray(self.normalized_effective_weights, dtype=np.float64).ravel()
        if normalized.size == 0 or not np.all(np.isfinite(normalized)):
            raise ValueError("normalized_effective_weights must be finite and non-empty")
        if np.any(normalized < 0.0) or not np.isclose(
            float(normalized.sum()), 1.0, atol=1e-12
        ):
            raise ValueError("normalized_effective_weights must be a probability vector")
        raw = (
            normalized.copy()
            if self.raw_effective_weights is None
            else np.asarray(self.raw_effective_weights, dtype=np.float64).ravel()
        )
        if raw.shape != normalized.shape or not np.all(np.isfinite(raw)) or np.any(raw < 0.0):
            raise ValueError("raw_effective_weights must match normalized_effective_weights")
        normalized = normalized.copy()
        raw = raw.copy()
        normalized.setflags(write=False)
        raw.setflags(write=False)
        object.__setattr__(self, "normalized_effective_weights", normalized)
        object.__setattr__(self, "raw_effective_weights", raw)

    @property
    def agent_weights(self) -> ArrayF:
        """Deprecated alias for normalized effective influence weights."""
        warnings.warn(
            "AggregationResult.agent_weights is deprecated; use "
            "normalized_effective_weights",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.normalized_effective_weights

    @agent_weights.setter
    def agent_weights(self, value: ArrayF) -> None:
        """Set the deprecated alias for compatibility with older adapters."""
        warnings.warn(
            "AggregationResult.agent_weights is deprecated; use "
            "normalized_effective_weights",
            DeprecationWarning,
            stacklevel=2,
        )
        self.normalized_effective_weights = value


@dataclass(frozen=True)
class AggregationConfig:
    """Validated, serializable configuration shared by every aggregation adapter.

    The defaults preserve :func:`aggregate` compatibility: ``method`` is the
    project log-linear pool, while selecting either robust method without
    overriding ``robustness`` uses the historical strength of ``1.0``.
    Individual low-level solvers continue to accept ``max_iter=0`` for
    diagnostic tests; the public configuration requires a positive iteration
    budget because a configured production run must be able to execute.
    """

    method: AggregationMethod = "naive"
    robustness: float = 1.0
    entropy_weight: float = TEMPERED_ENTROPY_WEIGHT_DEFAULT
    max_iter: int = 64
    tol: float = 1e-9
    multistart: bool = True

    def __post_init__(self) -> None:
        if self.method not in ("naive", "robust", "variational"):
            raise ValueError(f"unknown method {self.method!r}; choose 'naive', 'robust' or 'variational'")
        for name in ("robustness", "entropy_weight", "tol"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"{name} must be a finite non-negative value")
            object.__setattr__(self, name, float(value))
        if (
            isinstance(self.max_iter, bool)
            or not isinstance(self.max_iter, (int, np.integer))
            or self.max_iter <= 0
        ):
            raise ValueError("max_iter must be a positive integer")
        object.__setattr__(self, "max_iter", int(self.max_iter))
        if not isinstance(self.multistart, bool):
            raise ValueError("multistart must be a boolean")

    def as_dict(self) -> dict[str, bool | float | int | str]:
        """Return the canonical JSON-compatible representation."""
        return {
            "method": self.method,
            "robustness": float(self.robustness),
            "entropy_weight": float(self.entropy_weight),
            "max_iter": int(self.max_iter),
            "tol": float(self.tol),
            "multistart": self.multistart,
        }

    @property
    def fingerprint(self) -> str:
        """SHA-256 binding used by transport and evidence receipts."""
        encoded = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return sha256(encoded).hexdigest()


class AggregatorProtocol(Protocol):
    """Callable server-side aggregation boundary used by federation adapters."""

    def __call__(
        self,
        local_posteriors: Iterable[ArrayF],
        *,
        config: AggregationConfig,
        base_weights: Iterable[float] | None = None,
    ) -> AggregationResult:
        """Fuse local posteriors according to ``config`` and return diagnostics."""


def _prefer_variational_candidate(
    candidate: AggregationResult,
    incumbent: AggregationResult | None,
) -> bool:
    """Rank multi-start results by convergence first, then final objective.

    A lower intermediate objective does not make an unfinished iterate a safer
    production result than a converged fixed point. If no start converges, the
    lowest observed objective is still the most informative bounded-budget
    fallback and retains ``converged=False`` for the caller.
    """
    if incumbent is None:
        return True
    if candidate.converged != incumbent.converged:
        return candidate.converged
    if not candidate.free_energy_history:
        return False
    if not incumbent.free_energy_history:
        return True
    return candidate.free_energy_history[-1] < incumbent.free_energy_history[-1]


def robust_aggregate(
    local_posteriors: Iterable[ArrayF] | None = None,
    *,
    base_weights: Iterable[float] | None = None,
    robustness: float = 0.0,
    max_iter: int = 32,
    tol: float = 1e-9,
    **legacy: Any,
) -> AggregationResult:
    """Robustly fuse agent beliefs by iterative divergence-reweighting.

    ``robustness`` (``c >= 0``) controls down-weighting: the effective weight of
    agent ``n`` is ``w_n * exp(-c * KL(q_n || q))``. With ``c = 0`` every
    multiplier is 1 and the result equals :func:`log_linear_pool` exactly (the
    project-local recovery identity). Under the documented bridge assumptions,
    that pool is a categorical Eq. 7 specialization rather than a complete
    Friston protocol reconstruction. With ``c > 0`` outliers can be suppressed
    in declared regimes, but the rule has no general bounded-influence guarantee.

    Returns an :class:`AggregationResult` carrying the final consensus, the
    converged per-agent effective weights (normalized, for influence plots), the
    iteration count and a convergence flag.
    """
    local_posteriors = _resolve_compat_alias(
        local_posteriors,
        legacy,
        canonical="local_posteriors",
        aliases=("beliefs",),
    )
    base_weights = _resolve_compat_alias(
        base_weights,
        legacy,
        canonical="base_weights",
        aliases=("weights",),
    )
    _reject_unexpected_keywords(legacy)
    if local_posteriors is None:
        raise TypeError("local_posteriors is required")
    robustness = _nonnegative_scalar(robustness, name="robustness")
    _validate_solver_controls(max_iter, tol)
    mat = _stack(local_posteriors)
    n = mat.shape[0]
    base = _weights(base_weights, n)
    log_mat = np.log(mat)

    consensus = softmax(base @ log_mat)
    if robustness == 0.0:
        eff = base.copy()
        return AggregationResult(
            consensus=consensus,
            raw_effective_weights=eff,
            normalized_effective_weights=eff / eff.sum(),
            iterations=0,
            converged=True,
            history=[consensus],
        )

    history = [consensus]
    converged = False
    fallback_events: list[str] = []
    eff = base.copy()
    it = 0
    for it in range(1, max_iter + 1):
        divs = np.array([kl_divergence(mat[i], consensus) for i in range(n)])
        eff = base * np.exp(-robustness * divs)
        used_fallback = bool(eff.sum() <= _EPS)
        if used_fallback:
            _log.warning(
                "robust_aggregate: all effective weights collapsed to zero at iteration %d "
                "(every agent vetoed); falling back to base weights. "
                "This may indicate extreme contamination or a pathological colony.",
                it,
            )
            eff = base.copy()
            fallback_events.append(_BASE_WEIGHT_FALLBACK)
        new_consensus = softmax(eff @ log_mat)
        history.append(new_consensus)
        if used_fallback:
            consensus = new_consensus
            break
        if np.max(np.abs(new_consensus - consensus)) < tol:
            consensus = new_consensus
            converged = True
            break
        consensus = new_consensus
    divs = np.array([kl_divergence(mat[i], consensus) for i in range(n)])
    eff = base * np.exp(-robustness * divs)
    if eff.sum() <= _EPS:
        eff = base.copy()
        converged = False
        if not fallback_events:
            fallback_events.append(_BASE_WEIGHT_FALLBACK)
    return AggregationResult(
        consensus=consensus,
        raw_effective_weights=eff,
        normalized_effective_weights=eff / eff.sum(),
        iterations=it,
        converged=converged,
        history=history,
        fallback_events=tuple(fallback_events),
    )


def _cross_entropies(consensus: ArrayF, log_mat: ArrayF) -> ArrayF:
    """Per-agent cross-entropy ``CE_n(q) = -sum_i q_i log s_{n,i}`` (length ``n``).

    ``log_mat`` is the ``(n, n_s)`` matrix of agent log-pmfs; ``consensus`` is the
    length-``n_s`` consensus pmf. Returns the vector of cross-entropies of the
    consensus relative to each agent — the data term of the aggregation free
    energy and the exponent of the rigorous weight update.
    """
    return -(log_mat @ np.asarray(consensus, dtype=np.float64).ravel())


def _entropy_regularized_pool(logits: ArrayF, entropy_weight: float) -> ArrayF:
    """Solve the categorical ``q`` block, including the zero-entropy boundary.

    For positive ``entropy_weight`` this is the usual tempered softmax. At zero
    the objective is linear on the simplex; the deterministic solution assigns
    equal mass to every tied maximizing state.
    """
    values = np.asarray(logits, dtype=np.float64).ravel()
    if entropy_weight > 0.0:
        return softmax(values / entropy_weight)
    maximizing = values == np.max(values)
    result = maximizing.astype(np.float64)
    return result / result.sum()


def aggregation_free_energy(
    consensus_posterior: ArrayF | None = None,
    raw_effective_weights: ArrayF | None = None,
    local_posteriors: Iterable[ArrayF] | None = None,
    *,
    base_weights: Iterable[float] | None = None,
    robustness: float,
    entropy_weight: float = 1.0,
    **legacy: Any,
) -> float:
    r"""Variational free energy minimized by :func:`variational_aggregate`.

    .. math::

        F_\lambda(q, a) = \sum_n a_n\,\mathrm{CE}(q, q_n) - \lambda H(q)
                  + \tfrac{1}{c}\,\mathrm{KL_{gen}}(a \,\|\, w)

    where :math:`\mathrm{CE}(q, q_n) = -\sum_i q_i \log q_{n,i}` is the
    cross-entropy of the consensus ``q`` relative to agent ``n``'s belief,
    :math:`H(q)` is the consensus entropy, ``c = robustness``,
    :math:`\lambda = entropy_weight`, and
    :math:`\mathrm{KL_{gen}}(a\|w) = \sum_n [a_n \log(a_n/w_n) - a_n + w_n]` is
    the generalized KL (I-divergence) between the effective weights ``a`` and the
    base weights ``w``. Defined for ``robustness > 0`` (the ``c -> 0`` limit pins
    ``a = w`` exactly via the diverging penalty and is handled separately by
    :func:`variational_aggregate`).

    This is a genuine, stated objective: block-coordinate minimization of ``F``
    over ``q`` (fixed ``a``) and over ``a`` (fixed ``q``) is exactly the iteration
    in :func:`variational_aggregate`. Every completed block update is therefore
    non-increasing in ``F``; a converged fixed point is coordinatewise stationary.
    """
    consensus_posterior = _resolve_compat_alias(
        consensus_posterior,
        legacy,
        canonical="consensus_posterior",
        aliases=("consensus",),
    )
    raw_effective_weights = _resolve_compat_alias(
        raw_effective_weights,
        legacy,
        canonical="raw_effective_weights",
        aliases=("agent_weights",),
    )
    local_posteriors = _resolve_compat_alias(
        local_posteriors,
        legacy,
        canonical="local_posteriors",
        aliases=("beliefs",),
    )
    _reject_unexpected_keywords(legacy)
    if consensus_posterior is None or raw_effective_weights is None or local_posteriors is None:
        raise TypeError("consensus_posterior, raw_effective_weights, and local_posteriors are required")
    robustness = _nonnegative_scalar(robustness, name="robustness")
    if robustness <= 0.0:
        raise ValueError("aggregation_free_energy is defined for robustness > 0")
    entropy_weight = _nonnegative_scalar(
        entropy_weight,
        name="entropy_weight",
    )
    mat = _stack(local_posteriors)
    w = _weights(base_weights, mat.shape[0])
    a = np.asarray(raw_effective_weights, dtype=np.float64).ravel()
    if a.shape[0] != mat.shape[0]:
        raise ValueError("raw_effective_weights length must match number of agents")
    if not np.all(np.isfinite(a)):
        raise ValueError("raw_effective_weights must contain only finite values")
    if np.any(a < 0):
        raise ValueError("raw_effective_weights must be non-negative")
    if np.any((w == 0.0) & (a > _EPS)):
        raise ValueError("raw_effective_weights must be zero where base_weights are zero")
    q = _closed_simplex(
        consensus_posterior,
        n_states=mat.shape[1],
        name="consensus",
    )
    log_mat = np.log(mat)

    ce = _cross_entropies(q, log_mat)
    positive_q = q > 0.0
    entropy = float(-np.sum(q[positive_q] * np.log(q[positive_q])))
    active = w > 0.0
    a_active = a[active]
    w_active = w[active]
    positive_a = a_active > 0.0
    kl_terms = w_active - a_active
    kl_terms[positive_a] += a_active[positive_a] * np.log(a_active[positive_a] / w_active[positive_a])
    kl_gen = float(np.sum(kl_terms))
    return float(np.sum(a * ce) - entropy_weight * entropy + kl_gen / robustness)


def variational_aggregate(
    local_posteriors: Iterable[ArrayF] | None = None,
    *,
    base_weights: Iterable[float] | None = None,
    robustness: float = 0.0,
    max_iter: int = 64,
    tol: float = 1e-9,
    multistart: bool = True,
    entropy_weight: float = 1.0,
    **legacy: Any,
) -> AggregationResult:
    r"""Objective-backed conservative fusion on the stated finite-simplex free energy.

    Unlike :func:`robust_aggregate` (a heuristic whose declared separable
    block-objective class is ruled out by a scoped MAJ-1 proposition, without a
    broader objective certificate), this aggregator performs block-coordinate
    descent on the variational free energy :func:`aggregation_free_energy`.
    Alternating the two exact block updates

    .. math::

        q   &\leftarrow \mathrm{softmax}\!\big(\textstyle\sum_n a_n \log q_n / \lambda\big) \\
        a_n &\leftarrow w_n \exp\!\big(-c\,\mathrm{CE}(q, q_n)\big)

    (``c = robustness``, ``CE`` the forward cross-entropy) makes ``F``
    non-increasing. A converged fixed point is coordinatewise stationary. Three
    properties hold and are tested:

    * **Recovery (the limit is the proof).** ``robustness = 0`` with the
      default ``entropy_weight=1`` returns the project
      :func:`log_linear_pool` *exactly* because every weight collapses to
      ``w_n``. Under the documented bridge assumptions, that pool is a
      categorical Eq. 7 specialization; it is not a complete source-protocol
      reconstruction. (Identical ``c -> 0`` limit to
      :func:`robust_aggregate`, so both robust families share this project
      recovery corner.)
    * **Bounded effective-weight update (honest-majority experiments).**
      ``a_n = w_n exp(-c CE(q, q_n)) <= w_n`` is unconditional, and ``a_n -> 0`` as
      agent ``n`` diverges from the *realized* consensus. This is a redescending
      raw-weight mechanism, not by itself a proof that the normalized consensus
      estimator is B-robust. Because ``F`` is biconvex, a near-one-hot adversary can capture the
      product-of-experts seed; on the tested honest-majority paths, multi-start
      descent finds a lower objective basin that vetoes the outlier for
      contamination bounded away from the simplex vertex (rate < 1). This is an
      empirical observation, not a global optimization guarantee. The residual
      precondition is fundamental to all robust fusion: a *dominant* agent with
      no honest majority to oppose it (e.g. a lone
      hyper-confident voter) legitimately carries the consensus.
    * **Conservatism (honest caveat).** The ``-lambda H(q)`` term makes this the
      maximum-entropy / conservative consensus: it trades peak point-accuracy for
      a principled objective and explicit weight control. It is *not* a drop-in
      accuracy-maximizer for the robustness sweep — that remains the sharp
      :func:`robust_aggregate` heuristic. The two are complementary, never
      conflated.

    ``multistart`` (default ``True``) descends from the pool, uniform, and
    arithmetic-mean seeds and keeps the lowest-observed-``F`` candidate among
    the converged starts. If no start converges within the declared budget, it
    returns the lowest-observed unfinished iterate with ``converged=False``.
    This reduces seed-capture risk without guaranteeing the global minimum;
    ``multistart=False`` uses
    only the log-linear-pool seed (the single-basin "naive -> robust" trajectory
    the descent figure plots, which can be captured at the simplex vertex).

    Returns an :class:`AggregationResult` whose ``free_energy_history`` records
    ``F`` after each nominal iteration (monotone non-increasing) for the descent
    figure. If numerical underflow collapses every effective weight, the
    implementation substitutes base weights only as a fail-visible stability
    path, records ``fallback_events``, and does not mark that substituted
    trajectory as converged; the exact block-descent certificate does not apply
    to that fallback step.
    """
    local_posteriors = _resolve_compat_alias(
        local_posteriors,
        legacy,
        canonical="local_posteriors",
        aliases=("beliefs",),
    )
    base_weights = _resolve_compat_alias(
        base_weights,
        legacy,
        canonical="base_weights",
        aliases=("weights",),
    )
    _reject_unexpected_keywords(legacy)
    if local_posteriors is None:
        raise TypeError("local_posteriors is required")
    robustness = _nonnegative_scalar(robustness, name="robustness")
    entropy_weight = _nonnegative_scalar(
        entropy_weight,
        name="entropy_weight",
    )
    _validate_solver_controls(max_iter, tol)
    mat = _stack(local_posteriors)
    n = mat.shape[0]
    base = _weights(base_weights, n)
    log_mat = np.log(mat)

    consensus = _entropy_regularized_pool(base @ log_mat, entropy_weight)
    if robustness == 0.0:
        # c -> 0: the diverging KL_gen penalty pins a = w, so the consensus is the
        # tempered log-linear pool softmax((sum_n w_n log q_n)/lambda). At lambda = 1
        # this is the exact project log-linear pool. Its Eq. 7 correspondence
        # is the documented categorical specialization, not a protocol reconstruction.
        # Free energy is undefined at c = 0.
        return AggregationResult(
            consensus=consensus,
            raw_effective_weights=base,
            normalized_effective_weights=base / base.sum(),
            iterations=0,
            converged=True,
            history=[consensus],
            free_energy_history=[],
        )

    # Multi-start block-coordinate descent. F is biconvex, not jointly convex, so
    # the natural log-linear-pool seed can land in a consensus-CAPTURE basin when a
    # near-one-hot adversary already dominates the product-of-experts (the seed is
    # itself captured). We descend from several seeds — the pool, the uniform
    # belief, and the arithmetic mean — and return the lowest-observed-F converged
    # candidate. On the tested paths this reaches an outlier-vetoing basin even
    # for near-vertex adversaries (within the rate < 1 regime). Candidates with
    # the same convergence status are ordered by objective; a converged start is
    # preferred to a lower-objective unfinished trace. The c = 0 recovery limit
    # above is unaffected.
    n_s = mat.shape[1]
    seeds: tuple[ArrayF, ...]
    if multistart:
        seeds = (consensus, np.full(n_s, 1.0 / n_s, dtype=np.float64), mat.mean(axis=0))
    else:
        # Single (log-linear-pool) start: the canonical "naive belief sharing
        # descends to the robust consensus" trajectory, used by the descent figure.
        # May land in a capture basin for a near-one-hot adversary — not for
        # production fusion, where the default multi-start escapes it.
        seeds = (consensus,)
    best: AggregationResult | None = None
    operation_used_fallback = False
    for q_init in seeds:
        q = np.clip(np.asarray(q_init, dtype=np.float64), _EPS, None)
        q = q / q.sum()
        history = [q]
        fe_history: list[float] = []
        converged = False
        start_used_fallback = False
        eff = base.copy()
        it = 0
        for it in range(1, max_iter + 1):
            # a-update (exact minimizer of F over a, fixed q): forward cross-entropy.
            ce = _cross_entropies(q, log_mat)
            eff = base * np.exp(-robustness * ce)
            if eff.sum() <= _EPS:  # numerical floor: every agent vetoed -> fall back
                _log.warning(
                    "variational_aggregate: all effective weights collapsed to zero at "
                    "iteration %d; falling back to base weights.",
                    it,
                )
                eff = base.copy()
                start_used_fallback = True
                operation_used_fallback = True
            # q-update (exact minimizer of F over q, fixed a): weighted product of experts.
            new_q = _entropy_regularized_pool(eff @ log_mat, entropy_weight)
            fe_history.append(
                aggregation_free_energy(
                    new_q,
                    eff,
                    mat,
                    base_weights=base,
                    robustness=robustness,
                    entropy_weight=entropy_weight,
                )
            )
            history.append(new_q)
            if start_used_fallback:
                q = new_q
                break
            if np.max(np.abs(new_q - q)) < tol:
                q = new_q
                converged = True
                break
            q = new_q
        final_eff = base * np.exp(-robustness * _cross_entropies(q, log_mat))
        if final_eff.sum() <= _EPS:
            final_eff = base.copy()
            converged = False
            start_used_fallback = True
            operation_used_fallback = True
        if fe_history:
            fe_history[-1] = aggregation_free_energy(
                q,
                final_eff,
                mat,
                base_weights=base,
                robustness=robustness,
                entropy_weight=entropy_weight,
            )
        candidate = AggregationResult(
            consensus=q,
            raw_effective_weights=final_eff,
            normalized_effective_weights=final_eff / final_eff.sum(),
            iterations=it,
            converged=converged,
            history=history,
            free_energy_history=fe_history,
            fallback_events=((_BASE_WEIGHT_FALLBACK,) if start_used_fallback else ()),
        )
        if _prefer_variational_candidate(candidate, best):
            best = candidate
    assert best is not None  # seeds is non-empty
    if operation_used_fallback and not best.fallback_events:
        best.fallback_events = (_BASE_WEIGHT_FALLBACK,)
    return best


def aggregate_result(
    local_posteriors: Iterable[ArrayF] | None = None,
    config: AggregationConfig | None = None,
    base_weights: Iterable[float] | None = None,
    **legacy: Any,
) -> AggregationResult:
    """Canonical aggregation dispatcher returning consensus and diagnostics.

    Args:
        local_posteriors: Agent probability vectors over a common finite state space.
        config: Validated method and solver configuration. Defaults to the
            naive project log-linear pool.
        base_weights: Optional non-negative per-agent base weights.

    Returns:
        An :class:`AggregationResult` for every method. The naive path records
        normalized base weights and zero iterations, providing a uniform
        diagnostics contract without changing :func:`aggregate`'s compatibility
        array-only return type.
    """
    local_posteriors = _resolve_compat_alias(
        local_posteriors,
        legacy,
        canonical="local_posteriors",
        aliases=("beliefs",),
    )
    base_weights = _resolve_compat_alias(
        base_weights,
        legacy,
        canonical="base_weights",
        aliases=("weights",),
    )
    _reject_unexpected_keywords(legacy)
    if local_posteriors is None:
        raise TypeError("local_posteriors is required")
    if config is not None and not isinstance(config, AggregationConfig):
        raise ValueError("config must be an AggregationConfig or None")
    resolved = config or AggregationConfig()
    raw_local_posteriors = list(local_posteriors)
    base = _weights(base_weights, len(raw_local_posteriors))
    if resolved.method == "naive":
        consensus = log_linear_pool(raw_local_posteriors, base_weights=base)
        return AggregationResult(
            consensus=consensus,
            raw_effective_weights=base,
            normalized_effective_weights=base / base.sum(),
            iterations=0,
            converged=True,
            history=[consensus],
            free_energy_history=[],
        )
    if resolved.method == "robust":
        return robust_aggregate(
            raw_local_posteriors,
            base_weights=base,
            robustness=resolved.robustness,
            max_iter=resolved.max_iter,
            tol=resolved.tol,
        )
    return variational_aggregate(
        raw_local_posteriors,
            base_weights=base,
        robustness=resolved.robustness,
        max_iter=resolved.max_iter,
        tol=resolved.tol,
        multistart=resolved.multistart,
        entropy_weight=resolved.entropy_weight,
    )


def aggregate(
    local_posteriors: Iterable[ArrayF] | None = None,
    *,
    method: str | None = None,
    config: AggregationConfig | None = None,
    **kwargs: Any,
) -> ArrayF:
    """Convenience dispatch returning just the consensus pmf.

    ``method='naive'`` -> :func:`log_linear_pool`; ``method='robust'`` ->
    :func:`robust_aggregate` (the empirically sharp heuristic;
    ``robustness`` defaults to 1.0); ``method='variational'`` ->
    :func:`variational_aggregate` (objective-backed for its stated
    finite-simplex objective; ``robustness`` defaults to 1.0).

    ``config`` is the additive public configuration path. It is mutually
    exclusive with the compatibility ``method`` and solver keyword path; ``weights``
    remains per-call data and may be supplied with either path.
    """
    if local_posteriors is None:
        local_posteriors = kwargs.pop("beliefs", None)
        if local_posteriors is not None:
            warnings.warn(
                "beliefs is deprecated; use local_posteriors",
                DeprecationWarning,
                stacklevel=2,
            )
    if local_posteriors is None:
        raise TypeError("local_posteriors is required")
    base_weights = kwargs.pop("base_weights", None)
    if "weights" in kwargs:
        if base_weights is not None:
            raise TypeError("base_weights and deprecated weights cannot both be supplied")
        base_weights = kwargs.pop("weights")
        warnings.warn(
            "weights is deprecated; use base_weights",
            DeprecationWarning,
            stacklevel=2,
        )
    if config is not None:
        if method is not None:
            raise ValueError("config and compatibility method are mutually exclusive")
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise ValueError(f"config and compatibility solver arguments are mutually exclusive: {names}")
        return aggregate_result(
            local_posteriors,
            config=config,
            base_weights=base_weights,
        ).consensus

    resolved_method = "naive" if method is None else method
    if resolved_method == "naive":
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise ValueError(f"unknown naive aggregation arguments: {names}")
        return log_linear_pool(local_posteriors, base_weights=base_weights)
    if resolved_method == "robust":
        kwargs.setdefault("robustness", 1.0)
        return robust_aggregate(
            local_posteriors,
            base_weights=base_weights,
            **kwargs,
        ).consensus
    if resolved_method == "variational":
        kwargs.setdefault("robustness", 1.0)
        kwargs.setdefault("entropy_weight", 1.0)
        return variational_aggregate(
            local_posteriors,
            base_weights=base_weights,
            **kwargs,
        ).consensus
    raise ValueError(f"unknown method {resolved_method!r}; choose 'naive', 'robust' or 'variational'")
