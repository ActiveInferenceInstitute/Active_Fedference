"""Federated belief-sharing across an ensemble of active-inference agents.

This is the operational bridge between FedGVI aggregation and the Friston et al.
(2024) sentinel scenario. Each agent broadcasts a posterior belief over a shared
latent factor; every agent then assimilates the others' broadcasts to form a
consensus. Following Friston (sensory attenuation — "agents do not hear
themselves"), an agent's heard consensus *excludes its own message* by default.

A single sharing round with the naive (log-linear) rule implements the
project's categorical posterior-log-potential specialization of the
"hive-mind" message-passing rule. That correspondence requires the documented
shared-support, admitted-potential, and fixed-weight assumptions; it does not
reconstruct the source factor graph, cavity structure, or full Eqs. 6--8
protocol. Swapping in the robust server heuristic changes the influence
weights. Seeded experiments test that behavior under declared contamination
regimes and retain both wins and reversals; they do not establish universal
contamination resistance.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

import numpy as np

from ._validation import as_nonnegative_weights, as_pmf_matrix
from .aggregation import (
    AggregationConfig,
    AggregationMethod,
    AggregationResult,
    aggregate_result,
)

ArrayF = np.ndarray
_EPS = 1e-12


@dataclass
class SharingDiagnostics:
    """Per-round outcome of belief sharing."""

    shared_posteriors: ArrayF  # (n_agents, n_states) post-sharing posteriors
    consensus: ArrayF  # global consensus (no self-exclusion)
    mean_surprise: float  # mean -log q(true_state) across agents
    mean_accuracy: float  # mean q(true_state) across agents
    normalized_effective_weights: ArrayF | None  # normalized influence (None if naive)

    @property
    def shared_beliefs(self) -> ArrayF:
        """Deprecated alias for :attr:`shared_posteriors`."""
        warnings.warn(
            "SharingDiagnostics.shared_beliefs is deprecated; use shared_posteriors",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.shared_posteriors

    @property
    def agent_weights(self) -> ArrayF | None:
        """Deprecated alias for normalized effective influence weights."""
        warnings.warn(
            "SharingDiagnostics.agent_weights is deprecated; use "
            "normalized_effective_weights",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.normalized_effective_weights


@dataclass(frozen=True)
class SharingRoundResult:
    """Complete diagnostics for one in-process belief-sharing round.

    ``global_aggregation`` records the all-agent server computation, while
    ``recipient_aggregations`` records the exact configured aggregation used
    for each recipient (including leave-one-out computations when
    ``exclude_self`` is enabled).  The compatibility :func:`share_round`
    adapter intentionally projects this richer result back to the historical
    :class:`SharingDiagnostics` shape.
    """

    global_aggregation: AggregationResult
    recipient_aggregations: tuple[AggregationResult, ...]
    shared_posteriors: ArrayF
    exclude_self: bool
    true_state: int | None
    mean_surprise: float | None
    mean_accuracy: float | None

    def __post_init__(self) -> None:
        shared = np.asarray(self.shared_posteriors, dtype=np.float64).copy()
        if shared.ndim != 2 or shared.size == 0 or not np.all(np.isfinite(shared)):
            raise ValueError("shared_posteriors must be a non-empty finite matrix")
        if len(self.recipient_aggregations) != shared.shape[0]:
            raise ValueError(
                "recipient_aggregations must contain one result per shared posterior"
            )
        if not isinstance(self.exclude_self, bool):
            raise ValueError("exclude_self must be a boolean")
        shared.setflags(write=False)
        object.__setattr__(self, "shared_posteriors", shared)
        object.__setattr__(
            self,
            "recipient_aggregations",
            tuple(self.recipient_aggregations),
        )

    @property
    def consensus(self) -> ArrayF:
        """Return the global all-agent consensus."""
        return self.global_aggregation.consensus

    @property
    def normalized_effective_weights(self) -> ArrayF:
        """Return the global normalized influence weights."""
        return self.global_aggregation.normalized_effective_weights


def _surprise(belief: ArrayF, true_state: int) -> float:
    return float(-np.log(max(belief[int(true_state)], _EPS)))


def _validate_true_state(true_state: int, n_states: int) -> int:
    if isinstance(true_state, (bool, np.bool_)):
        raise ValueError("true_state must be an integer index")
    try:
        raw = float(true_state)
    except (TypeError, ValueError) as exc:
        raise ValueError("true_state must be an integer index") from exc
    if not np.isfinite(raw) or raw != np.floor(raw):
        raise ValueError("true_state must be an integer index")
    index = int(raw)
    if not 0 <= index < n_states:
        raise ValueError(f"true_state must lie in [0, {n_states})")
    return index


def share_round_result(
    local_posteriors: Iterable[ArrayF] | None = None,
    *,
    method: str | None = None,
    base_weights: Iterable[float] | None = None,
    robustness: float | None = None,
    config: AggregationConfig | None = None,
    exclude_self: bool = True,
    true_state: int | None = None,
    **legacy: object,
) -> SharingRoundResult:
    """Run one federated belief-sharing round over a shared factor.

    ``local_posteriors`` : ``(n_agents, n_states)`` array of broadcast pmfs.
    ``method``        : compatibility method selector: ``'naive'`` (the
                        project log-linear pool; categorical Eq. 7
                        specialization under documented assumptions),
                        ``'robust'`` (server heuristic), or
                        ``'variational'`` (objective-backed conservative rule).
    ``config``        : validated public aggregation configuration; mutually
                        exclusive with compatibility ``method`` / ``robustness``.
    ``exclude_self``  : if True each agent's consensus omits its own broadcast.
    ``true_state``    : optional ground-truth index for surprise/accuracy.

    Returns :class:`SharingRoundResult` with the complete global and
    recipient-specific aggregation diagnostics.
    """
    if "agent_beliefs" in legacy:
        if local_posteriors is not None:
            raise TypeError("local_posteriors and deprecated agent_beliefs cannot both be supplied")
        local_posteriors = legacy.pop("agent_beliefs")  # type: ignore[assignment]
        warnings.warn(
            "agent_beliefs is deprecated; use local_posteriors",
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
        names = ", ".join(sorted(legacy))
        raise TypeError(f"unexpected keyword argument(s): {names}")
    if local_posteriors is None:
        raise TypeError("local_posteriors is required")
    raw_local_posteriors = list(local_posteriors)
    posterior_matrix = as_pmf_matrix(raw_local_posteriors, name="local_posteriors")
    n_agents, n_states = posterior_matrix.shape
    w = None if base_weights is None else as_nonnegative_weights(base_weights, n_agents)
    if config is not None and not isinstance(config, AggregationConfig):
        raise ValueError("config must be an AggregationConfig or None")
    if config is not None and (method is not None or robustness is not None):
        raise ValueError("config and compatibility method/robustness arguments are mutually exclusive")
    if config is None:
        resolved_method = "naive" if method is None else method
        resolved_robustness = 1.0 if robustness is None else robustness
        config = AggregationConfig(
            method=cast(AggregationMethod, resolved_method),
            robustness=resolved_robustness,
            max_iter=64 if resolved_method == "variational" else 32,
        )

    def fuse(idx_set: ArrayF) -> AggregationResult:
        # Pass each original row through the canonical aggregator exactly once.
        # Reusing ``posterior_matrix`` here would normalize once at this sharing
        # boundary and again inside ``aggregate_result``, creating avoidable
        # sub-ULP drift from a direct/process/socket call on the same input.
        sub = [raw_local_posteriors[int(index)] for index in idx_set]
        sub_w = None if w is None else w[idx_set]
        return aggregate_result(sub, config=config, base_weights=sub_w)

    all_idx = np.arange(n_agents)
    global_aggregation = fuse(all_idx)

    shared = np.empty_like(posterior_matrix)
    recipient_aggregations: list[AggregationResult] = []
    for n in range(n_agents):
        if exclude_self and n_agents > 1:
            idx = all_idx[all_idx != n]
            recipient = fuse(idx)
        else:
            recipient = global_aggregation
        recipient_aggregations.append(recipient)
        shared[n] = recipient.consensus

    if true_state is None:
        resolved_true_state = None
        mean_surprise = None
        mean_accuracy = None
    else:
        resolved_true_state = _validate_true_state(true_state, n_states)
        mean_surprise = float(
            np.mean(
                [
                    _surprise(shared[n], resolved_true_state)
                    for n in range(n_agents)
                ]
            )
        )
        mean_accuracy = float(
            np.mean([shared[n, resolved_true_state] for n in range(n_agents)])
        )

    return SharingRoundResult(
        global_aggregation=global_aggregation,
        recipient_aggregations=tuple(recipient_aggregations),
        shared_posteriors=shared,
        exclude_self=exclude_self,
        true_state=resolved_true_state,
        mean_surprise=mean_surprise,
        mean_accuracy=mean_accuracy,
    )


def share_round(
    local_posteriors: Iterable[ArrayF] | None = None,
    *,
    method: str | None = None,
    base_weights: Iterable[float] | None = None,
    robustness: float | None = None,
    config: AggregationConfig | None = None,
    exclude_self: bool = True,
    true_state: int | None = None,
    **legacy: object,
) -> SharingDiagnostics:
    """Run one round and return the historical compatibility diagnostics."""
    result = share_round_result(
        local_posteriors,
        method=method,
        base_weights=base_weights,
        robustness=robustness,
        config=config,
        exclude_self=exclude_self,
        true_state=true_state,
        **legacy,
    )
    global_weights = (
        None
        if config is not None and config.method == "naive"
        else result.normalized_effective_weights
    )
    if config is None:
        resolved_method = "naive" if method is None else method
        if resolved_method == "naive":
            global_weights = None
    return SharingDiagnostics(
        shared_posteriors=result.shared_posteriors,
        consensus=result.consensus,
        mean_surprise=(
            float("nan") if result.mean_surprise is None else result.mean_surprise
        ),
        mean_accuracy=(
            float("nan") if result.mean_accuracy is None else result.mean_accuracy
        ),
        normalized_effective_weights=global_weights,
    )
