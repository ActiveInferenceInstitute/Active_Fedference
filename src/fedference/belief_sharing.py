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
from .aggregation import AggregationConfig, AggregationMethod, aggregate_result

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

    Returns :class:`SharingDiagnostics`.
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
    posterior_matrix = as_pmf_matrix(local_posteriors, name="local_posteriors")
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

    def fuse(idx_set: ArrayF) -> tuple[ArrayF, ArrayF | None]:
        sub = posterior_matrix[idx_set]
        sub_w = None if w is None else w[idx_set]
        result = aggregate_result(sub, config=config, base_weights=sub_w)
        diagnostics = None if config.method == "naive" else result.normalized_effective_weights
        return result.consensus, diagnostics

    all_idx = np.arange(n_agents)
    consensus, global_weights = fuse(all_idx)

    shared = np.empty_like(posterior_matrix)
    for n in range(n_agents):
        if exclude_self and n_agents > 1:
            idx = all_idx[all_idx != n]
            shared[n], _ = fuse(idx)
        else:
            shared[n] = consensus

    if true_state is None:
        mean_surprise = float("nan")
        mean_accuracy = float("nan")
    else:
        state = _validate_true_state(true_state, n_states)
        mean_surprise = float(np.mean([_surprise(shared[n], state) for n in range(n_agents)]))
        mean_accuracy = float(np.mean([shared[n, state] for n in range(n_agents)]))

    return SharingDiagnostics(
        shared_posteriors=shared,
        consensus=consensus,
        mean_surprise=mean_surprise,
        mean_accuracy=mean_accuracy,
        normalized_effective_weights=global_weights,
    )
