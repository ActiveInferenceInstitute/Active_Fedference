"""Shared trial kernels for hierarchical-world experiment harnesses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .aggregation import log_linear_pool
from .belief_updating import infer_states

ArrayF = np.ndarray
_EPS = 1e-12


@dataclass(frozen=True)
class FlatVsNlevelMetrics:
    """Per-trial flat vs hierarchical location accuracy and free energy."""

    flat_loc_correct: float
    nlevel_loc_correct: float
    flat_fe: float
    nlevel_fe: float
    top_correct: float | None = None


def compare_flat_vs_nlevel(
    *,
    A_base: ArrayF,
    per_agent_obs: list[int],
    true_state: int,
    n_iters: int,
    infer_fn: Callable[..., dict],
    infer_kwargs: dict,
    nlevel_key: str = "nlevel",
    top_level_index: int | None = None,
    true_top: int | None = None,
) -> FlatVsNlevelMetrics:
    """Compare flat log-linear pooling against an N-level infer + pool path.

    Args:
        A_base: L1 likelihood matrix.
        per_agent_obs: One observation index per agent.
        true_state: Ground-truth L1 location index.
        n_iters: Alternating-minimization iterations for ``infer_fn``.
        infer_fn: ``hierarchical_infer`` or ``nlevel_infer``.
        infer_kwargs: Extra kwargs passed to ``infer_fn`` (world dict).
        nlevel_key: Unused label hook for callers; kept for API stability.
        top_level_index: When set, read ``q_levels[top_level_index]`` for top accuracy.
        true_top: Ground-truth top-level state when ``top_level_index`` is set.

    Returns:
        Per-trial metrics accumulated by callers over ``n_trials``.
    """
    _ = nlevel_key
    n_s = A_base.shape[1]
    flat_log_prior = np.log(np.full(n_s, 1.0 / n_s))
    flat_local_posteriors = [
        infer_states(A_base, o, flat_log_prior) for o in per_agent_obs
    ]
    flat_consensus = log_linear_pool(flat_local_posteriors)

    nlevel_results = [
        infer_fn(A_base, o, n_iters=n_iters, **infer_kwargs) for o in per_agent_obs
    ]
    if "q_loc" in nlevel_results[0]:
        l1_local_posteriors = [r["q_loc"] for r in nlevel_results]
    else:
        l1_local_posteriors = [r["q_levels"][-1] for r in nlevel_results]
    nlevel_consensus = log_linear_pool(l1_local_posteriors)

    top_correct: float | None = None
    if top_level_index is not None and true_top is not None:
        if "q_ctx" in nlevel_results[0]:
            top_local_posteriors = [r["q_ctx"] for r in nlevel_results]
        else:
            top_local_posteriors = [
                r["q_levels"][top_level_index] for r in nlevel_results
            ]
        top_consensus = log_linear_pool(top_local_posteriors)
        top_correct = float(np.argmax(top_consensus) == true_top)

    return FlatVsNlevelMetrics(
        flat_loc_correct=float(np.argmax(flat_consensus) == true_state),
        nlevel_loc_correct=float(np.argmax(nlevel_consensus) == true_state),
        flat_fe=float(-np.log(np.clip(flat_consensus[true_state], _EPS, None))),
        nlevel_fe=float(-np.log(np.clip(nlevel_consensus[true_state], _EPS, None))),
        top_correct=top_correct,
    )


__all__ = ["FlatVsNlevelMetrics", "compare_flat_vs_nlevel"]
