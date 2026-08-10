"""Colony construction helpers for experiment harnesses."""

from __future__ import annotations

import numpy as np

ArrayF = np.ndarray


def soft_colony(
    true_state: int,
    n_agents: int,
    n_s: int,
    confidence: float,
    rng: np.random.Generator,
    jitter: float,
) -> ArrayF:
    """Build ``n_agents`` soft healthy beliefs peaked on ``true_state``."""
    colony = np.empty((n_agents, n_s), dtype=np.float64)
    for n in range(n_agents):
        conf = float(np.clip(confidence + rng.uniform(-jitter, jitter), 0.05, 0.95))
        belief = np.full(n_s, (1.0 - conf) / (n_s - 1), dtype=np.float64)
        belief[true_state] = conf
        colony[n] = belief
    return colony


def healthy_colony(
    true_state: int,
    n_agents: int,
    n_s: int,
    confidence: float,
    *,
    rng: np.random.Generator | None = None,
    jitter: float = 0.0,
) -> ArrayF:
    """Build a colony of healthy beliefs; optional jitter when ``rng`` is given."""
    if rng is not None and jitter > 0.0:
        return soft_colony(true_state, n_agents, n_s, confidence, rng, jitter)
    colony = np.empty((n_agents, n_s), dtype=np.float64)
    for n in range(n_agents):
        belief = np.full(n_s, (1.0 - confidence) / (n_s - 1), dtype=np.float64)
        belief[true_state] = confidence
        colony[n] = belief
    return colony


__all__ = ["soft_colony", "healthy_colony"]
