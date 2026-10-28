"""Typed backend execution records and exact categorical request projection."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .types import CategoricalModel, JSONValue


@dataclass(frozen=True, slots=True)
class BackendConfig:
    upstream_root: Path
    python: Path
    lock_path: Path = Path(__file__).resolve().parents[1] / "config" / "gnn-backend-lock.json"
    steps: int = 4
    seed: int = 731



@dataclass(frozen=True, slots=True)
class BackendResult:
    beliefs: tuple[tuple[float, ...], ...]
    policies: tuple[tuple[float, ...], ...]
    actions: tuple[int, ...]
    observations: tuple[int, ...]
    trace_digest: str
    model_digest: str
    replay_max_error: float
    receipt: Path



def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()



def _spec(model: CategoricalModel, config: BackendConfig) -> dict[str, JSONValue]:
    c = model.categorical
    matrices: dict[str, JSONValue] = {
        name: np.array(t.values).reshape(t.shape).tolist()
        for name, t in (("A", c.A), ("B", c.B), ("C", c.C), ("D", c.D))
    }
    params: dict[str, JSONValue] = dict(
        num_states=len(c.states),
        num_observations=len(c.observations),
        num_actions=len(c.actions),
        num_timesteps=config.steps,
        random_seed=config.seed,
        batch_size=1,
        policy_len=1,
        gamma=16.0,
        alpha=16.0,
        b_tensor_order="next_state_previous_state_action",
    )
    return {
        "model_name": model.metadata.name,
        "initialparameterization": matrices,
        "model_parameters": params,
    }
