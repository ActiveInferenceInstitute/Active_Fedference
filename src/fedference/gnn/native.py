"""Typed native adapters for the Friston et al. (2024) categorical specialization."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from ..belief_updating import infer_states
from .admission import GNNAdmissionError, dumps, loads
from .types import CategoricalModel, Tensor


class NativeWorld(TypedDict):
    A: list[NDArray[np.float64]]
    B: list[NDArray[np.float64]]
    C: list[NDArray[np.float64]]
    D: list[NDArray[np.float64]]
    n_states: int
    n_obs: int
    n_controls: int
    control_labels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SentinelParameters:
    """Explicit overrides; None retains the numeric source value."""

    acuity: float | None = None
    goal_bonus: float | None = None


def parameterize_sentinel(
    model: CategoricalModel,
    parameters: SentinelParameters,
) -> CategoricalModel:
    """Adjust the declared sensor and sole preferred outcome, retaining topology.

    Off-diagonal relative likelihoods come from the source asset. No transition,
    prior, label or preference-location formula is reconstructed here.
    """
    c = model.categorical
    a = np.array(c.A.values).reshape(c.A.shape)
    preferences = np.array(c.C.values)
    if parameters.acuity is not None:
        acuity = parameters.acuity
        if not np.isfinite(acuity) or not 0.0 < acuity <= 1.0:
            raise GNNAdmissionError("GNN_ACUITY", "acuity must lie in (0, 1]")
        if a.shape[0] != a.shape[1]:
            raise GNNAdmissionError("GNN_SHAPE", "sentinel.A")
        off = a.copy()
        np.fill_diagonal(off, 0)
        if np.any(off.sum(axis=0) <= 0):
            raise GNNAdmissionError("GNN_PARAMETERIZATION", "sentinel.A")
        a = off / off.sum(axis=0) * (1.0 - acuity)
        np.fill_diagonal(a, acuity)
        a = a / a.sum(axis=0)
    if parameters.goal_bonus is not None:
        if not np.isfinite(parameters.goal_bonus):
            raise GNNAdmissionError("GNN_NONFINITE", "goal_bonus")
        goals = np.flatnonzero(preferences)
        if goals.size != 1:
            raise GNNAdmissionError("GNN_PARAMETERIZATION", "sentinel.C")
        preferences[goals[0]] = parameters.goal_bonus
    parameterized = replace(
        model,
        categorical=replace(
            c,
            A=Tensor(c.A.axes, c.A.shape, tuple(a.ravel())),
            C=Tensor(c.C.axes, c.C.shape, tuple(preferences)),
        ),
    )
    # Preserve source provenance while the full IR digest binds all overrides.
    admitted = loads(dumps(parameterized))
    if not isinstance(admitted, CategoricalModel):
        raise GNNAdmissionError("GNN_KIND", "sentinel")
    return admitted


def native_world(model: CategoricalModel) -> NativeWorld:
    """Create writable native arrays from admitted immutable model data."""
    dumps(model)
    c = model.categorical
    return {
        "A": [np.array(c.A.values).reshape(c.A.shape)],
        "B": [np.array(c.B.values).reshape(c.B.shape)],
        "C": [np.array(c.C.values).reshape(-1, 1)],
        "D": [np.array(c.D.values).reshape(-1, 1)],
        "n_states": len(c.states),
        "n_obs": len(c.observations),
        "n_controls": len(c.actions),
        "control_labels": c.actions,
    }


def native_posterior(
    model: CategoricalModel,
    observation: int,
    log_prior: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Execute the existing native inference operation on GNN-derived values."""
    return infer_states(native_world(model)["A"], observation, log_prior)
