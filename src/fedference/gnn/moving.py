"""Native distinct-factor moving mapping and declared topology parameterization."""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from .admission import GNNAdmissionError
from .structured_admission import StructuredProgram, _array, _native
from .types import MovingObserverModel, Tensor


@dataclass(frozen=True, slots=True)
class MovingParameters:
    """Explicit topology parameters; numeric sensor profiles remain source-owned."""

    n_positions: int | None = None
    n_agents: int | None = None
    fov_width: int | None = None
    seed: int = 0



class MovingWorld(TypedDict):
    A: list[NDArray[np.float64]]
    B: NDArray[np.float64]
    D: NDArray[np.float64]
    observation_map: NDArray[np.float64]
    position_prior: NDArray[np.float64]
    agent_positions: list[int]
    n_actions: int
    actions: tuple[str, ...]
    n_states: int
    n_positions: int
    n_agents: int
    fov_width: int
    seed: int
    model_digest: str



def moving_world(program: StructuredProgram, *, seed: int = 0) -> MovingWorld:
    """Consume distinct threat and position tensors without flattening them."""
    native = _native(program)
    model = program.model
    if not isinstance(model, MovingObserverModel):
        raise GNNAdmissionError("GNN_KIND", "moving")
    raw_positions = native["agent_positions"]
    assert isinstance(raw_positions, list)
    positions = [p for p in raw_positions if type(p) is int]
    width = native["fov_width"]
    assert isinstance(width, int)
    observation_map = _array(model.observation_map)
    return {
        "A": [observation_map[:, :, p].copy() for p in positions],
        "B": _array(model.position_transition),
        "D": _array(model.threat.D),
        "observation_map": observation_map,
        "position_prior": _array(model.position_prior),
        "agent_positions": positions,
        "n_actions": len(model.threat.actions),
        "actions": model.threat.actions,
        "n_states": len(model.threat.states),
        "n_positions": len(model.positions),
        "n_agents": len(positions),
        "fov_width": width,
        "seed": seed,
        "model_digest": program.digest,
    }



def parameterize_moving(program: StructuredProgram, parameters: MovingParameters) -> StructuredProgram:
    """Resize only the declared line template, retaining source sensor/prior values."""
    _native(program)
    model = program.model
    if not isinstance(model, MovingObserverModel):
        raise GNNAdmissionError("GNN_KIND", "moving")
    native = _native(program)
    raw_positions = native["agent_positions"]
    assert isinstance(raw_positions, list)
    n = len(model.positions) if parameters.n_positions is None else parameters.n_positions
    agents = len(raw_positions) if parameters.n_agents is None else parameters.n_agents
    width = native["fov_width"] if parameters.fov_width is None else parameters.fov_width
    if type(n) is not int or type(agents) is not int or agents < 1 or n < agents:
        raise GNNAdmissionError("GNN_POSITION", "need one position per agent")
    if type(width) is not int or width < 1:
        raise GNNAdmissionError("GNN_POSITION", "fov_width")
    native = _native(program)
    if n != len(model.positions):
        source_n = len(model.positions)
        expected_controls = np.zeros((source_n, source_n, 3))
        for position in range(source_n):
            targets = (position, max(0, position - 1), min(source_n - 1, position + 1))
            for action, target in enumerate(targets):
                expected_controls[target, position, action] = 1
        if not np.array_equal(_array(model.position_transition), expected_controls):
            raise GNNAdmissionError("GNN_PARAMETERIZATION", "non-template controls cannot resize")
        if not np.array_equal(_array(model.position_prior), np.full(source_n, 1 / source_n)):
            raise GNNAdmissionError("GNN_PARAMETERIZATION", "nonuniform position prior cannot resize")
        source = _array(model.observation_map)
        half = len(model.positions) // 2
        if (
            half < 1
            or not np.array_equal(source[:, :, :half], np.repeat(source[:, :, :1], half, axis=2))
            or not np.array_equal(
                source[:, :, half:],
                np.repeat(source[:, :, half : half + 1], len(model.positions) - half, axis=2),
            )
        ):
            raise GNNAdmissionError("GNN_PARAMETERIZATION", "heterogeneous sensors cannot resize")
        b = np.zeros((n, n, 3))
        for p in range(n):
            for action, target in enumerate((p, max(0, p - 1), min(n - 1, p + 1))):
                b[target, p, action] = 1
        observation_map = np.stack([source[:, :, 0 if p < n // 2 else half] for p in range(n)], axis=2)
        model = replace(
            model,
            positions=tuple(str(p) for p in range(n)),
            position_transition=Tensor(model.position_transition.axes, b.shape, tuple(b.ravel())),
            position_prior=Tensor(("position",), (n,), tuple(np.full(n, 1 / n))),
            observation_map=Tensor(
                model.observation_map.axes, observation_map.shape, tuple(observation_map.ravel())
            ),
            threat=replace(
                model.threat,
                A=Tensor(model.threat.A.axes, model.threat.A.shape, tuple(observation_map[:, :, 0].ravel())),
            ),
        )
    if parameters.n_positions is not None or parameters.n_agents is not None:
        native["agent_positions"] = [i * (n // agents) for i in range(agents)]
    native["fov_width"] = width
    result = StructuredProgram(model, json.dumps(native, sort_keys=True))
    _native(result)
    return result
