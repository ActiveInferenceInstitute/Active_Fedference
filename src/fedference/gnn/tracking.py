"""Source-role mapping into the bounded native hybrid tracking consumer."""
from __future__ import annotations

import json
from dataclasses import asdict, fields

import numpy as np

from ..hybrid_tracking import HybridTrackingConfig, run_hybrid_tracking
from ._documents import _json, _mapping
from .admission import GNNAdmissionError
from .scientific_admission import ScientificProgram, _native
from .types import HybridGaussianModel, JSONValue


def tracking_config(program: ScientificProgram) -> HybridTrackingConfig:
    """Bind target/noise/cue inputs; refuse unsupported dynamical semantics."""
    native = _native(program)
    model = program.model
    if not isinstance(model, HybridGaussianModel):
        raise GNNAdmissionError("GNN_KIND", "tracking")
    if model.coordinates != ("target_position", "velocity_observation_bias") or model.units != ("m", "m/s"):
        raise GNNAdmissionError("GNN_UNIT", "tracking coordinates")
    covariance = np.array(model.covariance.values).reshape(2, 2)
    if covariance[0, 1] != 0 or covariance[1, 0] != 0 or model.mean.values[1] != 0:
        raise GNNAdmissionError("GNN_CAPABILITY", "correlation or velocity bias")
    c = model.categorical
    a = np.array(c.A.values).reshape(c.A.shape)
    if (
        c.states != ("constant-velocity", "accelerating")
        or c.observations != ("constant-cue", "accelerating-cue")
        or c.actions != ("native-controller",)
        or c.B.values != (1.0, 0.0, 0.0, 1.0)
        or c.C.values != (0.0, 0.0)
        or c.D.values != (0.5, 0.5)
        or a.shape != (2, 2)
        or not np.array_equal(a, a[::-1, ::-1])
    ):
        raise GNNAdmissionError("GNN_CAPABILITY", "tracking categorical protocol")
    settings = _mapping(native["tracking"], "tracking settings")
    derived = {"target_position", "position_observation_var", "velocity_observation_var", "context_accuracy"}
    expected = {field.name for field in fields(HybridTrackingConfig)} - derived
    if set(settings) != expected:
        raise GNNAdmissionError("GNN_FIELDS", "tracking settings")
    numbers: dict[str, float | int] = {}
    for key, value in settings.items():
        if type(value) not in (int, float) or not isinstance(value, (int, float)):
            raise GNNAdmissionError("GNN_NUMBER", key)
        numbers[key] = value
    if numbers["robustness"] != 0 or numbers["outlier_bias"] != 0:
        raise GNNAdmissionError("GNN_CAPABILITY", "clean naive tracking protocol")
    return HybridTrackingConfig(
        n_agents=_integer(numbers["n_agents"], "n_agents"),
        horizon=_integer(numbers["horizon"], "horizon"),
        target_position=model.mean.values[0],
        position_observation_var=float(covariance[0, 0]),
        velocity_observation_var=float(covariance[1, 1]),
        context_accuracy=float(a[0, 0]),
        process_var=numbers["process_var"],
        action_gain=numbers["action_gain"],
        max_acceleration=numbers["max_acceleration"],
        robustness=numbers["robustness"],
        outlier_bias=numbers["outlier_bias"],
    )



def _integer(value: float | int, name: str) -> int:
    if type(value) is not int:
        raise GNNAdmissionError("GNN_NUMBER", name)
    return int(value)



def run_tracking(program: ScientificProgram, *, seed: int) -> dict[str, JSONValue]:
    """Run the real native tracking consumer; seed is the explicit caller input."""
    config = tracking_config(program)
    result = _mapping(
        _json(
            json.dumps(
                run_hybrid_tracking(
                    seed,
                    method="naive",
                    contaminate_one_agent=False,
                    config=config,
                ),
                allow_nan=False,
            )
        ),
        "tracking result",
    )
    return {"model_digest": program.digest, "config": _json(json.dumps(asdict(config))), "result": result}
