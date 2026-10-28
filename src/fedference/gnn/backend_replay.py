"""Verify qualified backend output against its request and native replay.

Only float32 batch-one, horizon-one state inference is compared. Learning,
planning and complete scientific protocol equivalence are not claimed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ._documents import _mapping
from .admission import GNNAdmissionError, dumps
from .backend_protocol import BackendConfig, BackendResult, _digest, _spec
from .backend_registry import ATOL, RTOL, load_backend_registry, read_backend_document
from .native import native_posterior, native_world
from .types import CategoricalModel, JSONValue


def _same_json(left: JSONValue, right: JSONValue) -> bool:
    try:
        return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
            right, sort_keys=True, allow_nan=False
        )
    except (ValueError, TypeError):
        return False



def _array(value: JSONValue, shape: tuple[int, ...]) -> np.ndarray:
    def numeric(item: JSONValue) -> bool:
        return all(numeric(v) for v in item) if isinstance(item, list) else type(item) in (int, float)

    if not numeric(value):
        raise GNNAdmissionError("GNN_BACKEND_OUTPUT", "trajectory must contain JSON numbers")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise GNNAdmissionError("GNN_BACKEND_OUTPUT", "numeric trajectory") from error
    if array.shape != shape or not np.isfinite(array).all():
        raise GNNAdmissionError("GNN_BACKEND_OUTPUT", "trajectory shape or finite values")
    return array



def verify_backend_output(
    model: CategoricalModel,
    config: BackendConfig,
    output: Path,
) -> BackendResult:
    """Read actual runtime files and reject refusal, truncation or replay mismatch."""
    lock, runtime, result = (
        load_backend_registry(config.lock_path).as_dict(),
        read_backend_document(output / "runtime.json"),
        read_backend_document(output / "simulation_results.json"),
    )
    profile_name = runtime.get("profile")
    profiles = _mapping(lock["profiles"], "profiles")
    if not isinstance(profile_name, str) or profile_name not in profiles:
        raise GNNAdmissionError("GNN_BACKEND_ENVIRONMENT", "profile")
    environment = dict(lock, **_mapping(profiles[profile_name], "profile"))
    request = read_backend_document(output / "request.json")
    if not _same_json(request.get("lock"), lock):
        raise GNNAdmissionError("GNN_BACKEND_BINDING", "lock")
    if runtime.get("request_sha256") != _digest((output / "request.json").read_bytes()) or request.get(
        "model_digest"
    ) != _digest(dumps(model).encode()):
        raise GNNAdmissionError("GNN_BACKEND_BINDING", "request/model")
    if not _same_json(request.get("spec"), _spec(model, config)):
        raise GNNAdmissionError("GNN_BACKEND_BINDING", "request/spec")
    if runtime.get("success") is not True or result.get("success") is not True:
        raise GNNAdmissionError("GNN_BACKEND_REFUSED", "success")
    for key in ("agent", "python", "distributions", "dtype", "device", "batch_size", "policy_horizon"):
        if not _same_json(runtime.get(key), environment.get(key)):
            raise GNNAdmissionError("GNN_BACKEND_ENVIRONMENT", key)
    if result.get("framework") != "PyMDP" or result.get("backend") != "jax":
        raise GNNAdmissionError("GNN_BACKEND_OUTPUT", "framework")
    if not _same_json(result.get("num_timesteps"), config.steps):
        raise GNNAdmissionError("GNN_BACKEND_OUTPUT", "num_timesteps")
    metadata = _mapping(result.get("runtime_metadata"), "runtime_metadata")
    params = _mapping(result.get("model_parameters"), "model_parameters")
    if (
        not _same_json(metadata.get("random_seed"), config.seed)
        or not _same_json(params.get("batch_size"), 1)
        or not _same_json(params.get("policy_len"), 1)
    ):
        raise GNNAdmissionError("GNN_BACKEND_OUTPUT", "runtime parameters")
    c = model.categorical
    beliefs = _array(result.get("beliefs"), (config.steps, len(c.states)))
    policies = _array(result.get("policy_posterior"), (config.steps, len(c.actions)))
    _array(result.get("expected_free_energy"), policies.shape)
    for array in (beliefs, policies):
        if np.any(array < 0) or not np.allclose(array.sum(axis=1), 1, rtol=RTOL, atol=ATOL):
            raise GNNAdmissionError("GNN_BACKEND_OUTPUT", "normalized probabilities")
    indices = []
    for key, size in (
        ("actions", len(c.actions)),
        ("observations", len(c.observations)),
        ("true_states", len(c.states)),
    ):
        values = result.get(key)
        array = _array(values, (config.steps + (key == "true_states"),))
        if not isinstance(values, list) or any(type(v) is not int for v in values):
            raise GNNAdmissionError("GNN_BACKEND_OUTPUT", key)
        if np.any(array < 0) or np.any(array >= size):
            raise GNNAdmissionError("GNN_BACKEND_OUTPUT", key)
        indices.append(tuple(int(v) for v in array))
    actions, observations = indices[:2]
    world = native_world(model)
    prior = world["D"][0].reshape(-1)
    replay = []
    for observation, action in zip(observations, actions):
        posterior = native_posterior(model, observation, np.log(np.maximum(prior, 1e-16)))
        replay.append(posterior.reshape(-1))
        prior = world["B"][0][:, :, action] @ posterior.reshape(-1)
    expected = np.asarray(replay)
    if not np.allclose(beliefs, expected, rtol=RTOL, atol=ATOL):
        raise GNNAdmissionError(
            "GNN_BACKEND_REPLAY", "posterior differs from fixed observation/action replay"
        )
    trace = {
        key: result[key]
        for key in ("beliefs", "observations", "actions", "policy_posterior", "expected_free_energy")
    }
    return BackendResult(
        tuple(tuple(float(v) for v in row) for row in beliefs),
        tuple(tuple(float(v) for v in row) for row in policies),
        actions,
        observations,
        _digest(json.dumps(trace, sort_keys=True, allow_nan=False).encode()),
        _digest(dumps(model).encode()),
        float(np.max(np.abs(beliefs - expected))),
        output / "receipt.json",
    )
