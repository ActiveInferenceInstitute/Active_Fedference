"""Explicit pinned PyMDP execution; Friston et al. (2024) categorical subset only.

Float32 replay agrees only on state inference, not learning or planning/EFE.
The caller supplies an upstream checkout, interpreter, lock and fresh output.
No installation, network discovery, native fallback or default import is used.
"""

# SIZE_OK: The embedded isolated child program belongs to this single process boundary.
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from .admission import GNNAdmissionError, dumps
from .backend_registry import ATOL, RTOL, load_backend_registry, read_backend_document
from .loader import _mapping
from .native import native_posterior, native_world
from .types import CategoricalModel, JSONValue, Model

_RUNNER: Final = r"""
import hashlib, json, sys
from pathlib import Path
site = Path(sys.executable).parent.parent / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"
sys.path.append(str(site / "site-packages"))
import importlib.metadata as md
root, output = map(Path, sys.argv[1:])
request_bytes = (output / "request.json").read_bytes()
request = json.loads(request_bytes)
registry = request["lock"]
versions = {d.metadata["Name"]: d.version for d in md.distributions()}
matches = [name for name, profile in registry["profiles"].items()
           if profile["distributions"] == versions and profile["python"] == sys.version.split()[0]]
if len(matches) != 1:
    raise RuntimeError("GNN_BACKEND_ENVIRONMENT: installed closure differs from qualified profiles")
profile_name = matches[0]
lock = dict(registry, **registry["profiles"][profile_name])
sys.path.insert(0, str(root / "src"))
from gnn.execute.pymdp.simulation import run_pymdp_simulation, _build_pymdp_agent
from pymdp.agent import Agent
import jax
import numpy as np
if jax.default_backend() != lock["device"]:
    raise RuntimeError("GNN_BACKEND_ENVIRONMENT: device differs from lock")
spec = request["spec"]
success, result = run_pymdp_simulation(spec, output)
m = spec["initialparameterization"]
agent = _build_pymdp_agent(A_np=np.array(m["A"]), B_np=np.array(m["B"]),
    C_np=np.array(m["C"]), D_np=np.array(m["D"]), E_np=None)
runtime = dict(success=success, profile=profile_name, agent=Agent.__module__+"."+Agent.__qualname__,
    python=sys.version.split()[0], distributions=versions, dtype=str(agent.A[0].dtype),
    device=jax.default_backend(), devices=[str(d) for d in jax.devices()],
    batch_size=agent.A[0].shape[0], policy_horizon=int(agent.policy_len),
    executable=sys.executable, prefix=sys.prefix, base_prefix=sys.base_prefix,
    request_sha256=hashlib.sha256(request_bytes).hexdigest())
(output / "runtime.json").write_text(json.dumps(runtime, allow_nan=False))
if not success:
    raise RuntimeError("GNN_BACKEND_REFUSED: " + repr(result))
"""


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


def run_pymdp(model: Model, config: BackendConfig, output: Path) -> BackendResult:
    """Execute admitted categorical data with a locked, isolated upstream process.

    One modality/factor, batch one, horizon one, static transitions, no learning.
    Upstream observes D at t=0 and predicts with each sampled action thereafter.
    Retained failure files are intentional; destinations must be newly owned.
    """
    dumps(model)
    if not isinstance(model, CategoricalModel):
        raise GNNAdmissionError("GNN_BACKEND_SEMANTICS", model.kind)
    if type(config.steps) is not int or not 1 <= config.steps <= 1000:
        raise GNNAdmissionError("GNN_BACKEND_CONFIG", "steps")
    if type(config.seed) is not int or not 0 <= config.seed < 2**32:
        raise GNNAdmissionError("GNN_BACKEND_CONFIG", "seed")
    if output.exists() or output.is_symlink():
        raise GNNAdmissionError("GNN_BACKEND_DESTINATION", str(output))
    lock = load_backend_registry(config.lock_path).as_dict()
    files = _mapping(lock.get("source_files"), "source_files")
    if not files:
        raise GNNAdmissionError("GNN_BACKEND_LOCK", "empty source closure")
    output.mkdir(parents=True, exist_ok=False)
    request = dict(
        lock=lock,
        model_digest=_digest(dumps(model).encode()),
        spec=_spec(model, config),
    )
    (output / "request.json").write_text(json.dumps(request, allow_nan=False), encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="gnn-backend-") as directory:
        snapshot = Path(directory)
        for relative, expected_hash in files.items():
            path = Path(relative)
            if not path.parts or path.is_absolute() or ".." in path.parts or path.parts[0] != "src":
                raise GNNAdmissionError("GNN_BACKEND_LOCK", relative)
            try:
                payload = (config.upstream_root / path).read_bytes()
            except OSError as error:
                raise GNNAdmissionError("GNN_BACKEND_SOURCE", relative) from error
            if _digest(payload) != expected_hash:
                raise GNNAdmissionError("GNN_BACKEND_SOURCE", relative)
            target = snapshot / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        command = [
            str(config.python.absolute()),
            "-I",
            "-S",
            "-B",
            "-c",
            _RUNNER,
            str(snapshot),
            str(output.resolve()),
        ]
        try:
            process = subprocess.run(command, cwd=snapshot, capture_output=True, text=True, timeout=180)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise GNNAdmissionError("GNN_BACKEND_PROCESS", str(error)) from error
        (output / "stdout.log").write_text(process.stdout, encoding="utf-8")
        (output / "stderr.log").write_text(process.stderr, encoding="utf-8")
        (output / "process.json").write_text(json.dumps(dict(argv=command, returncode=process.returncode)))
        if process.returncode != 0:
            raise GNNAdmissionError("GNN_BACKEND_PROCESS", process.stderr)
    result = verify_backend_output(model, config, output)
    result.receipt.write_text(
        json.dumps(
            dict(
                model_digest=result.model_digest,
                trace_digest=result.trace_digest,
                replay_max_error=result.replay_max_error,
                rtol=RTOL,
                atol=ATOL,
                lock_digest=_digest(config.lock_path.read_bytes()),
                scope="categorical state inference replay; no learning/planning equivalence",
            )
        )
    )
    return result
