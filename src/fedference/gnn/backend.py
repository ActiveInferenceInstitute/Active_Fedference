"""Explicit pinned PyMDP process execution (Friston et al., 2024 subset).

No installation, network discovery, native fallback or default optional import.
Output admission belongs to backend_replay; registry admission is independent.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Final

from ._documents import _mapping
from .admission import GNNAdmissionError, dumps
from .backend_protocol import BackendConfig as BackendConfig
from .backend_protocol import BackendResult as BackendResult
from .backend_protocol import _digest, _spec
from .backend_registry import ATOL, RTOL, load_backend_registry
from .backend_replay import verify_backend_output as verify_backend_output
from .types import CategoricalModel, Model

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
