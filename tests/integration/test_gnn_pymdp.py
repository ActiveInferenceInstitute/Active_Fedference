"""Real optional upstream runtime and adversarial boundary tests; no mocks."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from fedference.gnn.admission import GNNAdmissionError
from fedference.gnn.backend import BackendConfig, run_pymdp, verify_backend_output
from fedference.gnn.loader import load_sentinel
from fedference.gnn.types import Categorical, CategoricalModel, HybridGaussianModel, SiteFactorModel, Tensor

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def config() -> BackendConfig:
    root = Path(os.environ["GNN_ROOT"]).resolve()
    selected = BackendConfig(root, Path(os.environ.get("GNN_PYTHON", str(root / ".venv/bin/python"))))
    if os.environ.get("GNN_BACKEND_LOCK"):
        selected = replace(selected, lock_path=Path(os.environ["GNN_BACKEND_LOCK"]))
    return selected


@pytest.fixture(scope="module")
def model() -> CategoricalModel:
    sentinel = load_sentinel()
    tiny = Categorical(
        ("s0", "s1"),
        ("o0", "o1"),
        ("stay", "switch"),
        Tensor(("observation", "state"), (2, 2), (0.8, 0.3, 0.2, 0.7)),
        Tensor(
            ("next_state", "previous_state", "action"), (2, 2, 2), (0.9, 0.2, 0.1, 0.8, 0.1, 0.8, 0.9, 0.2)
        ),
        Tensor(("observation",), (2,), (0.0, 1.0)),
        Tensor(("state",), (2,), (0.4, 0.6)),
    )
    return replace(sentinel, categorical=tiny)


@pytest.fixture(scope="module")
def executed(config: BackendConfig, model: CategoricalModel, tmp_path_factory: pytest.TempPathFactory):
    output = tmp_path_factory.mktemp("pymdp-real") / "run"
    result = run_pymdp(model, config, output)
    return output, result


def test_real_agent_inference_policy_and_action(config, model, executed):
    output, result = executed
    runtime = json.loads((output / "runtime.json").read_text())
    assert runtime["agent"] == "pymdp.agent.Agent"
    assert runtime["dtype"] == "float32" and runtime["device"] == "cpu"
    assert runtime["batch_size"] == runtime["policy_horizon"] == 1
    assert len(result.beliefs) == len(result.policies) == len(result.actions) == 4
    assert result.replay_max_error < 1e-6
    assert result == verify_backend_output(model, config, output)
    print(
        json.dumps(
            dict(
                runtime=runtime,
                observations=result.observations,
                actions=result.actions,
                beliefs=result.beliefs,
                policies=result.policies,
                replay_max_error=result.replay_max_error,
                receipt=str(result.receipt),
            )
        )
    )


def test_seeded_replay_and_preference_sensitivity(config, model, executed, tmp_path):
    _, baseline = executed
    replay = run_pymdp(model, config, tmp_path / "replay")
    assert replay.trace_digest == baseline.trace_digest
    changed = replace(
        model, categorical=replace(model.categorical, C=Tensor(("observation",), (2,), (1.0, 2.0)))
    )
    shifted = run_pymdp(changed, config, tmp_path / "changed")
    assert shifted.trace_digest != baseline.trace_digest
    assert shifted.model_digest != baseline.model_digest
    assert shifted.observations == baseline.observations
    assert shifted.beliefs == baseline.beliefs
    print(
        json.dumps(
            dict(baseline=baseline.trace_digest, replay=replay.trace_digest, changed=shifted.trace_digest)
        )
    )


def test_missing_backend_and_semantic_mismatch(config, model, tmp_path):
    missing = replace(config, python=tmp_path / "missing-python")
    with pytest.raises(GNNAdmissionError, match="GNN_BACKEND_PROCESS") as caught:
        run_pymdp(model, missing, tmp_path / "missing")
    print(str(caught.value))
    bad = replace(model, metadata=replace(model.metadata, schedule=("observe", "predict")))
    with pytest.raises(GNNAdmissionError, match="GNN_SCHEDULE") as caught:
        run_pymdp(bad, config, tmp_path / "semantics")
    assert not (tmp_path / "semantics").exists()
    print(str(caught.value))
    site = SiteFactorModel(
        replace(model.metadata, capabilities=("fedference.gnn/site-factor/1",), schedule=("site-add",)),
        ("x",),
        ("m",),
        Tensor(("coordinate", "coordinate_dual"), (1, 1), (1.0,)),
        Tensor(("coordinate",), (1,), (0.0,)),
    )
    with pytest.raises(GNNAdmissionError, match="GNN_BACKEND_SEMANTICS"):
        run_pymdp(site, config, tmp_path / "site")
    assert not (tmp_path / "site").exists()
    hybrid = HybridGaussianModel(
        replace(model.metadata, capabilities=("fedference.gnn/hybrid-gaussian/1",)),
        model.categorical,
        ("x",),
        ("m",),
        Tensor(("coordinate",), (1,), (0.0,)),
        Tensor(("coordinate", "coordinate_dual"), (1, 1), (1.0,)),
    )
    with pytest.raises(GNNAdmissionError, match="GNN_BACKEND_SEMANTICS"):
        run_pymdp(hybrid, config, tmp_path / "hybrid")
    assert not (tmp_path / "hybrid").exists()
    with pytest.raises(GNNAdmissionError, match="installed closure differs"):
        run_pymdp(model, replace(config, python=Path(sys.executable)), tmp_path / "missing-deps")


@pytest.mark.parametrize(
    "field,value",
    [
        ("success", False),
        ("beliefs", []),
        ("beliefs", [[float("nan")] * 2] * 4),
        ("actions", [0, 0, 0, 5]),
        ("observations", [False] * 4),
        ("policy_posterior", [[0.0, 0.0]] * 4),
        ("beliefs", [[0.5, 0.5]] * 4),
    ],
)
def test_corrupted_output_rejected(field, value, config, model, executed, tmp_path):
    original, _ = executed
    output = tmp_path / "corrupt"
    shutil.copytree(original, output)
    path = output / "simulation_results.json"
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(GNNAdmissionError):
        verify_backend_output(model, config, output)


def test_missing_and_truncated_output(config, model, executed, tmp_path):
    with pytest.raises(GNNAdmissionError, match="GNN_BACKEND_OUTPUT"):
        verify_backend_output(model, config, tmp_path)
    original, _ = executed
    output = tmp_path / "truncated"
    shutil.copytree(original, output)
    (output / "runtime.json").write_text("{")
    with pytest.raises(GNNAdmissionError, match="GNN_JSON"):
        verify_backend_output(model, config, output)


def test_nonempty_destination_preserved(config, model, tmp_path):
    protected = tmp_path / "keep.txt"
    protected.write_text("keep")
    with pytest.raises(GNNAdmissionError, match="GNN_BACKEND_DESTINATION"):
        run_pymdp(model, config, tmp_path)
    assert protected.read_text() == "keep"
    assert list(tmp_path.iterdir()) == [protected]


def test_source_drift_before_execution(config, model, tmp_path):
    root = tmp_path / "upstream"
    root.mkdir()
    with pytest.raises(GNNAdmissionError, match="GNN_BACKEND_SOURCE"):
        run_pymdp(model, replace(config, upstream_root=root), tmp_path / "refused")
    assert not (tmp_path / "refused/runtime.json").exists()


@pytest.mark.parametrize("steps,seed", [(0, 1), (True, 1), (4, -1), (4, True)])
def test_invalid_configuration(config, model, tmp_path, steps, seed):
    with pytest.raises(GNNAdmissionError, match="GNN_BACKEND_CONFIG"):
        run_pymdp(model, replace(config, steps=steps, seed=seed), tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize(
    "file,field,value",
    [
        ("request.json", "lock", {}),
        ("runtime.json", "success", False),
        ("runtime.json", "dtype", "float64"),
        ("simulation_results.json", "framework", "native"),
        ("simulation_results.json", "num_timesteps", 99),
        ("simulation_results.json", "runtime_metadata", {"random_seed": 99}),
        ("simulation_results.json", "beliefs", ["not numeric"]),
    ],
)
def test_runtime_metadata_rejected(file, field, value, config, model, executed, tmp_path):
    output = tmp_path / "bad-runtime"
    shutil.copytree(executed[0], output)
    path = output / file
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(GNNAdmissionError):
        verify_backend_output(model, config, output)


@pytest.mark.parametrize(
    "field,value",
    [
        ("revision", "changed"),
        ("source_files", {}),
        ("source_files", {"../escape.py": "0" * 64}),
        ("source_files", {"src/gnn/__init__.py": "0" * 64}),
    ],
)
def test_lock_refusal(field, value, config, model, tmp_path):
    data = json.loads(config.lock_path.read_text())
    data[field] = value
    lock = tmp_path / "bad-lock.json"
    lock.write_text(json.dumps(data))
    with pytest.raises(GNNAdmissionError):
        run_pymdp(model, replace(config, lock_path=lock), tmp_path / "bad-lock-run")


def test_closed_copy_ignores_unbound_source(config, model, executed, tmp_path):
    root = tmp_path / "copy"
    for relative in json.loads(config.lock_path.read_text())["source_files"]:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(config.upstream_root / relative, target)
    (root / "src/psutil.py").write_text('raise RuntimeError("unbound shadow executed")')
    command = "import sys; sys.path.insert(0, sys.argv[1]); import psutil"
    probe = subprocess.run(
        [str(config.python), "-I", "-S", "-B", "-c", command, str(root / "src")],
        capture_output=True,
        text=True,
    )
    assert probe.returncode and "unbound shadow executed" in probe.stderr
    result = run_pymdp(model, replace(config, upstream_root=root), tmp_path / "closed-run")
    assert result.trace_digest == executed[1].trace_digest


def test_model_and_request_binding(config, model, executed, tmp_path):
    output = tmp_path / "binding"
    shutil.copytree(executed[0], output)
    changed = replace(
        model, categorical=replace(model.categorical, C=Tensor(("observation",), (2,), (10.0, 20.0)))
    )
    with pytest.raises(GNNAdmissionError, match="GNN_BACKEND_BINDING"):
        verify_backend_output(changed, config, output)
    request = output / "request.json"
    request.write_text(request.read_text() + " ")
    with pytest.raises(GNNAdmissionError, match="GNN_BACKEND_BINDING"):
        verify_backend_output(model, config, output)


def test_startup_ignores_executable_pth(config, model, executed, tmp_path):
    env = tmp_path / "runtime"
    (env / "bin").mkdir(parents=True)
    (env / "bin/python").symlink_to(config.python.resolve())
    shutil.copyfile(config.python.parent.parent / "pyvenv.cfg", env / "pyvenv.cfg")
    version = ".".join(json.loads((executed[0] / "runtime.json").read_text())["python"].split(".")[:2])
    site = env / "lib" / ("python" + version) / "site-packages"
    site.mkdir(parents=True)
    original_site = config.python.parent.parent / "lib" / ("python" + version) / "site-packages"
    for path in original_site.iterdir():
        (site / path.name).symlink_to(path)
    marker = tmp_path / "pth-executed"
    (site / "zz_gnn_probe.pth").write_text(
        f'import pathlib; pathlib.Path({str(marker)!r}).open("a").write("x")\n'
    )
    probe = subprocess.run(
        [str(env / "bin/python"), "-I", "-B", "-c", "print('startup positive control')"],
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr
    before = marker.read_bytes()
    result = run_pymdp(model, replace(config, python=env / "bin/python"), tmp_path / "isolated")
    assert marker.read_bytes() == before
    assert result.trace_digest == executed[1].trace_digest


@pytest.mark.parametrize("change", ["A", "B", "C", "D", "name", "seed", "steps", "gamma"])
def test_rehashed_request_spec_cannot_claim_original_model(change, config, model, executed, tmp_path):
    import hashlib

    output = tmp_path / "rehashed-request"
    shutil.copytree(executed[0], output)
    path = output / "request.json"
    request = json.loads(path.read_text())
    spec = request["spec"]
    matrices = spec["initialparameterization"]
    if change == "A":
        matrices["A"][0][0], matrices["A"][1][0] = 0.7, 0.3
    elif change == "B":
        matrices["B"][0][0][0], matrices["B"][1][0][0] = 0.8, 0.2
    elif change == "C":
        matrices["C"][1] = 2.0
    elif change == "D":
        matrices["D"] = [0.5, 0.5]
    elif change == "name":
        spec["model_name"] += " changed"
    else:
        field = {"seed": "random_seed", "steps": "num_timesteps", "gamma": "gamma"}[change]
        spec["model_parameters"][field] += 1
    path.write_text(json.dumps(request))
    runtime_path = output / "runtime.json"
    runtime = json.loads(runtime_path.read_text())
    runtime["request_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    runtime_path.write_text(json.dumps(runtime))
    with pytest.raises(GNNAdmissionError, match="GNN_BACKEND_BINDING"):
        verify_backend_output(model, config, output)


@pytest.mark.parametrize("field", ["beliefs", "policy_posterior", "expected_free_energy"])
def test_numeric_strings_are_not_backend_numbers(field, config, model, executed, tmp_path):
    output = tmp_path / "string-numbers"
    shutil.copytree(executed[0], output)
    path = output / "simulation_results.json"
    data = json.loads(path.read_text())
    data[field] = [[str(value) for value in row] for row in data[field]]
    path.write_text(json.dumps(data))
    with pytest.raises(GNNAdmissionError, match="GNN_BACKEND_OUTPUT"):
        verify_backend_output(model, config, output)


def test_unrepresentable_backend_number_has_typed_refusal(config, model, executed, tmp_path):
    output = tmp_path / "overflow-number"
    shutil.copytree(executed[0], output)
    path = output / "simulation_results.json"
    data = json.loads(path.read_text())
    data["beliefs"][0][0] = 10 ** 400
    path.write_text(json.dumps(data))
    with pytest.raises(GNNAdmissionError, match="GNN_BACKEND_OUTPUT"):
        verify_backend_output(model, config, output)



@pytest.mark.parametrize(
    "fault",
    ["old-schema", "empty-profiles", "bad-python", "bad-distributions",
     "bad-release", "boolean-common", "duplicate-runtime", "unknown-field", "qualification"],
)
def test_runtime_registry_refuses_invalid_profiles(config, model, tmp_path, fault):
    data = json.loads(config.lock_path.read_text())
    name = next(iter(data["profiles"]))
    if fault == "old-schema":
        data["schema"] = "fedference.gnn-backend/1"
    elif fault == "empty-profiles":
        data["profiles"] = {}
    elif fault == "bad-python":
        data["profiles"][name]["python"] = True
    elif fault == "bad-distributions":
        data["profiles"][name]["distributions"] = []
    elif fault == "bad-release":
        package = next(iter(data["profiles"][name]["distributions"]))
        data["profiles"][name]["distributions"][package] = 1
    elif fault == "boolean-common":
        data["batch_size"] = True
    elif fault == "duplicate-runtime":
        data["profiles"]["duplicate"] = data["profiles"][name]
    elif fault == "qualification":
        data["profiles"][name]["qualification"] = "unqualified"
    else:
        data["unrecognized"] = True
    lock = tmp_path / "registry.json"
    lock.write_text(json.dumps(data))
    output = tmp_path / "refused"
    with pytest.raises(GNNAdmissionError, match="GNN_BACKEND_LOCK"):
        run_pymdp(model, replace(config, lock_path=lock), output)
    assert not output.exists()


@pytest.mark.parametrize("field,value", [("profile", "unknown"), ("batch_size", True)])
def test_runtime_profile_metadata_refuses_forgery(config, model, executed, tmp_path, field, value):
    output = tmp_path / "forged-profile"
    shutil.copytree(executed[0], output)
    runtime_path = output / "runtime.json"
    runtime = json.loads(runtime_path.read_text())
    runtime[field] = value
    runtime_path.write_text(json.dumps(runtime))
    with pytest.raises(GNNAdmissionError, match="GNN_BACKEND_ENVIRONMENT"):
        verify_backend_output(model, config, output)


def test_registry_rejects_untyped_source_hash_before_execution(tmp_path: Path) -> None:
    from fedference.gnn.backend import load_backend_registry

    lock = BackendConfig(Path("."), Path(".")).lock_path
    raw = json.loads(lock.read_text())
    raw["source_files"][next(iter(raw["source_files"]))] = 1
    path = tmp_path / "invalid-registry.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(GNNAdmissionError, match="GNN_BACKEND_LOCK"):
        load_backend_registry(path)
