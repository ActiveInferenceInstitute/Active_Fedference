"""Real child-process profile controls and exact required-shard assignment."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.ci_shards import digest, validate_plan, verify_results
from scripts.validation_profiles import GNN_MODULES

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("profile", ["gnn", "full"])
def test_required_profile_refuses_missing_runtime(tmp_path: Path, profile: str) -> None:
    # Given no caller-selected real upstream runtime.
    environment = dict(os.environ)
    for name in ("GNN_ROOT", "GNN_PARSER_DEPENDENCIES", "PYTEST_ADDOPTS"):
        environment.pop(name, None)
    command = [sys.executable, "-m", "pytest", "-c", str(ROOT / "pyproject.toml"),
               f"--rootdir={ROOT}", "tests/integration/test_gnn_pymdp.py",
               f"--validation-profile={profile}", "-o", f"cache_dir={tmp_path / 'cache'}", "-q"]
    # When an explicit required profile is executed.
    result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True,
                            text=True, timeout=120, check=False)
    (tmp_path / "process.json").write_text(json.dumps({
        "argv": command, "returncode": result.returncode,
        "stdout": result.stdout, "stderr": result.stderr,
    }, indent=2))
    # Then execution refuses before fixture skips or a KeyError.
    assert result.returncode == 4, result.stdout + result.stderr
    assert "GNN_PROFILE_RUNTIME_REQUIRED" in result.stderr
    assert "KeyError" not in result.stdout


def test_native_profile_avoids_backend_import(tmp_path: Path) -> None:
    # Given a native process with no upstream configuration.
    driver = tmp_path / "driver.py"
    cache = repr("cache_dir=" + str(tmp_path / "cache"))
    driver.write_text(
        "import sys,pytest\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        f"code=pytest.main(['tests/','--collect-only','-q','-o',{cache}])\n"
        "assert not {'gnn','pymdp','fedference.gnn.backend'} & sys.modules.keys()\n"
        "raise SystemExit(code)\n"
    )
    environment = dict(os.environ)
    for name in ("GNN_ROOT", "GNN_PARSER_DEPENDENCIES", "PYTEST_ADDOPTS"):
        environment.pop(name, None)
    # When the real process collects the entire default profile.
    result = subprocess.run([sys.executable, str(driver)], cwd=ROOT, env=environment,
                            capture_output=True, text=True, timeout=120, check=False)
    (tmp_path / "native.json").write_text(json.dumps({
        "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
    }, indent=2))
    # Then the complete native inventory collects without importing the external backend.
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("fault", ["missing", "duplicate", "wrong", "missing-receipt", "deleted-metadata"])
def test_profile_union_rejects_incomplete_or_misassigned_execution(fault: str) -> None:
    # Given a plan whose required profiles cover every node exactly once.
    native = "tests/test_native.py::test_native"
    gnn = sorted(GNN_MODULES)[0] + "::test_real"
    nodes = sorted([native, gnn])
    plan = {"schema": 2, "nodes": nodes, "groups": [[native], [gnn]],
            "profiles": ["native", "gnn"], "collection_digest": digest(nodes)}
    validate_plan(plan)
    broken = copy.deepcopy(plan)
    # When profile metadata or a required receipt is lost or corrupted.
    if fault == "missing":
        broken["profiles"].pop()
    elif fault == "duplicate":
        broken["profiles"] = ["native", "native"]
    elif fault == "wrong":
        broken["profiles"] = ["gnn", "native"]
    elif fault == "deleted-metadata":
        del broken["profiles"]
    else:
        receipts = [{"index": 0, "plan_digest": digest(plan), "exit_code": 0,
                     "finished": [native], "skipped": []}]
        # Then a green native-only result cannot fulfill the required union.
        with pytest.raises(ValueError):
            verify_results(plan, receipts)
        return
    # Then no malformed assignment is accepted.
    with pytest.raises(ValueError):
        validate_plan(broken)



@pytest.mark.parametrize("fault", ["existing-destination", "source-drift"])
def test_bootstrap_refuses_before_runtime_install(tmp_path: Path, fault: str) -> None:
    # Given a protected destination or a source tree outside the pinned closure.
    destination = tmp_path / "runtime"
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    if fault == "existing-destination":
        destination.mkdir()
        (destination / "keep.txt").write_text("keep")
    if fault == "source-drift":
        registry = json.loads((ROOT / "src/fedference/config/gnn-backend-lock.json").read_text())
        first = next(iter(registry["source_files"]))
        altered = upstream / first
        altered.parent.mkdir(parents=True)
        altered.write_text("altered pinned source")
    command = [sys.executable, str(ROOT / "scripts/bootstrap_gnn.py"),
               "--upstream-root", str(upstream), "--destination", str(destination)]
    # When bootstrap is invoked on the actual subprocess boundary.
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                            timeout=60, check=False)
    (tmp_path / "bootstrap-refusal.json").write_text(json.dumps({
        "argv": command, "returncode": result.returncode,
        "stdout": result.stdout, "stderr": result.stderr,
    }, indent=2))
    # Then no environment is installed and protected input survives.
    assert result.returncode != 0
    assert not (destination / ".venv").exists()
    if fault == "existing-destination":
        assert (destination / "keep.txt").read_text() == "keep"
    else:
        assert not destination.exists()
        assert "Pinned upstream source differs" in result.stderr


@pytest.mark.parametrize("revision", ["main", "0" * 40])
def test_bootstrap_uses_the_same_pin_reader_before_provisioning(tmp_path: Path, revision: str) -> None:
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github/gnn-pin.json").write_text(json.dumps({
        "schema": "gnn-importer-pin/v1", "commit": revision,
    }))
    destination = tmp_path / "runtime"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/bootstrap_gnn.py"),
         "--project-root", str(tmp_path),
         "--lock", str(ROOT / "src/fedference/config/gnn-backend-lock.json"),
         "--upstream-root", str(tmp_path), "--destination", str(destination)],
        cwd=ROOT, capture_output=True, text=True, timeout=60, check=False,
    )
    assert result.returncode != 0
    assert not destination.exists()
    expected = (
        "GNN pin must declare gnn-importer-pin/v1 and an exact commit SHA"
        if revision == "main" else "GNN pin revision differs from the qualified backend lock"
    )
    assert expected in result.stderr



@pytest.mark.parametrize("fault", ["missing", "duplicate"])
def test_shard_cli_rejects_missing_or_duplicate_profiles(tmp_path: Path, fault: str) -> None:
    # Given an otherwise exact node partition with corrupted required-profile metadata.
    nodes = ["tests/test_native.py::test", sorted(GNN_MODULES)[0] + "::test"]
    plan = {"schema": 2, "nodes": sorted(nodes), "groups": [[nodes[0]], [nodes[1]]],
            "collection_digest": digest(sorted(nodes))}
    if fault == "duplicate":
        plan["profiles"] = ["native", "native"]
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    output = tmp_path / "results"
    command = [sys.executable, str(ROOT / "scripts/ci_shards.py"), "run",
               "--project-root", str(ROOT), "--plan", str(path), "--results", str(output)]
    # When the actual shard command consumes the malformed plan.
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True,
                            timeout=60, check=False)
    (tmp_path / "profile-refusal.json").write_text(json.dumps({
        "argv": command, "returncode": result.returncode,
        "stdout": result.stdout, "stderr": result.stderr,
    }, indent=2))
    # Then refusal precedes test execution and output creation.
    assert result.returncode != 0
    assert "validation profile" in result.stderr
    assert not output.exists()
