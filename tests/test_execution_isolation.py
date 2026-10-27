"""Real-process regression probes for test roots and complete CI inventories."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(arguments: list[str], cwd: Path, environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(arguments, cwd=cwd, env=environment, capture_output=True,
                               text=True, stdin=subprocess.DEVNULL, timeout=120, check=False)
    (cwd / f"subprocess-{uuid.uuid4().hex}.json").write_text(json.dumps({
        "argv": arguments, "cwd": str(cwd), "returncode": completed.returncode,
        "stdout": completed.stdout, "stderr": completed.stderr,
    }, indent=2))
    return completed


def test_outside_checkout_and_complete_shard_inventory(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    identity = _run([sys.executable, "-c", "import json,sys,fedference; "
                     "print(json.dumps({'prefix':sys.prefix,'base':sys.base_prefix,"
                     "'module':fedference.__file__}))"], tmp_path, environment)
    assert identity.returncode == 0, identity.stdout + identity.stderr
    observed = json.loads(identity.stdout)
    assert observed["prefix"] == sys.prefix
    assert observed["base"] == sys.base_prefix
    assert Path(observed["module"]).resolve().is_relative_to(ROOT / "src")

    ancestor = tmp_path / "ancestor"
    ancestor.mkdir()
    config = ancestor / "pytest.ini"
    config.write_text("[pytest]\naddopts = --ancestor-option-must-not-be-used\n")
    environment["PYTEST_ADDOPTS"] = f"-c {config} --rootdir={ancestor}"
    plan_path = tmp_path / "plan.json"
    planned = _run([sys.executable, str(ROOT / "scripts" / "ci_shards.py"), "plan",
                    "--project-root", str(ROOT), "--plan", str(plan_path)], ancestor, environment)
    assert planned.returncode == 0, planned.stdout + planned.stderr
    plan = json.loads(plan_path.read_text())
    collected = _run([sys.executable, "-m", "pytest", "-c", str(ROOT / "pyproject.toml"),
                      f"--rootdir={ROOT}", str(ROOT / "tests"), "--collect-only", "-qq"],
                     ancestor, environment)
    assert collected.returncode == 0, collected.stdout + collected.stderr
    nodes = [line for line in collected.stdout.splitlines() if line.startswith("tests/") and "::" in line]
    assert nodes
    assert Counter(plan["nodes"]) == Counter(nodes)
    assert Counter(node for group in plan["groups"] for node in group) == Counter(nodes)
    assert all(count == 1 for count in Counter(nodes).values())


def test_ancestor_git_and_duplicate_or_missing_shard(tmp_path: Path) -> None:
    ancestor = tmp_path / "workspace"
    ancestor.mkdir()
    environment = dict(os.environ)
    for key in ("PYTEST_ADDOPTS", "GIT_CEILING_DIRECTORIES", "UV_PYTHON", "PYTHONPATH"):
        environment.pop(key, None)
    initialized = _run(["git", "init", "-q", str(ancestor)], tmp_path, environment)
    assert initialized.returncode == 0, initialized.stderr
    (ancestor / "pytest.ini").write_text("[pytest]\naddopts = --ancestor-option-must-not-be-used\n")
    suite = ancestor / "suite"
    tests = suite / "tests"
    tests.mkdir(parents=True)
    (suite / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    (suite / "isolation_probe.py").write_text("VALUE = 42\n")
    shutil.copyfile(ROOT / "tests" / "conftest.py", tests / "conftest.py")
    (tests / "test_child.py").write_text(
        "import json,os,pathlib,subprocess,sys\n"
        "from isolation_probe import VALUE\n"
        "def test_child(tmp_path,pytestconfig):\n"
        "    assert VALUE == 42\n"
        "    git=subprocess.run(['git','rev-parse','--show-toplevel'],"
        "cwd=tmp_path,capture_output=True,text=True)\n"
        "    assert git.returncode != 0, git.stdout\n"
        "    (tmp_path/'test_tiny.py').write_text('def test_ok(): assert 2+2 == 4\\n')\n"
        "    child=subprocess.run([sys.executable,'-m','pytest','test_tiny.py','-q'],"
        "cwd=tmp_path,capture_output=True,text=True)\n"
        "    assert child.returncode == 0, child.stdout+child.stderr\n"
        "    assert os.environ['UV_PYTHON'] == sys.executable\n"
        "    pathlib.Path('identity.json').write_text(json.dumps({'prefix':sys.prefix,"
        "'root':str(pytestconfig.rootpath),'git_status':git.returncode,"
        "'pytest_status':child.returncode}))\n"
    )
    child = _run([sys.executable, "-m", "pytest", "-c", str(suite / "pyproject.toml"),
                  f"--rootdir={suite}", "--basetemp", str(ancestor / "scratch"),
                  str(tests), "-q"], ancestor, environment)
    assert child.returncode == 0, child.stdout + child.stderr
    identity = json.loads((ancestor / "identity.json").read_text())
    assert identity["prefix"] == sys.prefix
    assert identity["root"] == str(suite)
    assert identity["git_status"] != 0
    assert identity["pytest_status"] == 0

    # Drive the public shard CLI with bad plans; rejection precedes execution/output.
    nodes = ["tests/test_a.py::test_one", "tests/test_b.py::test_two"]
    for fault, groups in (("missing", [[nodes[0]]]), ("duplicate", [nodes, [nodes[0]]])):
        plan_path = tmp_path / f"{fault}.json"
        plan_path.write_text(json.dumps({"nodes": nodes, "groups": groups}))
        results = tmp_path / f"{fault}-results"
        failed = _run([sys.executable, str(ROOT / "scripts" / "ci_shards.py"), "run",
                       "--project-root", str(ROOT), "--plan", str(plan_path),
                       "--results", str(results)], ancestor, environment)
        assert failed.returncode != 0
        assert "Shard union differs from complete test collection" in failed.stderr
        assert not results.exists()
