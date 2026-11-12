"""Complete CI partitioning and fail-closed execution evidence; no mocks."""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.ci_shards import digest, partition, validate_plan, verify_results, verify_shard_artifacts


def _plan() -> dict:
    nodes = ["tests/a.py::test_one", "tests/a.py::test_two", "tests/b.py::test_one"]
    return {"nodes": nodes, "groups": partition(nodes, {"tests/a.py": 10}, 2),
            "collection_digest": digest(nodes)}


def _results(plan: dict) -> list[dict]:
    return [{"index": index, "plan_digest": digest(plan), "exit_code": 0,
             "finished": group.copy(), "skipped": []}
            for index, group in enumerate(plan["groups"])]


def test_balanced_partition_preserves_modules_and_is_deterministic() -> None:
    nodes = [f"tests/{name}.py::test_{i}" for name in "abcd" for i in range(2)]
    groups = partition(nodes, {"tests/a.py": 9, "tests/b.py": 8, "tests/c.py": 2,
                               "tests/d.py": 1}, 2)
    assert groups == partition(list(reversed(nodes)), {"tests/a.py": 9, "tests/b.py": 8,
                                                     "tests/c.py": 2, "tests/d.py": 1}, 2)
    assert {node.split("::")[0] for node in groups[0]} == {"tests/a.py", "tests/d.py"}
    assert sorted(node for group in groups for node in group) == sorted(nodes)


@pytest.mark.parametrize("weights,count", [({}, 0), ({}, 3), ({"tests/a.py": -1}, 2),
                                          ({"tests/a.py": float("nan")}, 2)])
def test_invalid_scheduling_inputs_rejected(weights: dict, count: int) -> None:
    with pytest.raises(ValueError):
        partition(_plan()["nodes"], weights, count)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "foreign", "split", "digest"])
def test_corrupt_plan_rejected(fault: str) -> None:
    plan = _plan()
    if fault == "missing":
        plan["groups"][0].pop()
    elif fault == "duplicate":
        plan["groups"][0].append(plan["groups"][0][0])
    elif fault == "foreign":
        plan["groups"][0][0] = "tests/foreign.py::test_bad"
    elif fault == "split":
        plan["groups"][1].append(plan["groups"][0].pop())
    else:
        plan["collection_digest"] = "invalid"
    with pytest.raises(ValueError):
        validate_plan(plan)


@pytest.mark.parametrize("fault", ["missing_shard", "duplicate_shard", "failure", "digest",
                                   "missing_test", "duplicate_test", "skip"])
def test_incomplete_or_unsuccessful_results_rejected(fault: str) -> None:
    plan = _plan()
    results = copy.deepcopy(_results(plan))
    if fault == "missing_shard":
        results.pop()
    elif fault == "duplicate_shard":
        results.append(results[0])
    elif fault == "failure":
        results[0]["exit_code"] = 1
    elif fault == "digest":
        results[0]["plan_digest"] = "foreign"
    elif fault == "missing_test":
        results[0]["finished"].pop()
    elif fault == "duplicate_test":
        results[0]["finished"].append(results[0]["finished"][0])
    else:
        results[0]["skipped"] = [results[0]["finished"][0]]
    with pytest.raises(ValueError):
        verify_results(plan, results)


def test_valid_results_accepted_in_any_arrival_order() -> None:
    plan = _plan()
    verify_results(plan, list(reversed(_results(plan))))


def test_real_pytest_plugin_selects_only_its_module_and_records_execution(tmp_path: Path) -> None:
    tests = tmp_path / "tests"
    tests.mkdir()
    for name in "ab":
        (tests / f"test_{name}.py").write_text("def test_one():\n    assert 2 + 2 == 4\n")
    script_dir = Path(__file__).resolve().parents[1] / "scripts"
    driver = tmp_path / "driver.py"
    driver.write_text(
        "import sys, json, pytest\n"
        f"sys.path.insert(0, {str(script_dir)!r})\n"
        "from ci_shards import Inventory, digest, verify_results\n"
        "nodes=['tests/test_a.py::test_one','tests/test_b.py::test_one']\n"
        "plan={'nodes':nodes,'groups':[[nodes[0]],[nodes[1]]], 'collection_digest':digest(nodes)}\n"
        "p=Inventory(plan, 1)\n"
        "code=pytest.main(['tests','-q'], plugins=[p])\n"
        "open('result.json','w').write(json.dumps({'code':int(code),'finished':p.finished,'nodes':p.nodes}))\n"
        "raise SystemExit(code)\n"
    )
    result = subprocess.run([sys.executable, str(driver)], cwd=tmp_path, capture_output=True,
                            text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads((tmp_path / "result.json").read_text())
    assert record["finished"] == ["tests/test_b.py::test_one"]
    assert len(record["nodes"]) == 2



@pytest.fixture(scope="module")
def bounded_shard_artifacts(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    """Real tiny pytest/coverage runs; these are not production-suite receipts."""
    root = tmp_path_factory.mktemp("bounded-shard-control")
    (root / "src").mkdir()
    (root / "tests").mkdir()
    (root / "src/arithmetic_sample.py").write_text(
        "def add(a, b):\n    return a + b\ndef multiply(a, b):\n    return a * b\n"
    )
    (root / "tests/test_a.py").write_text(
        "from arithmetic_sample import add\ndef test_add():\n    assert add(2, 3) == 5\n"
    )
    (root / "tests/test_b.py").write_text(
        "import os,pytest\nfrom arithmetic_sample import multiply\n"
        "def test_multiply():\n"
        "    if os.environ.get('SKIP_SAMPLE'):\n        pytest.skip('negative skip control')\n"
        "    assert multiply(2, 3) == 6\n"
    )
    nodes = ["tests/test_a.py::test_add", "tests/test_b.py::test_multiply"]
    plan = {"nodes": nodes, "groups": [[nodes[0]], [nodes[1]]],
            "collection_digest": digest(nodes), "scope": "bounded-runner-control"}
    (root / "plan.json").write_text(json.dumps(plan))
    source = Path(__file__).resolve().parents[1]
    (root / "driver.py").write_text(
        "import json,os,sys,pytest\nfrom pathlib import Path\n"
        f"sys.path.insert(0, {str(source)!r})\n"
        "sys.path.insert(0,str(Path.cwd()/'src'))\n"
        "from scripts.ci_shards import Inventory,record_shard_result\n"
        "plan=json.loads(Path('plan.json').read_text())\n"
        "index=int(sys.argv[1]); directory=Path(sys.argv[2])/f'shard-{index}'\n"
        "directory.mkdir(parents=True)\n"
        "os.environ['COVERAGE_FILE']=str((directory/f'.coverage.shard-{index}').resolve())\n"
        "inventory=Inventory(plan,index)\n"
        "status=pytest.main(['tests','--cov=src','--cov-fail-under=0','-q'],plugins=[inventory])\n"
        "record_shard_result(plan,index,directory,inventory,int(status))\n"
        "raise SystemExit(status)\n"
    )
    for index in range(2):
        command = [sys.executable, str(root / "driver.py"), str(index), str(root / "results")]
        result = subprocess.run(command, cwd=root, capture_output=True, text=True,
                                timeout=60, check=False)
        (root / f"process-{index}.json").write_text(json.dumps({
            "argv": command, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr,
        }, indent=2))
        assert result.returncode == 0, result.stdout + result.stderr
    return root, plan


def test_real_receipts_bind_complete_fixture_execution(bounded_shard_artifacts: tuple[Path, dict]) -> None:
    # Given two genuine fixture shard runs and coverage databases.
    root, plan = bounded_shard_artifacts
    # When the production artifact verifier consumes their receipts.
    verify_shard_artifacts(plan, root / "results")
    # Then the recorded nodes are exactly the bounded fixture's universe.
    observed = [json.loads(path.read_text())["finished"]
                for path in sorted((root / "results").glob("shard-*/receipt.json"))]
    assert observed == plan["groups"]
    environment = dict(os.environ, COVERAGE_FILE=str(root / ".coverage"))
    commands = [
        [sys.executable, "-m", "coverage", "combine", "--keep",
         str(root / "results/shard-0"), str(root / "results/shard-1")],
        [sys.executable, "-m", "coverage", "report", "--fail-under=90"],
        [sys.executable, "-m", "coverage", "json", "-o", str(root / "combined.json")],
    ]
    for index, command in enumerate(commands):
        result = subprocess.run(command, cwd=root, env=environment, capture_output=True,
                                text=True, timeout=60, check=False)
        (root / f"combine-{index}.json").write_text(json.dumps({
            "argv": command, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr,
        }, indent=2))
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("fault", ["missing", "duplicate", "coverage", "skipped"])
def test_real_artifact_verifier_refuses_incomplete_evidence(
    bounded_shard_artifacts: tuple[Path, dict], tmp_path: Path, fault: str,
) -> None:
    # Given actual fixture receipts and their original coverage bytes.
    root, plan = bounded_shard_artifacts
    target = tmp_path / "results"
    shutil.copytree(root / "results", target)
    # When a required artifact is withheld, duplicated, altered, or a test truly skips.
    if fault == "missing":
        (target / "shard-1").rename(target / "withheld-shard-1")
    elif fault == "duplicate":
        shutil.copytree(target / "shard-1", target / "shard-extra")
    elif fault == "coverage":
        with (target / "shard-1/.coverage.shard-1").open("ab") as stream:
            stream.write(b"tampered")
    else:
        (target / "shard-1").rename(target / "original-shard-1")
        environment = dict(os.environ, SKIP_SAMPLE="1")
        command = [sys.executable, str(root / "driver.py"), "1", str(target)]
        result = subprocess.run(command, cwd=root, env=environment, capture_output=True,
                                text=True, timeout=60, check=False)
        (tmp_path / "skip-process.json").write_text(json.dumps({
            "argv": command, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr,
        }, indent=2))
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads((target / "shard-1/receipt.json").read_text())["skipped"]
    # Then green subprocess status cannot substitute for complete evidence.
    with pytest.raises(ValueError):
        verify_shard_artifacts(plan, target)



@pytest.mark.parametrize("mode", ["run", "verify"])
@pytest.mark.parametrize("fault", ["missing", "duplicate", "valid"])
def test_cli_admits_plan_before_git_discovery(tmp_path: Path, mode: str, fault: str) -> None:
    """Malformed plans retain their diagnostics in a real Git-free export."""
    plan = _plan()
    if fault == "missing":
        plan["groups"][0].pop()
    elif fault == "duplicate":
        plan["groups"][0].append(plan["groups"][0][0])
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    results = tmp_path / "results"
    environment = dict(os.environ, GIT_CEILING_DIRECTORIES=str(tmp_path.parent))
    for selector in ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR"):
        environment.pop(selector, None)
    source = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(source / "scripts/ci_shards.py"), mode,
               "--project-root", str(tmp_path), "--plan", str(plan_path),
               "--results", str(results)]
    result = subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True,
                            text=True, timeout=60, check=False)
    assert result.returncode != 0, result.stdout + result.stderr
    if fault == "valid":
        assert "not a git repository" in result.stderr
    else:
        assert "Shard union differs from complete test collection" in result.stderr
        assert "not a git repository" not in result.stderr
    assert not results.exists()



@pytest.mark.parametrize("valid", [False, True])
def test_hosted_gnn_checkout_uses_admitted_lock_revision(tmp_path: Path, valid: bool) -> None:
    """Execute the actual hosted admission script; a moving ref cannot pass."""
    import yaml

    source = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load((source / ".github/workflows/ci.yml").read_text())
    steps = workflow["jobs"]["gnn-shard"]["steps"]
    admission = next(step for step in steps if step.get("id") == "gnn-source")
    checkout = next(step for step in steps if step.get("with", {}).get("repository")
                    == "ActiveInferenceInstitute/Generalized_Notation_Notation")
    assert checkout["with"]["ref"] == "${{ steps.gnn-source.outputs.revision }}"
    pin = json.loads((source / ".github/gnn-pin.json").read_text())
    fixture = tmp_path / ".github/gnn-pin.json"
    fixture.parent.mkdir(parents=True)
    revision = pin["commit"] if valid else "main"
    fixture.write_text(json.dumps({"schema": "gnn-importer-pin/v1", "commit": revision}))
    output = tmp_path / "github-output"
    environment = dict(os.environ, GITHUB_OUTPUT=str(output))
    environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment["PATH"]
    result = subprocess.run(["bash", "-eu", "-c", admission["run"]], cwd=tmp_path,
                            env=environment, capture_output=True, text=True, timeout=30, check=False)
    if valid:
        assert result.returncode == 0, result.stdout + result.stderr
        assert output.read_text() == f"revision={revision}\n"
    else:
        assert result.returncode != 0
        assert "GNN pin must declare gnn-importer-pin/v1 and an exact commit SHA" in result.stderr
        assert not output.exists()
