"""Complete CI partitioning and fail-closed execution evidence; no mocks."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.ci_shards import digest, partition, validate_plan, verify_results


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
