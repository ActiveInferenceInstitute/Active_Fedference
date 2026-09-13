#!/usr/bin/env python3
"""Plan, execute and verify complete module-preserving CI test partitions.

Scheduling weights affect placement only. Every invocation collects the current
suite, and verification rejects missing, duplicated or foreign test results.
Scientific computation and assertions remain owned by the unchanged tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pytest


def digest(value: Any) -> str:
    """Hash a canonical JSON value for cross-job binding."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def checkout_digest() -> str:
    """Bind declared validation inputs, including a dirty local candidate."""
    # Do not import src/ before pytest-cov starts in the test process: doing so
    # would make import-time statements disappear from measured coverage.
    command = [sys.executable, "-c",
               "import sys,json; from pathlib import Path; sys.path.insert(0,sys.argv[1]); "
               "from publication.validation_receipt import validation_input_hashes; "
               "print(json.dumps(validation_input_hashes(Path.cwd())))",
               str(Path(__file__).resolve().parents[1] / "src")]
    return digest(json.loads(subprocess.check_output(command, text=True)))


def partition(nodes: list[str], weights: dict[str, float], count: int) -> list[list[str]]:
    """Greedily balance whole modules; never split a fixture's module."""
    if len(nodes) != len(set(nodes)) or not nodes or count < 1:
        raise ValueError("Expected nonempty unique test IDs and a positive shard count")
    modules: dict[str, list[str]] = {}
    for node in nodes:
        modules.setdefault(node.split("::", 1)[0], []).append(node)
    if count > len(modules):
        raise ValueError("More shards than test modules")
    costs = {module: float(weights.get(module, len(items))) for module, items in modules.items()}
    if any(not math.isfinite(value) or value < 0 for value in costs.values()):
        raise ValueError("Scheduling costs must be finite and nonnegative")
    groups: list[list[str]] = [[] for _ in range(count)]
    loads = [0.0] * count
    for module in sorted(modules, key=lambda name: (-costs[name], name)):
        index = min(range(count), key=lambda i: (loads[i], len(groups[i]), i))
        groups[index].extend(modules[module])
        loads[index] += costs[module]
    return [sorted(group) for group in groups]


def validate_plan(plan: dict[str, Any]) -> None:
    """Reject corrupt partitions before they can select tests."""
    nodes, groups = plan["nodes"], plan["groups"]
    if not nodes or len(nodes) != len(set(nodes)) or not groups or not all(groups):
        raise ValueError("Empty or duplicate plan inventory")
    if Counter(node for group in groups for node in group) != Counter(nodes):
        raise ValueError("Shard union differs from complete test collection")
    owners: dict[str, int] = {}
    for index, group in enumerate(groups):
        for node in group:
            module = node.split("::", 1)[0]
            if owners.setdefault(module, index) != index:
                raise ValueError("A module spans multiple shards")
    if plan["collection_digest"] != digest(nodes):
        raise ValueError("Collection digest differs")


class Inventory:
    """Use pytest's actual collection and lifecycle, including setup failures."""

    def __init__(self, plan: dict[str, Any] | None = None, index: int = 0) -> None:
        self.nodes: list[str] = []
        self.finished: list[str] = []
        self.skipped: list[str] = []
        self.plan, self.index = plan, index

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, config: Any, items: list[Any]) -> None:
        """Check the complete inventory before selecting this shard's tests."""
        self.nodes = sorted(item.nodeid for item in items)
        if self.plan is None:
            return
        if self.nodes != self.plan["nodes"]:
            missing = sorted(set(self.plan["nodes"]) - set(self.nodes))
            extra = sorted(set(self.nodes) - set(self.plan["nodes"]))
            raise pytest.UsageError(
                f"Current collection differs from plan: missing={missing!r}; extra={extra!r}"
            )
        selected = set(self.plan["groups"][self.index])
        rejected = [item for item in items if item.nodeid not in selected]
        items[:] = [item for item in items if item.nodeid in selected]
        config.hook.pytest_deselected(items=rejected)

    def pytest_runtest_logfinish(self, nodeid: str) -> None:
        """Record completion even when setup or the test itself has failed."""
        self.finished.append(nodeid)

    def pytest_runtest_logreport(self, report: Any) -> None:
        """Retain skips so the complete-execution verifier can reject them."""
        if report.skipped:
            self.skipped.append(report.nodeid)


def verify_results(plan: dict[str, Any], results: list[dict[str, Any]]) -> None:
    """Require every planned test exactly once and every shard successful."""
    validate_plan(plan)
    if sorted(result["index"] for result in results) != list(range(len(plan["groups"]))):
        raise ValueError("Missing, duplicate or foreign shard receipts")
    for result in results:
        expected = plan["groups"][result["index"]]
        if result["plan_digest"] != digest(plan) or result["exit_code"] != 0:
            raise ValueError("Unsuccessful or mismatched shard receipt")
        if Counter(result["finished"]) != Counter(expected) or result["skipped"]:
            raise ValueError("Missing, duplicated or skipped test execution")


def main() -> int:
    """Execute one explicit planning, shard execution or verification operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "run", "verify"))
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=Path(".github/ci/module-times.json"))
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--results", type=Path, default=Path(".tmp/ci-shards"))
    args = parser.parse_args()
    default_root = Path(__file__).resolve().parents[1]
    # Resolve through the shared boundary in a child process, preserving
    # pytest-cov's ability to measure import-time execution in this process.
    command = [sys.executable, "-c",
               "import sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]+'/src'); "
               "from project_paths import resolve_script_project_root; "
               "print(resolve_script_project_root(Path(sys.argv[1]), "
               "Path(sys.argv[2]) if sys.argv[2] else None))",
               str(default_root), str(args.project_root) if args.project_root else ""]
    os.chdir(subprocess.check_output(command, text=True).strip())
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if args.mode == "plan":
        inventory = Inventory()
        status = pytest.main(["tests/", "--collect-only", "-q"], plugins=[inventory])
        if status:
            return int(status)
        weights = json.loads(args.weights.read_text())["modules"]
        plan = {"schema": 1, "head": head, "source_digest": checkout_digest(), "nodes": inventory.nodes,
                "collection_digest": digest(inventory.nodes),
                "groups": partition(inventory.nodes, weights, args.count)}
        validate_plan(plan)
        args.plan.parent.mkdir(parents=True, exist_ok=True)
        with args.plan.open("x") as stream:
            json.dump(plan, stream, indent=2)
        return 0
    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    if head != plan["head"]:
        raise ValueError("Checkout HEAD differs from plan HEAD")
    if checkout_digest() != plan["source_digest"]:
        raise ValueError("Validation inputs differ from plan inputs")
    if args.mode == "verify":
        results = []
        for path in args.results.glob("shard-*/receipt.json"):
            result = json.loads(path.read_text())
            files = list(path.parent.glob(".coverage*"))
            expected = path.parent / f".coverage.shard-{result['index']}"
            if (files != [expected]
                    or hashlib.sha256(expected.read_bytes()).hexdigest() != result["coverage_sha256"]):
                raise ValueError("Missing, foreign or corrupted shard coverage data")
            results.append(result)
        verify_results(plan, results)
        print(f"Verified {len(plan['nodes'])} test IDs exactly once in {len(results)} shards")
        return 0
    if not 0 <= args.index < len(plan["groups"]):
        raise ValueError("Shard index outside plan")
    directory = args.results / f"shard-{args.index}"
    directory.mkdir(parents=True, exist_ok=False)
    os.environ["COVERAGE_FILE"] = str((directory / f".coverage.shard-{args.index}").resolve())
    inventory = Inventory(plan, args.index)
    # Partial shards cannot individually cover all src/. The final aggregate
    # gate combines every shard and enforces the unchanged >=90% threshold.
    status = pytest.main(["tests/", "--cov=src", "--cov-fail-under=0", "-q",
                          "--durations=0", f"--junitxml={directory / 'junit.xml'}"],
                         plugins=[inventory])
    coverage_file = directory / f".coverage.shard-{args.index}"
    result = {"index": args.index, "plan_digest": digest(plan), "exit_code": int(status),
              "finished": inventory.finished, "skipped": inventory.skipped,
              "coverage_sha256": hashlib.sha256(coverage_file.read_bytes()).hexdigest()
              if coverage_file.is_file() else None}
    (directory / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    return int(status)


if __name__ == "__main__":
    raise SystemExit(main())
