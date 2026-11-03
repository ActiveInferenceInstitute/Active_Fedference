"""Regression tests for the CI gate guard.

The gate job must not paint a superseded run red, and it must still fail on a
genuine failure. Both branches run as real subprocesses over synthetic
``needs`` payloads, so the guard is exercised off the green path.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ci_gate.py"
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def run_gate(
    results: dict[str, str], *, superseded: bool = False
) -> subprocess.CompletedProcess[str]:
    payload = json.dumps({name: {"result": result} for name, result in results.items()})
    command = [sys.executable, str(SCRIPT), "--results", payload]
    if superseded:
        command.append("--superseded")
    return subprocess.run(command, capture_output=True, text=True, check=False)


def test_live_run_passes_when_every_prerequisite_succeeds() -> None:
    completed = run_gate({"fast-checks": "success", "test-shards": "success"})
    assert completed.returncode == 0
    assert "live run" in completed.stdout


def test_live_run_rejects_a_failed_prerequisite() -> None:
    completed = run_gate({"fast-checks": "failure"})
    assert completed.returncode == 1
    assert "fast-checks" in completed.stderr


def test_live_run_rejects_cancelled_and_skipped_prerequisites() -> None:
    completed = run_gate({"test-shards": "cancelled", "compatibility": "skipped"})
    assert completed.returncode == 1
    assert "test-shards" in completed.stderr
    assert "compatibility" in completed.stderr


def test_superseded_run_accepts_cancelled_and_skipped_prerequisites() -> None:
    completed = run_gate(
        {"test-shards": "cancelled", "compatibility": "skipped"}, superseded=True
    )
    assert completed.returncode == 0
    assert "superseded" in completed.stdout


def test_superseded_run_still_rejects_a_failed_prerequisite() -> None:
    completed = run_gate({"fast-checks": "failure"}, superseded=True)
    assert completed.returncode == 1
    assert "fast-checks" in completed.stderr


def test_gate_job_skips_cancelled_runs_and_uses_the_checked_in_guard() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    gate = workflow["jobs"]["gate"]
    assert gate["if"] == "always() && !cancelled()"
    guard = next(
        step
        for step in gate["steps"]
        if step.get("name") == "Require every prerequisite job to pass"
    )
    assert "scripts/ci_gate.py" in guard["run"]
    assert "toJSON(needs)" in guard["env"]["RESULTS"]
