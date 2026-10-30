"""Shared real-file fixtures and subprocess drivers for analysis stage tests."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from analysis.stage_workspace import prepare_workspace
from tests.analysis.test_workflow import _make_project

SOURCE = Path(__file__).resolve().parents[2]
SCRIPT = SOURCE / "scripts/analysis_stages.py"


@pytest.fixture
def prepared(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    project.mkdir()
    shutil.copytree(SOURCE / "src", project / "src")
    shutil.copytree(SOURCE / "scripts", project / "scripts")
    for name in ("pyproject.toml", "uv.lock", "experiment_plan.yaml"):
        shutil.copy2(SOURCE / name, project / name)
    _make_project(project)
    workspace = tmp_path / "workspace"
    prepare_workspace(project, workspace, profile="smoke")
    return project, workspace


def cli(workspace: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, "-IB", str(SCRIPT), *arguments, "--workspace", str(workspace)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = process.communicate(timeout=150)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.communicate(timeout=10)
        raise
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def run(workspace: Path, stage: str = "belief_sharing", seconds: str = "20") -> dict:
    result = cli(workspace, "run", "--stage", stage, "--max-seconds", seconds)
    assert result.returncode == 0, (result.stdout, result.stderr)
    return json.loads(result.stdout)


def wait_for(path: Path, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 15
    while not path.exists() and process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    assert path.exists(), f"real worker did not reach {path.name}"
