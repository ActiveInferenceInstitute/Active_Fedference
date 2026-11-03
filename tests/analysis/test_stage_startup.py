"""Observe real worker startup without replacing imports or numerical operations."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from analysis.stage_io import encode
from analysis.stage_process import workspace_lock
from analysis.stage_workspace import admit_workspace
from tests.analysis.stage_fixtures import prepared as prepared


def test_expired_worker_budget_refuses_before_numerical_imports(prepared: tuple[Path, Path]) -> None:
    _, workspace = prepared
    token = "a" * 32
    attempt = workspace / "attempts" / token
    attempt.mkdir(parents=True)
    (attempt / "started.json").write_bytes(
        encode(
            {
                "stage": "belief_sharing",
                "identity": admit_workspace(workspace)["identity"],
                "deadline_monotonic": time.monotonic() - 1,
            }
        )
    )
    marker = attempt / "unexpected-numerical-import"
    script = workspace / "source/scripts/analysis_stages.py"
    observer = """
import runpy, sys
from pathlib import Path
marker, script, workspace, token, descriptor = sys.argv[1:]
def observe(event, args):
    if event == 'import' and args[0] == 'numpy':
        Path(marker).write_text('real numpy import began')
sys.addaudithook(observe)
sys.argv = [script, 'worker', '--workspace', workspace, '--stage', 'belief_sharing',
            '--attempt', token, '--lease-fd', descriptor]
runpy.run_path(script, run_name='__main__')
"""
    with workspace_lock(workspace) as descriptor:
        result = subprocess.run(
            [
                sys.executable,
                "-IB",
                "-c",
                observer,
                str(marker),
                str(script),
                str(workspace),
                token,
                str(descriptor),
            ],
            pass_fds=(descriptor,),
            capture_output=True,
            text=True,
            timeout=10,
        )
    assert not marker.exists(), "numerical imports preceded the worker's own budget admission"
    assert result.returncode == 124 and "budget expired" in result.stderr, result.stderr
    assert not (attempt / "producer-started.json").exists()
