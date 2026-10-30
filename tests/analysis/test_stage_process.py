"""Real staged-analysis boundary regressions."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from analysis.stage_io import StageError, encode, read_json
from analysis.stage_process import request_cancel, workspace_lock
from tests.analysis.stage_fixtures import SCRIPT, cli, run, wait_for
from tests.analysis.stage_fixtures import prepared as prepared


@pytest.mark.parametrize("method", ["request", "supervisor_signal"])
def test_cooperative_cancel_retains_partial_attempt_and_completed_peer(
    prepared: tuple[Path, Path],
    method: str,
) -> None:
    _, workspace = prepared
    run(workspace)
    completed = (workspace / "completed/belief_sharing/report.json").read_bytes()
    process = subprocess.Popen(
        [
            sys.executable,
            "-IB",
            str(SCRIPT),
            "run",
            "--workspace",
            str(workspace),
            "--stage",
            "robustness_review_grid",
            "--max-seconds",
            "20",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        wait_for(workspace / "active.json", process)
        # Previous completed-stage active.json may exist: wait for this actual attempt.
        deadline = time.monotonic() + 15
        active = read_json(workspace / "active.json")
        while active["stage"] != "robustness_review_grid" and time.monotonic() < deadline:
            time.sleep(0.02)
            active = read_json(workspace / "active.json")
        assert active["status"] == "running"
        attempt = workspace / "attempts" / active["attempt"]
        wait_for(attempt / "producer-started.json", process)
        rejected = cli(workspace, "run", "--stage", "emergence", "--max-seconds", "5")
        assert rejected.returncode != 0 and "active writer" in rejected.stderr
        if method == "request":
            assert request_cancel(workspace).exists()
        else:
            os.kill(process.pid, signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 130, (stdout, stderr)
        receipt = read_json(attempt / "result.json")
        assert receipt["status"] == "cancelled"
        assert receipt["returncode"] != 0
        assert not (workspace / "completed/robustness_review_grid").exists()
        assert (workspace / "completed/belief_sharing/report.json").read_bytes() == completed
        # No lock is held after cancellation and the already completed stage is reusable.
        with workspace_lock(workspace):
            pass
        assert run(workspace)["status"] == "reused"
        resumed = run(workspace, "robustness_review_grid")
        assert resumed["status"] == "complete"
        assert attempt.exists() and (attempt / "stderr").exists()
    finally:
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=10)


def test_deadline_is_incomplete_then_resume_uses_new_attempt(prepared: tuple[Path, Path]) -> None:
    _, workspace = prepared
    result = cli(workspace, "run", "--stage", "belief_sharing", "--max-seconds", "0.001")
    assert result.returncode == 124, (result.stdout, result.stderr)
    receipt = json.loads(result.stdout)
    assert receipt["status"] == "timed_out"
    failed = workspace / "attempts" / receipt["attempt"]
    assert failed.exists() and not (workspace / "completed/belief_sharing").exists()
    assert run(workspace)["status"] == "complete"
    assert failed.exists()


@pytest.mark.parametrize("seconds", ["0", "-1", "nan", "inf", "3601"])
def test_invalid_budgets_are_refused(prepared: tuple[Path, Path], seconds: str) -> None:
    _, workspace = prepared
    result = cli(workspace, "run", "--stage", "belief_sharing", "--max-seconds", seconds)
    assert result.returncode != 0 and "max_seconds" in result.stderr
    assert not (workspace / "attempts").exists()


def test_cancel_without_running_attempt_is_refused(prepared: tuple[Path, Path]) -> None:
    _, workspace = prepared
    assert cli(workspace, "cancel").returncode != 0


def test_memory_budget_stops_only_its_worker(prepared: tuple[Path, Path]) -> None:
    _, workspace = prepared
    result = cli(workspace, "run", "--stage", "belief_sharing", "--max-seconds", "10", "--max-rss-mb", "1")
    assert result.returncode == 1, (result.stdout, result.stderr)
    receipt = json.loads(result.stdout)
    assert receipt["status"] == "memory_budget"
    assert not (workspace / "completed/belief_sharing").exists()
    assert (workspace / "attempts" / receipt["attempt"] / "result.json").exists()


def test_live_orphan_blocks_resume_without_signalling_it(prepared: tuple[Path, Path]) -> None:
    """Use a real live process identity; no fabricated PID is signalled."""
    from analysis.stage_process import _process_start, _refuse_live_previous_worker

    _, workspace = prepared
    (workspace / "active.json").write_bytes(
        encode(
            {
                "status": "running",
                "pid": os.getpid(),
                "process_start": _process_start(os.getpid()),
            }
        )
    )
    with pytest.raises(StageError, match="previous worker is still live"):
        _refuse_live_previous_worker(workspace)
    # A positively different birth identifies an old incarnation, without
    # treating the current unrelated process as owned or signalling it.
    (workspace / "active.json").write_bytes(
        encode(
            {
                "status": "running",
                "pid": os.getpid(),
                "process_start": "different incarnation",
            }
        )
    )
    _refuse_live_previous_worker(workspace)


@pytest.mark.parametrize("kind", ["hardlink", "fifo"])
def test_workspace_writer_lock_must_be_a_private_regular_file(prepared: tuple[Path, Path], kind: str) -> None:
    _, workspace = prepared
    path = workspace / "writer.lock"
    if kind == "hardlink":
        original = workspace / "original-lock"
        original.write_text("retained")
        os.link(original, path)
    else:
        os.mkfifo(path)
    with pytest.raises(StageError, match="private regular"):
        with workspace_lock(workspace):
            pytest.fail("invalid lock admitted")


@pytest.mark.parametrize("value", [None, True, "not-a-budget"])
def test_python_callers_cannot_bypass_budget_types(prepared: tuple[Path, Path], value: object) -> None:
    from analysis.stage_runner import run_stage

    _, workspace = prepared
    with pytest.raises(StageError, match="max_seconds"):
        run_stage(workspace, "belief_sharing", "belief_sharing", max_seconds=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("termination", ["signal", "own_deadline"])
def test_worker_keeps_workspace_lock_when_supervisor_and_active_record_are_lost(
    prepared: tuple[Path, Path],
    termination: str,
) -> None:
    from analysis.stage_process import _process_start

    _, workspace = prepared
    process = subprocess.Popen(
        [
            sys.executable,
            "-IB",
            str(SCRIPT),
            "run",
            "--workspace",
            str(workspace),
            "--stage",
            "robustness_review_grid",
            "--max-seconds",
            "5",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    active = None
    try:
        wait_for(workspace / "active.json", process)
        active = read_json(workspace / "active.json")
        wait_for(workspace / "attempts" / active["attempt"] / "producer-started.json", process)
        (workspace / "active.json").rename(workspace / "retained-active-before-loss.json")
        process.kill()  # Only this test's supervisor; its worker must retain the kernel lock.
        process.communicate(timeout=5)
        assert process.returncode == -signal.SIGKILL
        assert _process_start(active["pid"]) == active["process_start"]
        with pytest.raises(StageError, match="active writer"):
            with workspace_lock(workspace):
                pytest.fail("orphan worker lost the workspace lock")
        status = cli(workspace, "status")
        assert status.returncode == 0, status.stderr
        observed = json.loads(status.stdout)
        assert observed["workspace_writer_held"] is True
        assert observed["observed_worker"] == "unknown"
        if termination == "own_deadline":
            from analysis.stage_process import writer_held

            deadline = time.monotonic() + 8
            while writer_held(workspace) and time.monotonic() < deadline:
                time.sleep(0.02)
            assert not writer_held(workspace), "orphan outlived its independent resource budget"
            attempt = workspace / "attempts" / active["attempt"]
            assert "budget expired" in (attempt / "stderr").read_text()
            assert not (workspace / "completed/robustness_review_grid").exists()
    finally:
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=10)
        if active is not None and _process_start(active["pid"]) == active["process_start"]:
            os.kill(active["pid"], signal.SIGTERM)
            deadline = time.monotonic() + 5
            while _process_start(active["pid"]) == active["process_start"] and time.monotonic() < deadline:
                time.sleep(0.02)
            if _process_start(active["pid"]) == active["process_start"]:
                os.kill(active["pid"], signal.SIGKILL)
    with workspace_lock(workspace):
        pass


def test_supervisor_reaps_its_term_ignoring_process_group(tmp_path: Path) -> None:
    """Exercise real escalation against an acknowledged, independently owned group."""
    from analysis.stage_process import _stop_owned

    acknowledged = tmp_path / "term-ignored"
    process = subprocess.Popen(
        [
            sys.executable,
            "-IB",
            "-c",
            "import signal,sys,time; from pathlib import Path; "
            "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
            "Path(sys.argv[1]).write_text('ready'); time.sleep(30)",
            str(acknowledged),
        ],
        start_new_session=True,
    )
    try:
        wait_for(acknowledged, process)
        _stop_owned(process)
        assert process.returncode == -signal.SIGKILL
        _stop_owned(process)  # Reaping an already terminal owned child is idempotent.
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
