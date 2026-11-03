"""Supervise exactly one explicitly bounded frozen analysis stage."""

from __future__ import annotations

import math
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from analysis.stage_checkpoint import load_checkpoint
from analysis.stage_io import StageError, atomic_bytes, confined, digest, encode, read_json
from analysis.stage_process import (
    StageCancelled,
    _process_start,
    _refuse_live_previous_worker,
    _stop_owned,
    workspace_lock,
)
from analysis.stage_workspace import admit_workspace


def run_stage(
    workspace: Path,
    stage: str,
    schema: str,
    *,
    max_seconds: float,
    max_rss_mb: int = 1024,
) -> dict[str, Any]:
    """Run or reuse exactly one requested stage, with fail-closed identity validation."""
    if (
        isinstance(max_seconds, bool)
        or not isinstance(max_seconds, (int, float))
        or not math.isfinite(max_seconds)
        or not 0 < max_seconds <= 3600
    ):
        raise StageError("max_seconds must be finite, positive and at most 3600")
    if type(max_rss_mb) is not int or max_rss_mb <= 0:
        raise StageError("max_rss_mb must be a positive integer")
    from analysis.stage_catalog import STAGE_SCHEMAS

    schemas = {**STAGE_SCHEMAS, "figures": "analysis_bundle"}
    if schemas.get(stage) != schema:
        raise StageError(f"unknown stage/schema: {stage}")
    with workspace_lock(workspace) as lease_fd:
        manifest = admit_workspace(workspace)
        _refuse_live_previous_worker(workspace)
        completed = confined(workspace, f"completed/{stage}")
        if completed.exists():
            load_checkpoint(workspace, stage, schema, manifest["identity"])
            return {"status": "reused", "stage": stage, "identity": manifest["identity"]}
        token = uuid.uuid4().hex
        attempt = confined(workspace, f"attempts/{token}")
        attempt.mkdir(parents=True)
        started = time.monotonic()
        info: dict[str, Any] = {
            "status": "running",
            "stage": stage,
            "attempt": token,
            "identity": manifest["identity"],
            "max_seconds": max_seconds,
            "deadline_monotonic": started + max_seconds,
            "max_rss_mb": max_rss_mb,
            "max_file_bytes": 16 * 1024 * 1024,
        }
        atomic_bytes(attempt / "started.json", encode(info))
        command = [
            sys.executable,
            "-IB",
            str(workspace / "source/scripts/analysis_stages.py"),
            "worker",
            "--workspace",
            str(workspace),
            "--stage",
            stage,
            "--attempt",
            token,
            "--lease-fd",
            str(lease_fd),
        ]
        process: subprocess.Popen[bytes] | None = None
        reason = "failed"
        peak_rss_kb = 0

        def cancelled(signum: int, frame: object) -> None:
            raise StageCancelled(f"stage supervisor received signal {signum}")

        handlers = {sig: signal.signal(sig, cancelled) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            with (attempt / "stdout").open("xb") as stdout, (attempt / "stderr").open("xb") as stderr:
                process = subprocess.Popen(
                    command, stdout=stdout, stderr=stderr, start_new_session=True, pass_fds=(lease_fd,)
                )
                info["pid"] = process.pid
                info["process_start"] = _process_start(process.pid)
                atomic_bytes(workspace / "active.json", encode(info))
                while process.poll() is None:
                    if confined(workspace, f"cancel-{token}.json").exists():
                        reason = "cancelled"
                        break
                    if time.monotonic() - started >= max_seconds:
                        reason = "timed_out"
                        break
                    # Sample the owned process only; this is not a hard allocation ceiling.
                    rss = subprocess.run(
                        ["ps", "-p", str(process.pid), "-o", "rss="],
                        capture_output=True,
                        text=True,
                        timeout=2,
                        check=False,
                    )
                    if rss.returncode == 0 and rss.stdout.strip():
                        peak_rss_kb = max(peak_rss_kb, int(rss.stdout.strip()))
                        if peak_rss_kb > max_rss_mb * 1024:
                            reason = "memory_budget"
                            break
                    time.sleep(0.05)
                _stop_owned(process)
                info["returncode"] = process.returncode
                if process.returncode == 124 and reason == "failed":
                    reason = "timed_out"
            if process.returncode == 0 and reason == "failed":
                # Recheck drift, schema and exact hash before exposing completion.
                admit_workspace(workspace)
                marker = read_json(attempt / "complete.json")
                data = read_json(attempt / "report.json")
                from analysis.report_schemas import validate_report

                if schema == "analysis_bundle":
                    from analysis.stage_bundle import validate_bundle

                    validate_bundle(workspace, data)
                else:
                    validate_report(schema, data)
                if marker != {
                    "schema": 1,
                    "stage": stage,
                    "identity": manifest["identity"],
                    "report_sha256": digest((attempt / "report.json").read_bytes()),
                }:
                    raise StageError("worker completion mismatch")
                info.update(
                    status="complete",
                    elapsed_seconds=time.monotonic() - started,
                    peak_sampled_rss_kb=peak_rss_kb,
                )
                atomic_bytes(attempt / "result.json", encode(info))
                marker["result_sha256"] = digest((attempt / "result.json").read_bytes())
                atomic_bytes(attempt / "complete.json", encode(marker))
                completed.parent.mkdir(exist_ok=True)
                os.rename(attempt, completed)
                directory = os.open(completed.parent, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
                attempt = completed
                reason = "complete"
        except BaseException as exc:
            if isinstance(exc, (StageCancelled, KeyboardInterrupt)):
                reason = "cancelled"
            if process is not None:
                _stop_owned(process)
                info["returncode"] = process.returncode
            info["failure_type"] = type(exc).__name__
            raise
        finally:
            # Restore process policy even if evidence persistence hits ENOSPC.
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            if reason != "complete":
                info.update(
                    status=reason, elapsed_seconds=time.monotonic() - started, peak_sampled_rss_kb=peak_rss_kb
                )
                atomic_bytes(attempt / "result.json", encode(info))
            atomic_bytes(workspace / "active.json", encode(info))
        return info
