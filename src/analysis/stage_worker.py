"""Execute one admitted frozen producer under its owned resource budget."""

from __future__ import annotations

import math
import signal
import time
from pathlib import Path

from analysis.stage_io import StageError, atomic_bytes, confined, digest, encode, read_json
from analysis.stage_process import StageCancelled, StageTimedOut, inherit_workspace_lock


def worker(workspace: Path, stage: str, token: str, lease_fd: int) -> None:
    """Execute the admitted frozen producer and commit its report before its marker."""
    import resource

    source = workspace / "source"
    if Path(__file__).resolve() != source / "src/analysis/stage_worker.py":
        raise StageError("executed worker source is not the admitted frozen source")
    attempt = confined(workspace, f"attempts/{token}")
    started = read_json(attempt / "started.json")
    if started.get("stage") != stage:
        raise StageError("attempt identity mismatch")
    inherit_workspace_lock(workspace, lease_fd)
    deadline = started.get("deadline_monotonic")
    if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline):
        raise StageError("invalid worker deadline")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise StageTimedOut("owned worker budget expired before producer admission")
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
    usage = resource.getrusage(resource.RUSAGE_SELF)
    cpu_soft = max(1, math.ceil(usage.ru_utime + usage.ru_stime + remaining))
    _, old_hard = resource.getrlimit(resource.RLIMIT_CPU)
    cpu_hard = cpu_soft + 1 if old_hard == resource.RLIM_INFINITY else min(old_hard, cpu_soft + 1)
    resource.setrlimit(resource.RLIMIT_CPU, (min(cpu_soft, cpu_hard), cpu_hard))

    def cancelled(signum: int, frame: object) -> None:
        raise StageCancelled(f"owned stage received signal {signum}")

    signal.signal(signal.SIGTERM, cancelled)
    signal.signal(signal.SIGINT, cancelled)

    def timed_out(signum: int, frame: object) -> None:
        raise StageTimedOut("owned worker wall budget expired")

    signal.signal(signal.SIGALRM, timed_out)
    signal.signal(signal.SIGXCPU, timed_out)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    # Only after independent resource policy is active may numerical imports,
    # runtime hashing, source admission or producer configuration begin.
    from analysis.stage_catalog import STAGE_SCHEMAS, report_producers
    from analysis.stage_workspace import admit_workspace
    from analysis.workflow import _write_json, effective_analysis_config

    manifest = admit_workspace(workspace)
    if started.get("identity") != manifest["identity"]:
        raise StageError("attempt identity mismatch")
    if (
        Path(report_producers.__code__.co_filename).resolve() != source / "src/analysis/stage_catalog.py"
        or Path(_write_json.__code__.co_filename).resolve() != source / "src/analysis/workflow.py"
    ):
        raise StageError("executed worker source is not the admitted frozen source")
    root = workspace / "source"
    config = effective_analysis_config(root, manifest["profile"])
    if stage == "figures":
        from analysis.stage_bundle import assemble_bundle

        payload = assemble_bundle(workspace)
    else:
        producer = report_producers(config, root, smoke=manifest["profile"] == "smoke")[stage]
        atomic_bytes(
            attempt / "producer-started.json", encode({"stage": stage, "identity": manifest["identity"]})
        )
        payload = producer()
    admit_workspace(workspace)
    if stage == "figures":
        atomic_bytes(attempt / "report.json", encode(payload))
    else:
        _write_json(payload, attempt / "report.json", schema=STAGE_SCHEMAS[stage])
    atomic_bytes(
        attempt / "complete.json",
        encode(
            {
                "schema": 1,
                "stage": stage,
                "identity": manifest["identity"],
                "report_sha256": digest((attempt / "report.json").read_bytes()),
            }
        ),
    )
    signal.setitimer(signal.ITIMER_REAL, 0)
