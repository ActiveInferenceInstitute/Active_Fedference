#!/usr/bin/env python3
"""Prepare, run, cancel, inspect or import one frozen external analysis stage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    """Expose explicit stage/resource selection; never launch a full analysis implicitly."""
    from analysis.artifacts import ANALYSIS_REPORT_FILENAMES
    from analysis.stage_io import StageError
    from analysis.stage_process import StageCancelled, StageTimedOut, request_cancel, workspace_lock

    # Keep private worker startup stdlib-only until its inherited lock and
    # independent budgets are admitted. Catalog/config imports load numerics.
    schemas = {Path(name).stem: Path(name).stem for name in ANALYSIS_REPORT_FILENAMES}
    schemas["figures"] = "analysis_bundle"
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run", "cancel", "status", "import", "worker"):
        item = sub.add_parser(name)
        item.add_argument("--workspace", type=Path, required=True)
        if name in {"prepare", "import"}:
            item.add_argument("--project-root", type=Path, default=ROOT)
        if name == "prepare":
            item.add_argument("--profile", choices=("smoke", "publication"), required=True)
        if name in {"run", "import", "worker"}:
            item.add_argument("--stage", choices=tuple(schemas), required=True)
        if name == "run":
            item.add_argument("--max-seconds", type=float, required=True)
            item.add_argument("--max-rss-mb", type=int, default=1024)
        if name == "worker":
            item.add_argument("--attempt", required=True)
            item.add_argument("--lease-fd", type=int, default=-1)
    args = parser.parse_args()
    try:
        workspace = args.workspace.absolute()
        if args.command == "worker":
            from analysis.stage_worker import worker

            worker(workspace, args.stage, args.attempt, args.lease_fd)
            return 0
        from analysis.stage_checkpoint import import_report
        from analysis.stage_runner import run_stage
        from analysis.stage_workspace import admit_workspace, prepare_workspace

        if args.command == "prepare":
            result = prepare_workspace(args.project_root, workspace, profile=args.profile)
            print(json.dumps({"status": "prepared", "identity": result["identity"]}))
        elif args.command == "run":
            result = run_stage(
                workspace,
                args.stage,
                schemas[args.stage],
                max_seconds=args.max_seconds,
                max_rss_mb=args.max_rss_mb,
            )
            print(json.dumps(result))
            return {"complete": 0, "reused": 0, "cancelled": 130, "timed_out": 124}.get(result["status"], 1)
        elif args.command == "cancel":
            print(json.dumps({"cancel_request": str(request_cancel(workspace))}))
        elif args.command == "status":
            manifest = admit_workspace(workspace)
            from analysis.stage_checkpoint import load_checkpoint
            from analysis.stage_io import confined, read_json
            from analysis.stage_process import _process_start, writer_held

            completed = []
            for stage, schema in schemas.items():
                if (workspace / "completed" / stage).exists():
                    load_checkpoint(workspace, stage, schema, manifest["identity"])
                    completed.append(stage)
            active_path = confined(workspace, "active.json")
            active = read_json(active_path) if active_path.exists() else None
            held = writer_held(workspace)
            liveness = "unknown" if held else "not_running"
            if active is not None and active.get("status") == "running":
                pid = active.get("pid")
                if type(pid) is not int or pid <= 0:
                    raise StageError("invalid recorded worker PID")
                actual = _process_start(pid)
                liveness = (
                    "absent"
                    if actual is None
                    else (
                        "unknown"
                        if active.get("process_start") is None
                        else ("live" if actual == active["process_start"] else "different_incarnation")
                    )
                )
            print(
                json.dumps(
                    {
                        "identity": manifest["identity"],
                        "completed": completed,
                        "pending": [s for s in schemas if s not in completed],
                        "last_attempt_record": active,
                        "observed_worker": liveness,
                        "workspace_writer_held": held,
                    }
                )
            )
        elif args.command == "import":
            with workspace_lock(workspace):
                if args.stage == "figures":
                    from analysis.stage_bundle import import_bundle

                    print(json.dumps(import_bundle(workspace, args.project_root)))
                    return 0
                path = import_report(workspace, args.project_root, args.stage, schemas[args.stage])
            print(json.dumps({"imported_report": str(path), "analysis_acceptance": False}))
    except StageTimedOut as exc:
        print(str(exc), file=sys.stderr)
        return 124
    except StageCancelled as exc:
        print(str(exc), file=sys.stderr)
        return 130
    except (StageError, OSError, ValueError) as exc:
        print(f"analysis stage refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
