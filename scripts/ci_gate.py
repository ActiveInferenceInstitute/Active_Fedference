#!/usr/bin/env python3
"""Fail a gate job when a prerequisite job did not succeed.

GitHub supplies the ``needs`` context as JSON. The gate job runs under
``if: always() && !cancelled()``: a superseded or manually cancelled run skips
the job instead of re-asserting a stale cancelled prerequisite as a failure.
In a live run every prerequisite must report ``success``; a ``cancelled``
result from matrix fail-fast or a ``skipped`` result from a failed upstream
dependency is still a failure.

``--superseded`` models the skipped-job outcome so the unit tests and a local
dry run can exercise both branches. CI does not pass the flag: the job
condition already prevents the call.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

SUPERSEDED_PASS = frozenset({"cancelled", "skipped"})


def parse_results(payload: str) -> dict[str, str]:
    """Parse the ``needs`` JSON into job name -> result, failing closed."""
    try:
        data: Any = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"ci gate: invalid needs JSON: {exc}") from exc
    if not isinstance(data, dict) or not data:
        raise SystemExit("ci gate: empty needs context")
    results: dict[str, str] = {}
    for name, job in data.items():
        if not isinstance(name, str):
            raise SystemExit("ci gate: needs context has a non-string job name")
        if not isinstance(job, dict) or not isinstance(job.get("result"), str):
            raise SystemExit(f"ci gate: malformed needs entry for {name!r}")
        results[name] = job["result"]
    return results


def failures(results: dict[str, str], *, superseded: bool) -> list[str]:
    """Return the jobs that block the gate.

    A superseded run tolerates only cancelled or skipped prerequisites. A
    failed prerequisite stays red even there, so relaxing the job condition
    cannot turn a broken matrix green.
    """
    if superseded:
        return sorted(
            name for name, result in results.items() if result not in SUPERSEDED_PASS
        )
    return sorted(name for name, result in results.items() if result != "success")


def main(argv: list[str] | None = None) -> int:
    """Decide whether the gate may pass from the ``needs`` JSON.

    Reads every prerequisite entry GitHub supplies. A live run is green only
    when all results are ``success``; ``--superseded`` tolerates
    ``cancelled``/``skipped`` but still rejects ``failure``. Exit 0 means the
    gate passes; a non-zero exit means at least one prerequisite did not
    succeed and the job must stay red.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="JSON needs context")
    parser.add_argument(
        "--superseded",
        action="store_true",
        help="model a cancelled (superseded) run that skips the gate job",
    )
    args = parser.parse_args(argv)
    blocking = failures(parse_results(args.results), superseded=args.superseded)
    if blocking:
        print(
            "ci gate: prerequisite jobs did not succeed: " + ", ".join(blocking),
            file=sys.stderr,
        )
        return 1
    state = "superseded" if args.superseded else "live"
    print(f"ci gate: every prerequisite job is acceptable ({state} run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
