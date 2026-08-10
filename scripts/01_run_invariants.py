#!/usr/bin/env python3
"""Thin orchestrator: run and serialise the fedference numerical invariants.

All checks and serialisation live in :func:`src.invariants` (pure compute, no
``infrastructure.*``); this script only wires the path, calls
:func:`invariants.write_invariants_report`, prints the report path, and maps the
overall pass verdict to an exit code.

Exit codes:
    0   all locked-core invariants hold (report written)
    1   one or more invariants failed
    2   unexpected error
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


def main() -> int:
    """Run the locked-core numerical invariants and write the report; 0/1 exit code."""
    from invariants import write_invariants_report
    from project_paths import resolve_env_project_root

    # Honor ACTIVE_FEDFERENCE_PROJECT_ROOT so subprocess tests write
    # output/reports/invariants.json into a scaffold, not the real tree.
    # An invalid override raises loudly (exit via traceback), never falls back.
    root = resolve_env_project_root(_PROJECT_ROOT)

    try:
        path, all_passed = write_invariants_report(root)
    except Exception as exc:  # noqa: BLE001 - top-level orchestrator guard
        print(f"invariants report failed: {exc}", file=sys.stderr)
        return 2

    print(path)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
