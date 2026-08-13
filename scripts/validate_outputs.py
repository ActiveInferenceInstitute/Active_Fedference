#!/usr/bin/env python3
"""Thin orchestrator: validate that all expected output figures and reports exist.

Checks every artifact expected from Stage 02 (figures, JSON reports, manuscript
variables) exists under ``output/`` and is non-zero in size.  Prints a
label-prefixed line per artifact found and exits non-zero if any are missing or
empty.

All discovery logic lives in :func:`src.analysis.artifacts.expected_artifacts`;
this script only prints results and maps the outcome to an exit code.

Exit codes:
    0   all expected artifacts present and non-empty
    1   one or more artifacts missing or empty
    2   unexpected error
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))


def main(argv: list[str] | None = None) -> int:
    """Validate output artifacts and report status; 0/1/2 exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None)
    args = parser.parse_args(argv)

    from project_paths import resolve_script_project_root

    root = resolve_script_project_root(_PROJECT_ROOT, args.project_root)
    try:
        from analysis.artifacts import expected_artifacts
    except ImportError as exc:
        print(f"expected_artifacts import failed: {exc}", file=sys.stderr)
        return 2

    try:
        artifacts = expected_artifacts(root)
    except Exception as exc:  # noqa: BLE001
        print(f"expected_artifacts failed: {exc}", file=sys.stderr)
        return 2

    missing: list[str] = []
    empty: list[str] = []
    for label, path in artifacts.items():
        p = Path(path)
        if not p.exists():
            missing.append(label)
            print(f"MISSING  {label}: {path}")
        elif p.stat().st_size == 0:
            empty.append(label)
            print(f"EMPTY    {label}: {path}")
        else:
            print(f"ok       {label}: {path}")

    if missing or empty:
        print(
            f"\nvalidation_result: FAIL ({len(missing)} missing, {len(empty)} empty)",
            file=sys.stderr,
        )
        return 1

    print(f"\nvalidation_result: PASS ({len(artifacts)} artifacts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
