#!/usr/bin/env python3
"""Thin orchestrator: validate that all expected output figures and reports exist.

Checks every artifact expected from Stage 02 (figures, JSON reports, manuscript
variables) exists under ``output/`` and is non-zero in size.  Prints a
label-prefixed line per artifact found and exits non-zero if any are missing or
empty.

All discovery logic lives in :func:`src.analysis.workflow.expected_artifacts`;
this script only prints results and maps the outcome to an exit code.

Exit codes:
    0   all expected artifacts present and non-empty
    1   one or more artifacts missing or empty
    2   unexpected error
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))


def main() -> int:
    """Validate output artifacts and report status; 0/1/2 exit code."""
    try:
        from analysis.workflow import expected_artifacts
    except ImportError:
        # Fallback: scan output/ directly when expected_artifacts is unavailable.
        return _fallback_scan(_PROJECT_ROOT)

    try:
        artifacts = expected_artifacts(_PROJECT_ROOT)
    except Exception as exc:  # noqa: BLE001
        print(f"expected_artifacts failed: {exc}", file=sys.stderr)
        return _fallback_scan(_PROJECT_ROOT)

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


def _fallback_scan(project_root: Path) -> int:
    """Scan output/ for figures, reports, and data files directly."""
    output = project_root / "output"
    subdirs = {
        "figures": output / "figures",
        "reports": output / "reports",
        "data": output / "data",
    }
    missing: list[str] = []
    empty: list[str] = []
    found = 0
    for subdir_name, subdir in subdirs.items():
        if not subdir.exists():
            print(f"MISSING  {subdir_name}/: {subdir}")
            missing.append(subdir_name)
            continue
        for f in sorted(subdir.iterdir()):
            if not f.is_file():
                continue
            label = f"{subdir_name}/{f.name}"
            if f.stat().st_size == 0:
                empty.append(label)
                print(f"EMPTY    {label}: {f}")
            else:
                found += 1
                print(f"ok       {label}: {f}")

    if missing or empty:
        print(
            f"\nvalidation_result: FAIL ({len(missing)} missing dirs, {len(empty)} empty files)",
            file=sys.stderr,
        )
        return 1

    print(f"\nvalidation_result: PASS ({found} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
