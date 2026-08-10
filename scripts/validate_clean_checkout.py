#!/usr/bin/env python3
"""Probe whether a checkout is clean, clone-correct, and importable."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


def main(argv: list[str] | None = None) -> int:
    """Run the clean-checkout probe and report every blocking condition."""
    from publication.clean_checkout import inspect_clean_checkout

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=_PROJECT_ROOT,
    )
    parser.add_argument(
        "--skip-imports",
        action="store_true",
        help="Only check Git cleanliness and the required tracking set.",
    )
    args = parser.parse_args(argv)
    report = inspect_clean_checkout(args.project_root, check_imports=not args.skip_imports)
    print(f"tracked_files: {report.tracked_files}")
    if report.ok:
        print("clean checkout: PASS")
        return 0
    print("clean checkout: FAIL")
    for finding in report.findings:
        print(f"- {finding}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
