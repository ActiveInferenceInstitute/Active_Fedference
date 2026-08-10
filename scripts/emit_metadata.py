#!/usr/bin/env python3
"""Thin orchestrator: check or regenerate the generated metadata surfaces.

All logic lives in :mod:`publication.metadata` (MED-2 canonical emitter).
This script only parses ``--check``/``--write`` and maps results to exit codes.

Exit codes:
    0   consistent (--check) or written successfully (--write)
    1   drift detected (--check)
    2   unexpected error
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


def main() -> int:
    """Check or write CITATION.cff/.zenodo.json/codemeta.json; 0/1/2 exit."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="report drift, write nothing")
    group.add_argument("--write", action="store_true", help="regenerate all surfaces")
    args = parser.parse_args()

    from publication.metadata import check_metadata, write_metadata

    try:
        if args.check:
            drifted = check_metadata(_PROJECT_ROOT)
            if drifted:
                print("DRIFT: " + ", ".join(drifted))
                return 1
            print("consistent: CITATION.cff, .zenodo.json, codemeta.json")
            return 0
        written = write_metadata(_PROJECT_ROOT)
        for rel in written:
            print(f"wrote {rel}")
        return 0
    except Exception as exc:  # noqa: BLE001 - top-level orchestrator guard
        print(f"metadata emitter failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
