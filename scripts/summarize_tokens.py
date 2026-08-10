#!/usr/bin/env python3
"""Thin orchestrator: print a summary of all manuscript tokens and their resolved values.

Reads ``output/data/manuscript_variables.json`` (produced by
``z_generate_manuscript_variables.py``) and prints every token in
alphabetical order.  Useful for auditing token resolution and for diffing
token sets across pipeline runs.

No computation lives here — all token generation is delegated to
:func:`src.manuscript_variables.generate_variables` when the JSON is absent.

Exit codes:
    0   tokens printed successfully
    1   JSON absent and live generation also failed
    2   unexpected error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))


def main() -> int:
    """Print token summary from JSON or by regenerating; 0/1/2 exit code."""
    parser = argparse.ArgumentParser(
        description="Summarise all manuscript {{TOKEN}} values for active_fedference"
    )
    parser.add_argument(
        "--regen",
        action="store_true",
        help="Regenerate tokens from src/ even if the JSON cache exists.",
    )
    parser.add_argument(
        "--filter",
        metavar="PREFIX",
        default="",
        help="Only print tokens whose key starts with PREFIX (case-insensitive).",
    )
    args = parser.parse_args()

    json_path = _PROJECT_ROOT / "output" / "data" / "manuscript_variables.json"

    variables: dict[str, str] | None = None

    if not args.regen and json_path.exists():
        try:
            variables = json.loads(json_path.read_text(encoding="utf-8"))
            print(f"source: {json_path}")
        except (json.JSONDecodeError, OSError) as exc:
            print(f"JSON read failed ({exc}); regenerating …", file=sys.stderr)

    if variables is None:
        try:
            from src.manuscript_variables import generate_variables

            variables = generate_variables(_PROJECT_ROOT)
            print(f"source: generated (no cache at {json_path})")
        except Exception as exc:  # noqa: BLE001
            print(f"token generation failed: {exc}", file=sys.stderr)
            return 1

    prefix = args.filter.upper()
    filtered = {k: v for k, v in variables.items() if k.startswith(prefix)} if prefix else variables

    print(f"token_count: {len(filtered)}")
    print()
    for key in sorted(filtered):
        print(f"{key}: {filtered[key]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
