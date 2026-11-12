#!/usr/bin/env python3
"""Read the estate pin locally until its revision includes ``gnn.importers``."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def read_gnn_pin(project_root: Path) -> str:
    """Admit the exact revision used by both CI checkout and local bootstrap."""
    pin = json.loads((project_root / ".github/gnn-pin.json").read_text(encoding="utf-8"))
    revision = pin.get("commit") if isinstance(pin, dict) else None
    if (
        not isinstance(pin, dict)
        or pin.get("schema") != "gnn-importer-pin/v1"
        or not isinstance(revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", revision) is None
    ):
        raise ValueError("GNN pin must declare gnn-importer-pin/v1 and an exact commit SHA")
    return revision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from project_paths import resolve_script_project_root

    revision = read_gnn_pin(resolve_script_project_root(root, args.project_root))
    if args.github_output is not None:
        with args.github_output.open("a", encoding="utf-8") as output:
            output.write(f"revision={revision}\n")
    print(revision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
