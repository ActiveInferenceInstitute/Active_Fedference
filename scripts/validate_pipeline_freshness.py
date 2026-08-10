#!/usr/bin/env python3
"""Validate the content-bound upstream/downstream pipeline receipts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


def main(argv: list[str] | None = None) -> int:
    """Validate the requested freshness dependency closure."""
    from publication.pipeline_freshness import (
        PIPELINE_STAGES,
        validate_publication_pipeline_freshness,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=tuple(stage.name for stage in PIPELINE_STAGES),
        default=None,
        help="Stages to validate; dependencies are included automatically.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=_PROJECT_ROOT,
    )
    args = parser.parse_args(argv)
    findings = validate_publication_pipeline_freshness(args.project_root, args.stages)
    if findings:
        print("pipeline freshness: FAIL")
        for finding in findings:
            print(f"- {finding}")
        return 1
    stages = args.stages or tuple(stage.name for stage in PIPELINE_STAGES)
    print(f"pipeline freshness: PASS ({', '.join(stages)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
