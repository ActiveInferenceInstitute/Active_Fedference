#!/usr/bin/env python3
"""Record content hashes for a completed publication-pipeline stage."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


def main(argv: list[str] | None = None) -> int:
    """Record one stage receipt and fail if its upstream chain is stale."""
    from publication.pipeline_freshness import record_pipeline_stage
    from publication.surface_validation import validate_rendered_surfaces

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("render",),
        help=(
            "external render boundary only; analysis and hydration receipts are "
            "written by their source-bound producers"
        ),
    )
    parser.add_argument(
        "--renderer",
        default=None,
        help="Optional external renderer label for the render-stage receipt.",
    )
    parser.add_argument(
        "--timestamp",
        help=(
            "optional completion time in canonical UTC form YYYY-MM-DDTHH:MM:SSZ; "
            "omit for a byte-reproducible receipt"
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=_PROJECT_ROOT,
    )
    args = parser.parse_args(argv)
    if args.renderer is not None and args.stage != "render":
        parser.error("--renderer is valid only for the render stage")
    from publication.release_manifest import timestamp_from_source_date_epoch

    source_date_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if args.timestamp is not None and source_date_epoch is not None:
        parser.error("--timestamp cannot be combined with SOURCE_DATE_EPOCH")
    timestamp = (
        timestamp_from_source_date_epoch(source_date_epoch)
        if source_date_epoch is not None
        else args.timestamp
    )
    surfaces = validate_rendered_surfaces(args.project_root)
    if not surfaces.ok:
        detail = "; ".join(surfaces.findings)
        raise ValueError(f"cannot record render before rendered-surface validation passes: {detail}")
    record = record_pipeline_stage(
        args.project_root,
        args.stage,
        renderer=args.renderer,
        timestamp=timestamp,
    )
    print(f"pipeline stage recorded: {record['stage']}")
    print(f"input_digest: {record['input_digest']}")
    print(f"output_digest: {record['output_digest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
