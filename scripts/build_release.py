#!/usr/bin/env python3
"""Thin orchestrator: build or verify the release checksum bundle (MED-1).

All checksum logic lives in :mod:`publication.release_manifest`. Before either
building or verifying a reviewer bundle, this entry point requires the current
publication-profile analysis, receipt-backed final hydration, rendered surface,
and full test/coverage receipt chain. That local provenance preflight does not
grant external release authority. Approved releases may pass ``--timestamp``
or set the standard ``SOURCE_DATE_EPOCH`` environment variable.

Exit codes:
    0   bundle built (default) or verified clean (--verify)
    1   verification found mismatched/missing artifacts
    2   unexpected error
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


def _require_current_metadata(root: Path) -> None:
    """Reject a bundle when generated publication metadata has drifted."""
    from publication.metadata import check_metadata

    drifted = check_metadata(root)
    if drifted:
        raise ValueError("generated publication metadata is stale: " + ", ".join(drifted))


def _require_current_rendered_surfaces(root: Path) -> None:
    """Run the live PDF/slide/web gate before bundling rendered artifacts."""
    from publication.surface_validation import validate_rendered_surfaces

    result = validate_rendered_surfaces(root)
    if not result.ok:
        raise ValueError("rendered surface validation failed: " + "; ".join(result.findings))


def _require_current_reviewer_snapshot(project_root: Path = _PROJECT_ROOT) -> None:
    """Require the source-current local evidence chain before bundle operations."""
    from publication.pipeline_freshness import (
        require_fresh_pipeline_stages,
        require_fresh_publication_analysis,
    )
    from publication.validation_receipt import require_fresh_validation_receipt

    root = Path(project_root).resolve()
    _require_current_metadata(root)
    _require_current_rendered_surfaces(root)
    require_fresh_publication_analysis(root)
    require_fresh_validation_receipt(root)
    require_fresh_pipeline_stages(root, ("render",))


def main(argv: list[str] | None = None) -> int:
    """Build output/release/ (default) or --verify it; 0/1/2 exit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="verify the existing bundle, write nothing")
    parser.add_argument(
        "--profile",
        choices=("publication",),
        default="publication",
        help="fixed profile for a receipt-backed reviewer bundle (default: publication)",
    )
    parser.add_argument(
        "--timestamp",
        help=(
            "approved release time in canonical UTC form YYYY-MM-DDTHH:MM:SSZ; "
            "omit for a byte-reproducible unreleased build"
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="standalone checkout to validate and bundle (default: this checkout)",
    )
    args = parser.parse_args(argv)

    from project_paths import resolve_script_project_root
    from publication.release_manifest import (
        build_release,
        timestamp_from_source_date_epoch,
        verify_release,
    )

    try:
        root = resolve_script_project_root(_PROJECT_ROOT, args.project_root)
        _require_current_reviewer_snapshot(root)
        if args.verify:
            if args.timestamp is not None:
                parser.error("--timestamp cannot be combined with --verify")
            bad = verify_release(root)
            if bad:
                print("MISMATCH: " + ", ".join(bad[:20]))
                return 1
            print("release bundle verified")
            return 0
        source_date_epoch = os.environ.get("SOURCE_DATE_EPOCH")
        if args.timestamp is not None and source_date_epoch is not None:
            parser.error("--timestamp cannot be combined with SOURCE_DATE_EPOCH")
        timestamp = (
            timestamp_from_source_date_epoch(source_date_epoch)
            if source_date_epoch is not None
            else args.timestamp
        )
        manifest = build_release(
            root,
            profile=args.profile,
            timestamp=timestamp,
        )
        print(f"output/release written: {manifest['n_artifacts']} artifacts, {manifest['total_bytes']} bytes")
        return 0
    except Exception as exc:  # noqa: BLE001 - top-level orchestrator guard
        print(f"release bundle failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
