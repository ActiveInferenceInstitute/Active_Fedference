#!/usr/bin/env python3
"""Thin orchestrator: generate and inject manuscript variables.

Reads experiment config and analysis outputs, writes
``output/data/manuscript_variables.json``, and substitutes
``{{TOKEN}}`` markers in manuscript sections into
``output/manuscript/`` for PDF rendering.

All computation lives in ``src/manuscript_variables``. This script calls
``src/`` only; no ``infrastructure.*`` imports here.

Exit codes:
    0   variables written and injected successfully
    1   unexpected error
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))


def main(argv: list[str] | None = None) -> int:
    """Generate and persist the manuscript {{TOKEN}} variables JSON; 0/1 exit code."""
    parser = argparse.ArgumentParser(description="Generate manuscript variables for active_fedference")
    parser.add_argument(
        "--allow-draft",
        action="store_true",
        help=(
            "Pass through to generate_variables: when True, missing analysis "
            "reports degrade tokens to N/A instead of raising an error."
        ),
    )
    parser.add_argument(
        "--provisional-validation",
        action="store_true",
        help=(
            "permit one non-draft provisional hydration after fresh analysis but "
            "before the full test-and-coverage receipt exists; final hydration "
            "must be rerun without this flag"
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="standalone checkout whose reports and manuscript tree are written",
    )
    args = parser.parse_args(argv)
    if args.allow_draft and args.provisional_validation:
        parser.error("--provisional-validation cannot be combined with --allow-draft")

    from manuscript_variables import generate_variables, render_manuscript_tree, save_variables
    from project_paths import resolve_script_project_root
    from publication.pipeline_freshness import (
        record_pipeline_stage,
        require_fresh_publication_analysis,
    )
    from publication.release_manifest import timestamp_from_source_date_epoch
    from publication.validation_receipt import require_fresh_validation_receipt

    # Honor ACTIVE_FEDFERENCE_PROJECT_ROOT so subprocess tests can redirect
    # manuscript_variables.json and output/manuscript/ writes into a scaffold
    # instead of the real committed tree. An invalid override raises loudly.
    root = resolve_script_project_root(_PROJECT_ROOT, args.project_root)

    # This is deliberately before *any* variable or manuscript write.  A
    # provisional render can occur after analysis to enable the full suite,
    # but release-facing non-draft hydration must consume the successful,
    # source- and analysis-bound test/coverage receipt.
    if not args.allow_draft:
        require_fresh_publication_analysis(root)
        if not args.provisional_validation:
            require_fresh_validation_receipt(root)

    variables = generate_variables(
        root,
        allow_draft=args.allow_draft,
        require_validation_receipt=(not args.allow_draft and not args.provisional_validation),
    )
    out_path = root / "output" / "data" / "manuscript_variables.json"
    save_variables(variables, out_path)
    manuscript_dir = render_manuscript_tree(root, variables)
    # Do not let draft, provisional, or minimal subprocess hydrations mint a
    # publication receipt. The provisional pass exists only to prepare inputs
    # for the full suite; it cannot make a pre-test render look release-ready.
    # The full standalone project records hydration only after final, receipt-
    # backed writes.
    if not args.allow_draft and not args.provisional_validation and (root / "src").is_dir():
        source_date_epoch = os.environ.get("SOURCE_DATE_EPOCH")
        timestamp = (
            timestamp_from_source_date_epoch(source_date_epoch) if source_date_epoch is not None else None
        )
        # Recheck after the output writes and immediately before recording the
        # hydration stage. Otherwise a source/manuscript/test/config edit during
        # hydration could be incorporated into a new stage hash even though the
        # successful full-suite receipt no longer binds it. A failed recheck
        # leaves no new hydration receipt, so downstream freshness remains
        # fail-closed.
        require_fresh_validation_receipt(root)
        record_pipeline_stage(root, "hydration", timestamp=timestamp)
    print(f"variables_json: {out_path}")
    print(f"manuscript_dir: {manuscript_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
