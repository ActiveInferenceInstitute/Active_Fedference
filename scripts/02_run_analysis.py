#!/usr/bin/env python3
"""Thin orchestrator: run the Active Fedference analysis pipeline.

Runs every study (:mod:`fedference.experiments` — reduced categorical,
source-inspired analogues related to Friston et al. (2024), plus the
robustness, moving-world, hierarchical, sensitivity, and parameter-recovery
extensions) via
:func:`analysis.workflow.run_analysis_pipeline`, which writes the JSON result
reports to ``output/reports/`` and all figures to ``output/figures/``.
All numerics live in ``src/`` (the ``fedference`` core and ``figures``);
this file only wires paths and prints the produced artifacts to stdout for
manifest collection.

Exit codes:
    0   reports and figures written successfully
    1   unexpected error
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
# src/ first (bare imports: ``from fedference...``, ``from figures...``),
# then the repo root for any optional infrastructure helpers.
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))


def main() -> int:
    """Run the analysis pipeline (reports + figures) and print artifact paths; 0/1 exit."""
    parser = argparse.ArgumentParser(description="Run the Active Fedference analysis pipeline")
    parser.add_argument(
        "--profile",
        choices=("publication", "smoke"),
        default=None,
        help="Override the config budget profile; default reads manuscript/config.yaml.",
    )
    args = parser.parse_args()

    from analysis.workflow import resolve_analysis_profile, run_analysis_pipeline
    from project_paths import resolve_env_project_root
    from publication.pipeline_freshness import record_publication_analysis_stage
    from publication.release_manifest import timestamp_from_source_date_epoch

    # Honor ACTIVE_FEDFERENCE_PROJECT_ROOT so subprocess tests can redirect all
    # output/ writes into a scaffold instead of the real committed tree. An
    # invalid override raises here (loud traceback, exit 1) — never a silent
    # fallback to the real root.
    root = resolve_env_project_root(_PROJECT_ROOT)

    try:
        # Resolve once, then pass the effective profile into the producer. This
        # prevents a config-selected smoke run (with no CLI override) from being
        # misclassified as publication merely because ``args.profile`` is None.
        effective_profile = resolve_analysis_profile(root, override=args.profile)
        paths = run_analysis_pipeline(project_root=root, profile=effective_profile)
        # Smoke subprocesses never create publication receipts. A publication
        # receipt is minted only after the producer has written the sidecar that
        # attests both configured and effective publication profiles.
        if effective_profile == "publication" and (root / "src").is_dir():
            source_date_epoch = os.environ.get("SOURCE_DATE_EPOCH")
            timestamp = (
                timestamp_from_source_date_epoch(source_date_epoch) if source_date_epoch is not None else None
            )
            record_publication_analysis_stage(root, timestamp=timestamp)
    except Exception as exc:  # noqa: BLE001 - top-level orchestrator guard
        print(f"analysis pipeline failed: {exc}", file=sys.stderr)
        return 1

    for label, path in paths.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
