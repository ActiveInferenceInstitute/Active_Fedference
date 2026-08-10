#!/usr/bin/env python3
"""Fail closed on structurally, textually, or visibly broken publication surfaces."""

from __future__ import annotations

import argparse
from pathlib import Path

from publication.surface_validation import validate_rendered_surfaces


def main(argv: list[str] | None = None) -> int:
    """Validate the combined manuscript, slides, and web package."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
    )
    args = parser.parse_args(argv)
    result = validate_rendered_surfaces(args.project_root)
    print(
        "rendered surfaces: "
        f"{int(result.manuscript_pdf)} combined PDF, "
        f"{result.manuscript_logs} manuscript logs, "
        f"{result.slide_pdfs} slide PDFs, {result.slide_tex} TeX sources, "
        f"{result.slide_logs} logs, {result.web.html_files} web HTML files, "
        f"{result.web.assets_checked} web assets"
    )
    if result.ok:
        print("rendered surfaces verified")
        return 0
    print("rendered surface findings:")
    for finding in result.findings:
        print(f"- {finding}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
