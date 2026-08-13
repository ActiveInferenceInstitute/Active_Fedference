#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


def main(argv: list[str] | None = None) -> int:
    """Prepare and validate the generated web package."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None)
    args = parser.parse_args(argv)

    from project_paths import resolve_script_project_root
    from publication.web_package import (
        mirror_web_figures,
        normalize_web_xrefs,
        sanitize_machine_paths,
        validate_web_package,
    )

    root = resolve_script_project_root(_PROJECT_ROOT, args.project_root)
    sanitized = sanitize_machine_paths(root)
    copied = mirror_web_figures(root)
    replacements = normalize_web_xrefs(root)
    result = validate_web_package(root)
    print(f"machine_paths_sanitized: {len(sanitized)}")
    print(f"web_figures_copied: {len(copied)}")
    print(f"web_xrefs_normalized: {replacements}")
    print(f"web_html_files: {result.html_files}")
    print(f"web_assets_checked: {result.assets_checked}")
    if result.ok:
        print("web_package: PASS")
        return 0
    for missing in result.missing_assets:
        print(f"MISSING_ASSET {missing}", file=sys.stderr)
    for raw in result.raw_xrefs:
        print(f"RAW_XREF {raw}", file=sys.stderr)
    for broken in result.broken_xrefs:
        print(f"BROKEN_XREF {broken}", file=sys.stderr)
    for malformed in result.malformed_markup:
        print(f"MALFORMED_MARKUP {malformed}", file=sys.stderr)
    print("web_package: FAIL", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
