#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


def main() -> int:
    """Validate generated web assets, references, markup, and accessibility."""
    from publication.web_package import validate_web_package

    result = validate_web_package(_PROJECT_ROOT)
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
    for issue in result.accessibility_issues:
        print(f"ACCESSIBILITY {issue}", file=sys.stderr)
    print("web_package: FAIL", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
