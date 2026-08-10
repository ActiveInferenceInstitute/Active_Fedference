#!/usr/bin/env python3
"""Thin orchestrator: generate API documentation.

Public entry-point — delegates entirely to the canonical implementation in
``_generate_api_docs.py``.  All generation logic lives in
:func:`src.documentation.run_api_doc_generation`; this script only wires the
path, calls it, prints the produced files, and maps the result to an exit code.

Exit codes:
    0   API reference (and best-effort glossary) written
    1   unexpected error
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def main() -> int:
    """Generate the API/glossary documentation artifacts; return a 0/1 exit code."""
    from project_paths import resolve_env_project_root
    from src.documentation import run_api_doc_generation

    # Honor ACTIVE_FEDFERENCE_PROJECT_ROOT so subprocess tests write
    # output/docs/*.md into a scaffold, not the real tree. An invalid
    # override raises loudly, never falls back to the real root.
    root = resolve_env_project_root(PROJECT_ROOT)

    try:
        docs_files = run_api_doc_generation(root)
    except Exception as exc:  # noqa: BLE001 - top-level orchestrator guard
        print(f"API documentation generation failed: {exc}", file=sys.stderr)
        return 1

    for file_path in docs_files.values():
        if file_path:
            print(file_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
