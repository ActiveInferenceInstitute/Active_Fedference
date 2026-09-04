#!/usr/bin/env python3
"""Pre-flight check for manuscript rendering prerequisites."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _load_preflight(template_root: Path):
    sys.path.insert(0, str(template_root))
    from infrastructure.rendering.preflight import run_manuscript_preflight

    return run_manuscript_preflight


def main(argv: list[str] | None = None) -> int:
    """Run the manuscript preflight check; return a 0/1 process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="standalone checkout whose manuscript should be checked",
    )
    parser.add_argument(
        "--template-root",
        type=Path,
        required=True,
        help="clean, source-locked Template checkout providing the renderer",
    )
    args = parser.parse_args(argv)
    from project_paths import resolve_script_project_root

    project_root = resolve_script_project_root(PROJECT_ROOT, args.project_root)
    template_root = args.template_root.expanduser().resolve()
    from publication.template_renderer import require_locked_template_renderer

    require_locked_template_renderer(project_root, template_root)
    run_manuscript_preflight = _load_preflight(template_root)
    manuscript_dir = project_root / "manuscript"
    ok, message = run_manuscript_preflight(manuscript_dir)
    if ok:
        return 0
    sys.stderr.write(message)
    return 1


if __name__ == "__main__":
    sys.exit(main())
