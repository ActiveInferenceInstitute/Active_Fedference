#!/usr/bin/env python3
"""Pre-flight check for manuscript rendering prerequisites."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# ``active_fedference`` is a standalone checkout nested below the shared
# template repository in this workspace.  The third parent is the template
# root itself; appending another ``template`` would silently disable the
# real preflight and turn a release prerequisite into a no-op.
TEMPLATE_ROOT = PROJECT_ROOT.parent.parent.parent


def _load_preflight():
    if not (TEMPLATE_ROOT / "infrastructure").is_dir():
        return None, f"template preflight unavailable: {TEMPLATE_ROOT}"
    sys.path.insert(0, str(TEMPLATE_ROOT))
    try:
        from infrastructure.rendering.preflight import run_manuscript_preflight
    except ImportError as exc:
        return None, f"template preflight unavailable: {exc}"

    return run_manuscript_preflight, ""


def main() -> int:
    """Run the manuscript preflight check; return a 0/1 process exit code."""
    run_manuscript_preflight, unavailable = _load_preflight()
    if run_manuscript_preflight is None:
        print(unavailable, file=sys.stderr)
        return 0
    manuscript_dir = PROJECT_ROOT / "manuscript"
    ok, message = run_manuscript_preflight(manuscript_dir)
    if ok:
        return 0
    sys.stderr.write(message)
    return 1


if __name__ == "__main__":
    sys.exit(main())
