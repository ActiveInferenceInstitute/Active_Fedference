#!/usr/bin/env python3
"""Compatibility shim for :mod:`scripts.generate_api_docs`.

The historical underscored entry point remains callable for local automation,
but the public command is ``generate_api_docs.py``. Keeping one implementation
prevents the two entry points from drifting in root selection, error handling,
or output reporting.

Exit codes:
    0   API reference (and best-effort glossary) written
    1   unexpected error
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def main(argv: list[str] | None = None) -> int:
    """Delegate to the canonical API-documentation entry point."""
    from generate_api_docs import main as generate_main

    return generate_main(argv)


if __name__ == "__main__":
    sys.exit(main())
