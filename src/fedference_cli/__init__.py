"""Installed command-line facade for evidence-bound Active Fedference runs.

The implementation is intentionally split by responsibility: ``_support``
owns safe filesystem and receipt mechanics, ``_commands`` owns domain-runner
dispatch, and ``_parser`` owns the stable process interface.  ``main`` and the
historical ``_report_fallbacks`` import remain available for compatibility.
"""

from __future__ import annotations

from ._parser import main
from ._support import _report_fallbacks

__all__ = ["_report_fallbacks", "main"]
