"""Installed command-line facade for evidence-bound Active Fedference runs.

The implementation is intentionally split by responsibility: ``_support``
owns safe filesystem and receipt mechanics, ``_commands`` owns domain-runner
dispatch, and ``_parser`` owns the stable process interface. ``main`` is the
public entry point; diagnostic helpers remain internal to ``_support``.
"""

from __future__ import annotations

from ._parser import main

__all__ = ["main"]
