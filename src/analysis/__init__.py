"""Active Fedference analysis package (re-export barrel).

Exposes the thin orchestrator that runs the :mod:`fedference.experiments`
categorical source-mechanism analogues and diagnostics and writes reports + figures to
``output/``.
"""

from __future__ import annotations

from .artifacts import expected_artifacts
from .workflow import main, run_analysis_pipeline

__all__ = ["expected_artifacts", "main", "run_analysis_pipeline"]
