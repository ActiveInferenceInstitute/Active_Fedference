"""Active Fedference analysis package (re-export barrel).

Exposes the thin orchestrator that runs the :mod:`fedference.experiments`
categorical source-mechanism analogues and diagnostics and writes reports + figures to
``output/``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .artifacts import expected_artifacts

if TYPE_CHECKING:
    from .workflow import main, run_analysis_pipeline

__all__ = ["expected_artifacts", "main", "run_analysis_pipeline"]


def __getattr__(name: str) -> Any:
    """Lazily expose the orchestrator without creating an import cycle.

    Publication freshness imports the small :mod:`analysis.artifacts` contract,
    while the workflow itself consumes publication freshness.  Importing the
    workflow eagerly from the package barrel would therefore make either
    import order fail.  The public convenience exports remain available, but
    their heavier dependency graph is loaded only when a caller requests one.
    """
    if name in {"main", "run_analysis_pipeline"}:
        from . import workflow

        return getattr(workflow, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
