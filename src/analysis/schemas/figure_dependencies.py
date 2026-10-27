"""Figure dependencies report contracts."""

from __future__ import annotations

from analysis.schemas.figure_evidence_dependencies import _FIGURE_EVIDENCE_DEPENDENCIES
from analysis.schemas.figure_study_dependencies import _FIGURE_STUDY_DEPENDENCIES
from analysis.schemas.figure_world_dependencies import _FIGURE_WORLD_DEPENDENCIES
from analysis.schemas.primitives import FigureDependencyContract

FIGURE_DEPENDENCY_CONTRACTS: dict[str, tuple[FigureDependencyContract, ...]] = {
    **_FIGURE_STUDY_DEPENDENCIES,
    **_FIGURE_WORLD_DEPENDENCIES,
    **_FIGURE_EVIDENCE_DEPENDENCIES,
}
