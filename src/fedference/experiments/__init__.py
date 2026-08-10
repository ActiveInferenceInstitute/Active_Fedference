"""Experiment harness for categorical source-mechanism analogues.

Thin, JSON-serialisable orchestrators wiring FedGVI / active-inference primitives.
Public API is re-exported from submodules for backward compatibility.
"""

from __future__ import annotations

from ..colonies import healthy_colony, soft_colony
from ..hierarchy_tasks import run_hierarchy_task_pilot, simulate_hierarchy_task
from ..single_machine import run_calibration_pilot, run_fedgvi_bnn_pilot
from ._common import (
    _EPS,
    _N_BOOT,
    ArrayF,
    _consensus_accuracy,
    _divergence_to_robustness,
    _finite_cohens_d,
    _finite_d_equivalent,
    _sample_observation,
)
from .belief_sharing import (
    run_belief_sharing,
    run_emergence,
    run_language_acquisition,
    summarize_language_acquisition,
)
from .complexity import run_complexity_scaling
from .conditional_world import (
    ConditionalScenario,
    conditional_scenario_grid,
    run_belief_quality_sensitivity,
    run_conditional_world_generalization,
)
from .cross_study import summarize_cross_study
from .diagnostics import (
    run_bnn_robustness_report,
    run_efe_decomposition_report,
    run_influence_weights_report,
    run_variational_aggregation_report,
)
from .gallery import (
    _DIRECTIONAL_KINDS,
    _ENTROPY_KINDS,
    _GALLERY_RELIABLE_WIN_FRACTION,
    run_contamination_gallery,
    run_robustness_onset,
)
from .heuristic_characterization import run_heuristic_characterization
from .navigation import run_disjoint_fov_world, run_efe_navigation_test
from .parameter_recovery import run_parameter_recovery
from .report_bundle import (
    disjoint_fov_report,
    hierarchical_world_report,
    moving_world_report,
    nlevel3_world_report,
)
from .review_grid import (
    DEFAULT_REVIEW_GRID_RATES,
    REVIEW_GRID_ATTACKS,
    run_review_grid,
)
from .robustness import run_robustness_sweep
from .sensitivity import run_belief_sharing_sensitivity, run_hierarchical_sensitivity
from .worlds import (
    run_3level_world,
    run_hierarchical_bmr,
    run_hierarchical_world,
    run_moving_world,
    run_nlevel_world,
)

# Backward-compatible aliases for tests and internal callers.
_soft_colony = soft_colony

__all__ = [
    "ArrayF",
    "_EPS",
    "_N_BOOT",
    "_consensus_accuracy",
    "_divergence_to_robustness",
    "_DIRECTIONAL_KINDS",
    "_ENTROPY_KINDS",
    "_finite_cohens_d",
    "_finite_d_equivalent",
    "_GALLERY_RELIABLE_WIN_FRACTION",
    "_sample_observation",
    "_soft_colony",
    "healthy_colony",
    "soft_colony",
    "run_3level_world",
    "run_hierarchical_bmr",
    "run_heuristic_characterization",
    "run_belief_sharing",
    "run_belief_sharing_sensitivity",
    "run_calibration_pilot",
    "run_bnn_robustness_report",
    "run_contamination_gallery",
    "run_complexity_scaling",
    "ConditionalScenario",
    "conditional_scenario_grid",
    "run_belief_quality_sensitivity",
    "run_conditional_world_generalization",
    "run_disjoint_fov_world",
    "run_efe_decomposition_report",
    "run_efe_navigation_test",
    "run_emergence",
    "run_hierarchical_sensitivity",
    "run_hierarchical_world",
    "run_hierarchy_task_pilot",
    "run_influence_weights_report",
    "run_language_acquisition",
    "summarize_language_acquisition",
    "run_moving_world",
    "run_nlevel_world",
    "run_parameter_recovery",
    "run_robustness_onset",
    "run_review_grid",
    "DEFAULT_REVIEW_GRID_RATES",
    "REVIEW_GRID_ATTACKS",
    "disjoint_fov_report",
    "hierarchical_world_report",
    "moving_world_report",
    "nlevel3_world_report",
    "run_robustness_sweep",
    "run_fedgvi_bnn_pilot",
    "simulate_hierarchy_task",
    "run_variational_aggregation_report",
    "summarize_cross_study",
]
