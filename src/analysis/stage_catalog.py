"""Lazy catalog of the existing analysis reports, preserving their scientific controls.

The Friston et al. (2024) and Mildner et al. (2025) interpretations remain with
existing producers. This module selects calls; it does not implement mathematics,
load configuration, clamp budgets, or persist reports.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from experiment_config import ExperimentConfig

STAGE_SCHEMAS: dict[str, str] = {
    "application_integrity_flow": "application_integrity_flow",
    "evidence_replication_map": "evidence_replication_map",
    "source_render_provenance": "source_render_provenance",
    "belief_sharing": "belief_sharing",
    "language_acquisition": "language_acquisition",
    "emergence": "emergence",
    "robustness_sweep": "robustness_sweep",
    "hierarchical_bmr": "hierarchical_bmr",
    "heuristic_characterization": "heuristic_characterization",
    "efe_decomposition": "efe_decomposition",
    "robust_influence_weights": "robust_influence_weights",
    "bnn_robustness": "bnn_robustness",
    "bnn_torch": "bnn_torch",
    "variational_aggregation": "variational_aggregation",
    "contamination_gallery": "contamination_gallery",
    "robustness_onset": "robustness_onset",
    "conditional_world": "conditional_world",
    "robustness_review_grid": "robustness_review_grid",
    "belief_quality": "belief_quality",
    "complexity_scaling": "complexity_scaling",
    "sensitivity": "sensitivity",
    "moving_world": "moving_world",
    "parameter_recovery": "parameter_recovery",
    "hierarchical_world": "hierarchical_world",
    "nlevel3_world": "nlevel3_world",
    "cross_study_summary": "cross_study_summary",
    "disjoint_fov_world": "disjoint_fov_world",
}


def report_producers(
    config: ExperimentConfig,
    project_root: Path,
    *,
    smoke: bool,
) -> dict[str, Callable[[], dict[str, Any]]]:
    """Bind lazy producers to an already effective configuration and explicit root.

    The caller owns smoke-budget clamping. Existing producer-specific smoke
    options are retained here; configuration/file access and numerical work occur
    only when the selected callable runs. Importing the workflow at call time
    avoids a cycle when the workflow itself consumes this catalog.
    """
    from . import workflow

    return {
        "application_integrity_flow": lambda: cast(
            dict[str, Any], workflow.application_integrity_flow_contract()
        ),
        "evidence_replication_map": lambda: cast(
            dict[str, Any], workflow.evidence_replication_map_contract()
        ),
        "source_render_provenance": lambda: cast(
            dict[str, Any], workflow.source_render_provenance_contract()
        ),
        "belief_sharing": lambda: workflow._belief_sharing_report(config),
        "language_acquisition": lambda: workflow._language_report(config),
        "emergence": lambda: workflow._emergence_report(config),
        "robustness_sweep": lambda: workflow._robustness_report(config),
        "hierarchical_bmr": lambda: workflow.run_hierarchical_bmr(),
        "heuristic_characterization": lambda: workflow.run_heuristic_characterization(config.seeds[0]),
        "efe_decomposition": lambda: workflow._efe_terms(config),
        "robust_influence_weights": lambda: workflow._influence_weights(config),
        "bnn_robustness": lambda: workflow._bnn_report(config, smoke=smoke),
        "bnn_torch": lambda: workflow._configured_bnn_torch_report(config, project_root=project_root),
        "variational_aggregation": lambda: workflow._variational_aggregation_report(config),
        "contamination_gallery": lambda: workflow._contamination_gallery_report(config, smoke=smoke),
        "robustness_onset": lambda: workflow._robustness_onset_report(config, smoke=smoke),
        "conditional_world": lambda: workflow._conditional_world_report(config),
        "robustness_review_grid": lambda: workflow._review_grid_report(config, smoke=smoke),
        "belief_quality": lambda: workflow._belief_quality_report(config),
        "complexity_scaling": lambda: workflow.run_complexity_scaling(
            config.complexity.for_smoke() if smoke else config.complexity
        ),
        "sensitivity": lambda: workflow._sensitivity_report(
            seed=config.seeds[0], n_trials=5 if smoke else 20
        ),
        "moving_world": lambda: workflow._moving_world_report(config),
        "parameter_recovery": lambda: workflow._parameter_recovery_report(config, smoke=smoke),
        "hierarchical_world": lambda: workflow.hierarchical_world_report(
            config.seeds[0], n_trials=config.n_trials, n_multiseed=config.replicate_seeds
        ),
        "nlevel3_world": lambda: workflow.nlevel3_world_report(
            config.seeds[0], n_trials=config.n_trials, n_multiseed=config.replicate_seeds
        ),
        "cross_study_summary": lambda: workflow.summarize_cross_study(
            config.seeds[0], n_seeds=config.replicate_seeds, n_trials=config.cross_study_n_trials
        ),
        "disjoint_fov_world": lambda: workflow.disjoint_fov_report(0, n_multiseed=config.replicate_seeds),
    }
