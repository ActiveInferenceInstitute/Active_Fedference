"""Canonical Stage-02 artifact declarations for Active Fedference.

The analysis producer and its audit command need the same expected-path
contract without running experiments or scanning whatever happens to be
present under output/. Keeping the declaration in a small importable module
makes the contract reusable from tests, CI, and sibling checkouts.
"""

from __future__ import annotations

from pathlib import Path

# These are the report files written by analysis.workflow.run_analysis_pipeline.
ANALYSIS_REPORT_FILENAMES: tuple[str, ...] = (
    "belief_quality.json",
    "belief_sharing.json",
    "bnn_robustness.json",
    "bnn_torch.json",
    "complexity_scaling.json",
    "conditional_world.json",
    "contamination_gallery.json",
    "cross_study_summary.json",
    "disjoint_fov_world.json",
    "efe_decomposition.json",
    "emergence.json",
    "heuristic_characterization.json",
    "hierarchical_bmr.json",
    "hierarchical_world.json",
    "language_acquisition.json",
    "moving_world.json",
    "nlevel3_world.json",
    "parameter_recovery.json",
    "robust_influence_weights.json",
    "robustness_onset.json",
    "robustness_review_grid.json",
    "robustness_sweep.json",
    "variational_aggregation.json",
)

# The pipeline writes these publication-facing figure pairs and the registry.
ANALYSIS_FIGURE_FILENAMES: tuple[str, ...] = (
    "aggregation_descent.png",
    "belief_heatmap.png",
    "belief_quality.png",
    "bnn_robustness.png",
    "bounded_influence.png",
    "complexity_scaling.png",
    "conditional_world.png",
    "contamination_gallery.png",
    "cross_study_summary.png",
    "descent_comparison.png",
    "disjoint_fov_world.png",
    "efe_decomposition.png",
    "emergence_bmr.png",
    "free_energy_comparison.png",
    "generative_model_schema.png",
    "graphical_abstract.png",
    "heuristic_breakdown.png",
    "hierarchical_bmr.png",
    "hierarchical_pomdp.png",
    "language_kl_decay.png",
    "message_passing.png",
    "moving_world.png",
    "parameter_recovery.png",
    "pomdp_loop.png",
    "robust_influence_weights.png",
    "robustness_onset.png",
    "robustness_review_grid.png",
    "robustness_sweep.png",
    "sensitivity_heatmap.png",
)

ANALYSIS_DATA_FILENAMES: tuple[str, ...] = (
    "analysis_execution.json",
    "manuscript_variables.json",
    "stage_timings.json",
)


def expected_artifacts(project_root: Path) -> dict[str, Path]:
    """Return the source-declared Stage-02 artifact paths.

    The function only declares paths; it does not inspect directory contents,
    run analysis, or accept an arbitrary file as evidence that a required
    artifact exists. The downstream manuscript-variable cache is included
    because the audit command historically validates the complete analysis
    hand-off rather than only the immediate report directory.
    """
    root = Path(project_root).resolve()
    reports = root / "output" / "reports"
    figures = root / "output" / "figures"
    data = root / "output" / "data"
    return {
        **{f"report:{name}": reports / name for name in ANALYSIS_REPORT_FILENAMES},
        **{f"figure:{name}": figures / name for name in ANALYSIS_FIGURE_FILENAMES},
        "figure_registry": figures / "figure_registry.json",
        **{f"data:{name}": data / name for name in ANALYSIS_DATA_FILENAMES},
    }
