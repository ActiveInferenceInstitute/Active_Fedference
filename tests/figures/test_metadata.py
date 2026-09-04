"""Figure provenance and publication-style contracts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.visual_contracts import COMPLEX_FIGURE_GENERATORS
from analysis.workflow import _belief_sharing_report, _sensitivity_report
from experiment_config import ExperimentConfig
from fedference.experiments import (
    run_conditional_world_generalization,
    run_heuristic_characterization,
    run_parameter_recovery,
)
from figures import FIGURE_METADATA, apply_style
from figures._common import (
    FIGURE_EXPORT_DPI,
    MIN_QUANTITATIVE_FONT_SIZE,
    MIN_SCHEMATIC_FONT_SIZE,
)

_EXACT_VALUE_FIGURES = {
    "belief_quality",
    "bnn_robustness",
    "complexity_scaling",
    "conditional_world",
    "cross_study_summary",
    "robustness_review_grid",
    "sensitivity_heatmap",
}


def test_every_figure_generator_has_complete_metadata() -> None:
    figure_dir = Path(__file__).resolve().parents[2] / "src" / "figures"
    generators = {
        path.stem
        for path in figure_dir.glob("*.py")
        if path.name not in {"__init__.py", "_common.py", "_metadata.py"}
    }
    assert generators == set(FIGURE_METADATA)
    required = {
        "status",
        "source_relation",
        "source_figure",
        "source_equation",
        "source_citation",
        "estimand",
        "unit",
        "uncertainty",
        "replication_unit",
        "alt_text",
    }
    for generator, metadata in FIGURE_METADATA.items():
        assert required <= set(metadata), generator
        assert all(isinstance(metadata[key], str) for key in required)
        assert metadata["status"]
        assert metadata["source_relation"]
        assert metadata["estimand"]
        assert metadata["unit"]
        assert metadata["uncertainty"]
        assert metadata["replication_unit"]
        assert metadata["alt_text"]
        assert len(metadata["alt_text"]) <= 500
        assert metadata["estimand"] != "project-specific diagnostic quantity"
        assert metadata["unit"] != "declared in the embedded caption"
        assert metadata["uncertainty"] != "caption declares the interval or deterministic status"
        assert metadata["replication_unit"] != "caption declares the replication unit"


def test_complex_figures_have_structured_long_descriptions() -> None:
    for generator in COMPLEX_FIGURE_GENERATORS:
        description = FIGURE_METADATA[generator].get("long_description")
        assert isinstance(description, str) and description, generator
        assert "\n\n" in description, generator
        assert len(description.split()) >= 90, generator
        assert any(
            boundary in description.lower() for boundary in ("does not", "not ", "no claim")
        ), generator


def test_reflowed_complex_descriptions_match_visible_panel_contracts() -> None:
    application = FIGURE_METADATA["application_integrity_flow"]
    application_alt = application["alt_text"]
    application_long = application["long_description"]
    assert "two-by-two key" in application_alt
    assert "neutral dashed exit" in application_alt
    assert "separates artifact integrity" in application_long
    assert "solid purple arrows" in application_long.lower()
    assert "dotted sequence" in application_long.lower()
    for solver_status in (
        "nominal",
        "converged_with_fallback",
        "not_converged",
        "not_converged_with_fallback",
    ):
        assert solver_status in application_long

    message = FIGURE_METADATA["message_passing"]["alt_text"]
    assert "Four-band protocol schematic" in message
    assert "three standard, heuristic, or variational fusion routes" in message
    assert "claim owners separate" in message
    assert "Three-lane" not in message

    evidence = FIGURE_METADATA["evidence_replication_map"]
    assert "fourteen numbered result-family rows" in evidence["alt_text"]
    assert "six directly labeled columns" in evidence["alt_text"]
    assert "row 1 through row 14" in evidence["long_description"]
    assert "paired free energy" in evidence["long_description"]
    assert "acuity recovery" in evidence["long_description"]
    assert "cross-study summary" in evidence["long_description"]

    review = FIGURE_METADATA["robustness_review_grid"]
    assert "full-width top heatmap" in review["alt_text"]
    assert "dedicated right label lane" in review["alt_text"]
    assert "Read the portrait figure from top to bottom" in review["long_description"]
    assert "reserved right-side label lane" in review["long_description"]

    cross_study = FIGURE_METADATA["cross_study_summary"]
    assert "two-column composition" in cross_study["alt_text"]
    assert "full left column" in cross_study["long_description"]
    assert "six signed accuracy" in cross_study["long_description"]
    assert "two signed information" in cross_study["long_description"]
    assert "one parameter-recovery" in cross_study["long_description"]
    assert "separate harmonized seed-level rerun" in cross_study["long_description"]
    assert "Study 4 row is the within-run display-selected maximum" in cross_study["long_description"]

    provenance = FIGURE_METADATA["source_render_provenance"]
    assert "five-input dependency matrix" in provenance["alt_text"]
    assert "crossing-free producer path" in provenance["alt_text"]
    assert "five source-owned inputs" in provenance["long_description"]
    assert "seven downstream targets" in provenance["long_description"]
    assert "fourteen changed owners" in provenance["long_description"]
    assert "immediate stale targets" in provenance["long_description"]
    assert "back to the changed owner" in provenance["long_description"]


def test_parameter_recovery_report_metadata_and_caption_agree_on_trial_unit() -> None:
    report = run_parameter_recovery(
        23,
        acuity_grid=(0.60, 0.80),
        n_observations=4,
        n_trials=2,
        fit_resolution=4,
    )
    metadata = FIGURE_METADATA["parameter_recovery"]
    caption_source = (
        Path(__file__).resolve().parents[2] / "manuscript" / "S15_results_parameter_recovery.md"
    ).read_text(encoding="utf-8")

    assert report["replication_unit"] == "synthetic trial"
    assert report["analysis_unit"] == metadata["replication_unit"]
    assert report["seed_role"] == metadata["seed_role"]
    assert "r_squared" in report
    assert "coefficient of determination" in metadata["estimand"]
    assert "R-sq" in metadata["unit"]
    generator_source = (
        Path(__file__).resolve().parents[2] / "src" / "figures" / "parameter_recovery.py"
    ).read_text(encoding="utf-8")
    assert "R²" in generator_source
    assert "coefficient of determination $R^2$" in caption_source
    assert "unitless" in caption_source
    assert "independent synthetic trials within each true-acuity grid point" in caption_source
    assert "base RNG stream" in caption_source
    assert "not the independent analysis unit" in caption_source


def test_heuristic_report_metadata_and_caption_agree_on_declared_grid_unit() -> None:
    report = run_heuristic_characterization(23)
    metadata = FIGURE_METADATA["heuristic_breakdown"]
    caption_source = (
        Path(__file__).resolve().parents[2]
        / "manuscript"
        / "S17_results_heuristic_characterization.md"
    ).read_text(encoding="utf-8")

    assert report["independent_unit"] in metadata["replication_unit"]
    assert "no population resampling" in metadata["replication_unit"]
    assert "declared seeded scenario row is a computational unit" in caption_source
    assert "not an exchangeable population replicate" in caption_source
    assert "no confidence interval is shown" in caption_source


def test_sensitivity_report_metadata_and_caption_agree_on_nested_trial_unit() -> None:
    report = _sensitivity_report(seed=23, n_trials=2)
    metadata = FIGURE_METADATA["sensitivity_heatmap"]
    caption_source = (
        Path(__file__).resolve().parents[2] / "manuscript" / "S13_results_sensitivity.md"
    ).read_text(encoding="utf-8")
    normalized_caption = " ".join(caption_source.split())

    assert report["replication_unit"] == metadata["replication_unit"]
    assert report["uncertainty"] == metadata["uncertainty"]
    assert "trials are nested within their configured cell" in normalized_caption
    assert "no resampling interval is shown" in normalized_caption


def test_declared_exact_value_figures_have_stable_fallback_identifiers() -> None:
    values = []
    for generator in _EXACT_VALUE_FIGURES:
        identifier = FIGURE_METADATA[generator].get("exact_value_fallback")
        assert isinstance(identifier, str) and identifier.startswith("fig-values:"), generator
        values.append(identifier)
    assert len(values) == len(set(values))


def test_publication_style_contract_is_readable() -> None:
    apply_style()
    assert plt.rcParams["savefig.dpi"] == FIGURE_EXPORT_DPI
    assert plt.rcParams["axes.titlesize"] >= 15
    assert plt.rcParams["axes.labelsize"] >= 13
    assert plt.rcParams["xtick.labelsize"] >= 11
    assert plt.rcParams["legend.fontsize"] >= MIN_QUANTITATIVE_FONT_SIZE
    assert MIN_SCHEMATIC_FONT_SIZE >= 8.5


def test_source_analogue_uncertainty_metadata_matches_captions() -> None:
    assert (
        FIGURE_METADATA["free_energy_comparison"]["uncertainty"]
        == "95-percent percentile-bootstrap interval of paired seed differences"
    )
    assert FIGURE_METADATA["emergence_bmr"]["uncertainty"] == (
        "none; deterministic closed-form comparison on a single posterior"
    )
    assert FIGURE_METADATA["emergence_bmr"]["replication_unit"] == "not applicable"


def test_configured_overview_metadata_preserves_quantitative_schematic_content() -> None:
    graphical = FIGURE_METADATA["graphical_abstract"]
    overview = FIGURE_METADATA["system_overview"]

    assert graphical["status"] == "deterministic formal/mechanistic schematic"
    assert "configured colony and outcome summaries" in graphical["source_relation"]
    assert "true-state probability mass" in graphical["estimand"]
    assert "probability mass" in graphical["unit"]
    assert "one configured explanatory colony" in graphical["replication_unit"]

    assert overview["status"] == "deterministic configured diagnostic schematic"
    assert "failure-and-repair diagnostic" in overview["source_relation"]
    assert overview["estimand"] == (
        "displayed posterior-mass and normalized influence-weight contrasts"
    )
    assert overview["unit"] == "probability mass or normalized server weight"
    assert "one deterministic configured colony" in overview["uncertainty"]


def test_belief_quality_metadata_and_caption_limit_visible_estimands() -> None:
    metadata = FIGURE_METADATA["belief_quality"]
    caption_source = (
        Path(__file__).resolve().parents[2]
        / "manuscript"
        / "28_supplement_extended_methods.md"
    ).read_text(encoding="utf-8")

    assert "displayed categorical log score and binned reliability coordinates" in metadata["estimand"]
    assert "report-only secondary diagnostics" in metadata["estimand"]
    assert metadata["unit"] == (
        "nats for log score; confidence and accuracy fractions for reliability coordinates"
    )
    assert "displayed control log scores" in metadata["uncertainty"]
    assert "Brier score and expected calibration error are retained as report-only" in caption_source
    assert "are not plotted in this two-panel figure" in caption_source


def test_bnn_proxy_metadata_denies_generalized_bayes_or_alpha_renyi_promotion() -> None:
    metadata = FIGURE_METADATA["bnn_robustness"]
    relation = metadata["source_relation"]
    manuscript_root = Path(__file__).resolve().parents[2] / "manuscript"
    design = (manuscript_root / "12_methods_experimental_design.md").read_text(
        encoding="utf-8"
    )
    syntax = (manuscript_root / "SYNTAX.md").read_text(encoding="utf-8")

    assert "two joint loss/L2 configurations" in relation
    assert "legacy AR selects stronger L2" in relation
    assert "rather than a weight-space divergence" in relation
    assert "composite loss-and-L2 contrast rather than an RCCE-only effect" in design
    assert "does not implement a FedGVI generalized-posterior" in design
    assert "isolating the rigorous axis" not in design
    descent_row = next(line for line in syntax.splitlines() if "{#fig:descent-comparison}" in line)
    assert "Variational/objective-backed server initialization diagnostic" in descent_row
    assert "heuristic axis" not in descent_row
    assert "generalized-Bayes" not in relation
    assert "joint NLL/L2=0.05 and RCCE/L2=0.10 configurations" in metadata["estimand"]
    assert "cannot isolate an RCCE-only effect" in metadata["alt_text"]


def test_free_energy_report_metadata_and_caption_agree_on_difference_sign() -> None:
    report = _belief_sharing_report(ExperimentConfig(n_agents=3, n_seeds=3))
    metadata = FIGURE_METADATA["free_energy_comparison"]
    caption_source = (
        Path(__file__).resolve().parents[2] / "manuscript" / "16_results_belief_sharing.md"
    ).read_text(encoding="utf-8")

    expected = np.asarray(report["incommunicado_free_energy"]) - np.asarray(
        report["communicating_free_energy"]
    )
    assert report["difference_definition"] == "incommunicado_minus_communicating"
    assert np.allclose(report["paired_free_energy_difference"], expected)
    assert metadata["estimand"] == "incommunicado minus communicating colony mean free energy"
    assert r"\bar F_{\text{solo}} - \bar F_{\text{share}}" in caption_source
    assert "positive values indicate lower free energy with communication" in caption_source


def test_conditional_world_report_metadata_and_caption_agree_on_nested_unit() -> None:
    report = run_conditional_world_generalization(seed=3, n_seeds=2, n_trials=1)
    metadata = FIGURE_METADATA["conditional_world"]
    caption_source = (
        Path(__file__).resolve().parents[2] / "manuscript" / "28_supplement_extended_methods.md"
    ).read_text(encoding="utf-8")

    assert report["independent_unit"] == "seeded world/scenario row"
    assert report["independent_unit"] in metadata["replication_unit"]
    assert "trials nested within row" in metadata["replication_unit"]
    assert "independent unit is the seeded world/scenario row" in caption_source
    assert "trials nested within each row" in caption_source
