"""Regression checks for the manuscript claim-boundary refactor."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANUSCRIPT = ROOT / "manuscript"


def _main_manuscript_text() -> str:
    paths = sorted(MANUSCRIPT.glob("*.md"))
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in paths
        if path.name not in {"AGENTS.md", "README.md", "SYNTAX.md"}
    ).lower()


def test_claim_audit_is_part_of_the_research_record() -> None:
    audit = ROOT / "docs" / "research" / "manuscript-claim-audit.md"
    text = audit.read_text(encoding="utf-8")
    assert "Load-bearing claims" in text
    assert "Conditional empirical" in text
    assert "Scoped implementation fact" in text


def test_visual_claim_audit_is_part_of_the_research_record() -> None:
    audit = ROOT / "docs" / "research" / "visual-claim-audit.md"
    text = audit.read_text(encoding="utf-8")
    for marker in ("graphical-abstract.png", "message_passing.png", "robust_aggregate", "POMDP"):
        assert marker in text


def test_manuscript_does_not_reintroduce_unbounded_claim_language() -> None:
    text = _main_manuscript_text()
    stale_phrases = (
        "model-class-agnostic",
        "arbitrary depth",
        "proof-by-test",
        "communication is necessary for consensus",
        "novel studies beyond",
        "reproduce each exactly",
        "neither has reached across the divide",
        "obstruction is structural",
        "yields no joint objective",
        "minimizes no closed-form free energy",
    )
    assert not [phrase for phrase in stale_phrases if phrase in text]


def test_section_titles_and_new_claim_boundary_sections_are_present() -> None:
    expected = {
        "01_introduction.md": "Introduction: from belief sharing to robust generalized Bayes",
        "02_gap.md": "Research gap and claim boundary",
        "07_methods_aggregation.md": (
            "Aggregation and message passing: standard pool, heuristic, and variational server"
        ),
        "09_methods_generative_model.md": (
            "Generative model: categorical states, observations, actions, and hierarchy"
        ),
        "19_results_robustness.md": (
            "Contamination sweep: regime-dependent server behavior under declared attacks"
        ),
        "20_results_baseline.md": "Exploratory generalized-Bayes logistic-regression baseline",
        "21_discussion_findings.md": "Discussion: what the evidence supports",
        "22_discussion_related_work.md": (
            "Related work: active inference, federated Bayes, and the scoped bridge"
        ),
        "25_conclusion.md": "Conclusion: a recovery-tested bridge with bounded claims",
    }
    for filename, title in expected.items():
        first_line = (MANUSCRIPT / filename).read_text(encoding="utf-8").splitlines()[0]
        assert title in first_line

    discussion = (MANUSCRIPT / "21_discussion_findings.md").read_text(encoding="utf-8")
    assert "{#sec:discussion-identifiability}" in discussion
    introduction = (MANUSCRIPT / "01_introduction.md").read_text(encoding="utf-8")
    assert "{#sec:intro-questions}" in introduction


def test_visual_scholarship_claim_language_stays_bounded() -> None:
    """Key diagnostics retain their declared evidence class in source prose."""
    emergence = (MANUSCRIPT / "18_results_emergence.md").read_text(encoding="utf-8").lower()
    hierarchical = (MANUSCRIPT / "S16_results_hierarchical_bmr.md").read_text(encoding="utf-8").lower()
    baseline = (MANUSCRIPT / "20_results_baseline.md").read_text(encoding="utf-8").lower()
    sharing = (MANUSCRIPT / "16_results_belief_sharing.md").read_text(encoding="utf-8")
    robustness = (MANUSCRIPT / "19_results_robustness.md").read_text(encoding="utf-8").lower()
    baseline_flat = " ".join(baseline.split())

    assert "configured bmr sign control" in emergence
    assert "emergence verdict" not in emergence
    assert "thresholded information-gating" in hierarchical
    assert "data, not the modeler" not in hierarchical
    assert "configured inputs, not parameters selected by this report" in baseline_flat
    assert "only the peak-margin contamination level is selected" in baseline_flat
    assert "points per class per client" in baseline_flat
    assert "cannot identify an rcce-only effect" in baseline_flat
    assert "does not establish leakage-free calibration" in baseline_flat
    assert "posterior uncertainty" in baseline_flat
    assert r"\bar F_{\text{solo}} - \bar F_{\text{share}}" in sharing
    assert "seed" in sharing.lower() and "agents are nested" in sharing.lower()
    assert "server presets" in robustness
    assert "trials nested" in robustness


def test_study_four_legacy_labels_are_documented_as_server_presets() -> None:
    paths = (
        MANUSCRIPT / "config.yaml",
        MANUSCRIPT / "config.yaml.example",
        ROOT / "src" / "experiment_config.py",
        ROOT / "docs" / "core" / "experiments-and-artifacts.md",
    )
    texts = {path: path.read_text(encoding="utf-8").casefold() for path in paths}

    for path, text in texts.items():
        assert "server preset" in text or "server-preset" in text, path
        assert "client-divergence labels" not in text, path
        assert "client divergence labels" not in text, path
    assert "legacy compatibility" in texts[MANUSCRIPT / "config.yaml"]
    assert "robust_aggregate" in texts[ROOT / "src" / "experiment_config.py"]


def test_robustness_axis_summary_does_not_make_bnn_own_server_weights() -> None:
    foundations = (
        ROOT / "docs" / "core" / "conceptual-foundations.md"
    ).read_text(encoding="utf-8")
    compact = " ".join(foundations.split()).casefold()

    assert "the influence-weights figure exercises axis 2" in compact
    assert "logistic-regression client-loss figure separately exercises axis 1" in compact
    assert "influence-weights and logistic-regression robustness figures exercise" not in compact


def test_methods_field_summary_is_bounded_to_reviewed_sources() -> None:
    methods = (MANUSCRIPT / "13_methods_statistics.md").read_text(encoding="utf-8")
    compact = " ".join(methods.split()).casefold()
    assert "among the active-inference sources reviewed here" in compact
    assert "where the active-inference community has typically reported" not in compact


def test_current_scholarship_surfaces_do_not_make_unbounded_priority_claims() -> None:
    """Historical audits may record earlier wording; current claims stay scoped."""
    paths = (
        ROOT / "docs" / "core" / "conceptual-foundations.md",
        *sorted(MANUSCRIPT.glob("*.md")),
    )
    stale = (
        "to our knowledge, had not",
        "had not been formally linked before this project",
        "the first project to",
        "no prior work has",
    )
    for path in paths:
        if path.name in {"AGENTS.md", "README.md", "SYNTAX.md"}:
            continue
        compact = " ".join(path.read_text(encoding="utf-8").casefold().split())
        assert not [phrase for phrase in stale if phrase in compact], path


def test_point_estimate_logistic_proxy_is_not_promoted_to_fedgvi_posterior() -> None:
    paths = (
        ROOT / "src" / "fedference" / "bnn_baseline.py",
        ROOT / "src" / "figures" / "bnn_robustness.py",
        ROOT / "src" / "analysis" / "workflow.py",
        MANUSCRIPT / "20_results_baseline.md",
        MANUSCRIPT / "11_methods_contamination.md",
        ROOT / "docs" / "core" / "conceptual-foundations.md",
        ROOT / "docs" / "core" / "architecture.md",
        ROOT / "README.md",
    )
    compact = "\n".join(path.read_text(encoding="utf-8") for path in paths).casefold()
    for forbidden in (
        "conjugate bernoulli analogue",
        "mean-field fedgvi logistic regression",
        "fedgvi-faithful per-agent generalized-bayes objective",
        "alpha-rényi client losses (`rcce`/`ar`",
    ):
        assert forbidden not in compact
    assert "does not evaluate alpha-renyi" in compact or "does not evaluate kl or alpha-rényi" in compact
    assert "point-estimate" in compact


def test_point_estimate_proxy_documents_bounded_logit_and_leverage_gradients() -> None:
    source = (ROOT / "src" / "fedference" / "bnn_baseline.py").read_text(
        encoding="utf-8"
    )
    compact = " ".join(source.casefold().split())

    assert "standard bounded ``(p - y)``" in compact
    assert "large parameter gradient" in compact
    assert "reducing its contribution relative to nll" in compact
    assert "recovers the nll loss and gradient" in compact
    for forbidden in (
        "produces a huge gradient",
        "flipped labels cannot dominate",
        "(standard bayes)",
    ):
        assert forbidden not in compact


def test_bnn_proxy_selection_and_identification_boundaries_agree_across_prose() -> None:
    baseline = " ".join(
        (MANUSCRIPT / "20_results_baseline.md").read_text(encoding="utf-8").casefold().split()
    )
    visual_audit = " ".join(
        (ROOT / "docs" / "research" / "visual-claim-audit.md")
        .read_text(encoding="utf-8")
        .casefold()
        .split()
    )

    for text in (baseline, visual_audit):
        assert "cannot identify an rcce-only effect" in text
        assert "points per class per client" in text
        assert "only" in text and "peak contamination" in text
    assert "configured inputs, not parameters selected by this report" in baseline
    assert "loss and shrinkage change together" in visual_audit


def test_project_bmr_surfaces_name_the_fixed_posterior_sign_control() -> None:
    surfaces = (
        MANUSCRIPT / "01_introduction.md",
        MANUSCRIPT / "03_contributions.md",
        MANUSCRIPT / "10_methods_learning.md",
        MANUSCRIPT / "12_methods_experimental_design.md",
        MANUSCRIPT / "18_results_emergence.md",
        MANUSCRIPT / "28_supplement_extended_methods.md",
        MANUSCRIPT / "SYNTAX.md",
    )
    compact = "\n".join(path.read_text(encoding="utf-8") for path in surfaces).casefold()
    assert "configured bmr sign control" in compact or "configured bmr sign-control" in compact
    for forbidden in (
        "our emergence study",
        "study 3 — emergence",
        "bayesian model reduction for structure emergence",
        "structure emergence by bayesian model reduction",
    ):
        assert forbidden not in compact


def test_multistart_descent_does_not_claim_global_optimality() -> None:
    text = (MANUSCRIPT / "27_supplement_aggregation_objective.md").read_text(
        encoding="utf-8"
    ).casefold()
    assert "solving the stated objective properly" not in text
    assert "optimal basin" not in text
    assert "does not certify a global optimum" in text
