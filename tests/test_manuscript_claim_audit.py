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
        "20_results_baseline.md": "Client-side robustness complement: categorical FedGVI baseline",
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
