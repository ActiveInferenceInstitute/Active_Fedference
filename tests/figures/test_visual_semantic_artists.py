"""Artist- and source-level gates for non-colour visual semantics.

The registry tests establish that semantic roles *have* distinct styles.  These
tests establish the second half of the contract: representative generators
must actually apply those markers, dashes, hatches, keylines, and direct labels
to the artists they draw.
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba

from analysis.report_schemas import ReportSchemaError, validate_report
from fedference.experiments import run_robustness_sweep
from figures import FIGURE_METADATA, system_overview
from figures._common import semantic_style

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FIGURE_ROOT = _PROJECT_ROOT / "src" / "figures"


def _source_tree(module_name: str) -> ast.Module:
    """Parse one figure generator so assertions inspect executable syntax."""
    return ast.parse((_FIGURE_ROOT / f"{module_name}.py").read_text(encoding="utf-8"))


def _semantic_roles(tree: ast.AST) -> set[str]:
    """Return literal roles passed to ``semantic_style`` in *tree*."""
    roles: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "semantic_style" or not node.args:
            continue
        role = node.args[0]
        if isinstance(role, ast.Constant) and isinstance(role.value, str):
            roles.add(role.value)
    return roles


def _keyword_attribute_uses(tree: ast.AST) -> set[tuple[str, str]]:
    """Return ``(keyword, attribute)`` pairs used in Matplotlib calls."""
    uses: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg is not None and isinstance(keyword.value, ast.Attribute):
                uses.add((keyword.arg, keyword.value.attr))
    return uses


def _called_artist_methods(tree: ast.AST) -> set[str]:
    """Return attribute method names called in one generator."""
    return {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }


def test_aggregation_descent_uses_variational_and_reference_rule_artists() -> None:
    tree = _source_tree("aggregation_descent")
    source = (_FIGURE_ROOT / "aggregation_descent.py").read_text(encoding="utf-8")

    assert _semantic_roles(tree) == {"reference_rule", "variational"}
    assert "objective-backed third robustness axis" in source
    assert "axis-2-made-rigorous" not in source
    uses = _keyword_attribute_uses(tree)
    assert ("marker", "marker") in uses
    assert ("linestyle", "dash") in uses
    assert ("markeredgecolor", "keyline") in uses


def test_descent_comparison_uses_neutral_condition_artists() -> None:
    tree = _source_tree("descent_comparison")

    assert _semantic_roles(tree) == {
        "condition_comparison",
        "condition_reference",
        "reference_rule",
    }
    assert not _semantic_roles(tree) & {"heuristic_robust", "naive", "variational"}
    uses = _keyword_attribute_uses(tree)
    assert ("marker", "marker") in uses
    assert ("linestyle", "dash") in uses
    assert ("markeredgecolor", "keyline") in uses


@pytest.mark.parametrize(
    "module_name",
    (
        "belief_quality",
        "bnn_robustness",
        "complexity_scaling",
        "robustness_onset",
        "robustness_sweep",
    ),
)
def test_named_multiseries_generators_apply_non_colour_styles_and_direct_labels(
    module_name: str,
) -> None:
    tree = _source_tree(module_name)
    uses = _keyword_attribute_uses(tree)
    artist_methods = _called_artist_methods(tree)

    assert ("marker", "marker") in uses, module_name
    assert ("linestyle", "dash") in uses, module_name
    assert ("markeredgecolor", "keyline") in uses, module_name
    assert artist_methods & {"annotate", "text"}, module_name


def test_system_overview_agent_artist_uses_hatch_keyline_and_direct_role_label() -> None:
    adversarial = semantic_style("adversarial")
    honest = semantic_style("honest")
    belief = np.full(system_overview.N_STATES, 1.0 / system_overview.N_STATES)
    fig, ax = plt.subplots()
    try:
        adversarial_artist = system_overview._draw_agent(
            ax,
            0.25,
            0.35,
            belief,
            adversarial,
            "A₁",
        )
        honest_artist = system_overview._draw_agent(
            ax,
            0.65,
            0.35,
            belief,
            honest,
            "H₁",
        )

        assert adversarial_artist.get_hatch() == adversarial.hatch
        assert honest_artist.get_hatch() == honest.hatch
        assert adversarial_artist.get_hatch() != honest_artist.get_hatch()
        assert adversarial_artist.get_edgecolor() == pytest.approx(
            to_rgba(adversarial.keyline)
        )
        assert honest_artist.get_edgecolor() == pytest.approx(to_rgba(honest.keyline))
        assert {text.get_text() for text in ax.texts} >= {"A₁", "H₁"}
    finally:
        plt.close(fig)


def test_system_overview_influence_bars_apply_role_hatches_and_labels() -> None:
    tree = _source_tree("system_overview")
    uses = _keyword_attribute_uses(tree)
    source = (_FIGURE_ROOT / "system_overview.py").read_text(encoding="utf-8")

    assert ("hatch", "hatch") in uses
    assert ("edgecolor", "keyline") in uses
    assert 'adv_labels = ["A₁", "A₂"]' in source
    assert 'hon_labels = ["H₁", "H₂", "H₃"]' in source
    assert "hatch=role_style.hatch" in source


def test_robustness_sweep_report_metadata_caption_and_schema_agree_on_trial_unit() -> None:
    report = run_robustness_sweep(
        23,
        rates=(0.0, 0.9),
        divergences=("KLD", "RKL"),
        n_trials=4,
    )
    metadata = FIGURE_METADATA["robustness_sweep"]
    caption_source = (_PROJECT_ROOT / "manuscript" / "19_results_robustness.md").read_text(
        encoding="utf-8"
    )
    normalized_caption = " ".join(caption_source.split()).lower()

    assert "matched trial" in report["analysis_unit"]
    assert "matched" in metadata["replication_unit"].lower()
    assert "trial" in metadata["replication_unit"].lower()
    assert "independent replication unit is a matched synthetic trial" in normalized_caption
    assert "fixed seeded true state and attack target" in normalized_caption
    assert "seed is the independent" not in normalized_caption
    assert "true state" in report["trial_structure"]
    assert "attack target" in report["trial_structure"]
    assert report["seed"] == 23
    assert report["n_trials"] == 4

    # The producer records the requested colony shape at the report boundary;
    # add the same source-owned fields before exercising that full schema here.
    report["n_agents"] = 7
    report["n_contaminated"] = 2
    contradictory = copy.deepcopy(report)
    contradictory["analysis_unit"] = "configured seed"
    with pytest.raises(ReportSchemaError, match="matched trials"):
        validate_report("robustness_sweep", contradictory)

    contradictory = copy.deepcopy(report)
    contradictory["trial_structure"] = "trials redraw every world variable"
    with pytest.raises(ReportSchemaError, match="fixed world variables"):
        validate_report("robustness_sweep", contradictory)
