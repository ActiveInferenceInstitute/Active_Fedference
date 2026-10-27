"""Characterized report bytes and real write-boundary rejection controls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TypeAlias

import pytest

from analysis.report_schemas import (
    ReportSchemaError,
    check_figure_contract,
)
from analysis.workflow import _write_json

MutationValue: TypeAlias = (
    None | bool | int | float | str | list["MutationValue"] | dict[str, "MutationValue"]
)

_REPORTS = Path(__file__).resolve().parents[2] / "output" / "reports"

_REPORT_HASHES = {
    "application_integrity_flow": (
        "f1e9a12e643e57d0d5bc0b668df6e39c3c1dbe38db0698cd1e11d294c2591a0c",
        "f1e9a12e643e57d0d5bc0b668df6e39c3c1dbe38db0698cd1e11d294c2591a0c",
    ),
    "belief_quality": (
        "e4592f8d897a3fa6ed7fee617ef4bccb18873b7c0389cef2395687202a47ccb2",
        "e4592f8d897a3fa6ed7fee617ef4bccb18873b7c0389cef2395687202a47ccb2",
    ),
    "belief_sharing": (
        "f878b3764cb545eb68d54a8cb9655af75d7094cf5b7418d2466bc416f1b0d66d",
        "f878b3764cb545eb68d54a8cb9655af75d7094cf5b7418d2466bc416f1b0d66d",
    ),
    "bnn_robustness": (
        "011bb4b34ab5bdc0ed8573916a032c83115cef80024931519c72362b0a632406",
        "011bb4b34ab5bdc0ed8573916a032c83115cef80024931519c72362b0a632406",
    ),
    "bnn_torch": (
        "0b0fd1169dfbff267692bf5f4628fac5a96cad8e82dc1e8a5fde67b738dc999d",
        "0b0fd1169dfbff267692bf5f4628fac5a96cad8e82dc1e8a5fde67b738dc999d",
    ),
    "complexity_scaling": (
        "35af65c9f67cd5124884da4400908f6b2eafaa2febcb04f5887fa64fc440db2e",
        "35af65c9f67cd5124884da4400908f6b2eafaa2febcb04f5887fa64fc440db2e",
    ),
    "conditional_world": (
        "d580ba7f7c3b307cfd72b3dd3145ff17a06a326e54d7a94959d3ee4c620a30ed",
        "d580ba7f7c3b307cfd72b3dd3145ff17a06a326e54d7a94959d3ee4c620a30ed",
    ),
    "contamination_gallery": (
        "154ea98465f2daf07c36b0a61ee0f73cf310979e9a8719583d3b5b5bc59ad562",
        "154ea98465f2daf07c36b0a61ee0f73cf310979e9a8719583d3b5b5bc59ad562",
    ),
    "cross_study_summary": (
        "8244f2c8ab6529e1561be892896791b3cf154763b6f57c98affbaa935d9e3a99",
        "8244f2c8ab6529e1561be892896791b3cf154763b6f57c98affbaa935d9e3a99",
    ),
    "disjoint_fov_world": (
        "9ea790ccc7693da531d566d1184405e83b3facec1bc8a3d47bcdfbdcb0b4490a",
        "9ea790ccc7693da531d566d1184405e83b3facec1bc8a3d47bcdfbdcb0b4490a",
    ),
    "efe_decomposition": (
        "028331e574a59ffc0666391c6433c11422e726e2f8f141da822403ec9de346f0",
        "028331e574a59ffc0666391c6433c11422e726e2f8f141da822403ec9de346f0",
    ),
    "emergence": (
        "1fef4cc26d63e10b8b14e0d7e9024d8c8108beebdaac402f937c77b99e1db765",
        "1fef4cc26d63e10b8b14e0d7e9024d8c8108beebdaac402f937c77b99e1db765",
    ),
    "evidence_replication_map": (
        "bac2109d8162fee23df3eb9ba69180ae65993b4500c6dd75f8bf99c5410088a4",
        "bac2109d8162fee23df3eb9ba69180ae65993b4500c6dd75f8bf99c5410088a4",
    ),
    "heuristic_characterization": (
        "1db9b866b64e0ed5a12c25061c9bfdb8e422bfced810007863cecd62ddd7bd2d",
        "1db9b866b64e0ed5a12c25061c9bfdb8e422bfced810007863cecd62ddd7bd2d",
    ),
    "hierarchical_bmr": (
        "207f8d9f7454d32085dbf827bbb4f7659d2c693ceea66bd36d2b6411307dd040",
        "207f8d9f7454d32085dbf827bbb4f7659d2c693ceea66bd36d2b6411307dd040",
    ),
    "hierarchical_world": (
        "81d795bf4ce3c47667e6291f0e0a8ff40adb8d07b8ffad92e72f441f2205031d",
        "81d795bf4ce3c47667e6291f0e0a8ff40adb8d07b8ffad92e72f441f2205031d",
    ),
    "language_acquisition": (
        "9aa575021186aa458c6b66335a539668f2b41e6206cae2fa7929141153dbd8f1",
        "9aa575021186aa458c6b66335a539668f2b41e6206cae2fa7929141153dbd8f1",
    ),
    "moving_world": (
        "949ac6b3da51efc160a31979d5426e574a39c2129cc437aa6b21f29bcbfafe3f",
        "949ac6b3da51efc160a31979d5426e574a39c2129cc437aa6b21f29bcbfafe3f",
    ),
    "nlevel3_world": (
        "c55398a55581cef22a29707f89d9ecb34120b1b35ca4e7262dff423a1af26acb",
        "c55398a55581cef22a29707f89d9ecb34120b1b35ca4e7262dff423a1af26acb",
    ),
    "parameter_recovery": (
        "80bd649f7ab2673f1fae7931770b67cf74e3f18013ea2fa217ee8bd193b83bd2",
        "80bd649f7ab2673f1fae7931770b67cf74e3f18013ea2fa217ee8bd193b83bd2",
    ),
    "robust_influence_weights": (
        "e523a34836395085b0fb91fa5984686deca99dd317ec5f1a8858c9d5a2d0520c",
        "e523a34836395085b0fb91fa5984686deca99dd317ec5f1a8858c9d5a2d0520c",
    ),
    "robustness_onset": (
        "9e080920bce969f3634217c2910f71c29b05240f755c851444ed2973ad5f8c89",
        "9e080920bce969f3634217c2910f71c29b05240f755c851444ed2973ad5f8c89",
    ),
    "robustness_review_grid": (
        "24e14eec098e4d8a1182af21c4ccc804d7f685e36ba2a21cd96ed06686c57983",
        "24e14eec098e4d8a1182af21c4ccc804d7f685e36ba2a21cd96ed06686c57983",
    ),
    "robustness_sweep": (
        "be0b03f7aa770ef6806bec0e1a648766d8c0b12a8f623864cf705f9ba0d0cef1",
        "be0b03f7aa770ef6806bec0e1a648766d8c0b12a8f623864cf705f9ba0d0cef1",
    ),
    "sensitivity": (
        "5912c0f93de0d89e2e6b9289b9382bea052bc0a81daa6257d445d8a8b43c02b0",
        "5912c0f93de0d89e2e6b9289b9382bea052bc0a81daa6257d445d8a8b43c02b0",
    ),
    "source_render_provenance": (
        "73caad46172beadbed66ad68901ab83e9429da96dcdbc39c5d4c90344c6c6e3c",
        "73caad46172beadbed66ad68901ab83e9429da96dcdbc39c5d4c90344c6c6e3c",
    ),
    "variational_aggregation": (
        "c85925d20cf5053fc48cc70d285bab24318cc051ff6c1612b4e64fc81b38ead7",
        "c85925d20cf5053fc48cc70d285bab24318cc051ff6c1612b4e64fc81b38ead7",
    ),
}

_INVALID_CASES = [
    (
        "belief_sharing",
        "schema_version",
        "missing",
        "belief_sharing payload missing required field 'schema_version' (expected str)",
    ),
    (
        "belief_sharing",
        "communicating_mean",
        "bool",
        "belief_sharing payload field 'communicating_mean' expected number, got bool",
    ),
    (
        "language_acquisition",
        "final_kl",
        "missing",
        "language_acquisition payload missing required field 'final_kl' (expected number)",
    ),
    (
        "language_acquisition",
        "final_kl",
        "bool",
        "language_acquisition payload field 'final_kl' expected number, got bool",
    ),
    (
        "emergence",
        "convergence",
        "missing",
        "emergence payload missing required field 'convergence' (expected bool)",
    ),
    (
        "emergence",
        "delta_F_redundant",
        "bool",
        "emergence payload field 'delta_F_redundant' expected number, got bool",
    ),
    (
        "robustness_sweep",
        "schema_version",
        "missing",
        "robustness_sweep payload missing required field 'schema_version' (expected str)",
    ),
    (
        "robustness_sweep",
        "accuracy_threshold",
        "bool",
        "robustness_sweep payload field 'accuracy_threshold' expected number, got bool",
    ),
    (
        "hierarchical_bmr",
        "acuity",
        "missing",
        "hierarchical_bmr payload missing required field 'acuity' (expected number)",
    ),
    (
        "hierarchical_bmr",
        "acuity",
        "bool",
        "hierarchical_bmr payload field 'acuity' expected number, got bool",
    ),
    (
        "heuristic_characterization",
        "breakdown",
        "missing",
        "heuristic_characterization payload missing required field 'breakdown' (expected dict)",
    ),
    (
        "heuristic_characterization",
        "seed",
        "bool",
        "heuristic_characterization payload field 'seed' expected int, got bool",
    ),
    (
        "efe_decomposition",
        "ambiguity",
        "missing",
        "efe_decomposition payload missing required field 'ambiguity' (expected number)",
    ),
    (
        "efe_decomposition",
        "ambiguity",
        "bool",
        "efe_decomposition payload field 'ambiguity' expected number, got bool",
    ),
    (
        "robust_influence_weights",
        "normalized_effective_weights",
        "missing",
        "robust_influence_weights payload missing required field 'normalized_effective_weights' "
        "(expected list)",
    ),
    (
        "robust_influence_weights",
        "n_agents",
        "bool",
        "robust_influence_weights payload field 'n_agents' expected int, got bool",
    ),
    (
        "bnn_robustness",
        "accuracy_by_config",
        "missing",
        "bnn_robustness payload missing required field 'accuracy_by_config' (expected dict)",
    ),
    (
        "bnn_robustness",
        "ci_percent",
        "bool",
        "bnn_robustness payload field 'ci_percent' expected int, got bool",
    ),
    (
        "variational_aggregation",
        "converged",
        "missing",
        "variational_aggregation payload missing required field 'converged' (expected bool)",
    ),
    (
        "variational_aggregation",
        "iterations",
        "bool",
        "variational_aggregation payload field 'iterations' expected int, got bool",
    ),
    (
        "contamination_gallery",
        "by_kind",
        "missing",
        "contamination_gallery payload missing required field 'by_kind' (expected dict)",
    ),
    (
        "contamination_gallery",
        "n_agents",
        "bool",
        "contamination_gallery payload field 'n_agents' expected int, got bool",
    ),
    (
        "robustness_onset",
        "by_kind",
        "missing",
        "robustness_onset payload missing required field 'by_kind' (expected dict)",
    ),
    (
        "robustness_onset",
        "n_agents",
        "bool",
        "robustness_onset payload field 'n_agents' expected int, got bool",
    ),
    (
        "moving_world",
        "accuracy",
        "missing",
        "moving_world payload missing required field 'accuracy' (expected dict)",
    ),
    ("moving_world", "n_agents", "bool", "moving_world payload field 'n_agents' expected int, got bool"),
    (
        "parameter_recovery",
        "abs_error",
        "missing",
        "parameter_recovery payload missing required field 'abs_error' (expected list)",
    ),
    (
        "parameter_recovery",
        "fit_resolution",
        "bool",
        "parameter_recovery payload field 'fit_resolution' expected int, got bool",
    ),
    (
        "hierarchical_world",
        "acuity",
        "missing",
        "hierarchical_world payload missing required field 'acuity' (expected number)",
    ),
    (
        "hierarchical_world",
        "acuity",
        "bool",
        "hierarchical_world payload field 'acuity' expected number, got bool",
    ),
    (
        "nlevel3_world",
        "acuity",
        "missing",
        "nlevel3_world payload missing required field 'acuity' (expected number)",
    ),
    ("nlevel3_world", "acuity", "bool", "nlevel3_world payload field 'acuity' expected number, got bool"),
    (
        "cross_study_summary",
        "n_seeds",
        "missing",
        "cross_study_summary payload missing required field 'n_seeds' (expected int)",
    ),
    (
        "cross_study_summary",
        "n_seeds",
        "bool",
        "cross_study_summary payload field 'n_seeds' expected int, got bool",
    ),
    (
        "disjoint_fov_world",
        "communicating_accuracy",
        "missing",
        "disjoint_fov_world payload missing required field 'communicating_accuracy' (expected number)",
    ),
    (
        "disjoint_fov_world",
        "communicating_accuracy",
        "bool",
        "disjoint_fov_world payload field 'communicating_accuracy' expected number, got bool",
    ),
    (
        "complexity_scaling",
        "analytic_specs",
        "missing",
        "complexity_scaling payload missing required field 'analytic_specs' (expected list)",
    ),
    ("complexity_scaling", "seed", "bool", "complexity_scaling payload field 'seed' expected int, got bool"),
    (
        "conditional_world",
        "by_scenario",
        "missing",
        "conditional_world payload missing required field 'by_scenario' (expected dict)",
    ),
    (
        "conditional_world",
        "n_agents",
        "bool",
        "conditional_world payload field 'n_agents' expected int, got bool",
    ),
    (
        "robustness_review_grid",
        "analysis_profile",
        "missing",
        "robustness_review_grid payload missing required field 'analysis_profile' (expected str)",
    ),
    (
        "robustness_review_grid",
        "n_agents",
        "bool",
        "robustness_review_grid payload field 'n_agents' expected int, got bool",
    ),
    (
        "belief_quality",
        "by_scenario",
        "missing",
        "belief_quality payload missing required field 'by_scenario' (expected dict)",
    ),
    ("belief_quality", "n_agents", "bool", "belief_quality payload field 'n_agents' expected int, got bool"),
    (
        "application_integrity_flow",
        "schema_version",
        "missing",
        "application_integrity_flow payload missing required field 'schema_version' (expected str)",
    ),
    (
        "evidence_replication_map",
        "schema_version",
        "missing",
        "evidence_replication_map payload missing required field 'schema_version' (expected str)",
    ),
    (
        "source_render_provenance",
        "schema_version",
        "missing",
        "source_render_provenance payload missing required field 'schema_version' (expected str)",
    ),
    (
        "sensitivity",
        "schema_version",
        "missing",
        "sensitivity payload missing required field 'schema_version' (expected str)",
    ),
    ("sensitivity", "seed", "bool", "sensitivity payload field 'seed' expected int, got bool"),
    ("belief_sharing", "n_seeds", 0, "belief_sharing paired arrays must each contain n_seeds >= 2 values"),
    ("hierarchical_bmr", "surprise_tol", 0, "hierarchical_bmr surprise_tol must be positive"),
    ("parameter_recovery", "n_trials", 0, "parameter_recovery grid, trial, and fit budgets must be positive"),
    (
        "robustness_sweep",
        "analysis_unit",
        "invalid",
        "robustness_sweep analysis_unit must identify matched trials",
    ),
    ("bnn_robustness", "n_seeds", 0, "bnn_robustness n_seeds must be >= 2"),
    ("robustness_review_grid", "n_seeds", 0, "robustness_review_grid requires n_seeds and n_trials >= 2"),
    (
        "application_integrity_flow",
        "study_status",
        "invalid",
        "application_integrity_flow study_status must identify an explanatory contract",
    ),
    (
        "evidence_replication_map",
        "study_status",
        "invalid",
        "evidence_replication_map study_status must identify a source-owned classification",
    ),
    (
        "source_render_provenance",
        "study_status",
        "invalid",
        "source_render_provenance study_status must identify a pipeline contract",
    ),
    ("sensitivity", "unit", "invalid", "sensitivity unit must be 'fraction'"),
    (
        "bnn_torch",
        "status",
        "invalid",
        "bnn_torch payload field 'status' must be 'ok' or start with 'skipped'",
    ),
]


@pytest.mark.parametrize("schema", sorted(_REPORT_HASHES))
def test_all_report_families_byte_stable(schema: str, tmp_path: Path) -> None:
    # Given a read-only committed report and pre-move source/canonical hashes.
    raw = (_REPORTS / f"{schema}.json").read_bytes()
    payload = json.loads(raw)
    expected_source, expected_canonical = _REPORT_HASHES[schema]
    assert hashlib.sha256(raw).hexdigest() == expected_source
    destination = tmp_path / schema / "report.json"

    # When the real canonical boundary writes the report.
    _write_json(payload, destination, schema=schema)

    # Then serialization matches the characterized bytes, with input untouched.
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == expected_canonical
    assert (_REPORTS / f"{schema}.json").read_bytes() == raw


@pytest.mark.parametrize(("schema", "field", "mutation", "diagnostic"), _INVALID_CASES)
def test_invalid_payload_before_write(
    schema: str,
    field: str,
    mutation: str | int,
    diagnostic: str,
    tmp_path: Path,
) -> None:
    # Given a real report with one characterized missing/type/semantic defect.
    payload = json.loads((_REPORTS / f"{schema}.json").read_bytes())
    if mutation == "missing":
        del payload[field]
    elif mutation == "bool":
        payload[field] = True
    else:
        payload[field] = mutation
    destination = tmp_path / "uncreated" / "report.json"

    # When the actual writer receives the invalid report.
    with pytest.raises(ReportSchemaError) as failure:
        _write_json(payload, destination, schema=schema)

    # Then the exact diagnostic survives and even the parent is not created.
    assert str(failure.value) == diagnostic
    assert not destination.parent.exists()


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_nested_nonfinite_rejected_before_write(nonfinite: float, tmp_path: Path) -> None:
    # Given a valid report with a nonfinite value in an additional nested field.
    payload = json.loads((_REPORTS / "belief_sharing.json").read_bytes())
    payload["extra"] = [{"value": nonfinite}]
    destination = tmp_path / "uncreated" / "report.json"

    # When the writer traverses all fields, including additional metadata.
    with pytest.raises(ReportSchemaError) as failure:
        _write_json(payload, destination, schema="belief_sharing")

    # Then finite-value rejection happens before any filesystem output.
    assert str(failure.value) == "payload.extra[0].value contains a non-finite JSON number"
    assert not destination.parent.exists()


@pytest.mark.parametrize(("generator", "report_name", "field"), [
    ("free_energy_comparison", "belief_sharing", "paired_difference_ci"),
    ("moving_world", "moving_world", "free_energy_gap"),
    ("robustness_sweep", "robustness_sweep", "per_rate_summary"),
    ("conditional_world", "conditional_world", "controls"),
])
def test_figure_contract_rejects_missing_consumed_field(
    generator: str, report_name: str, field: str
) -> None:
    # Given actual report data missing a field required by its figure consumer.
    payload = json.loads((_REPORTS / f"{report_name}.json").read_bytes())
    del payload[field]
    # When the runtime contract is applied, it rejects that missing field.
    with pytest.raises(ReportSchemaError, match=field):
        check_figure_contract(generator, report_name, payload)


_VISUAL_FIXTURES = {
    "application_integrity_flow": "reports/application_integrity_flow.json",
    "belief_sharing": "reports/belief_sharing.json",
    "figure_exact_values": "figures/figure_exact_values.json",
    "presentation_manifest": "figures/free_energy_comparison.slides.json",
    "robustness_review_grid": "reports/robustness_review_grid.json",
    "sensitivity": "reports/sensitivity.json",
    "source_render_provenance": "reports/source_render_provenance.json",
}

_NESTED_CASES = [
    ("figure_exact_values", ("schema_version",), "9", "figure_exact_values schema_version must be '1.0'"),
    (
        "figure_exact_values",
        ("generated_by",),
        "other",
        "figure_exact_values generated_by is not the canonical producer",
    ),
    ("figure_exact_values", ("tables",), [], "figure_exact_values requires at least one table"),
    ("figure_exact_values", ("tables", 0), 0, "figure_exact_values tables[0] must be a mapping"),
    (
        "figure_exact_values",
        ("tables", 0, "unexpected"),
        True,
        "figure_exact_values tables[0] has unknown or missing fields",
    ),
    (
        "figure_exact_values",
        ("tables", 0, "identifier"),
        "wrong",
        "figure_exact_values identifier/generator mismatch at tables[0]",
    ),
    (
        "figure_exact_values",
        ("tables", 0, "source_report"),
        "wrong.json",
        "figure_exact_values source_report mismatch at tables[0]",
    ),
    (
        "figure_exact_values",
        ("tables", 0, "note"),
        "",
        "figure_exact_values tables[0] note must be non-empty",
    ),
    (
        "figure_exact_values",
        ("tables", 0, "columns"),
        [],
        "figure_exact_values tables[0] has invalid columns",
    ),
    ("figure_exact_values", ("tables", 0, "rows"), [], "figure_exact_values tables[0] has no rows"),
    (
        "figure_exact_values",
        ("tables", 0, "rows", 0),
        {},
        "figure_exact_values tables[0].rows[0] does not match columns",
    ),
    (
        "figure_exact_values",
        ("tables", 0, "rows", 0),
        "NONSCALAR",
        "figure_exact_values tables[0].rows[0] contains a non-scalar value",
    ),
    ("figure_exact_values", ("tables",), "DUPLICATE", "figure_exact_values tables must be sorted and unique"),
    (
        "figure_exact_values",
        ("tables",),
        "INVENTORY",
        "figure_exact_values generator inventory mismatch: missing=['belief_quality'], extra=[]",
    ),
    (
        "presentation_manifest",
        ("schema_version",),
        "9",
        "presentation_manifest requires version 1.0 and exactly panels",
    ),
    ("presentation_manifest", ("panels",), [], "presentation_manifest requires 1 to 64 panels"),
    ("presentation_manifest", ("panels", 0), {}, "presentation_manifest panel fields are invalid"),
    (
        "presentation_manifest",
        ("panels", 0, "src"),
        "../escape.png",
        "presentation_manifest panel source is unsafe or repeated",
    ),
    (
        "presentation_manifest",
        ("panels", 0, "alt"),
        " ",
        "presentation_manifest panel alternative is invalid",
    ),
    (
        "presentation_manifest",
        ("panels", 0, "sha256"),
        "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        "presentation_manifest panel digest is invalid",
    ),
    (
        "presentation_manifest",
        ("panels", 0, "minimum_label_px"),
        0,
        "presentation_manifest label measurement must be finite and positive",
    ),
    (
        "belief_sharing",
        ("communicating_mean",),
        999,
        "belief_sharing communicating_mean disagrees with source values",
    ),
    (
        "belief_sharing",
        ("incommunicado_mean",),
        999,
        "belief_sharing incommunicado_mean disagrees with source values",
    ),
    (
        "belief_sharing",
        ("paired_difference_mean",),
        999,
        "belief_sharing paired_difference_mean disagrees with paired values",
    ),
    (
        "belief_sharing",
        ("free_energy_gap",),
        999,
        "belief_sharing free_energy_gap must equal paired_difference_mean",
    ),
    (
        "belief_sharing",
        ("communication_helps",),
        False,
        "belief_sharing communication_helps disagrees with paired means",
    ),
    (
        "belief_sharing",
        ("paired_difference_ci",),
        [],
        "belief_sharing paired_difference_ci must be ordered around the mean",
    ),
    (
        "belief_sharing",
        ("paired_difference_ci",),
        [100, 101],
        "belief_sharing paired_difference_ci must be ordered around the mean",
    ),
    (
        "application_integrity_flow",
        ("panels", 0, "label"),
        "",
        "application_integrity_flow payload field panels[0].label must be a non-empty string",
    ),
    (
        "application_integrity_flow",
        ("panels", 0, "nodes"),
        None,
        "application_integrity_flow payload field 'panels[0].nodes' must be a list",
    ),
    (
        "application_integrity_flow",
        ("panels", 0, "nodes", 0),
        None,
        "application_integrity_flow payload field panels[0].nodes[0] must be a mapping",
    ),
    (
        "application_integrity_flow",
        ("panels", 0, "nodes", 0, "id"),
        "",
        "application_integrity_flow payload field panels[0].nodes[0].id must be a non-empty string",
    ),
    (
        "application_integrity_flow",
        ("panels", 0, "edges", 0, "source"),
        "missing",
        "application_integrity_flow payload field panels[].edges[0] references an unknown node",
    ),
    (
        "application_integrity_flow",
        ("panels", 0, "edges", 0, "label"),
        "",
        "application_integrity_flow payload field panels[].edges[0].label must be non-empty",
    ),
    (
        "application_integrity_flow",
        ("panels", 0, "edges", 0, "disposition"),
        "missing",
        "application_integrity_flow payload field panels[].edges[0] has invalid disposition",
    ),
    (
        "robustness_review_grid",
        ("divergences",),
        ["KLD"],
        "robustness_review_grid must declare one or more unique non-KLD methods",
    ),
    (
        "robustness_review_grid",
        ("rates",),
        [-1],
        "robustness_review_grid rates must be non-empty and lie in [0, 1]",
    ),
    (
        "robustness_review_grid",
        ("rates",),
        [0.4, 0.2],
        "robustness_review_grid rates must be strictly increasing",
    ),
    (
        "robustness_review_grid",
        ("conditional_world", "by_scenario"),
        {},
        "robustness_review_grid conditional_world has no cells",
    ),
    (
        "robustness_review_grid",
        ("conditional_world", "by_scenario", "s0_t1_o45_clean_full", "attack"),
        None,
        "robustness_review_grid conditional cell 's0_t1_o45_clean_full' lacks attack",
    ),
    (
        "robustness_review_grid",
        ("conditional_world", "by_scenario", "s0_t1_o45_clean_full", "n_seeds"),
        0,
        "robustness_review_grid conditional cell 's0_t1_o45_clean_full' has a mismatched budget",
    ),
    (
        "robustness_review_grid",
        ("conditional_world", "by_scenario", "s0_t1_o45_clean_full", "contrast_by_seed"),
        [],
        "robustness_review_grid conditional cell 's0_t1_o45_clean_full' has wrong seed count",
    ),
    (
        "robustness_review_grid",
        ("conditional_world", "by_scenario", "s0_t1_o45_clean_full", "contrast_ci"),
        [],
        "robustness_review_grid conditional cell 's0_t1_o45_clean_full' has invalid CI",
    ),
    (
        "robustness_review_grid",
        ("conditional_world", "by_scenario", "s0_t1_o45_clean_full", "contrast_ci"),
        [999, 1000],
        "robustness_review_grid conditional cell 's0_t1_o45_clean_full' has unordered CI",
    ),
    (
        "robustness_review_grid",
        ("rate_profiles", "by_kind"),
        {},
        "robustness_review_grid rate profile mechanisms differ from directional_mechanisms",
    ),
    (
        "robustness_review_grid",
        ("rate_profiles", "by_kind", "byzantine", "rates"),
        [],
        "robustness_review_grid profile 'byzantine' has rates inconsistent with top level",
    ),
    (
        "robustness_review_grid",
        ("statistics", "selection_free"),
        False,
        "robustness_review_grid statistics must declare selection_free=true",
    ),
    (
        "robustness_review_grid",
        ("statistics", "fdr_alpha"),
        0,
        "robustness_review_grid statistics fdr_alpha must lie in (0, 1)",
    ),
    (
        "robustness_review_grid",
        ("statistics", "planning_alternative"),
        "bad",
        "robustness_review_grid statistics planning_alternative is unsupported",
    ),
    (
        "robustness_review_grid",
        ("statistics", "by_mechanism"),
        {},
        "robustness_review_grid statistics mechanisms differ from directional_mechanisms",
    ),
    (
        "robustness_review_grid",
        ("statistics", "by_mechanism", "byzantine", "methods"),
        [],
        "robustness_review_grid statistics methods differ for 'byzantine'",
    ),
    (
        "robustness_review_grid",
        ("statistics", "by_mechanism", "byzantine", "by_rate"),
        {},
        "robustness_review_grid statistics rate rows differ for 'byzantine'",
    ),
    (
        "robustness_review_grid",
        ("statistics", "by_mechanism", "byzantine", "by_rate", "0", "n_seeds"),
        0,
        "robustness_review_grid statistics row has a mismatched seed count: 'byzantine'/'0'",
    ),
    (
        "robustness_review_grid",
        ("statistics", "by_mechanism", "byzantine", "by_rate", "0", "methods"),
        {},
        "robustness_review_grid statistics row has missing or unexpected methods: 'byzantine'/'0'",
    ),
    (
        "robustness_review_grid",
        ("statistics", "by_mechanism", "byzantine", "by_rate", "0", "methods", "AR", "contrast_by_seed"),
        [],
        "robustness_review_grid method contrast has a wrong seed count: 'byzantine'/'0'/'AR'",
    ),
    (
        "robustness_review_grid",
        ("statistics", "by_mechanism", "byzantine", "by_rate", "0", "methods", "AR", "summary", "n"),
        0,
        "robustness_review_grid summary has a mismatched seed count: 'byzantine'/'0'/'AR'",
    ),
    (
        "robustness_review_grid",
        ("statistics", "by_mechanism", "byzantine", "by_rate", "0", "methods", "AR", "summary", "ci_lo"),
        999,
        "robustness_review_grid summary has invalid uncertainty ordering: 'byzantine'/'0'/'AR'",
    ),
    (
        "robustness_review_grid",
        ("statistics", "by_mechanism", "byzantine", "by_rate", "0", "methods", "AR", "contrast_ci"),
        [],
        "robustness_review_grid method interval must have exactly two values: 'byzantine'/'0'/'AR'",
    ),
    (
        "robustness_review_grid",
        ("statistics", "by_mechanism", "byzantine", "by_rate", "0", "methods", "AR", "contrast_ci"),
        [999, 1000],
        "robustness_review_grid method interval disagrees with its summary: 'byzantine'/'0'/'AR'",
    ),
    (
        "robustness_review_grid",
        ("controls", "observed_attack_mechanisms"),
        ["bad"],
        "robustness_review_grid observed attack controls do not match the emitted components",
    ),
    (
        "robustness_review_grid",
        ("controls", "all_declared_attack_controls_present"),
        False,
        "robustness_review_grid controls must prove declared attack coverage",
    ),
    (
        "robustness_review_grid",
        ("controls", "conditional_zero_robustness_control_passed"),
        False,
        "robustness_review_grid controls must retain the zero-robustness check",
    ),
    (
        "robustness_review_grid",
        ("precision_plan", "observed_max_mcse"),
        -1,
        "robustness_review_grid precision plan must report a non-negative MCSE",
    ),
    (
        "robustness_review_grid",
        ("precision_plan", "target_status"),
        "bad",
        "robustness_review_grid precision plan has an unmet MCSE target",
    ),
    (
        "robustness_review_grid",
        ("directional_mechanisms",),
        ["drift", "drift"],
        "robustness_review_grid field 'directional_mechanisms' must not contain duplicate strings",
    ),
    (
        "robustness_review_grid",
        ("statistics", "by_mechanism", "byzantine"),
        None,
        "robustness_review_grid field 'statistics.by_mechanism.byzantine' must be a mapping",
    ),
    (
        "robustness_review_grid",
        ("statistics", "by_mechanism", "byzantine", "by_rate", "0", "methods", "AR", "summary", "mcse"),
        "bad",
        "robustness_review_grid field 'statistics summary mcse' must be a finite number",
    ),
    ("sensitivity", ("n_agents_values",), [0], "sensitivity n_agents_values must contain positive integers"),
    ("sensitivity", ("acuity_values",), [2], "sensitivity acuity_values must lie in (0, 1]"),
    ("sensitivity", ("n_trials",), 0, "sensitivity n_trials must be positive"),
    (
        "sensitivity",
        ("belief_sharing", "acuity_values"),
        [],
        "sensitivity belief_sharing acuity axis does not match the top-level axis",
    ),
    (
        "sensitivity",
        ("belief_sharing", "n_agents_values"),
        [],
        "sensitivity belief_sharing colony-size axis does not match the top-level axis",
    ),
    (
        "sensitivity",
        ("belief_sharing", "seed"),
        -1,
        "sensitivity belief_sharing execution identity does not match the report",
    ),
    (
        "sensitivity",
        ("belief_sharing", "accuracy_gap_grid"),
        [],
        "sensitivity belief_sharing accuracy_gap_grid has the wrong row count",
    ),
    (
        "sensitivity",
        ("belief_sharing", "accuracy_gap_grid", 0),
        [],
        "sensitivity belief_sharing accuracy_gap_grid row 0 does not match the colony axis",
    ),
    ("sensitivity", ("estimands",), {}, "sensitivity estimands must name both displayed panels"),
    (
        "sensitivity",
        ("claim_boundary",),
        "bad",
        "sensitivity claim_boundary must explain the hatch semantics",
    ),
    (
        "source_render_provenance",
        ("inputs",),
        [],
        "source_render_provenance input inventory or order is invalid",
    ),
    (
        "source_render_provenance",
        ("stages",),
        [],
        "source_render_provenance producer-stage inventory or order is invalid",
    ),
    (
        "source_render_provenance",
        ("terminals",),
        [],
        "source_render_provenance terminal inventory or order is invalid",
    ),
    (
        "source_render_provenance",
        ("terminals", 0, "authorization_required"),
        False,
        "source_render_provenance publication terminals must require authorization",
    ),
    (
        "source_render_provenance",
        ("producer_edges",),
        [],
        "source_render_provenance producer edge inventory, order, or disposition is invalid",
    ),
    (
        "source_render_provenance",
        ("invalidation_edges",),
        [],
        "source_render_provenance invalidation edge inventory, order, or disposition is invalid",
    ),
    (
        "source_render_provenance",
        ("authorization_boundary",),
        "bad",
        "source_render_provenance must deny automatic publication authorization",
    ),
    (
        "source_render_provenance",
        ("manifest_cycle_boundary",),
        "bad",
        "source_render_provenance must deny a generated-manifest dependency cycle",
    ),
    (
        "source_render_provenance",
        ("no_claims",),
        [],
        "source_render_provenance must retain engineering and accessibility no-claims",
    ),
]


@pytest.mark.parametrize(("schema", "field_path", "value", "diagnostic"), _NESTED_CASES)
def test_nested_contract_rejected_before_write(
    schema: str,
    field_path: tuple[str | int, ...],
    value: MutationValue,
    diagnostic: str,
    tmp_path: Path,
) -> None:
    # Given a committed payload with one malformed nested contract.
    payload = json.loads((_REPORTS.parent / _VISUAL_FIXTURES[schema]).read_bytes())
    if value == "NONSCALAR":
        value = dict(payload["tables"][0]["rows"][0])
        value[next(iter(value))] = []
    elif value == "DUPLICATE":
        value = payload["tables"] + payload["tables"][:1]
    elif value == "INVENTORY":
        value = payload["tables"][1:]
    cursor = payload
    for key in field_path[:-1]:
        cursor = cursor[key]
    cursor[field_path[-1]] = value
    destination = tmp_path / "uncreated" / "report.json"

    # When the real report writer receives that malformed payload.
    with pytest.raises(ReportSchemaError) as failure:
        _write_json(payload, destination, schema=schema)

    # Then the characterized failure precedes filesystem creation.
    assert str(failure.value) == diagnostic
    assert not destination.parent.exists()


@pytest.mark.parametrize("schema", ["figure_exact_values", "presentation_manifest"])
def test_visual_payload_round_trip(schema: str, tmp_path: Path) -> None:
    # Given an actual committed table or slide manifest.
    raw = (_REPORTS.parent / _VISUAL_FIXTURES[schema]).read_bytes()
    payload = json.loads(raw)
    destination = tmp_path / "report.json"

    # When the real writer validates and serializes it.
    _write_json(payload, destination, schema=schema)

    # Then the complete payload survives the write boundary.
    assert (
        destination.read_bytes()
        == (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    )
