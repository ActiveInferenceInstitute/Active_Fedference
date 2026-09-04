"""Per-schema unit tests for the typed report and figure-registry validator.

No mocks: real dict payloads flow through the real validator
(``analysis.report_schemas.validate_report`` / ``check_figure_contract``), and
the stage-contract tests read the real committed artifacts under
``output/reports/`` and ``output/figures/figure_registry.json``. Valid payloads
are derived mechanically from the schema definitions themselves, so every
schema the module declares is exercised — a schema added without a test here
still gets accept/reject coverage automatically.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import pytest

from analysis import report_schemas
from analysis.report_schemas import (
    FIGURE_DEPENDENCY_CONTRACTS,
    FigureDependencyContract,
    ReportSchemaError,
    check_figure_contract,
    validate_report,
)
from analysis.visual_contracts import (
    application_integrity_flow_contract,
    evidence_replication_map_contract,
    source_render_provenance_contract,
)
from analysis.workflow import _PROJECT_ROOT, _sensitivity_report, _write_json
from fedference.experiments import (
    run_bnn_robustness_report,
    run_hierarchical_bmr,
    run_parameter_recovery,
    run_review_grid,
    run_robustness_sweep,
)

REPORTS_DIR = _PROJECT_ROOT / "output" / "reports"
FIGURE_REGISTRY_PATH = _PROJECT_ROOT / "output" / "figures" / "figure_registry.json"

#: One representative value per shallow type tag.
_SAMPLE_VALUES: dict[str, object] = {
    "bool": True,
    "dict": {"key": 1},
    "int": 3,
    "list": [1.0, 2.0],
    "number": 1.5,
    "str": "value",
}

#: One value guaranteed NOT to match each type tag. ``int`` deliberately gets a
#: bool and ``number`` a string: the checker must reject ``True`` as an int
#: (bool is a subclass of int) and must not coerce numeric strings.
_MISMATCHED_VALUES: dict[str, object] = {
    "bool": "yes",
    "dict": [1, 2],
    "int": True,
    "list": {"a": 1},
    "number": "1.5",
    "str": 7,
}

_SCHEMA_NAMES = sorted(report_schemas._REPORT_SCHEMAS)

_CONTRACT_CASES: list[tuple[str, FigureDependencyContract]] = [
    (generator, contract)
    for generator, contracts in sorted(FIGURE_DEPENDENCY_CONTRACTS.items())
    for contract in contracts
]
_CONTRACT_IDS = [f"{generator}-{contract.report_name}" for generator, contract in _CONTRACT_CASES]


def _fresh(tag: str, values: dict[str, object] = _SAMPLE_VALUES) -> object:
    value = values[tag]
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, list):
        return list(value)
    return value


def _version_for_spec(spec: report_schemas.SchemaDefinition) -> str | None:
    for name, candidate in report_schemas._REPORT_SCHEMAS.items():
        if candidate is spec:
            versions = report_schemas._SUPPORTED_REPORT_SCHEMA_VERSIONS.get(name)
            if versions:
                return sorted(versions)[0]
    return None


def _valid_payload(spec: report_schemas.SchemaDefinition) -> dict[str, object]:
    payload = {name: _fresh(tag) for name, tag in spec.required.items()}
    payload.update({name: _fresh(tag) for name, tag in spec.optional.items()})
    version = _version_for_spec(spec)
    if version is not None:
        payload["schema_version"] = version
    return payload


@lru_cache(maxsize=None)
def _base_valid_payload_for_name(name: str) -> dict[str, object]:
    """Build a real payload where a schema deliberately validates nested data."""
    if name == "belief_sharing":
        return {
            "schema_version": "1.0",
            "analysis_unit": "configured seed with within-seed paired conditions",
            "communicating_free_energy": [1.0, 2.0],
            "communicating_mean": 1.5,
            "communication_helps": True,
            "difference_definition": "incommunicado_minus_communicating",
            "free_energy_gap": 1.5,
            "incommunicado_free_energy": [2.0, 4.0],
            "incommunicado_mean": 3.0,
            "interval_method": "paired-seed percentile bootstrap",
            "interval_percent": 95,
            "n_agents": 4,
            "n_seeds": 2,
            "paired_difference_ci": [1.0, 2.0],
            "paired_difference_mean": 1.5,
            "paired_free_energy_difference": [1.0, 2.0],
            "replication_unit": "configured seed",
        }
    if name == "bnn_robustness":
        return run_bnn_robustness_report(
            3,
            n_seeds=2,
            n_per=8,
            contamination_levels=(0.0, 0.3),
        )
    if name == "hierarchical_bmr":
        return run_hierarchical_bmr()
    if name == "parameter_recovery":
        return run_parameter_recovery(
            5,
            acuity_grid=(0.65, 0.8),
            n_observations=5,
            n_trials=2,
            fit_resolution=4,
        )
    if name == "application_integrity_flow":
        return dict(application_integrity_flow_contract())
    if name == "evidence_replication_map":
        return dict(evidence_replication_map_contract())
    if name == "source_render_provenance":
        return dict(source_render_provenance_contract())
    if name == "robustness_review_grid":
        return run_review_grid(
            seed=11,
            n_seeds=2,
            n_trials=2,
            n_agents=3,
            rates=(0.0, 0.5),
            divergences=("KLD", "RKL"),
            target_max_mcse=1.0,
        )
    if name == "robustness_sweep":
        report = run_robustness_sweep(
            7,
            rates=(0.0, 0.5),
            divergences=("KLD", "RKL"),
            n_agents=3,
            n_contaminated=1,
            n_trials=2,
        )
        report["n_agents"] = 3
        report["n_contaminated"] = 1
        return report
    if name == "sensitivity":
        return _sensitivity_report(seed=0, n_trials=2)
    return _valid_payload(report_schemas._REPORT_SCHEMAS[name])


def _valid_payload_for_name(name: str) -> dict[str, object]:
    """Return an isolated copy so mutation tests cannot poison cached producers."""
    return deepcopy(_base_valid_payload_for_name(name))


def _valid_bnn_torch_ok_payload() -> dict[str, object]:
    return {
        "accuracy_by_config": {"beta->0 (standard)": [0.85], "beta=0.5 (robust)": [0.84]},
        "beta": 0.5,
        "consensus_max_simplex_deviation": 2.2e-16,
        "contamination_levels": [0.0, 0.2, 0.4],
        "deterministic": True,
        "hidden_dim": 16,
        "n_clients": 5,
        "n_steps": 200,
        "reported_contamination": 0.4,
        "robust_accuracy": 0.55,
        "robustness": 0.5,
        "seed": 0,
        "standard_accuracy": 0.56,
        "status": "ok",
        "torch_version": "2.12.1",
    }


def _valid_figure_entry(label: str = "fig:belief-heatmap") -> dict[str, str]:
    return {
        "label": label,
        "filename": "belief_heatmap.png",
        "path": "output/figures/belief_heatmap.png",
        "source_manuscript": "manuscript/16_results_belief_sharing.md",
        "caption": "Belief heatmap caption.",
        "generated_by": "belief_heatmap",
        "status": "generated",
        "source_relation": "analogue",
        "source_figure": "Figure 2",
        "source_equation": "none",
        "source_citation": "friston2024federated",
        "estimand": "posterior belief per agent",
        "unit": "probability",
        "uncertainty": "none",
        "replication_unit": "seed",
        "alt_text": "Heatmap alternative text.",
    }


def _valid_figure_registry_payload() -> dict[str, object]:
    return {
        "schema_version": "1.2",
        "generated_by": "analysis.workflow.run_analysis_pipeline",
        "figures": [_valid_figure_entry()],
    }


# ---------------------------------------------------------------------------
# Per-schema accept / reject coverage (every schema the module defines)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _SCHEMA_NAMES)
def test_valid_payload_accepted(name: str) -> None:
    validate_report(name, _valid_payload_for_name(name))


@pytest.mark.parametrize("name", [name for name in _SCHEMA_NAMES if name != "robustness_review_grid"])
def test_required_only_payload_accepted(name: str) -> None:
    # Declared optional fields may be absent entirely.
    spec = report_schemas._REPORT_SCHEMAS[name]
    payload = _valid_payload_for_name(name)
    for field in spec.optional:
        payload.pop(field, None)
    version = _version_for_spec(spec)
    if version is not None:
        payload["schema_version"] = version
    validate_report(name, payload)


@pytest.mark.parametrize("name", _SCHEMA_NAMES)
def test_each_missing_required_field_rejected_with_named_diagnostics(name: str) -> None:
    spec = report_schemas._REPORT_SCHEMAS[name]
    for field in spec.required:
        payload = _valid_payload_for_name(name)
        del payload[field]
        with pytest.raises(ReportSchemaError) as excinfo:
            validate_report(name, payload)
        message = str(excinfo.value)
        assert name in message, f"error must name the report, got: {message}"
        assert repr(field) in message, f"error must name the field, got: {message}"
        assert "missing required field" in message


@pytest.mark.parametrize("name", _SCHEMA_NAMES)
def test_each_mistyped_required_field_rejected_with_named_diagnostics(name: str) -> None:
    spec = report_schemas._REPORT_SCHEMAS[name]
    for field, tag in spec.required.items():
        payload = _valid_payload_for_name(name)
        payload[field] = _fresh(tag, _MISMATCHED_VALUES)
        with pytest.raises(ReportSchemaError) as excinfo:
            validate_report(name, payload)
        message = str(excinfo.value)
        assert name in message and repr(field) in message
        assert f"expected {tag}" in message


@pytest.mark.parametrize(
    "name",
    [name for name in _SCHEMA_NAMES if report_schemas._REPORT_SCHEMAS[name].optional],
)
def test_present_optional_field_with_wrong_type_rejected(name: str) -> None:
    spec = report_schemas._REPORT_SCHEMAS[name]
    for field, tag in spec.optional.items():
        payload = _valid_payload_for_name(name)
        payload[field] = _fresh(tag, _MISMATCHED_VALUES)
        with pytest.raises(ReportSchemaError) as excinfo:
            validate_report(name, payload)
        assert repr(field) in str(excinfo.value)


def test_bool_is_not_an_int_and_int_is_a_number() -> None:
    # 'seed' is declared int: True must be rejected even though bool <: int.
    spec = report_schemas._REPORT_SCHEMAS["emergence"]
    payload = _valid_payload(spec)
    payload["seed"] = True
    with pytest.raises(ReportSchemaError, match="'seed'"):
        validate_report("emergence", payload)
    # A plain int is an acceptable 'number' (JSON round-trips 1.0 as 1).
    payload = _valid_payload(spec)
    payload["delta_F_redundant"] = 2
    validate_report("emergence", payload)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nested_nonfinite_json_numbers_are_rejected(value: float) -> None:
    payload = _valid_payload_for_name("language_acquisition")
    payload["kl_trajectory"] = [value]
    with pytest.raises(ReportSchemaError, match="non-finite JSON number"):
        validate_report("language_acquisition", payload)


def test_unknown_schema_name_rejected() -> None:
    with pytest.raises(ReportSchemaError, match="Unknown report schema"):
        validate_report("no_such_report", {"anything": 1})


def test_review_grid_deep_validator_rejects_a_missing_interval() -> None:
    payload = _valid_payload_for_name("robustness_review_grid")
    del payload["statistics"]["by_mechanism"]["confident_wrong"]["by_rate"]["0"]["methods"]["RKL"][
        "contrast_ci"
    ]
    with pytest.raises(ReportSchemaError, match="contrast_ci"):
        validate_report("robustness_review_grid", payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("observed_max_mcse", 0.0, "maximum MCSE disagrees"),
        ("n_signed_method_rate_cells", 999, "mismatched signed method-rate cell count"),
    ),
)
def test_review_grid_precision_receipt_is_bound_to_all_signed_cells(
    field: str,
    value: object,
    message: str,
) -> None:
    payload = deepcopy(_valid_payload_for_name("robustness_review_grid"))
    payload["precision_plan"][field] = value
    with pytest.raises(ReportSchemaError, match=message):
        validate_report("robustness_review_grid", payload)


def test_belief_sharing_paired_estimand_is_recomputed_from_source_arrays() -> None:
    payload = _valid_payload_for_name("belief_sharing")
    payload["paired_free_energy_difference"][0] += 0.25
    with pytest.raises(ReportSchemaError, match="must equal"):
        validate_report("belief_sharing", payload)


def test_belief_sharing_rejects_seed_as_an_unpaired_replication_unit() -> None:
    payload = _valid_payload_for_name("belief_sharing")
    payload["analysis_unit"] = "unpaired observation"
    with pytest.raises(ReportSchemaError, match="within-seed pairing"):
        validate_report("belief_sharing", payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("trial_count", "wrong n_trials"),
        ("fit_bounds", "fit_bounds"),
        ("seed_role", "seed_role"),
    ),
)
def test_parameter_recovery_deep_contract_rejects_semantic_drift(
    mutation: str,
    message: str,
) -> None:
    payload = _valid_payload_for_name("parameter_recovery")
    if mutation == "trial_count":
        payload["recovered_acuity_by_trial"][0].pop()
    elif mutation == "fit_bounds":
        payload["fit_bounds"][1] = 0.75
    else:
        payload["seed_role"] = "independent replication unit"
    with pytest.raises(ReportSchemaError, match=message):
        validate_report("parameter_recovery", payload)


def test_hierarchical_bmr_rejects_threshold_verdict_drift() -> None:
    payload = _valid_payload_for_name("hierarchical_bmr")
    payload["surprise_tol"] = 1.0
    with pytest.raises(ReportSchemaError, match="prunable flag disagrees"):
        validate_report("hierarchical_bmr", payload)


def test_hierarchical_bmr_rejects_secondary_diagnostic_as_decision_owner() -> None:
    payload = _valid_payload_for_name("hierarchical_bmr")
    payload["secondary_diagnostic"] = "redundancy_delta_F determines pruning"
    with pytest.raises(ReportSchemaError, match="must not own"):
        validate_report("hierarchical_bmr", payload)


def test_bnn_legacy_stem_cannot_be_promoted_to_a_posterior_bnn() -> None:
    payload = _valid_payload_for_name("bnn_robustness")
    payload["model_family"] = "bayesian_neural_network"
    with pytest.raises(ReportSchemaError, match="point_estimate_logistic_regression"):
        validate_report("bnn_robustness", payload)


def test_bnn_selected_operating_point_is_bound_to_full_sweep() -> None:
    payload = _valid_payload_for_name("bnn_robustness")
    payload["peak_margin"] += 0.1
    with pytest.raises(ReportSchemaError, match="within-sweep selection"):
        validate_report("bnn_robustness", payload)


def test_bnn_proxy_configuration_labels_cannot_restore_divergence_claims() -> None:
    payload = _valid_payload_for_name("bnn_robustness")
    for field in (
        "accuracy_by_config",
        "accuracy_ci_by_config",
        "accuracy_seed_values_by_config",
    ):
        mapping = payload[field]
        assert isinstance(mapping, dict)
        mapping["rcce / AR (robust)"] = mapping.pop("rcce / L2=0.10 (exploratory proxy)")

    with pytest.raises(ReportSchemaError, match="retain both declared baselines"):
        validate_report("bnn_robustness", payload)


def test_bnn_proxy_configuration_boundary_pins_l2_semantics() -> None:
    payload = _valid_payload_for_name("bnn_robustness")
    payload["configuration_boundary"] = "KLD and AR are evaluated divergences"

    with pytest.raises(ReportSchemaError, match="expose the proxy's L2 semantics"):
        validate_report("bnn_robustness", payload)


def test_bnn_proxy_rejects_seed_row_mean_drift() -> None:
    payload = _valid_payload_for_name("bnn_robustness")
    rows = payload["accuracy_seed_values_by_config"]
    assert isinstance(rows, dict)
    config_rows = rows["nll / L2=0.05 (standard proxy)"]
    assert isinstance(config_rows, list) and isinstance(config_rows[0], list)
    config_rows[0][0] = 0.0

    with pytest.raises(ReportSchemaError, match="reported mean disagrees"):
        validate_report("bnn_robustness", payload)


@pytest.mark.parametrize("invalid_accuracy", (-0.01, 1.01))
def test_bnn_proxy_rejects_out_of_range_seed_accuracies(invalid_accuracy: float) -> None:
    payload = _valid_payload_for_name("bnn_robustness")
    rows = payload["accuracy_seed_values_by_config"]
    assert isinstance(rows, dict)
    config_rows = rows["nll / L2=0.05 (standard proxy)"]
    assert isinstance(config_rows, list) and isinstance(config_rows[0], list)
    config_rows[0][0] = invalid_accuracy

    with pytest.raises(ReportSchemaError, match="seed accuracy row"):
        validate_report("bnn_robustness", payload)


@pytest.mark.parametrize(
    ("field", "invalid_value", "message"),
    (
        ("analysis_unit", "observation", "analysis_unit"),
        ("replication_unit", "client", "replication_unit"),
        ("interval_method", "normal interval over clients", "interval_method"),
        ("ci_percent", 90, "95-percent"),
        ("n_bootstrap", 1, "5000-resample"),
        ("n_per", 0, "positive points-per-class"),
        ("robust_loss_param", -0.1, r"\[0, 1\]"),
        ("robust_loss_param", 1.1, r"\[0, 1\]"),
    ),
)
def test_bnn_proxy_pins_seed_level_design_fields(
    field: str,
    invalid_value: object,
    message: str,
) -> None:
    payload = _valid_payload_for_name("bnn_robustness")
    payload[field] = invalid_value

    with pytest.raises(ReportSchemaError, match=message):
        validate_report("bnn_robustness", payload)


def test_bnn_proxy_rejects_interval_inventory_and_bounds_drift() -> None:
    payload = _valid_payload_for_name("bnn_robustness")
    intervals = payload["accuracy_ci_by_config"]
    assert isinstance(intervals, dict)
    config_intervals = intervals["nll / L2=0.05 (standard proxy)"]
    assert isinstance(config_intervals, list)
    config_intervals.append(list(config_intervals[0]))

    with pytest.raises(ReportSchemaError, match="must match operating points"):
        validate_report("bnn_robustness", payload)

    payload = _valid_payload_for_name("bnn_robustness")
    intervals = payload["accuracy_ci_by_config"]
    assert isinstance(intervals, dict)
    config_intervals = intervals["nll / L2=0.05 (standard proxy)"]
    assert isinstance(config_intervals, list) and isinstance(config_intervals[0], list)
    config_intervals[0][0] = -0.01

    with pytest.raises(ReportSchemaError, match=r"interval must lie in \[0, 1\]"):
        validate_report("bnn_robustness", payload)


def test_bnn_proxy_selection_disclosure_cannot_promote_configured_inputs() -> None:
    payload = _valid_payload_for_name("bnn_robustness")
    payload["selection_disclosure"] = (
        "q and n_per were selected within the displayed sweep for the best result"
    )

    with pytest.raises(ReportSchemaError, match="limit selection to peak contamination"):
        validate_report("bnn_robustness", payload)


def test_robustness_sweep_rejects_server_preset_label_drift() -> None:
    payload = _valid_payload_for_name("robustness_sweep")
    payload["server_robustness_by_label"]["RKL"] = 0.0
    with pytest.raises(ReportSchemaError, match="positive robust values"):
        validate_report("robustness_sweep", payload)


def test_robustness_sweep_rejects_unmatched_per_rate_budget() -> None:
    payload = _valid_payload_for_name("robustness_sweep")
    payload["per_rate_summary"]["0"]["n"] += 1
    with pytest.raises(ReportSchemaError, match="matched trial budgets"):
        validate_report("robustness_sweep", payload)


# ---------------------------------------------------------------------------
# bnn_torch: executed "ok" payload vs declared "skipped" degradation
# ---------------------------------------------------------------------------


def test_bnn_torch_ok_payload_accepted() -> None:
    validate_report("bnn_torch", _valid_bnn_torch_ok_payload())


def test_bnn_torch_skipped_single_key_payload_accepted() -> None:
    # The PyTorch-optional degradation path writes only a status string.
    validate_report("bnn_torch", {"status": "skipped: torch not installed"})


def test_bnn_torch_missing_status_rejected() -> None:
    with pytest.raises(ReportSchemaError) as excinfo:
        validate_report("bnn_torch", {"seed": 0})
    message = str(excinfo.value)
    assert "bnn_torch" in message and "'status'" in message


def test_bnn_torch_non_string_status_rejected() -> None:
    with pytest.raises(ReportSchemaError, match="'status'"):
        validate_report("bnn_torch", {"status": 1})


def test_bnn_torch_unrecognised_status_rejected() -> None:
    with pytest.raises(ReportSchemaError, match="must be 'ok' or start with 'skipped'"):
        validate_report("bnn_torch", {"status": "maybe"})


def test_bnn_torch_ok_payload_missing_field_rejected() -> None:
    payload = _valid_bnn_torch_ok_payload()
    del payload["torch_version"]
    with pytest.raises(ReportSchemaError) as excinfo:
        validate_report("bnn_torch", payload)
    message = str(excinfo.value)
    assert "bnn_torch" in message and "'torch_version'" in message


# ---------------------------------------------------------------------------
# figure_registry: top-level payload plus per-entry metadata
# ---------------------------------------------------------------------------


def test_figure_registry_valid_payload_accepted() -> None:
    validate_report("figure_registry", _valid_figure_registry_payload())


def test_figure_registry_missing_top_level_field_rejected() -> None:
    payload = _valid_figure_registry_payload()
    del payload["schema_version"]
    with pytest.raises(ReportSchemaError) as excinfo:
        validate_report("figure_registry", payload)
    message = str(excinfo.value)
    assert "figure_registry" in message and "'schema_version'" in message


def test_figure_registry_non_dict_entry_rejected() -> None:
    payload = _valid_figure_registry_payload()
    payload["figures"] = ["not-a-dict"]
    with pytest.raises(ReportSchemaError, match="expected list of dict"):
        validate_report("figure_registry", payload)


def test_figure_registry_empty_inventory_rejected() -> None:
    payload = _valid_figure_registry_payload()
    payload["figures"] = []
    with pytest.raises(ReportSchemaError, match="at least one figure"):
        validate_report("figure_registry", payload)


def test_figure_registry_entry_missing_field_names_figure_label() -> None:
    entry = _valid_figure_entry()
    del entry["caption"]
    payload = _valid_figure_registry_payload()
    payload["figures"] = [entry]
    with pytest.raises(ReportSchemaError) as excinfo:
        validate_report("figure_registry", payload)
    message = str(excinfo.value)
    assert "'fig:belief-heatmap'" in message and "'caption'" in message


def test_figure_registry_entry_mistyped_field_rejected() -> None:
    entry = _valid_figure_entry()
    entry["status"] = 3  # type: ignore[assignment]
    payload = _valid_figure_registry_payload()
    payload["figures"] = [entry]
    with pytest.raises(ReportSchemaError) as excinfo:
        validate_report("figure_registry", payload)
    message = str(excinfo.value)
    assert "'status'" in message and "expected str" in message


def test_figure_registry_rejects_wrong_schema_or_producer() -> None:
    payload = _valid_figure_registry_payload()
    payload["schema_version"] = "1.1"
    with pytest.raises(ReportSchemaError, match="schema_version must be '1.2'"):
        validate_report("figure_registry", payload)

    payload = _valid_figure_registry_payload()
    payload["generated_by"] = "manual"
    with pytest.raises(ReportSchemaError, match="canonical producer"):
        validate_report("figure_registry", payload)


def test_figure_registry_rejects_duplicate_labels() -> None:
    first = _valid_figure_entry("fig:a")
    second = _valid_figure_entry("fig:b")
    second["filename"] = "other.png"
    second["path"] = "output/figures/other.png"
    second["generated_by"] = "other"
    second["label"] = first["label"]
    payload = _valid_figure_registry_payload()
    payload["figures"] = [first, second]

    with pytest.raises(ReportSchemaError, match="duplicate label"):
        validate_report("figure_registry", payload)


def test_figure_registry_rejects_duplicate_generator_and_filename() -> None:
    first = _valid_figure_entry("fig:a")
    second = _valid_figure_entry("fig:b")
    payload = _valid_figure_registry_payload()
    payload["figures"] = [first, second]

    with pytest.raises(ReportSchemaError, match="duplicate filename"):
        validate_report("figure_registry", payload)


def test_figure_registry_fallback_requires_matching_artifact() -> None:
    entry = _valid_figure_entry()
    entry["exact_value_fallback"] = "fig-values:belief-heatmap"
    payload = _valid_figure_registry_payload()
    payload["figures"] = [entry]
    with pytest.raises(ReportSchemaError, match="require exact_value_artifact"):
        validate_report("figure_registry", payload)

    payload["exact_value_artifact"] = {
        "json_path": "output/figures/figure_exact_values.json",
        "markdown_path": "output/figures/figure_exact_values.md",
        "identifiers": ["fig-values:other"],
    }
    with pytest.raises(ReportSchemaError, match="do not match"):
        validate_report("figure_registry", payload)


def test_figure_registry_rejects_noncanonical_exact_value_paths() -> None:
    payload = _valid_figure_registry_payload()
    payload["exact_value_artifact"] = {
        "json_path": "elsewhere/figure_exact_values.json",
        "markdown_path": "output/figures/figure_exact_values.md",
        "identifiers": [],
    }
    with pytest.raises(ReportSchemaError, match="JSON path is not canonical"):
        validate_report("figure_registry", payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.__setitem__("acuity_values", [0.5, 0.5]), "strictly increasing"),
        (lambda payload: payload.__setitem__("n_agents_values", [4, 2]), "strictly increasing"),
        (lambda payload: payload.__setitem__("noise_floor", 1.1), r"must lie in \[0, 1\]"),
    ],
)
def test_sensitivity_report_rejects_ambiguous_axes_or_band(mutation, message: str) -> None:
    payload = _sensitivity_report(seed=0, n_trials=2)
    mutation(payload)
    with pytest.raises(ReportSchemaError, match=message):
        validate_report("sensitivity", payload)


def test_sensitivity_report_rejects_impossible_accuracy_gap() -> None:
    payload = _sensitivity_report(seed=0, n_trials=2)
    payload["belief_sharing"]["accuracy_gap_grid"][0][0] = 1.1  # type: ignore[index]
    with pytest.raises(ReportSchemaError, match=r"must lie in \[-1, 1\]"):
        validate_report("sensitivity", payload)


# ---------------------------------------------------------------------------
# Figure dependency contracts (unit level)
# ---------------------------------------------------------------------------


def test_check_figure_contract_unknown_generator_rejected() -> None:
    with pytest.raises(ReportSchemaError, match="Unknown figure contract"):
        check_figure_contract("no_such_figure", "belief_sharing", {})


def test_check_figure_contract_undeclared_report_rejected() -> None:
    with pytest.raises(ReportSchemaError, match="does not declare a dependency"):
        check_figure_contract("emergence_bmr", "belief_sharing", {})


@pytest.mark.parametrize(("generator", "contract"), _CONTRACT_CASES, ids=_CONTRACT_IDS)
def test_contract_required_fields_are_a_subset_of_the_write_schema(
    generator: str, contract: FigureDependencyContract
) -> None:
    # Every field a figure declares must be one the write boundary guarantees,
    # so a report that passes _write_json validation always feeds its figures.
    spec = report_schemas._REPORT_SCHEMAS[contract.report_name]
    for field, tag in contract.required_fields.items():
        assert field in spec.required, f"{generator} requires undeclared field {field!r}"
        assert spec.required[field] == tag
    for field in contract.optional_fields:
        assert field in spec.required or field in spec.optional


# ---------------------------------------------------------------------------
# Stage contracts against the REAL committed artifacts on disk
# ---------------------------------------------------------------------------


@pytest.mark.publication
@pytest.mark.parametrize("name", _SCHEMA_NAMES + ["bnn_torch"])
def test_real_report_on_disk_satisfies_write_schema(name: str) -> None:
    report_path = REPORTS_DIR / f"{name}.json"
    assert report_path.exists(), f"committed report missing: {report_path}"
    validate_report(name, json.loads(report_path.read_text(encoding="utf-8")))


@pytest.mark.publication
def test_real_figure_registry_on_disk_satisfies_schema() -> None:
    assert FIGURE_REGISTRY_PATH.exists(), f"committed registry missing: {FIGURE_REGISTRY_PATH}"
    validate_report("figure_registry", json.loads(FIGURE_REGISTRY_PATH.read_text(encoding="utf-8")))


@pytest.mark.publication
@pytest.mark.parametrize(("generator", "contract"), _CONTRACT_CASES, ids=_CONTRACT_IDS)
def test_real_report_on_disk_satisfies_figure_contract(
    generator: str, contract: FigureDependencyContract
) -> None:
    report_path = REPORTS_DIR / f"{contract.report_name}.json"
    assert report_path.exists(), f"committed report missing: {report_path}"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    check_figure_contract(generator, contract.report_name, report)


@pytest.mark.publication
@pytest.mark.parametrize(("generator", "contract"), _CONTRACT_CASES, ids=_CONTRACT_IDS)
def test_stripped_real_report_fails_figure_contract_with_named_field(
    generator: str, contract: FigureDependencyContract
) -> None:
    report_path = REPORTS_DIR / f"{contract.report_name}.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    field = sorted(contract.required_fields)[0]
    stripped = dict(report)
    del stripped[field]
    with pytest.raises(ReportSchemaError) as excinfo:
        check_figure_contract(generator, contract.report_name, stripped)
    message = str(excinfo.value)
    assert generator in message, f"error must name the figure, got: {message}"
    assert contract.report_name in message, f"error must name the report, got: {message}"
    assert repr(field) in message, f"error must name the field, got: {message}"


# ---------------------------------------------------------------------------
# Pipeline write boundary: _write_json rejects before anything lands on disk
# ---------------------------------------------------------------------------


def test_write_boundary_rejects_missing_field_and_writes_nothing(tmp_path: Path) -> None:
    spec = report_schemas._REPORT_SCHEMAS["belief_sharing"]
    payload = _valid_payload(spec)
    del payload["n_agents"]
    target = tmp_path / "reports" / "belief_sharing.json"
    with pytest.raises(ReportSchemaError) as excinfo:
        _write_json(payload, target, schema="belief_sharing")
    message = str(excinfo.value)
    assert "belief_sharing" in message and "'n_agents'" in message
    assert not target.exists(), "malformed report must never land on disk"
    assert not target.parent.exists(), "validation must precede directory creation"


def test_write_boundary_rejects_mistyped_field_and_writes_nothing(tmp_path: Path) -> None:
    spec = report_schemas._REPORT_SCHEMAS["emergence"]
    payload = _valid_payload(spec)
    payload["n_states"] = "four"
    target = tmp_path / "emergence.json"
    with pytest.raises(ReportSchemaError) as excinfo:
        _write_json(payload, target, schema="emergence")
    message = str(excinfo.value)
    assert "emergence" in message and "'n_states'" in message and "expected int" in message
    assert not target.exists()
    assert list(tmp_path.iterdir()) == [], "no partial artifact may be written"


def test_write_boundary_accepts_valid_payload_and_round_trips(tmp_path: Path) -> None:
    spec = report_schemas._REPORT_SCHEMAS["emergence"]
    payload = _valid_payload(spec)
    target = tmp_path / "emergence.json"
    written = _write_json(payload, target, schema="emergence")
    assert written == target
    assert json.loads(target.read_text(encoding="utf-8")) == payload


def test_write_boundary_rejects_nonfinite_payload_before_replacing_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "report.json"
    target.write_text("original\n", encoding="utf-8")
    with pytest.raises(ValueError):
        _write_json({"value": math.nan}, target)
    assert target.read_text(encoding="utf-8") == "original\n"


def test_write_boundary_serialization_failure_preserves_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "report.json"
    target.write_text("original\n", encoding="utf-8")
    with pytest.raises(TypeError):
        _write_json({"value": object()}, target)
    assert target.read_text(encoding="utf-8") == "original\n"
