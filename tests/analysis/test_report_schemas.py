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
from analysis.workflow import _PROJECT_ROOT, _write_json
from fedference.experiments import run_review_grid

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


def _valid_payload_for_name(name: str) -> dict[str, object]:
    """Build a real payload where a schema deliberately validates nested data."""
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
    return _valid_payload(report_schemas._REPORT_SCHEMAS[name])


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
        "schema_version": "1.1",
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
    payload = {field: _fresh(tag) for field, tag in spec.required.items()}
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
