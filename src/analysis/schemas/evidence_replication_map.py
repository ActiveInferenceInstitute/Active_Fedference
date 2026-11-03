"""Evidence replication map report contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict

from analysis.schemas.primitives import ReportSchemaError, SchemaDefinition, _mapping
from analysis.schemas.records import _check_edges, _check_nonempty_string_fields, _record_ids, _record_list


class EvidenceReplicationMapReport(TypedDict):
    """Source-owned payload for the evidence and replication map."""

    schema_version: str
    study_status: str
    source_relation: str
    evidence_classes: list[dict[str, object]]
    lanes: list[dict[str, object]]
    nesting: list[dict[str, object]]
    nesting_edges: list[dict[str, object]]
    no_claims: list[str]


def _check_evidence_replication_map(payload: Mapping[str, object]) -> None:
    """Keep evidence classes and independent-unit ownership in separate lanes."""
    schema = "evidence_replication_map"
    if payload["study_status"] != "source_owned_evidence_classification":
        raise ReportSchemaError(f"{schema} study_status must identify a source-owned classification")
    classes = _record_list(payload["evidence_classes"], schema=schema, field="evidence_classes")
    class_ids = _record_ids(classes, schema=schema, field="evidence_classes")
    expected_classes = [
        "formal_executable",
        "source_conditional",
        "conditional_empirical",
        "scoped_implementation",
        "open",
    ]
    if class_ids != expected_classes:
        raise ReportSchemaError(f"{schema} evidence-class inventory or order is invalid")
    for index, evidence_class in enumerate(classes):
        _check_nonempty_string_fields(
            evidence_class,
            schema=schema,
            field=f"evidence_classes[{index}]",
            names=("label", "meaning"),
        )
    lanes = _record_list(payload["lanes"], schema=schema, field="lanes")
    lane_ids = _record_ids(lanes, schema=schema, field="lanes")
    expected_lane_classes = {
        "project_identity": "formal_executable",
        "client_loss": "source_conditional",
        "belief_sharing_free_energy": "conditional_empirical",
        "likelihood_learning": "conditional_empirical",
        "bmr_sign_control": "formal_executable",
        "client_loss_baseline": "conditional_empirical",
        "server_heuristic": "conditional_empirical",
        "variational_server": "formal_executable",
        "moving_disjoint_fov": "conditional_empirical",
        "hierarchy_sensitivity": "conditional_empirical",
        "parameter_recovery": "conditional_empirical",
        "protocol_integrity": "scoped_implementation",
        "application_provenance": "scoped_implementation",
        "external_confirmation": "open",
    }
    if lane_ids != list(expected_lane_classes):
        raise ReportSchemaError(f"{schema} claim-owner lanes are incomplete or reordered")
    for index, lane in enumerate(lanes):
        _check_nonempty_string_fields(
            lane,
            schema=schema,
            field=f"lanes[{index}]",
            names=(
                "headline",
                "evidence_class",
                "estimand",
                "unit",
                "replication_unit",
                "nesting",
                "permitted_interpretation",
                "prohibited_generalization",
            ),
        )
        if lane["evidence_class"] not in set(class_ids):
            raise ReportSchemaError(f"{schema} lane references an unknown evidence class")
        expected_class = expected_lane_classes[str(lane["id"])]
        if lane["evidence_class"] != expected_class:
            raise ReportSchemaError(
                f"{schema} lane {lane['id']!r} must retain evidence class {expected_class!r}"
            )
        display = _mapping(lane.get("display"), schema=schema, field=f"lanes[{index}].display")
        if set(display) != {
            "estimand_unit",
            "replication",
            "nesting",
            "permitted",
            "prohibited",
        }:
            raise ReportSchemaError(f"{schema} lane display summary has an invalid field set")
        _check_nonempty_string_fields(
            display,
            schema=schema,
            field=f"lanes[{index}].display",
            names=("estimand_unit", "replication", "nesting", "permitted", "prohibited"),
        )
    nesting = _record_list(payload["nesting"], schema=schema, field="nesting")
    nesting_ids = _record_ids(nesting, schema=schema, field="nesting")
    if nesting_ids != ["study", "seed", "trial", "agent", "step"]:
        raise ReportSchemaError(f"{schema} nesting inventory or order is invalid")
    _check_edges(
        _record_list(payload["nesting_edges"], schema=schema, field="nesting_edges"),
        schema=schema,
        field="nesting_edges",
        valid_nodes=set(nesting_ids),
        allowed_dispositions={"nesting"},
    )
    no_claims = payload["no_claims"]
    if (
        not isinstance(no_claims, list)
        or len(no_claims) < 3
        or any(not isinstance(item, str) or not item for item in no_claims)
    ):
        raise ReportSchemaError(f"{schema} must retain explicit non-transfer boundaries")


_EVIDENCE_REPLICATION_MAP_SCHEMA = SchemaDefinition(
    required={
        "schema_version": "str",
        "study_status": "str",
        "source_relation": "str",
        "evidence_classes": "list",
        "lanes": "list",
        "nesting": "list",
        "nesting_edges": "list",
        "no_claims": "list",
    }
)
