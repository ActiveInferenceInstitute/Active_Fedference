"""Source render provenance report contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict

from analysis.schemas.primitives import ReportSchemaError, SchemaDefinition
from analysis.schemas.records import _check_edges, _check_nonempty_string_fields, _record_ids, _record_list
from analysis.visual_contracts import (
    SOURCE_RENDER_INPUT_INVENTORY,
    SOURCE_RENDER_INVALIDATION_EDGE_INVENTORY,
    SOURCE_RENDER_PRODUCER_EDGE_INVENTORY,
    SOURCE_RENDER_STAGE_INVENTORY,
)


class SourceRenderProvenanceReport(TypedDict):
    """Source-owned payload for producer order and stale invalidation."""

    schema_version: str
    study_status: str
    source_relation: str
    inputs: list[dict[str, object]]
    stages: list[dict[str, object]]
    terminals: list[dict[str, object]]
    producer_edges: list[dict[str, object]]
    invalidation_edges: list[dict[str, object]]
    authorization_boundary: str
    manifest_cycle_boundary: str
    no_claims: list[str]


def _edge_inventory(records: list[Mapping[str, object]]) -> tuple[tuple[str, str, str], ...]:
    """Return ordered endpoint/disposition triples for an exact flow inventory."""

    return tuple((str(edge["source"]), str(edge["target"]), str(edge["disposition"])) for edge in records)


def _check_source_render_provenance(payload: Mapping[str, object]) -> None:
    """Validate producer order, reverse invalidation, and authorization gates."""
    schema = "source_render_provenance"
    if payload["study_status"] != "source_owned_pipeline_contract":
        raise ReportSchemaError(f"{schema} study_status must identify a pipeline contract")
    inputs = _record_list(payload["inputs"], schema=schema, field="inputs")
    stages = _record_list(payload["stages"], schema=schema, field="stages")
    terminals = _record_list(payload["terminals"], schema=schema, field="terminals")
    input_ids = _record_ids(inputs, schema=schema, field="inputs")
    stage_ids = _record_ids(stages, schema=schema, field="stages")
    terminal_ids = _record_ids(terminals, schema=schema, field="terminals")
    if input_ids != list(SOURCE_RENDER_INPUT_INVENTORY):
        raise ReportSchemaError(f"{schema} input inventory or order is invalid")
    if stage_ids != list(SOURCE_RENDER_STAGE_INVENTORY):
        raise ReportSchemaError(f"{schema} producer-stage inventory or order is invalid")
    if terminal_ids != ["github", "zenodo"]:
        raise ReportSchemaError(f"{schema} terminal inventory or order is invalid")
    for index, terminal in enumerate(terminals):
        _check_nonempty_string_fields(
            terminal,
            schema=schema,
            field=f"terminals[{index}]",
            names=("label", "role"),
        )
        if terminal.get("authorization_required") is not True:
            raise ReportSchemaError(f"{schema} publication terminals must require authorization")
    all_nodes = set(input_ids) | set(stage_ids) | set(terminal_ids)
    producer_edges = _record_list(payload["producer_edges"], schema=schema, field="producer_edges")
    _check_edges(
        producer_edges,
        schema=schema,
        field="producer_edges",
        valid_nodes=all_nodes,
        allowed_dispositions={"producer", "gate", "receipt", "authorization"},
    )
    if _edge_inventory(producer_edges) != SOURCE_RENDER_PRODUCER_EDGE_INVENTORY:
        raise ReportSchemaError(f"{schema} producer edge inventory, order, or disposition is invalid")
    producer_pairs = {(edge["source"], edge["target"]) for edge in producer_edges}
    if ("release_manifest", "github") not in producer_pairs or ("github", "zenodo") not in producer_pairs:
        raise ReportSchemaError(f"{schema} must retain both publication authorization gates")
    invalidation_edges = _record_list(
        payload["invalidation_edges"], schema=schema, field="invalidation_edges"
    )
    _check_edges(
        invalidation_edges,
        schema=schema,
        field="invalidation_edges",
        valid_nodes=all_nodes,
        allowed_dispositions={"invalidation"},
    )
    if _edge_inventory(invalidation_edges) != SOURCE_RENDER_INVALIDATION_EDGE_INVENTORY:
        raise ReportSchemaError(f"{schema} invalidation edge inventory, order, or disposition is invalid")
    producer_pairs = {(str(edge["source"]), str(edge["target"])) for edge in producer_edges}
    for index, edge in enumerate(invalidation_edges):
        pair = (str(edge["source"]), str(edge["target"]))
        if pair not in producer_pairs:
            raise ReportSchemaError(
                f"{schema} payload field invalidation_edges[{index}] must name an "
                "immediate producer dependency"
            )
    authorization = payload["authorization_boundary"]
    cycle_boundary = payload["manifest_cycle_boundary"]
    assert isinstance(authorization, str) and isinstance(cycle_boundary, str)
    if "do not automatically authorize" not in authorization:
        raise ReportSchemaError(f"{schema} must deny automatic publication authorization")
    if "never reads" not in cycle_boundary or "release manifest" not in cycle_boundary:
        raise ReportSchemaError(f"{schema} must deny a generated-manifest dependency cycle")
    no_claims = payload["no_claims"]
    if (
        not isinstance(no_claims, list)
        or len(no_claims) < 4
        or any(not isinstance(item, str) or not item for item in no_claims)
    ):
        raise ReportSchemaError(f"{schema} must retain engineering and accessibility no-claims")


_SOURCE_RENDER_PROVENANCE_SCHEMA = SchemaDefinition(
    required={
        "schema_version": "str",
        "study_status": "str",
        "source_relation": "str",
        "inputs": "list",
        "stages": "list",
        "terminals": "list",
        "producer_edges": "list",
        "invalidation_edges": "list",
        "authorization_boundary": "str",
        "manifest_cycle_boundary": "str",
        "no_claims": "list",
    }
)
