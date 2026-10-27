"""Application integrity flow report contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict

from analysis.schemas.primitives import ReportSchemaError, SchemaDefinition
from analysis.schemas.records import _check_edges, _check_nonempty_string_fields, _record_ids, _record_list
from analysis.visual_contracts import (
    APPLICATION_ARTIFACT_WRITE_ORDER,
    APPLICATION_FLOW_EDGE_INVENTORY,
    APPLICATION_PANEL_NODE_INVENTORY,
)


class ApplicationIntegrityFlowReport(TypedDict):
    """Source-owned payload for the application integrity flow figure."""

    schema_version: str
    study_status: str
    source_relation: str
    panels: list[dict[str, object]]
    solver_statuses: list[dict[str, object]]
    verification_levels: list[dict[str, object]]
    no_claims: list[str]


def _check_application_integrity_flow(payload: Mapping[str, object]) -> None:
    """Bind the application flow to implemented status and verification states."""
    schema = "application_integrity_flow"
    if payload["study_status"] != "source_owned_explanatory_contract":
        raise ReportSchemaError(f"{schema} study_status must identify an explanatory contract")
    panels = _record_list(payload["panels"], schema=schema, field="panels")
    panel_ids = _record_ids(panels, schema=schema, field="panels")
    if panel_ids != ["validation", "solver_health", "receipt"]:
        raise ReportSchemaError(f"{schema} panels must preserve validation, solver, receipt order")
    global_nodes: set[str] = set()
    edges: list[Mapping[str, object]] = []
    edge_inventory: list[tuple[str, str, str, str]] = []
    for index, panel in enumerate(panels):
        panel_id = str(panel["id"])
        _check_nonempty_string_fields(
            panel,
            schema=schema,
            field=f"panels[{index}]",
            names=("label",),
        )
        nodes = _record_list(panel.get("nodes"), schema=schema, field=f"panels[{index}].nodes")
        node_ids = _record_ids(nodes, schema=schema, field=f"panels[{index}].nodes")
        expected_node_ids = dict(APPLICATION_PANEL_NODE_INVENTORY)[panel_id]
        if node_ids != list(expected_node_ids):
            raise ReportSchemaError(f"{schema} panel {panel_id!r} node inventory or order is invalid")
        overlap = global_nodes & set(node_ids)
        if overlap:
            raise ReportSchemaError(f"{schema} nodes are duplicated across panels: {sorted(overlap)}")
        global_nodes.update(node_ids)
        for node_index, node in enumerate(nodes):
            _check_nonempty_string_fields(
                node,
                schema=schema,
                field=f"panels[{index}].nodes[{node_index}]",
                names=("label", "role"),
            )
        panel_edges = _record_list(panel.get("edges"), schema=schema, field=f"panels[{index}].edges")
        edges.extend(panel_edges)
        edge_inventory.extend(
            (
                panel_id,
                str(edge.get("source")),
                str(edge.get("target")),
                str(edge.get("disposition")),
            )
            for edge in panel_edges
        )
        if panel_id == "receipt":
            write_order = panel.get("write_order")
            if not isinstance(write_order, list) or tuple(write_order) != APPLICATION_ARTIFACT_WRITE_ORDER:
                raise ReportSchemaError(f"{schema} receipt artifact write order is invalid")
        elif "write_order" in panel:
            raise ReportSchemaError(f"{schema} write_order belongs only to the receipt panel")
    _check_edges(
        edges,
        schema=schema,
        field="panels[].edges",
        valid_nodes=global_nodes,
        allowed_dispositions={"normal", "rejected", "retained_warning", "data_dependency"},
    )
    if tuple(edge_inventory) != APPLICATION_FLOW_EDGE_INVENTORY:
        raise ReportSchemaError(f"{schema} edge inventory, panel ownership, or order is invalid")
    statuses = _record_list(payload["solver_statuses"], schema=schema, field="solver_statuses")
    status_ids = _record_ids(statuses, schema=schema, field="solver_statuses")
    expected_status = {
        "nominal": (True, False),
        "converged_with_fallback": (True, True),
        "not_converged": (False, False),
        "not_converged_with_fallback": (False, True),
    }
    if status_ids != list(expected_status):
        raise ReportSchemaError(f"{schema} solver status order or inventory is invalid")
    for status in statuses:
        identifier = str(status["id"])
        if (
            status.get("label") != identifier
            or (status.get("converged"), status.get("fallbacks")) != expected_status[identifier]
        ):
            raise ReportSchemaError(f"{schema} solver state {identifier!r} has invalid semantics")
    levels = _record_list(payload["verification_levels"], schema=schema, field="verification_levels")
    if _record_ids(levels, schema=schema, field="verification_levels") != [
        "artifact_integrity",
        "source_equivalence",
        "nominal_solver",
    ]:
        raise ReportSchemaError(f"{schema} verification levels are incomplete or reordered")
    for index, level in enumerate(levels):
        _check_nonempty_string_fields(
            level,
            schema=schema,
            field=f"verification_levels[{index}]",
            names=("label", "establishes", "does_not_establish"),
        )
    no_claims = payload["no_claims"]
    if not isinstance(no_claims, list) or set(no_claims) != {
        "calibration",
        "domain suitability",
        "scientific validity",
        "downstream decision correctness",
        "acceptance",
    }:
        raise ReportSchemaError(f"{schema} must retain the complete receipt no-claim boundary")


_APPLICATION_INTEGRITY_FLOW_SCHEMA = SchemaDefinition(
    required={
        "schema_version": "str",
        "study_status": "str",
        "source_relation": "str",
        "panels": "list",
        "solver_statuses": "list",
        "verification_levels": "list",
        "no_claims": "list",
    }
)
