"""Source-bound tests for explanatory visual contracts and their schemas."""

from __future__ import annotations

from copy import deepcopy

import pytest

from analysis.report_schemas import ReportSchemaError, check_figure_contract, validate_report
from analysis.visual_contracts import (
    APPLICATION_ARTIFACT_WRITE_ORDER,
    APPLICATION_FLOW_EDGE_INVENTORY,
    APPLICATION_PANEL_NODE_INVENTORY,
    SOURCE_RENDER_INPUT_INVENTORY,
    SOURCE_RENDER_INVALIDATION_EDGE_INVENTORY,
    SOURCE_RENDER_PRODUCER_EDGE_INVENTORY,
    SOURCE_RENDER_STAGE_INVENTORY,
    application_integrity_flow_contract,
    evidence_replication_map_contract,
    source_render_provenance_contract,
)


@pytest.mark.parametrize(
    ("schema", "generator", "producer"),
    (
        (
            "application_integrity_flow",
            "application_integrity_flow",
            application_integrity_flow_contract,
        ),
        (
            "evidence_replication_map",
            "evidence_replication_map",
            evidence_replication_map_contract,
        ),
        (
            "source_render_provenance",
            "source_render_provenance",
            source_render_provenance_contract,
        ),
    ),
)
def test_visual_contracts_validate_against_write_and_figure_boundaries(
    schema: str,
    generator: str,
    producer,
) -> None:
    payload = producer()
    validate_report(schema, payload)
    check_figure_contract(generator, schema, payload)


@pytest.mark.parametrize(
    "producer",
    (
        application_integrity_flow_contract,
        evidence_replication_map_contract,
        source_render_provenance_contract,
    ),
)
def test_visual_contract_producers_return_defensive_copies(producer) -> None:
    first = producer()
    second = producer()
    first["schema_version"] = "tampered"
    assert second["schema_version"] == "1.0"


def test_application_contract_rejects_solver_status_semantic_drift() -> None:
    payload = application_integrity_flow_contract()
    payload["solver_statuses"][0]["fallbacks"] = True
    with pytest.raises(ReportSchemaError, match="solver state 'nominal'"):
        validate_report("application_integrity_flow", payload)


def test_application_contract_rejects_receipt_claim_promotion() -> None:
    payload = application_integrity_flow_contract()
    payload["no_claims"].remove("scientific validity")
    with pytest.raises(ReportSchemaError, match="complete receipt no-claim boundary"):
        validate_report("application_integrity_flow", payload)


def test_application_contract_exposes_the_exact_node_edge_and_write_inventories() -> None:
    payload = application_integrity_flow_contract()

    observed_nodes = tuple(
        (str(panel["id"]), tuple(str(node["id"]) for node in panel["nodes"]))
        for panel in payload["panels"]
    )
    observed_edges = tuple(
        (
            str(panel["id"]),
            str(edge["source"]),
            str(edge["target"]),
            str(edge["disposition"]),
        )
        for panel in payload["panels"]
        for edge in panel["edges"]
    )
    receipt = next(panel for panel in payload["panels"] if panel["id"] == "receipt")

    assert observed_nodes == APPLICATION_PANEL_NODE_INVENTORY
    assert observed_edges == APPLICATION_FLOW_EDGE_INVENTORY
    assert tuple(receipt["write_order"]) == APPLICATION_ARTIFACT_WRITE_ORDER


def test_application_contract_keeps_request_and_destination_failures_distinct() -> None:
    payload = application_integrity_flow_contract()
    validation = next(panel for panel in payload["panels"] if panel["id"] == "validation")
    node_ids = {node["id"] for node in validation["nodes"]}
    rejection_pairs = {
        (edge["source"], edge["target"])
        for edge in validation["edges"]
        if edge["disposition"] == "rejected"
    }

    assert {"invalid_request", "unsafe_destination", "destination_safety"} <= node_ids
    assert rejection_pairs == {
        ("strict_validation", "invalid_request"),
        ("destination_safety", "unsafe_destination"),
    }


def test_application_contract_rejects_node_omission() -> None:
    payload = application_integrity_flow_contract()
    payload["panels"][0]["nodes"].pop()

    with pytest.raises(ReportSchemaError, match="node inventory or order"):
        validate_report("application_integrity_flow", payload)


def test_application_contract_rejects_edge_omission() -> None:
    payload = application_integrity_flow_contract()
    payload["panels"][2]["edges"].pop()

    with pytest.raises(ReportSchemaError, match="edge inventory"):
        validate_report("application_integrity_flow", payload)


def test_application_contract_rejects_valid_node_endpoint_substitution() -> None:
    payload = application_integrity_flow_contract()
    payload["panels"][0]["edges"][0]["target"] = "rich_result"

    with pytest.raises(ReportSchemaError, match="edge inventory"):
        validate_report("application_integrity_flow", payload)


def test_application_contract_rejects_cross_panel_edge_reownership() -> None:
    payload = application_integrity_flow_contract()
    moved = payload["panels"][0]["edges"].pop()
    payload["panels"][1]["edges"].insert(0, moved)

    with pytest.raises(ReportSchemaError, match="panel ownership"):
        validate_report("application_integrity_flow", payload)


def test_application_contract_rejects_duplicate_pair_under_new_label() -> None:
    payload = application_integrity_flow_contract()
    duplicate = dict(payload["panels"][0]["edges"][0])
    duplicate["label"] = "same route under another label"
    payload["panels"][0]["edges"][1] = duplicate

    with pytest.raises(ReportSchemaError, match="duplicate edges"):
        validate_report("application_integrity_flow", payload)


def test_application_contract_rejects_write_order_drift() -> None:
    payload = application_integrity_flow_contract()
    payload["panels"][2]["write_order"] = [
        "result_json",
        "request_json",
        "receipt_json",
    ]

    with pytest.raises(ReportSchemaError, match="artifact write order"):
        validate_report("application_integrity_flow", payload)


@pytest.mark.parametrize(
    ("lane_id", "expected_class", "wrong_valid_class"),
    (
        ("project_identity", "formal_executable", "source_conditional"),
        ("client_loss", "source_conditional", "conditional_empirical"),
        ("belief_sharing_free_energy", "conditional_empirical", "formal_executable"),
        ("likelihood_learning", "conditional_empirical", "source_conditional"),
        ("bmr_sign_control", "formal_executable", "conditional_empirical"),
        ("client_loss_baseline", "conditional_empirical", "source_conditional"),
        ("server_heuristic", "conditional_empirical", "formal_executable"),
        ("variational_server", "formal_executable", "conditional_empirical"),
        ("moving_disjoint_fov", "conditional_empirical", "scoped_implementation"),
        ("hierarchy_sensitivity", "conditional_empirical", "formal_executable"),
        ("parameter_recovery", "conditional_empirical", "formal_executable"),
        ("protocol_integrity", "scoped_implementation", "formal_executable"),
        ("application_provenance", "scoped_implementation", "conditional_empirical"),
        ("external_confirmation", "open", "conditional_empirical"),
    ),
)
def test_evidence_contract_pins_each_lane_to_its_exact_valid_class(
    lane_id: str,
    expected_class: str,
    wrong_valid_class: str,
) -> None:
    payload = evidence_replication_map_contract()
    lane = next(item for item in payload["lanes"] if item["id"] == lane_id)
    assert lane["evidence_class"] == expected_class
    lane["evidence_class"] = wrong_valid_class

    with pytest.raises(ReportSchemaError, match="must retain evidence class"):
        validate_report("evidence_replication_map", payload)


def test_evidence_contract_rejects_duplicate_lane_ids() -> None:
    payload = evidence_replication_map_contract()
    payload["lanes"][1]["id"] = payload["lanes"][0]["id"]
    with pytest.raises(ReportSchemaError, match="duplicate ids"):
        validate_report("evidence_replication_map", payload)


def test_bnn_evidence_lane_owns_only_the_composite_configuration_contrast() -> None:
    payload = evidence_replication_map_contract()
    lane = next(item for item in payload["lanes"] if item["id"] == "client_loss_baseline")

    assert "joint NLL/L2=0.05 and RCCE/L2=0.10 configurations" in lane["estimand"]
    assert "RCCE-only effect" in lane["prohibited_generalization"]
    assert "composite configuration contrast" in lane["display"]["permitted"]


def test_evidence_contract_covers_headline_result_families_and_public_boundaries() -> None:
    payload = evidence_replication_map_contract()

    assert "headline-result and claim-owner map" in payload["source_relation"]
    assert len(payload["lanes"]) == 14
    assert {lane["id"] for lane in payload["lanes"]} >= {
        "belief_sharing_free_energy",
        "likelihood_learning",
        "bmr_sign_control",
        "moving_disjoint_fov",
        "hierarchy_sensitivity",
        "parameter_recovery",
    }


def test_source_render_contract_rejects_automatic_publication_terminal() -> None:
    payload = source_render_provenance_contract()
    payload["terminals"][0]["authorization_required"] = False
    with pytest.raises(ReportSchemaError, match="must require authorization"):
        validate_report("source_render_provenance", payload)


def test_source_render_contract_rejects_manifest_dependency_cycle() -> None:
    payload = source_render_provenance_contract()
    payload["manifest_cycle_boundary"] = "the figure reads the final manifest"
    with pytest.raises(ReportSchemaError, match="generated-manifest dependency cycle"):
        validate_report("source_render_provenance", payload)


def test_source_render_contract_rejects_unknown_invalidation_target() -> None:
    payload = source_render_provenance_contract()
    mutated = deepcopy(payload)
    mutated["invalidation_edges"][0]["target"] = "untracked_output"
    with pytest.raises(ReportSchemaError, match="unknown node"):
        validate_report("source_render_provenance", mutated)


def test_source_render_contract_rejects_non_immediate_invalidation_target() -> None:
    payload = source_render_provenance_contract()
    payload["invalidation_edges"][2]["target"] = "figures"

    with pytest.raises(ReportSchemaError, match="invalidation edge inventory"):
        validate_report("source_render_provenance", payload)


@pytest.mark.parametrize("field", ("producer_edges", "invalidation_edges"))
def test_source_render_contract_rejects_edge_omission(field: str) -> None:
    payload = source_render_provenance_contract()
    payload[field].pop()

    with pytest.raises(ReportSchemaError, match=rf"{field.removesuffix('_edges')} edge inventory"):
        validate_report("source_render_provenance", payload)


@pytest.mark.parametrize("field", ("producer_edges", "invalidation_edges"))
def test_source_render_contract_rejects_endpoint_substitution(field: str) -> None:
    payload = source_render_provenance_contract()
    payload[field][0]["target"] = "surfaces"

    with pytest.raises(ReportSchemaError, match=rf"{field.removesuffix('_edges')} edge inventory"):
        validate_report("source_render_provenance", payload)


@pytest.mark.parametrize("field", ("producer_edges", "invalidation_edges"))
def test_source_render_contract_rejects_duplicate_pair_under_new_label(field: str) -> None:
    payload = source_render_provenance_contract()
    duplicate = dict(payload[field][0])
    duplicate["label"] = "same dependency under a different label"
    payload[field][1] = duplicate

    with pytest.raises(ReportSchemaError, match="contains duplicate edges"):
        validate_report("source_render_provenance", payload)


def test_source_render_contract_exposes_exact_inputs_stages_and_branching_dag() -> None:
    payload = source_render_provenance_contract()

    assert tuple(item["id"] for item in payload["inputs"]) == SOURCE_RENDER_INPUT_INVENTORY
    assert tuple(item["id"] for item in payload["stages"]) == SOURCE_RENDER_STAGE_INVENTORY
    assert tuple(
        (edge["source"], edge["target"], edge["disposition"])
        for edge in payload["producer_edges"]
    ) == SOURCE_RENDER_PRODUCER_EDGE_INVENTORY
    assert tuple(
        (edge["source"], edge["target"], edge["disposition"])
        for edge in payload["invalidation_edges"]
    ) == SOURCE_RENDER_INVALIDATION_EDGE_INVENTORY

    source_targets = {
        edge["target"]
        for edge in payload["producer_edges"]
        if edge["source"] == "source"
    }
    assert source_targets == {
        "reports",
        "figures",
        "provisional",
        "coverage",
        "final_hydration",
    }
    assert any(
        edge["source"] == "template_renderer" and edge["target"] == "render"
        for edge in payload["producer_edges"]
    )
