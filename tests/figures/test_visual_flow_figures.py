"""Rendering tests for the source-owned application and provenance maps."""

from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import pytest

from analysis.visual_contracts import (
    APPLICATION_FLOW_EDGE_INVENTORY,
    SOURCE_RENDER_PRODUCER_EDGE_INVENTORY,
    application_integrity_flow_contract,
    source_render_provenance_contract,
)
from figures.application_integrity_flow import (
    _APPLICATION_PANEL_A_LAYOUT,
    _APPLICATION_ROUTE_GEOMETRY,
    _application_edge_inventory,
    _solver_status_key,
    _wrap_label,
    generate_application_integrity_flow,
)
from figures.source_render_provenance import (
    _DISPLAY_LABELS,
    _MATRIX_COLUMN_INVENTORY,
    _MATRIX_ROW_INVENTORY,
    _MATRIX_ROW_LABELS,
    _PROCESS_DAG_EDGE_PAIRS,
    _derive_immediate_invalidation_routes,
    _representation_inventory,
    _reverse_invalidation_arrow_endpoints,
    _split_producer_representations,
    generate_source_render_provenance,
)


@pytest.mark.parametrize(
    ("producer", "generator", "is_landscape"),
    (
        (application_integrity_flow_contract, generate_application_integrity_flow, False),
        (source_render_provenance_contract, generate_source_render_provenance, False),
    ),
)
def test_visual_flow_generators_write_deterministic_pairs(
    tmp_path: Path,
    producer,
    generator,
    is_landscape: bool,
) -> None:
    report = producer()
    first = generator(report, project_root=tmp_path)
    first_png = first.read_bytes()
    first_pdf = first.with_suffix(".pdf").read_bytes()

    second = generator(report, project_root=tmp_path)
    pixels = mpimg.imread(second)

    assert second.read_bytes() == first_png
    assert second.with_suffix(".pdf").read_bytes() == first_pdf
    assert (pixels.shape[1] > pixels.shape[0]) is is_landscape


def test_application_flow_rejects_reordered_panels(tmp_path: Path) -> None:
    report = application_integrity_flow_contract()
    report["panels"][0], report["panels"][1] = report["panels"][1], report["panels"][0]

    with pytest.raises(ValueError, match="must preserve"):
        generate_application_integrity_flow(report, project_root=tmp_path)


def test_application_flow_preserves_exact_solver_codes_and_intentional_lines() -> None:
    report = application_integrity_flow_contract()
    key = _solver_status_key(report["solver_statuses"])

    assert [row[3] for row in key] == [
        "nominal",
        "converged_with_fallback",
        "not_converged",
        "not_converged_with_fallback",
    ]
    assert _wrap_label("Labeled JSON or\nPython request", width=80) == ("Labeled JSON or\nPython request")

    report["solver_statuses"][0]["label"] = "healthy"
    with pytest.raises(ValueError, match="exact public codes"):
        _solver_status_key(report["solver_statuses"])


def test_application_flow_geometry_is_owned_by_every_exact_typed_edge() -> None:
    report = application_integrity_flow_contract()
    observed = _application_edge_inventory(report["panels"])

    assert observed == APPLICATION_FLOW_EDGE_INVENTORY
    assert set(_APPLICATION_ROUTE_GEOMETRY) == {
        (source, target) for _, source, target, _ in APPLICATION_FLOW_EDGE_INVENTORY
    }


def test_application_flow_failure_routes_use_clear_dedicated_lanes() -> None:
    destination = _APPLICATION_PANEL_A_LAYOUT["destination_safety"]
    aggregate = _APPLICATION_PANEL_A_LAYOUT["aggregate_result"]
    invalid_route = _APPLICATION_ROUTE_GEOMETRY[("strict_validation", "invalid_request")]
    unsafe_route = _APPLICATION_ROUTE_GEOMETRY[("destination_safety", "unsafe_destination")]

    invalid_xs = (invalid_route[1][0], invalid_route[2][0])
    assert destination[0] + destination[2] < min(invalid_xs)
    assert max(invalid_xs) < aggregate[0]
    assert invalid_route[4] is False

    unsafe_xs = (unsafe_route[1][0], unsafe_route[2][0])
    assert destination[0] < min(unsafe_xs) <= max(unsafe_xs) < destination[0] + destination[2]
    assert unsafe_route[4] is False


def test_application_flow_rejects_dependency_omission(tmp_path: Path) -> None:
    report = application_integrity_flow_contract()
    report["panels"][2]["edges"].pop()

    with pytest.raises(ValueError, match="dependency edge inventory"):
        generate_application_integrity_flow(report, project_root=tmp_path)


def test_application_flow_rejects_valid_endpoint_substitution(tmp_path: Path) -> None:
    report = application_integrity_flow_contract()
    report["panels"][0]["edges"][0]["target"] = "rich_result"

    with pytest.raises(ValueError, match="dependency edge inventory"):
        generate_application_integrity_flow(report, project_root=tmp_path)


def test_application_flow_rejects_cross_panel_edge_reownership(tmp_path: Path) -> None:
    report = application_integrity_flow_contract()
    moved = report["panels"][0]["edges"].pop()
    report["panels"][1]["edges"].insert(0, moved)

    with pytest.raises(ValueError, match="dependency edge inventory"):
        generate_application_integrity_flow(report, project_root=tmp_path)


def test_application_flow_rejects_write_order_drift(tmp_path: Path) -> None:
    report = application_integrity_flow_contract()
    report["panels"][2]["write_order"] = [
        "result_json",
        "request_json",
        "receipt_json",
    ]

    with pytest.raises(ValueError, match="write order"):
        generate_application_integrity_flow(report, project_root=tmp_path)


def test_source_render_flow_rejects_unknown_edge_node(tmp_path: Path) -> None:
    report = source_render_provenance_contract()
    report["producer_edges"][0]["target"] = "not_a_registered_node"

    with pytest.raises(ValueError, match="unknown node"):
        generate_source_render_provenance(report, project_root=tmp_path)


def test_source_render_matrix_and_process_dag_cover_every_typed_edge_once() -> None:
    report = source_render_provenance_contract()
    matrix_edges, process_edges = _split_producer_representations(report["producer_edges"])
    matrix_inventory = {(edge["source"], edge["target"], edge["disposition"]) for edge in matrix_edges}
    process_inventory = {(edge["source"], edge["target"], edge["disposition"]) for edge in process_edges}

    assert matrix_inventory.isdisjoint(process_inventory)
    assert _representation_inventory(matrix_edges, process_edges) == set(
        SOURCE_RENDER_PRODUCER_EDGE_INVENTORY
    )
    assert {(edge["source"], edge["target"]) for edge in process_edges} == set(_PROCESS_DAG_EDGE_PAIRS)
    assert len(matrix_edges) == 18
    assert len(process_edges) == 9


def test_source_render_dependency_matrix_names_all_five_exact_inputs() -> None:
    assert _MATRIX_ROW_INVENTORY[:5] == (
        "source",
        "config",
        "lockfile",
        "registry",
        "template_renderer",
    )
    assert set(_MATRIX_COLUMN_INVENTORY) == {
        "reports",
        "figures",
        "provisional",
        "coverage",
        "final_hydration",
        "render",
        "release_manifest",
    }
    assert _MATRIX_ROW_LABELS["template_renderer"] == "I5  Exact clean Template commit"
    assert _DISPLAY_LABELS["github"].startswith("A1")
    assert _DISPLAY_LABELS["zenodo"].startswith("A2")


def test_source_render_invalidation_arrows_point_back_to_changed_owner() -> None:
    start, end = _reverse_invalidation_arrow_endpoints(0.025, 0.304)

    assert start[0] > end[0], "reverse dependency must point right-to-left"
    assert start[1] == end[1]


def test_source_render_rejects_substituted_producer_route(tmp_path: Path) -> None:
    report = source_render_provenance_contract()
    report["producer_edges"][0]["target"] = "surfaces"
    report["invalidation_edges"][0]["target"] = "surfaces"

    with pytest.raises(ValueError, match="producer edge inventory"):
        generate_source_render_provenance(report, project_root=tmp_path)


def test_source_render_rejects_invalidation_shortcut(tmp_path: Path) -> None:
    report = source_render_provenance_contract()
    report["invalidation_edges"][2]["target"] = "figures"

    with pytest.raises(ValueError, match="invalidation edge inventory"):
        generate_source_render_provenance(report, project_root=tmp_path)


@pytest.mark.parametrize("field", ("producer_edges", "invalidation_edges"))
def test_source_render_rejects_omitted_edge(field: str, tmp_path: Path) -> None:
    report = source_render_provenance_contract()
    report[field].pop()

    with pytest.raises(ValueError, match=rf"{field.removesuffix('_edges')} edge inventory"):
        generate_source_render_provenance(report, project_root=tmp_path)


@pytest.mark.parametrize("field", ("producer_edges", "invalidation_edges"))
def test_source_render_rejects_duplicate_pair_under_new_label(
    field: str,
    tmp_path: Path,
) -> None:
    report = source_render_provenance_contract()
    duplicate = dict(report[field][0])
    duplicate["label"] = "duplicate under another label"
    report[field][1] = duplicate

    with pytest.raises(ValueError, match="duplicate endpoint pairs"):
        generate_source_render_provenance(report, project_root=tmp_path)


def test_source_render_derives_all_immediate_invalidation_routes() -> None:
    report = source_render_provenance_contract()
    routes = _derive_immediate_invalidation_routes(
        report["producer_edges"],
        report["invalidation_edges"],
    )

    expected: dict[str, list[str]] = {}
    for edge in report["invalidation_edges"]:
        expected.setdefault(edge["source"], []).append(edge["target"])

    assert routes == [(source, tuple(targets)) for source, targets in expected.items()]


def test_source_render_rejects_a_non_immediate_route_before_drawing() -> None:
    report = source_render_provenance_contract()
    report["invalidation_edges"][2]["target"] = "figures"

    with pytest.raises(ValueError, match="immediate producer dependency"):
        _derive_immediate_invalidation_routes(
            report["producer_edges"],
            report["invalidation_edges"],
        )
