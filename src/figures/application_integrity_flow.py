"""Application integrity, solver-health, and receipt-verification flow."""

from __future__ import annotations

import textwrap
from collections.abc import Mapping
from pathlib import Path

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from analysis.visual_contracts import (
    APPLICATION_ARTIFACT_WRITE_ORDER,
    APPLICATION_FLOW_EDGE_INVENTORY,
    APPLICATION_PANEL_NODE_INVENTORY,
)

from ._common import (
    COLOR_ARROW,
    COLOR_DEEP,
    COLOR_EDGE_PANEL,
    COLOR_NAIVE,
    COLOR_PANEL_BG,
    COLOR_PANEL_FAIL,
    COLOR_PANEL_GOOD,
    COLOR_PANEL_NOTE,
    COLOR_ROBUST,
    COLOR_VARIATE,
    COLOR_WHITE,
    apply_style,
    figures_dir,
    plt,
    save_figure_pair,
)

_NODE_DISPLAY_LABELS = {
    "request": "Labeled JSON or\nPython request",
    "strict_validation": "Strict schema + 1-D\ncategorical shape",
    "canonical_request": "Canonical request\n+ SHA-256",
    "destination_safety": "Destination + source\nprovenance safety",
    "aggregate_result": "aggregate_result\ncalled once",
    "invalid_request": "Exit 2 · invalid request\nno directory created",
    "unsafe_destination": "Exit 2 · unsafe destination\nor provenance unavailable",
    "rich_result": "From A\nConsensus, weights,\nand histories",
    "solver_status": "Classify convergence\n× fallback",
    "non_nominal": "Exit 1 · retain valid\nnon-nominal artifacts",
    "request_json": "From A\n1  request.json",
    "result_json": "From B\n2  result.json",
    "receipt_json": "3  receipt.json",
    "verification": "Verify receipt",
    "mismatch": "Mismatch · report exact finding; do not reinterpret",
}

_STATUS_KEY_LAYOUT = (
    (True, False, "converged · no fallback", "nominal"),
    (True, True, "converged · fallback", "converged_with_fallback"),
    (False, False, "not converged · no fallback", "not_converged"),
    (False, True, "not converged · fallback", "not_converged_with_fallback"),
)

_APPLICATION_PANEL_A_LAYOUT = {
    "request": (0.04, 0.63, 0.23, 0.19),
    "strict_validation": (0.34, 0.63, 0.26, 0.19),
    "canonical_request": (0.67, 0.63, 0.24, 0.19),
    "destination_safety": (0.04, 0.32, 0.28, 0.19),
    "aggregate_result": (0.62, 0.32, 0.28, 0.19),
    "unsafe_destination": (0.03, 0.06, 0.31, 0.14),
    "invalid_request": (0.37, 0.06, 0.30, 0.14),
}

# Geometry is keyed by typed dependency endpoints. The renderer iterates report
# edges and rejects any route without an exact geometry entry; it never draws a
# semantically independent arrow. Cross-panel dependencies enter at the top of
# the receiving panel with their source panel named directly.
_APPLICATION_ROUTE_GEOMETRY: dict[
    tuple[str, str], tuple[int, tuple[float, float], tuple[float, float], float, bool]
] = {
    ("request", "strict_validation"): (0, (0.27, 0.725), (0.34, 0.725), 0.0, False),
    ("strict_validation", "canonical_request"): (0, (0.60, 0.725), (0.67, 0.725), 0.0, False),
    ("strict_validation", "invalid_request"): (0, (0.47, 0.63), (0.52, 0.20), 0.0, False),
    ("canonical_request", "destination_safety"): (0, (0.79, 0.63), (0.18, 0.51), -0.12, False),
    ("destination_safety", "unsafe_destination"): (
        0,
        (0.18, 0.32),
        (0.185, 0.20),
        0.0,
        False,
    ),
    ("destination_safety", "aggregate_result"): (0, (0.32, 0.415), (0.62, 0.415), 0.0, False),
    ("aggregate_result", "rich_result"): (1, (0.18, 0.91), (0.18, 0.84), 0.0, False),
    ("rich_result", "solver_status"): (1, (0.325, 0.75), (0.395, 0.75), 0.0, False),
    ("solver_status", "non_nominal"): (1, (0.665, 0.75), (0.735, 0.75), 0.0, True),
    ("canonical_request", "request_json"): (2, (0.095, 0.89), (0.095, 0.75), 0.0, False),
    ("rich_result", "result_json"): (2, (0.275, 0.89), (0.275, 0.75), 0.0, False),
    ("request_json", "receipt_json"): (2, (0.095, 0.57), (0.455, 0.57), -0.28, False),
    ("result_json", "receipt_json"): (2, (0.275, 0.57), (0.455, 0.57), -0.22, False),
    ("receipt_json", "verification"): (2, (0.525, 0.66), (0.615, 0.66), 0.0, False),
    ("verification", "mismatch"): (2, (0.945, 0.57), (0.945, 0.20), 0.0, True),
}


def _records(report: Mapping[str, object], key: str) -> list[Mapping[str, object]]:
    value = report.get(key)
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"{key} must be a list of mappings")
    return list(value)


def _label(record: Mapping[str, object]) -> str:
    value = record.get("label")
    if not isinstance(value, str) or not value:
        raise ValueError("flow records must contain a non-empty label")
    return value


def _id_map(records: list[Mapping[str, object]]) -> dict[str, Mapping[str, object]]:
    mapped: dict[str, Mapping[str, object]] = {}
    for record in records:
        identifier = record.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in mapped:
            raise ValueError("flow records must contain unique non-empty ids")
        mapped[identifier] = record
    return mapped


def _solver_status_key(
    statuses: list[Mapping[str, object]],
) -> list[tuple[bool, bool, str, str]]:
    """Return the four exact solver codes in the fixed visual key order."""
    by_health: dict[tuple[bool, bool], str] = {}
    for status in statuses:
        converged = status.get("converged")
        fallbacks = status.get("fallbacks")
        if not isinstance(converged, bool) or not isinstance(fallbacks, bool):
            raise ValueError("solver statuses must contain Boolean health flags")
        key = (converged, fallbacks)
        if key in by_health:
            raise ValueError("solver statuses must contain unique health combinations")
        by_health[key] = _label(status)
    required = {(converged, fallbacks) for converged, fallbacks, _, _ in _STATUS_KEY_LAYOUT}
    if set(by_health) != required:
        raise ValueError("solver statuses must contain all four health combinations")
    result: list[tuple[bool, bool, str, str]] = []
    for converged, fallbacks, condition, expected_code in _STATUS_KEY_LAYOUT:
        if by_health[(converged, fallbacks)] != expected_code:
            raise ValueError("solver status labels must preserve the exact public codes")
        result.append((converged, fallbacks, condition, expected_code))
    return result


def _wrap_label(text: str, *, width: int) -> str:
    """Wrap each intentional line independently instead of erasing its break."""
    return "\n".join(textwrap.fill(line, width=width) for line in text.splitlines())


def _box(
    ax: "plt.Axes",
    *,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    role: str,
    wrap: int = 25,
) -> None:
    fills = {
        "input": COLOR_WHITE,
        "gate": COLOR_PANEL_NOTE,
        "artifact": COLOR_PANEL_BG,
        "domain": COLOR_PANEL_GOOD,
        "result": COLOR_PANEL_GOOD,
        "health": COLOR_PANEL_NOTE,
        "receipt": COLOR_PANEL_BG,
        "verification": COLOR_PANEL_NOTE,
        "rejected": COLOR_PANEL_FAIL,
        "retained_warning": COLOR_PANEL_NOTE,
    }
    edges = {
        "gate": COLOR_ROBUST,
        "domain": COLOR_VARIATE,
        "result": COLOR_VARIATE,
        "health": COLOR_ROBUST,
        "verification": COLOR_ROBUST,
        "rejected": COLOR_NAIVE,
        "retained_warning": COLOR_ARROW,
    }
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=fills.get(role, COLOR_WHITE),
        edgecolor=edges.get(role, COLOR_EDGE_PANEL),
        linewidth=1.8 if role in edges else 1.2,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        _wrap_label(text, width=wrap),
        ha="center",
        va="center",
        fontsize=9.2,
        color=COLOR_DEEP,
        linespacing=1.18,
        zorder=3,
    )


def _arrow(
    ax: "plt.Axes",
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    label: str = "",
    disposition: str = "normal",
    curvature: float = 0.0,
) -> None:
    color = (
        COLOR_NAIVE
        if disposition == "rejected"
        else COLOR_VARIATE
        if disposition == "data_dependency"
        else COLOR_DEEP
        if disposition == "write_order"
        else COLOR_ARROW
    )
    linestyle = (
        ":"
        if disposition == "write_order"
        else "--"
        if disposition in {"rejected", "retained_warning"}
        else "-"
    )
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=1.35,
        linestyle=linestyle,
        color=color,
        connectionstyle=f"arc3,rad={curvature}",
        zorder=1,
    )
    ax.add_patch(arrow)
    if label:
        ax.text(
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2 + 0.035,
            textwrap.fill(label, width=19),
            ha="center",
            va="bottom",
            fontsize=8.7,
            # Line color carries the route disposition; annotation text stays
            # dark enough to meet the normal-text contrast floor.
            color=COLOR_DEEP,
            bbox={"boxstyle": "round,pad=0.12", "facecolor": COLOR_WHITE, "edgecolor": "none"},
            zorder=4,
        )


def _application_edge_inventory(
    panels: list[Mapping[str, object]],
) -> tuple[tuple[str, str, str, str], ...]:
    """Return the exact panel-owned dependency inventory from a report."""

    inventory: list[tuple[str, str, str, str]] = []
    for panel in panels:
        panel_id = panel.get("id")
        if not isinstance(panel_id, str):
            raise ValueError("application panels require string ids")
        for edge in _records(panel, "edges"):
            inventory.append(
                (
                    panel_id,
                    str(edge.get("source")),
                    str(edge.get("target")),
                    str(edge.get("disposition")),
                )
            )
    return tuple(inventory)


def _draw_application_edges(
    axes: list["plt.Axes"],
    panels: list[Mapping[str, object]],
) -> None:
    """Draw every and only the exact typed application dependency edges."""

    if _application_edge_inventory(panels) != APPLICATION_FLOW_EDGE_INVENTORY:
        raise ValueError("application dependency edge inventory is invalid")
    edge_records = [edge for panel in panels for edge in _records(panel, "edges")]
    endpoint_pairs = {(str(edge["source"]), str(edge["target"])) for edge in edge_records}
    if endpoint_pairs != set(_APPLICATION_ROUTE_GEOMETRY):
        raise ValueError("application dependency geometry does not match the typed edge inventory")
    for edge in edge_records:
        pair = (str(edge["source"]), str(edge["target"]))
        panel_index, start, end, curvature, show_label = _APPLICATION_ROUTE_GEOMETRY[pair]
        label = str(edge["label"]) if show_label else ""
        _arrow(
            axes[panel_index],
            start,
            end,
            label=label,
            disposition=str(edge["disposition"]),
            curvature=curvature,
        )


def _panel_setup(ax: "plt.Axes", title: str) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.006, 0.02),
            0.988,
            0.91,
            boxstyle="round,pad=0.008,rounding_size=0.015",
            facecolor=COLOR_WHITE,
            edgecolor=COLOR_EDGE_PANEL,
            linewidth=1.0,
            zorder=0,
        )
    )
    ax.text(0.025, 0.955, title, ha="left", va="top", fontsize=12.3, fontweight="bold", color=COLOR_DEEP)


def generate_application_integrity_flow(
    report: Mapping[str, object],
    *,
    project_root: Path | None = None,
    filename: str = "application_integrity_flow.png",
) -> Path:
    """Render the source-owned labeled-application and receipt boundary map."""
    panels = _records(report, "panels")
    if [panel.get("id") for panel in panels] != ["validation", "solver_health", "receipt"]:
        raise ValueError("panels must preserve validation, solver_health, receipt order")
    expected_nodes = dict(APPLICATION_PANEL_NODE_INVENTORY)
    for panel in panels:
        panel_id = str(panel["id"])
        observed_nodes = tuple(_id_map(_records(panel, "nodes")))
        if observed_nodes != expected_nodes[panel_id]:
            raise ValueError(f"application panel {panel_id!r} node inventory is invalid")
    statuses = _records(report, "solver_statuses")
    levels = _records(report, "verification_levels")
    no_claims = report.get("no_claims")
    if not isinstance(no_claims, list) or any(not isinstance(item, str) for item in no_claims):
        raise ValueError("no_claims must be a list of strings")
    status_key = _solver_status_key(statuses)

    apply_style()
    fig = plt.figure(figsize=(7.6, 10.2), facecolor=COLOR_WHITE)
    # This portrait schematic owns its panel spacing explicitly. Long
    # methodological qualifications belong to the self-contained caption;
    # the image keeps only execution states and the exact solver-status key.
    fig.set_layout_engine("none")
    grid = fig.add_gridspec(3, 1, height_ratios=(1.12, 1.18, 1.0), hspace=0.13)
    axes = [fig.add_subplot(grid[index, 0]) for index in range(3)]
    fig.suptitle(
        "Application integrity and numerical health remain separate evidence layers",
        x=0.5,
        y=0.995,
        ha="center",
        va="top",
        fontsize=15.2,
        fontweight="bold",
        color=COLOR_DEEP,
    )

    # Panel A: validated request to the single domain delegation.
    panel = panels[0]
    _panel_setup(axes[0], _label(panel))
    nodes = _id_map(_records(panel, "nodes"))
    for identifier, (x, y, width, height) in _APPLICATION_PANEL_A_LAYOUT.items():
        record = nodes[identifier]
        _box(
            axes[0],
            xy=(x, y),
            width=width,
            height=height,
            text=_NODE_DISPLAY_LABELS[identifier],
            role=str(record.get("role", "artifact")),
            wrap=28,
        )
    # Panel B: rich result, retained warning exit, and a roomy exact-code key.
    panel = panels[1]
    _panel_setup(axes[1], _label(panel))
    nodes = _id_map(_records(panel, "nodes"))
    top_boxes = {
        "rich_result": (0.035, 0.66, 0.29),
        "solver_status": (0.395, 0.66, 0.27),
        "non_nominal": (0.735, 0.66, 0.23),
    }
    for identifier, (x, y, width) in top_boxes.items():
        record = nodes[identifier]
        _box(
            axes[1],
            xy=(x, y),
            width=width,
            height=0.18,
            text=_NODE_DISPLAY_LABELS[identifier],
            role=str(record.get("role", "result")),
            wrap=30,
        )
    axes[1].text(
        0.035,
        0.58,
        "Exact solver_status key",
        ha="left",
        va="center",
        fontsize=9.2,
        fontweight="bold",
        color=COLOR_DEEP,
    )
    chip_positions = ((0.035, 0.35), (0.515, 0.35), (0.035, 0.12), (0.515, 0.12))
    for (converged, fallbacks, condition, status_code), (x, y) in zip(
        status_key,
        chip_positions,
        strict=True,
    ):
        fill = COLOR_PANEL_GOOD if converged and not fallbacks else COLOR_PANEL_NOTE
        edge = COLOR_VARIATE if converged and not fallbacks else COLOR_ARROW
        axes[1].add_patch(
            FancyBboxPatch(
                (x, y),
                0.45,
                0.17,
                boxstyle="round,pad=0.008,rounding_size=0.012",
                facecolor=fill,
                edgecolor=edge,
                linewidth=1.2,
            )
        )
        axes[1].text(
            x + 0.225,
            y + 0.118,
            condition,
            ha="center",
            va="center",
            fontsize=8.7,
            color=COLOR_DEEP,
        )
        axes[1].text(
            x + 0.225,
            y + 0.055,
            status_code,
            ha="center",
            va="center",
            fontsize=8.7,
            fontfamily="monospace",
            color=COLOR_DEEP,
        )

    # Panel C: ordered artifacts plus a dedicated verification-level lane.
    panel = panels[2]
    _panel_setup(axes[2], _label(panel))
    nodes = _id_map(_records(panel, "nodes"))
    write_order = panel.get("write_order")
    if not isinstance(write_order, list) or tuple(write_order) != APPLICATION_ARTIFACT_WRITE_ORDER:
        raise ValueError("receipt panel artifact write order is invalid")
    ordered = (
        ("request_json", 0.025),
        ("result_json", 0.205),
        ("receipt_json", 0.385),
        ("verification", 0.615),
    )
    widths = {
        "request_json": 0.14,
        "result_json": 0.14,
        "receipt_json": 0.14,
        "verification": 0.34,
    }
    for identifier, x in ordered:
        record = nodes[identifier]
        _box(
            axes[2],
            xy=(x, 0.57),
            width=widths[identifier],
            height=0.18,
            text=_NODE_DISPLAY_LABELS[identifier],
            role=str(record.get("role", "artifact")),
            wrap=25,
        )
    artifact_centers = {
        identifier: (x + widths[identifier] / 2.0, 0.79)
        for identifier, x in ordered
        if identifier in APPLICATION_ARTIFACT_WRITE_ORDER
    }
    for source, target in zip(write_order, write_order[1:], strict=False):
        start_x = artifact_centers[str(source)][0]
        end_x = artifact_centers[str(target)][0]
        _arrow(
            axes[2],
            (start_x, 0.79),
            (end_x, 0.79),
            disposition="write_order",
        )
    axes[2].text(
        0.50,
        0.805,
        "dotted = atomic write order · solid = data dependency",
        ha="center",
        va="bottom",
        fontsize=8.7,
        color=COLOR_DEEP,
    )
    level_positions = (0.025, 0.32, 0.615)
    for index, (x, level) in enumerate(zip(level_positions, levels, strict=True), start=1):
        _box(
            axes[2],
            xy=(x, 0.29),
            width=0.25,
            height=0.15,
            text=f"{index}  {_label(level)}",
            role="verification",
            wrap=24,
        )
    _box(
        axes[2],
        xy=(0.615, 0.07),
        width=0.34,
        height=0.13,
        text=_NODE_DISPLAY_LABELS["mismatch"],
        role="rejected",
        wrap=38,
    )
    _draw_application_edges(axes, panels)
    fig.subplots_adjust(left=0.025, right=0.975, top=0.965, bottom=0.025)
    canonical = save_figure_pair(fig, figures_dir(project_root) / filename)
    from ._presentation_flows import generate_application_presentation

    generate_application_presentation(report, canonical)
    return canonical


__all__ = ["generate_application_integrity_flow"]
