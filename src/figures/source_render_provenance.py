"""Source-to-render producer order, invalidation, and publication gates."""

from __future__ import annotations

import textwrap
from collections.abc import Mapping
from pathlib import Path

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from analysis.visual_contracts import (
    SOURCE_RENDER_INPUT_INVENTORY,
    SOURCE_RENDER_INVALIDATION_EDGE_INVENTORY,
    SOURCE_RENDER_PRODUCER_EDGE_INVENTORY,
    SOURCE_RENDER_STAGE_INVENTORY,
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
    COLOR_PURPLE,
    COLOR_VARIATE,
    COLOR_WHITE,
    apply_style,
    figures_dir,
    plt,
    save_figure_pair,
)

_DISPLAY_LABELS = {
    "source": "I1  Source · tests · manuscript",
    "config": "I2  Configuration · claim contracts",
    "lockfile": "I3  Lockfile · runtime metadata",
    "registry": "I4  Figure · accessibility registry",
    "template_renderer": "I5  Exact clean Template commit",
    "reports": "P1\nTyped\nreports",
    "figures": "P2\nFigures\n+ a11y",
    "provisional": "P3\nProvisional\nhydration",
    "coverage": "G4\nTests\n+ coverage",
    "final_hydration": "P5\nFinal\nhydration",
    "render": "E6\nClean\nrender",
    "surfaces": "S7\nPDF · HTML\nslides",
    "surface_validation": "G8\nSurface\nQA",
    "provenance": "R9\nProvenance",
    "release_manifest": "R10\nManifest\n+ checksums",
    "github": "A1  GitHub tag · release",
    "zenodo": "A2  Zenodo publication",
}

_NODE_BOX_PAD = 0.012
_STAGE_LABEL_WRAP_WIDTH = 12

_MATRIX_ROW_INVENTORY = (
    "source",
    "config",
    "lockfile",
    "registry",
    "template_renderer",
    "reports",
    "figures",
    "surfaces",
)
_MATRIX_COLUMN_INVENTORY = (
    "reports",
    "figures",
    "provisional",
    "coverage",
    "final_hydration",
    "render",
    "release_manifest",
)
_PROCESS_DAG_EDGE_PAIRS = (
    ("provisional", "coverage"),
    ("coverage", "final_hydration"),
    ("final_hydration", "render"),
    ("render", "surfaces"),
    ("surfaces", "surface_validation"),
    ("surface_validation", "provenance"),
    ("provenance", "release_manifest"),
    ("release_manifest", "github"),
    ("github", "zenodo"),
)

_MATRIX_COLUMN_LABELS = {
    "reports": "P1",
    "figures": "P2",
    "provisional": "P3",
    "coverage": "G4",
    "final_hydration": "P5",
    "render": "E6",
    "release_manifest": "R10",
}

_MATRIX_ROW_LABELS = {
    "source": "I1  Source · tests · manuscript",
    "config": "I2  Configuration · claim contracts",
    "lockfile": "I3  Lockfile · runtime metadata",
    "registry": "I4  Figure · accessibility registry",
    "template_renderer": "I5  Exact clean Template commit",
    "reports": "P1  Typed reports",
    "figures": "P2  Figures + accessibility metadata",
    "surfaces": "S7  Rendered surfaces",
}

_INVALIDATION_CODES = {
    "source": "I1",
    "config": "I2",
    "lockfile": "I3",
    "registry": "I4",
    "template_renderer": "I5",
    "reports": "P1",
    "figures": "P2",
    "provisional": "P3",
    "coverage": "G4",
    "final_hydration": "P5",
    "render": "E6",
    "surfaces": "S7",
    "surface_validation": "G8",
    "provenance": "R9",
    "release_manifest": "R10",
}

_INVALIDATION_CHIP_WIDTH = 0.304
_INVALIDATION_CHIP_HEIGHT = 0.028


def _derive_immediate_invalidation_routes(
    producer_edges: list[Mapping[str, object]],
    invalidation_edges: list[Mapping[str, object]],
) -> list[tuple[str, tuple[str, ...]]]:
    """Group each changed owner with all asserted immediate stale targets.

    The typed report retains an explicit invalidation inventory so omissions
    remain reviewable. Every target is an assertion against ``producer_edges``;
    branching owners are represented once with all immediate downstream
    targets instead of being forced into a false single-target chain.
    """

    producer_pairs = {(str(edge["source"]), str(edge["target"])) for edge in producer_edges}
    grouped: dict[str, list[str]] = {}
    for index, edge in enumerate(invalidation_edges):
        source = str(edge["source"])
        target = str(edge["target"])
        if (source, target) not in producer_pairs:
            raise ValueError(f"invalidation edge {index} must name an immediate producer dependency")
        grouped.setdefault(source, []).append(target)
    return [(source, tuple(targets)) for source, targets in grouped.items()]


def _split_producer_representations(
    producer_edges: list[Mapping[str, object]],
) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    """Partition typed dependencies into matrix cells and process arrows.

    Fan-in dependencies are clearer as a row-by-column incidence matrix. The
    temporal production spine and both authorization transitions remain a
    directed acyclic graph. Every record must be representable exactly once;
    this helper fails closed if a future contract edge has no declared visual
    location.
    """

    process_pairs = set(_PROCESS_DAG_EDGE_PAIRS)
    matrix_rows = set(_MATRIX_ROW_INVENTORY)
    matrix_columns = set(_MATRIX_COLUMN_INVENTORY)
    matrix_edges: list[Mapping[str, object]] = []
    process_edges: list[Mapping[str, object]] = []
    seen_process_pairs: set[tuple[str, str]] = set()

    for index, edge in enumerate(producer_edges):
        pair = (str(edge["source"]), str(edge["target"]))
        if pair in process_pairs:
            process_edges.append(edge)
            seen_process_pairs.add(pair)
        elif pair[0] in matrix_rows and pair[1] in matrix_columns:
            matrix_edges.append(edge)
        else:
            raise ValueError(f"producer edge {index} has no matrix or process-DAG location")

    if seen_process_pairs != process_pairs:
        raise ValueError("process-DAG representation omits a declared spine dependency")
    if len(matrix_edges) + len(process_edges) != len(producer_edges):
        raise ValueError("producer dependency representation must be exhaustive and disjoint")
    return matrix_edges, process_edges


def _representation_inventory(
    matrix_edges: list[Mapping[str, object]],
    process_edges: list[Mapping[str, object]],
) -> set[tuple[str, str, str]]:
    """Return the exact dependencies encoded by cells and arrows together."""

    return set(_edge_inventory([*matrix_edges, *process_edges]))


def _reverse_invalidation_arrow_endpoints(
    x: float,
    y: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return a right-to-left dependency arrow inside one invalidation chip.

    Producer arrows in Panel A point forward.  These deliberately point from
    the earliest stale target on the right back to the changed owner on the
    left, making the reverse dependency direction visible rather than relying
    on a border style or textual arrow glyph.
    """
    center_y = y + 0.011
    return (x + 0.236, center_y), (x + 0.068, center_y)


def _records(report: Mapping[str, object], key: str) -> list[Mapping[str, object]]:
    value = report.get(key)
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"{key} must be a list of mappings")
    return list(value)


def _indexed(records: list[Mapping[str, object]]) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for record in records:
        identifier = record.get("id")
        label = record.get("label")
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in result
            or not isinstance(label, str)
            or not label
        ):
            raise ValueError("flow records require unique ids and non-empty labels")
        result[identifier] = record
    return result


def _validate_edges(
    records: list[Mapping[str, object]],
    *,
    known_nodes: set[str],
    allowed_dispositions: set[str],
) -> None:
    seen: set[tuple[str, str]] = set()
    for index, record in enumerate(records):
        source = record.get("source")
        target = record.get("target")
        label = record.get("label")
        if source not in known_nodes or target not in known_nodes:
            raise ValueError(f"edge {index} references an unknown node")
        if not isinstance(label, str) or not label:
            raise ValueError(f"edge {index} requires a non-empty label")
        if record.get("disposition") not in allowed_dispositions:
            raise ValueError(f"edge {index} has an unexpected disposition")
        identity = (str(source), str(target))
        if identity in seen:
            raise ValueError("edge inventory contains duplicate endpoint pairs")
        seen.add(identity)


def _edge_inventory(
    records: list[Mapping[str, object]],
) -> tuple[tuple[str, str, str], ...]:
    """Return the ordered endpoint/disposition inventory used by the renderer."""

    return tuple(
        (str(record["source"]), str(record["target"]), str(record["disposition"])) for record in records
    )


def _node(
    ax: "plt.Axes",
    record: Mapping[str, object],
    xy: tuple[float, float],
    *,
    width: float = 0.165,
    height: float = 0.115,
    wrap_width: int = 22,
) -> None:
    role = str(record.get("role", "producer"))
    fill, edge = {
        "upstream input": (COLOR_PANEL_BG, COLOR_ARROW),
        "producer": (COLOR_PANEL_GOOD, COLOR_VARIATE),
        "external producer": (COLOR_PANEL_GOOD, COLOR_PURPLE),
        "rendered surface": (COLOR_PANEL_NOTE, COLOR_VARIATE),
        "gate": (COLOR_PANEL_NOTE, COLOR_ARROW),
        "receipt": (COLOR_WHITE, COLOR_PURPLE),
        "authorization-gated terminal": (COLOR_PANEL_FAIL, COLOR_NAIVE),
        "irreversible authorization-gated terminal": (COLOR_PANEL_FAIL, COLOR_NAIVE),
    }.get(role, (COLOR_WHITE, COLOR_EDGE_PANEL))
    ax.add_patch(
        FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle=f"round,pad={_NODE_BOX_PAD},rounding_size=0.015",
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.55,
            zorder=3,
        )
    )
    identifier = record.get("id")
    label = _DISPLAY_LABELS.get(str(identifier), record["label"])
    assert isinstance(label, str)
    wrapped_label = "\n".join(
        textwrap.fill(
            part,
            width=wrap_width,
            break_long_words=False,
            break_on_hyphens=False,
        )
        for part in label.splitlines()
    )
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        wrapped_label,
        ha="center",
        va="center",
        fontsize=10.8,
        color=COLOR_DEEP,
        linespacing=1.13,
        zorder=4,
    )


def _arrow(
    ax: "plt.Axes",
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLOR_ARROW,
    dashed: bool = False,
    curvature: float = 0.0,
    zorder: int = 2,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.35,
            linestyle="--" if dashed else "-",
            color=color,
            connectionstyle=f"arc3,rad={curvature}",
            zorder=zorder,
        )
    )


def _disposition_style(disposition: str) -> tuple[str, bool, str]:
    """Return color, line style, and visible code for one dependency role."""

    styles = {
        "producer": (COLOR_ARROW, False, "P"),
        "gate": (COLOR_ARROW, True, "G"),
        "receipt": (COLOR_PURPLE, False, "R"),
        "authorization": (COLOR_NAIVE, False, "A"),
    }
    try:
        return styles[disposition]
    except KeyError as exc:  # pragma: no cover - protected by report validation
        raise ValueError(f"unsupported producer disposition: {disposition}") from exc


def _draw_dependency_matrix(
    ax: "plt.Axes",
    matrix_edges: list[Mapping[str, object]],
) -> None:
    """Draw every fan-in edge as one directly coded matrix cell."""

    left = 0.365
    right = 0.985
    row_top = 0.802
    row_step = 0.025
    column_step = (right - left) / len(_MATRIX_COLUMN_INVENTORY)
    column_centers = {
        identifier: left + column_step * (index + 0.5)
        for index, identifier in enumerate(_MATRIX_COLUMN_INVENTORY)
    }
    row_centers = {
        identifier: row_top - row_step * index for index, identifier in enumerate(_MATRIX_ROW_INVENTORY)
    }

    ax.text(
        0.025,
        0.868,
        "Fan-in dependency matrix · P producer · G gate · R receipt · blank = no direct edge",
        ha="left",
        va="center",
        fontsize=10.8,
        fontweight="bold",
        color=COLOR_DEEP,
    )
    for identifier, center_x in column_centers.items():
        ax.text(
            center_x,
            0.840,
            _MATRIX_COLUMN_LABELS[identifier],
            ha="center",
            va="top",
            fontsize=9.5,
            fontweight="bold",
            color=COLOR_DEEP,
            linespacing=1.0,
        )
    for index, identifier in enumerate(_MATRIX_ROW_INVENTORY):
        center_y = row_centers[identifier]
        if index % 2 == 0:
            ax.add_patch(
                FancyBboxPatch(
                    (0.018, center_y - row_step * 0.46),
                    0.968,
                    row_step * 0.92,
                    boxstyle="square,pad=0",
                    facecolor=COLOR_PANEL_BG,
                    edgecolor="none",
                    zorder=0,
                )
            )
        ax.text(
            0.025,
            center_y,
            _MATRIX_ROW_LABELS[identifier],
            ha="left",
            va="center",
            fontsize=9.7,
            color=COLOR_DEEP,
        )

    bottom = row_centers[_MATRIX_ROW_INVENTORY[-1]] - row_step / 2
    top = row_centers[_MATRIX_ROW_INVENTORY[0]] + row_step / 2
    for index in range(len(_MATRIX_COLUMN_INVENTORY) + 1):
        x = left + index * column_step
        ax.plot([x, x], [bottom, top], color=COLOR_EDGE_PANEL, linewidth=0.7, zorder=1)
    for index in range(len(_MATRIX_ROW_INVENTORY) + 1):
        y = top - index * row_step
        ax.plot([left, right], [y, y], color=COLOR_EDGE_PANEL, linewidth=0.7, zorder=1)

    matrix_pairs: set[tuple[str, str]] = set()
    for edge in matrix_edges:
        source = str(edge["source"])
        target = str(edge["target"])
        disposition = str(edge["disposition"])
        pair = (source, target)
        if pair in matrix_pairs:
            raise ValueError("dependency matrix cannot encode a duplicate endpoint pair")
        matrix_pairs.add(pair)
        color, _, code = _disposition_style(disposition)
        ax.text(
            column_centers[target],
            row_centers[source],
            code,
            ha="center",
            va="center",
            fontsize=9.7,
            fontweight="bold",
            color=color,
            zorder=2,
        )


def _draw_process_dag(
    ax: "plt.Axes",
    process_edges: list[Mapping[str, object]],
    nodes: Mapping[str, Mapping[str, object]],
) -> None:
    """Draw the temporal, receipt, and authorization spine without crossings."""

    upper_ids = ("provisional", "coverage", "final_hydration", "render", "surfaces")
    lower_ids = ("zenodo", "github", "release_manifest", "provenance", "surface_validation")
    node_x = (0.020, 0.220, 0.420, 0.620, 0.820)
    width = 0.150
    height = 0.055
    upper_y = 0.480
    lower_y = 0.395
    positions = {identifier: (x, upper_y) for identifier, x in zip(upper_ids, node_x, strict=True)}
    positions.update({identifier: (x, lower_y) for identifier, x in zip(lower_ids, node_x, strict=True)})

    ax.text(
        0.025,
        0.588,
        "Process DAG — production spine, receipts, and separate authorization",
        ha="left",
        va="center",
        fontsize=10.8,
        fontweight="bold",
        color=COLOR_DEEP,
    )
    ax.text(
        0.025,
        0.563,
        "P producer · G gate · R receipt · A authorization",
        ha="left",
        va="center",
        fontsize=9.8,
        color=COLOR_DEEP,
    )

    for identifier in (*upper_ids, *lower_ids):
        _node(
            ax,
            nodes[identifier],
            positions[identifier],
            width=width,
            height=height,
            wrap_width=_STAGE_LABEL_WRAP_WIDTH,
        )

    edge_by_pair = {(str(edge["source"]), str(edge["target"])): edge for edge in process_edges}
    for source, target in _PROCESS_DAG_EDGE_PAIRS:
        edge = edge_by_pair[(source, target)]
        source_x, source_y = positions[source]
        target_x, target_y = positions[target]
        if source_y == target_y:
            if source_x < target_x:
                start = (source_x + width + _NODE_BOX_PAD, source_y + height / 2)
                end = (target_x - _NODE_BOX_PAD, target_y + height / 2)
            else:
                start = (source_x - _NODE_BOX_PAD, source_y + height / 2)
                end = (target_x + width + _NODE_BOX_PAD, target_y + height / 2)
        else:
            start = (source_x + width / 2, source_y - _NODE_BOX_PAD)
            end = (target_x + width / 2, target_y + height + _NODE_BOX_PAD)
        color, dashed, code = _disposition_style(str(edge["disposition"]))
        _arrow(ax, start, end, color=color, dashed=dashed)
        label_x = (start[0] + end[0]) / 2
        label_y = (start[1] + end[1]) / 2
        ax.text(
            label_x,
            label_y,
            code,
            ha="center",
            va="center",
            fontsize=9.7,
            fontweight="bold",
            color=color,
            bbox={"boxstyle": "round,pad=0.12", "facecolor": COLOR_WHITE, "edgecolor": "none"},
            zorder=5,
        )


def generate_source_render_provenance(
    report: Mapping[str, object],
    *,
    project_root: Path | None = None,
    filename: str = "source_render_provenance.png",
) -> Path:
    """Render the source-owned publication pipeline without reading its manifest."""
    inputs = _indexed(_records(report, "inputs"))
    stages = _indexed(_records(report, "stages"))
    terminals = _indexed(_records(report, "terminals"))
    producer_edges = _records(report, "producer_edges")
    invalidation_edges = _records(report, "invalidation_edges")
    no_claims = report.get("no_claims")
    authorization = report.get("authorization_boundary")
    cycle_boundary = report.get("manifest_cycle_boundary")
    if not isinstance(no_claims, list) or any(not isinstance(item, str) for item in no_claims):
        raise ValueError("no_claims must be a list of strings")
    if not isinstance(authorization, str) or not authorization:
        raise ValueError("authorization_boundary must be a non-empty string")
    if not isinstance(cycle_boundary, str) or not cycle_boundary:
        raise ValueError("manifest_cycle_boundary must be a non-empty string")

    expected_inputs = set(SOURCE_RENDER_INPUT_INVENTORY)
    expected_stages = set(SOURCE_RENDER_STAGE_INVENTORY)
    if set(inputs) != expected_inputs or set(stages) != expected_stages:
        raise ValueError("source-to-render contract has an unexpected input or stage set")
    if set(terminals) != {"github", "zenodo"}:
        raise ValueError("source-to-render terminals must be GitHub and Zenodo")
    known_nodes = set(inputs) | set(stages) | set(terminals)
    _validate_edges(
        producer_edges,
        known_nodes=known_nodes,
        allowed_dispositions={"producer", "gate", "receipt", "authorization"},
    )
    _validate_edges(
        invalidation_edges,
        known_nodes=known_nodes,
        allowed_dispositions={"invalidation"},
    )
    if _edge_inventory(producer_edges) != SOURCE_RENDER_PRODUCER_EDGE_INVENTORY:
        raise ValueError("source-to-render producer edge inventory is invalid")
    if _edge_inventory(invalidation_edges) != SOURCE_RENDER_INVALIDATION_EDGE_INVENTORY:
        raise ValueError("source-to-render invalidation edge inventory is invalid")
    invalidation_routes = _derive_immediate_invalidation_routes(
        producer_edges,
        invalidation_edges,
    )
    matrix_edges, process_edges = _split_producer_representations(producer_edges)
    if _representation_inventory(matrix_edges, process_edges) != set(SOURCE_RENDER_PRODUCER_EDGE_INVENTORY):
        raise ValueError("matrix and process-DAG representations must cover every producer edge")

    apply_style()
    plt.rcParams["figure.autolayout"] = False
    # A portrait canvas keeps every 10.8-point label above the effective
    # manuscript-scale floor. Paragraph explanations remain in the caption and
    # long description; stable codes carry the same graph semantics in-image.
    fig, ax = plt.subplots(figsize=(7.5, 10.8), facecolor=COLOR_WHITE)
    fig.set_layout_engine("none")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.set_facecolor(COLOR_WHITE)

    fig.text(
        0.5,
        0.986,
        "Source-to-render provenance and publication gates",
        ha="center",
        va="top",
        fontsize=16.5,
        fontweight="bold",
        color=COLOR_DEEP,
    )
    ax.text(
        0.5,
        0.946,
        "I input   ·   P producer   ·   E external render   ·   S surface\n"
        "G gate   ·   R receipt   ·   A authorization",
        ha="center",
        va="center",
        fontsize=10.8,
        fontweight="bold",
        color=COLOR_DEEP,
    )
    ax.text(
        0.025,
        0.910,
        "A  Source-owned producer order",
        fontsize=12.2,
        fontweight="bold",
        color=COLOR_DEEP,
    )

    all_nodes = {**inputs, **stages, **terminals}
    _draw_dependency_matrix(ax, matrix_edges)
    _draw_process_dag(ax, process_edges, all_nodes)

    ax.text(
        0.025,
        0.360,
        "B  Stale invalidation — regenerate from the earliest changed owner",
        fontsize=12.2,
        fontweight="bold",
        color=COLOR_DEEP,
    )
    chip_x = (0.025, 0.348, 0.671)
    chip_y = (0.322, 0.288, 0.254, 0.220, 0.186)
    for index, (source, targets) in enumerate(invalidation_routes):
        x = chip_x[index % 3]
        y = chip_y[index // 3]
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                _INVALIDATION_CHIP_WIDTH,
                _INVALIDATION_CHIP_HEIGHT,
                boxstyle="round,pad=0.003,rounding_size=0.004",
                facecolor=COLOR_PANEL_BG,
                edgecolor=COLOR_NAIVE,
                linewidth=1.15,
                linestyle="--",
            )
        )
        ax.text(
            x + 0.043,
            y + 0.011,
            f"Δ{_INVALIDATION_CODES[source]}",
            ha="center",
            va="center",
            fontsize=9.8,
            fontweight="bold",
            color=COLOR_DEEP,
        )
        ax.text(
            x + 0.263,
            y + 0.011,
            "·".join(_INVALIDATION_CODES[target] for target in targets),
            ha="center",
            va="center",
            fontsize=9.0,
            fontweight="bold",
            color=COLOR_DEEP,
        )
        arrow_start, arrow_end = _reverse_invalidation_arrow_endpoints(x, y)
        _arrow(
            ax,
            arrow_start,
            arrow_end,
            color=COLOR_NAIVE,
            dashed=True,
            zorder=2,
        )
        ax.text(
            x + 0.152,
            y + 0.023,
            "stale dependency",
            ha="center",
            va="center",
            fontsize=9.2,
            color=COLOR_DEEP,
        )

    ax.text(
        0.025,
        0.158,
        "C  Authorization and no-claim keys",
        fontsize=12.2,
        fontweight="bold",
        color=COLOR_DEEP,
    )
    ax.text(
        0.025,
        0.130,
        "Source-owned contract → downstream manifest; never manifest → this figure.\n"
        "Green gates ≠ release authority.",
        fontsize=10.8,
        color=COLOR_DEEP,
    )
    concise_no_claims = (
        "NC1  Green build ≠ scientific validation",
        "NC2  Tagged structure ≠ PDF/UA conformance",
        "NC3  Enhanced HTML ≠ WCAG conformance",
        "NC4  Release or DOI ≠ larger claim surface",
    )
    for index, label in enumerate(concise_no_claims):
        x = 0.025 + (index % 2) * 0.487
        y = 0.064 - (index // 2) * 0.056
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                0.463,
                0.046,
                boxstyle="round,pad=0.005,rounding_size=0.006",
                facecolor=COLOR_PANEL_FAIL,
                edgecolor=COLOR_NAIVE,
                linewidth=1.15,
            )
        )
        ax.text(
            x + 0.015,
            y + 0.023,
            textwrap.fill(label, width=43, break_long_words=False, break_on_hyphens=False),
            ha="left",
            va="center",
            fontsize=10.8,
            color=COLOR_DEEP,
            linespacing=0.95,
        )
    fig.subplots_adjust(left=0.010, right=0.990, top=0.998, bottom=0.010)
    return save_figure_pair(fig, figures_dir(project_root) / filename)


__all__ = ["generate_source_render_provenance"]
