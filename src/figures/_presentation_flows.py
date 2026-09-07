"""Complete semantic edge panels for the two source-owned provenance flows."""

from __future__ import annotations

import textwrap
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from ._common import (
    COLOR_ARROW,
    COLOR_DEEP,
    COLOR_NAIVE,
    COLOR_PANEL_BG,
    COLOR_VARIATE,
    COLOR_WHITE,
    apply_style,
    plt,
)
from ._presentation import PresentationPanel, save_presentation_panels


def _text_panel(identifier: str, text: str) -> PresentationPanel:
    apply_style()
    fig = Figure(figsize=(7.0, 2.1), facecolor=COLOR_WHITE)
    FigureCanvasAgg(fig)
    fig.set_layout_engine("none")
    wrapped = "\n".join(
        textwrap.fill(line, width=36, break_long_words=False, break_on_hyphens=False)
        for line in text.splitlines()
    )
    fig.text(0.5, 0.5, wrapped, ha="center", va="center", fontsize=22, color=COLOR_DEEP)
    return PresentationPanel(identifier, fig, text)


def _text_panels(identifier: str, text: str) -> list[PresentationPanel]:
    """Retain a full source disclosure in ordered, four-line reading panels."""
    lines = textwrap.wrap(text, width=36, break_long_words=False, break_on_hyphens=False)
    return [
        _text_panel(f"{identifier}-{index // 4 + 1}", "\n".join(lines[index : index + 4]))
        for index in range(0, len(lines), 4)
    ]


def _edge_panel(
    identifier: str,
    source: str,
    target: str,
    label: str,
    disposition: str,
) -> PresentationPanel:
    apply_style()
    fig = Figure(figsize=(7.0, 2.1), facecolor=COLOR_WHITE)
    FigureCanvasAgg(fig)
    fig.set_layout_engine("none")
    axis = fig.add_axes((0, 0, 1, 1))
    axis.axis("off")
    if disposition in {"naive", "heuristic_robust", "variational"}:
        from ._common import semantic_style

        color = semantic_style(disposition).keyline
        dash = "--"
        code = {"naive": "N", "heuristic_robust": "H", "variational": "V"}[disposition]
        label = f"{code} · {label}"
    elif disposition in {"producer", "gate", "receipt", "authorization"}:
        from .source_render_provenance import _disposition_style

        color, dashed, code = _disposition_style(disposition)
        dash = "--" if dashed else "-"
        label = f"{code} · {label}"
    else:
        color = (
            COLOR_NAIVE
            if disposition in {"rejected", "invalidation"}
            else COLOR_VARIATE
            if disposition == "data_dependency"
            else COLOR_DEEP
            if disposition == "write_order"
            else COLOR_ARROW
        )
        dash = (
            "--"
            if disposition in {"rejected", "retained_warning", "invalidation"}
            else ":"
            if disposition == "write_order"
            else "-"
        )
    renderer = cast(FigureCanvasAgg, fig.canvas).get_renderer()
    font = FontProperties(size=22)

    def wrap_node(text: str) -> str:
        lines: list[str] = []
        for authored_line in text.splitlines():
            current = ""
            for word in authored_line.split():
                candidate = f"{current} {word}".strip()
                width = renderer.get_text_width_height_descent(candidate, font, ismath=False)[0]
                if width > 0.39 * fig.get_figwidth() * fig.dpi and current:
                    lines.append(current)
                    current = word
                else:
                    current = candidate
            lines.append(current)
        return "\n".join(lines)

    for x, text in ((0.025, source), (0.555, target)):
        axis.add_patch(
            FancyBboxPatch(
                (x, 0.25),
                0.42,
                0.72,
                boxstyle="round,pad=0.005",
                facecolor=COLOR_PANEL_BG,
                edgecolor=color,
                linewidth=2,
                linestyle=dash,
            )
        )
        artist = axis.text(
            x + 0.21, 0.61, wrap_node(text), ha="center", va="center", fontsize=22, color=COLOR_DEEP
        )
        extent = artist.get_window_extent(renderer)
        if (
            extent.width > 0.40 * fig.get_figwidth() * fig.dpi
            or extent.height > 0.70 * fig.get_figheight() * fig.dpi
        ):
            plt.close(fig)
            raise ValueError(f"presentation node needs a semantic split: {text!r}")
    axis.add_patch(
        FancyArrowPatch(
            (0.452, 0.62),
            (0.548, 0.62),
            arrowstyle="-|>",
            mutation_scale=20,
            color=color,
            linewidth=2,
            linestyle=dash,
        )
    )
    axis.text(
        0.5, 0.12, textwrap.fill(label, width=38), ha="center", va="center", fontsize=22, color=COLOR_DEEP
    )
    alt = f"{source.replace(chr(10), ' ')} → {target.replace(chr(10), ' ')}: {label}; {disposition}."
    return PresentationPanel(identifier, fig, alt)


def generate_application_presentation(report: Mapping[str, object], canonical_path: Path) -> Path:
    """Preserve every declared application edge, status code, and boundary."""
    from analysis.report_schemas import validate_report

    from .application_integrity_flow import _NODE_DISPLAY_LABELS, _records, _solver_status_key

    validate_report("application_integrity_flow", report)
    panels: list[PresentationPanel] = []
    for group in _records(report, "panels"):
        for edge in _records(group, "edges"):
            source, target = str(edge["source"]), str(edge["target"])
            panels.append(
                _edge_panel(
                    f"{source}-{target}".replace("_", "-"),
                    _NODE_DISPLAY_LABELS[source],
                    _NODE_DISPLAY_LABELS[target],
                    str(edge["label"]),
                    str(edge["disposition"]),
                )
            )
    receipt = _records(report, "panels")[2]
    order = cast(list[str], receipt["write_order"])
    for source, target in zip(order, order[1:]):
        panels.append(
            _edge_panel(
                f"write-{source}-{target}".replace("_", "-"),
                _NODE_DISPLAY_LABELS[source],
                _NODE_DISPLAY_LABELS[target],
                "Atomic write order",
                "write_order",
            )
        )
    for _, _, condition, code in _solver_status_key(_records(report, "solver_statuses")):
        panels.append(_text_panel(f"status-{code}".replace("_", "-"), f"{condition}\n{code}"))
    for level in _records(report, "verification_levels"):
        identifier = str(level["id"]).replace("_", "-")
        for field, prefix in (("establishes", "Establishes"), ("does_not_establish", "Does not establish")):
            panels.append(
                _text_panel(
                    f"{identifier}-{field}".replace("_", "-"),
                    f"{level['label']}\n{prefix}: {level[field]}",
                )
            )
    no_claims = cast(list[str], report["no_claims"])
    for index, claim in enumerate(no_claims, start=1):
        panels.append(_text_panel(f"no-claim-{index}", f"Application receipt does not establish:\n{claim}"))
    return save_presentation_panels(
        panels,
        canonical_path=canonical_path,
        expected_identifiers=application_presentation_identifiers(),
    )


def application_presentation_identifiers() -> tuple[str, ...]:
    """Declare artifact ownership independently of the report supplied to draw."""
    from analysis.visual_contracts import APPLICATION_ARTIFACT_WRITE_ORDER, APPLICATION_FLOW_EDGE_INVENTORY

    return (
        *(f"{source}-{target}".replace("_", "-") for _, source, target, _ in APPLICATION_FLOW_EDGE_INVENTORY),
        *(
            f"write-{source}-{target}".replace("_", "-")
            for source, target in zip(
                APPLICATION_ARTIFACT_WRITE_ORDER,
                APPLICATION_ARTIFACT_WRITE_ORDER[1:],
            )
        ),
        "status-nominal",
        "status-converged-with-fallback",
        "status-not-converged",
        "status-not-converged-with-fallback",
        *(
            f"{level}-{field}"
            for level in ("artifact-integrity", "source-equivalence", "nominal-solver")
            for field in ("establishes", "does-not-establish")
        ),
        *(f"no-claim-{index}" for index in range(1, 6)),
    )


def source_render_presentation_identifiers() -> tuple[str, ...]:
    """Declare all producer and reverse-invalidation relations, without I/O."""
    from analysis.visual_contracts import (
        SOURCE_RENDER_INVALIDATION_EDGE_INVENTORY,
        SOURCE_RENDER_PRODUCER_EDGE_INVENTORY,
    )

    return (
        *(
            f"producer-{source}-{target}".replace("_", "-")
            for source, target, _ in SOURCE_RENDER_PRODUCER_EDGE_INVENTORY
        ),
        *(
            f"invalidation-{source}-{target}".replace("_", "-")
            for source, target, _ in SOURCE_RENDER_INVALIDATION_EDGE_INVENTORY
        ),
        "authorization-boundary",
        "manifest-cycle-boundary",
        *(f"no-claim-{index}" for index in range(1, 5)),
    )


def generate_source_render_presentation(report: Mapping[str, object], canonical_path: Path) -> Path:
    """Draw every typed dependency and reverse invalidation as a readable pair."""
    from analysis.report_schemas import validate_report

    from .application_integrity_flow import _records
    from .source_render_provenance import _DISPLAY_LABELS

    validate_report("source_render_provenance", report)
    panels: list[PresentationPanel] = []
    for kind in ("producer", "invalidation"):
        for edge in _records(report, f"{kind}_edges"):
            source, target = str(edge["source"]), str(edge["target"])
            left = _DISPLAY_LABELS[source]
            right = _DISPLAY_LABELS[target]
            label = str(edge["label"])
            if kind == "invalidation":
                left, right = right, left
                label = "Stale dependency after change"
            panels.append(
                _edge_panel(
                    f"{kind}-{source}-{target}".replace("_", "-"),
                    left,
                    right,
                    label,
                    str(edge["disposition"]),
                )
            )
    for field in ("authorization_boundary", "manifest_cycle_boundary"):
        panels.append(_text_panel(field.replace("_", "-"), str(report[field])))
    for index, claim in enumerate(cast(list[str], report["no_claims"]), start=1):
        panels.append(_text_panel(f"no-claim-{index}", claim))
    return save_presentation_panels(
        panels,
        canonical_path=canonical_path,
        expected_identifiers=source_render_presentation_identifiers(),
    )


def generate_evidence_presentation(report: Mapping[str, object], canonical_path: Path) -> Path:
    """Retain all fourteen evidence lanes and the complete nesting grammar."""
    from analysis.report_schemas import validate_report

    from .evidence_replication_map import _CLASS_CODE, _display, _records, _text

    validate_report("evidence_replication_map", report)
    classes = _records(report, "evidence_classes")
    lanes = _records(report, "lanes")
    nesting = _records(report, "nesting")
    edges = _records(report, "nesting_edges")
    panels: list[PresentationPanel] = []
    expected: list[str] = []
    for record in classes:
        identity = _text(record, "id")
        identifier = "class-" + identity.replace("_", "-")
        expected.append(identifier)
        panels.append(
            _text_panel(
                identifier, f"{_CLASS_CODE[identity]}: {_text(record, 'label')}. {_text(record, 'meaning')}."
            )
        )
    for index, lane in enumerate(lanes):
        display = _display(lane)
        heading = f"{index + 1}. {_text(lane, 'headline')} [{_CLASS_CODE[_text(lane, 'evidence_class')]}]"
        groups = (
            ("estimand", f"Estimand / unit: {_text(display, 'estimand_unit')}"),
            (
                "replication",
                f"Independent unit: {_text(display, 'replication')}. Nesting: {_text(display, 'nesting')}",
            ),
            (
                "claims",
                f"May support: {_text(display, 'permitted')}. "
                f"Does not support: {_text(display, 'prohibited')}",
            ),
        )
        for suffix, body in groups:
            identifier = f"lane-{index + 1}-{suffix}"
            expected.append(identifier)
            panels.append(_text_panel(identifier, f"{heading}\n{body}"))
    nodes = {_text(node, "id"): _text(node, "label") for node in nesting}
    for index, edge in enumerate(edges):
        identifier = f"nesting-{index + 1}"
        expected.append(identifier)
        panels.append(
            _edge_panel(
                identifier,
                nodes[_text(edge, "source")],
                nodes[_text(edge, "target")],
                _text(edge, "label"),
                "nesting",
            )
        )
    panels.append(
        _text_panel(
            "nesting-scope",
            "Nesting grammar: arrows mean contains, pairs, or repeats. "
            "Use only the levels declared in each evidence row.",
        )
    )
    expected.append("nesting-scope")
    for index, claim in enumerate(cast(list[str], report["no_claims"])):
        identifier = f"no-transfer-{index + 1}"
        expected.append(identifier)
        panels.append(_text_panel(identifier, f"NT{index + 1}: {claim}"))
    return save_presentation_panels(panels, canonical_path=canonical_path, expected_identifiers=expected)
