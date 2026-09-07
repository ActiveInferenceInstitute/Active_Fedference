"""Source-owned projection layouts for model and transport schematics."""

from __future__ import annotations

from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from ._common import COLOR_DEEP, COLOR_PANEL_BG, COLOR_PANEL_NOTE, COLOR_WHITE, apply_style
from ._presentation import PresentationPanel, save_presentation_panels
from ._presentation_flows import _edge_panel, _text_panel


def _grid_panel(identifier: str, title: str, highlight: int, *, den: bool = False) -> PresentationPanel:
    """Show a schematic nine-cell topology, never invented probabilities."""
    apply_style()
    fig = Figure(figsize=(7.0, 2.1), facecolor=COLOR_WHITE)
    FigureCanvasAgg(fig)
    fig.set_layout_engine("none")
    axis = fig.add_axes((0.38, 0.03, 0.24, 0.80))
    axis.set_xlim(0, 3)
    axis.set_ylim(0, 3)
    axis.set_aspect("equal")
    axis.axis("off")
    for index in range(9):
        col, row = index % 3, 2 - index // 3
        axis.add_patch(
            Rectangle(
                (col, row),
                1,
                1,
                facecolor=COLOR_PANEL_NOTE if index == highlight else COLOR_PANEL_BG,
                edgecolor=COLOR_DEEP,
                linewidth=1.2,
                hatch="//" if index == highlight else None,
            )
        )
        axis.text(
            col + 0.5,
            row + 0.5,
            "D" if den and index == highlight else str(index + 1),
            fontsize=22,
            ha="center",
            va="center",
            color=COLOR_DEEP,
        )
    fig.text(0.5, 0.98, title, fontsize=22, ha="center", va="top", color=COLOR_DEEP)
    return PresentationPanel(
        identifier,
        fig,
        f"{title}: nine-cell topology; cell {highlight + 1} is hatched. "
        "Schematic only, no measured probability scale.",
    )


def pomdp_presentation(path: Path) -> Path:
    """Keep the private-sensing, recipient-excluded sharing and optional control paths."""
    panels = [_grid_panel("world", "Shared world; D = den", 4, den=True)]
    identifiers = ["world"]
    for agent in ("1", "2", "n"):
        identifier = f"visual-field-{agent}"
        panels.append(
            _edge_panel(
                identifier, f"Sentinel {agent}", "Shared hidden location", "Visual field", "retained_warning"
            )
        )
        identifiers.append(identifier)
    for agent, highlight in (("1", 0), ("2", 4), ("n", 8)):
        identifier = f"local-posterior-{agent}"
        panels.append(_grid_panel(identifier, f"Schematic local posterior q{agent}", highlight))
        identifiers.append(identifier)
    relations = (
        ("message-1", "Local posterior q1", "Server", "Posterior message", "data_dependency"),
        ("message-2", "Local posterior q2", "Server", "Posterior message", "data_dependency"),
        ("message-n", "Local posterior qn", "Server", "Posterior message", "data_dependency"),
        (
            "return",
            "Qualified Eq. 7 pool / robust route",
            "Posterior excluding recipient",
            "Server return",
            "data_dependency",
        ),
        (
            "recipient",
            "Posterior excluding recipient",
            "Recipient n; m ≠ n",
            "Self-exclusion",
            "retained_warning",
        ),
        ("sensing", "Hidden location s(t)", "Private outcome o(t)", "A = P(o|s)", "data_dependency"),
        ("inference", "Private outcome o(t)", "Local posterior q(t)", "Local inference", "data_dependency"),
        (
            "policy",
            "Local posterior q(t)",
            "Action: still / left / right",
            "C preferences / EFE",
            "data_dependency",
        ),
        ("transition", "Action u(t)", "Next hidden state s(t+1)", "B = P(s′|s,u)", "data_dependency"),
        (
            "next-step",
            "Next hidden state s(t+1)",
            "Hidden location next step",
            "Temporal loop",
            "data_dependency",
        ),
    )
    for relation in relations:
        panels.append(_edge_panel(*relation))
        identifiers.append(relation[0])
    notes = (
        (
            "privacy",
            "Agents share beliefs about one hidden location. Private outcomes and controls "
            "remain local; no raw sensory data are pooled.",
        ),
        (
            "optional-control",
            "Flat federation uses inference and communication. The moving-world extension "
            "also executes B and EFE-guided control.",
        ),
        (
            "scope",
            "Model schematic, not an empirical result. Hatched cells only illustrate "
            "topology or posterior emphasis; they are not measured probability values.",
        ),
    )
    for identifier, text in notes:
        panels.append(_text_panel(identifier, text))
        identifiers.append(identifier)
    return save_presentation_panels(panels, canonical_path=path, expected_identifiers=identifiers)


def message_presentation(path: Path) -> Path:
    """Keep the full client, transport, server-route and recipient-specific ordering."""
    panels: list[PresentationPanel] = []
    expected: list[str] = []
    for agent, highlight in (("1", 0), ("2", 4), ("n", 8)):
        identifier = f"posterior-{agent}"
        panels.append(_grid_panel(identifier, f"Private update: local q{agent}", highlight))
        expected.append(identifier)
        identifier = f"broadcast-{agent}"
        panels.append(
            _edge_panel(
                identifier,
                f"Local posterior q{agent}",
                "Posterior-only broadcast",
                "Private outcome stays local",
                "data_dependency",
            )
        )
        expected.append(identifier)
    notes = (
        ("client-recovery", "Client route: NLL/KLD/β = 0 is a project recovery identity."),
        (
            "client-theorem",
            "Robust client losses: source theorem under its assumptions. Client guarantees do "
            "not transfer to server routes.",
        ),
        (
            "transport",
            "Protocol-v1 payload: categorical posterior q(n). No private outcome o(n) is broadcast.",
        ),
    )
    for identifier, text in notes:
        panels.append(_text_panel(identifier, text))
        expected.append(identifier)
    routes = (
        (
            "naive",
            "Log-linear pool",
            "N: softmax of the weighted sum of log posteriors. Qualified categorical Eq. 7 specialization.",
        ),
        (
            "heuristic_robust",
            "Heuristic robust",
            "H: robust_aggregate uses configured divergence reweighting. Conditional heuristic evidence.",
        ),
        (
            "variational",
            "Variational server",
            "V: variational_aggregate uses q,a block updates of F(q,a). Objective-backed weight property.",
        ),
    )
    for role, label, description in routes:
        stem = role.replace("_", "-")
        panels.extend(
            (
                _edge_panel(f"route-{stem}", "Posterior-only broadcast", label, "Server route", role),
                _text_panel(f"owner-{stem}", description),
                _edge_panel(f"return-{stem}", label, "Recipient n", "Senders m ≠ n only", role),
            )
        )
        expected.extend((f"route-{stem}", f"owner-{stem}", f"return-{stem}"))
    panels.append(
        _text_panel(
            "scope",
            "Deterministic protocol schematic; no empirical estimate or uncertainty interval. "
            "Self-exclusion is preserved. The Eq. 7 bridge does not reconstruct the full source "
            "protocol.",
        )
    )
    expected.append("scope")
    return save_presentation_panels(panels, canonical_path=path, expected_identifiers=expected)


def colony_presentation(path: Path, *, network: bool = False) -> Path:
    """Draw every source-derived colony probability and effective weight."""
    from typing import cast

    import numpy as np

    from ._common import COLOR_CORRECT, COLOR_MUTED, semantic_style
    from ._presentation_diagnostics import _axes
    from .system_overview import SYSTEM_OVERVIEW_METADATA, TRUE_STATE, build_data

    data = build_data()
    metadata = SYSTEM_OVERVIEW_METADATA
    adversaries = metadata["n_adversarial"]
    labels = [
        f"A{i + 1}" if i < adversaries else f"H{i - adversaries + 1}" for i in range(metadata["n_agents"])
    ]
    panels: list[PresentationPanel] = []
    expected: list[str] = []
    rows = [*data["local_posteriors"], data["naive"], data["robust"]]
    names = [*labels, "Naive consensus", "Heuristic consensus"]
    for row_index, (values, title) in enumerate(zip(rows, names, strict=True)):
        peak = int(np.argmax(values))
        for start in range(0, len(values), 3):
            stop = min(start + 3, len(values))
            axis = _axes(title, "State", "Mass")
            indices = np.arange(start, stop)
            colors = [COLOR_CORRECT if state == TRUE_STATE else COLOR_MUTED for state in indices]
            axis.bar(
                indices,
                values[start:stop],
                color=colors,
                edgecolor=COLOR_DEEP,
                hatch="xx" if row_index < adversaries else None,
                width=0.65,
            )
            for state in indices:
                if state == peak:
                    # Attach the arrow to the numeric label's measured boundary,
                    # so its shaft cannot cross the displayed posterior mass.
                    axis.annotate(
                        f"{values[state]:.3f}",
                        xy=(float(state), float(values[state])),
                        xytext=(float(state), 1.1),
                        fontsize=30,
                        ha="center",
                        va="top",
                        color=COLOR_DEEP,
                        arrowprops={"arrowstyle": "->", "color": COLOR_DEEP, "lw": 2},
                    )
                else:
                    axis.text(
                        float(state),
                        float(values[state]) + 0.04,
                        f"{values[state]:.3f}",
                        fontsize=30,
                        ha="center",
                        va="bottom",
                        color=COLOR_DEEP,
                    )
            axis.set_ylim(0, 1.15)
            axis.set_xticks(indices, [str(state + 1) for state in indices])
            axis.set_yticks([0, 1])
            identifier = f"row-{row_index + 1}-states-{start + 1}-{stop}"
            panels.append(
                PresentationPanel(
                    identifier,
                    cast(Figure, axis.figure),
                    f"{title}: source-derived posterior masses for states {start + 1} to {stop}, "
                    f"common 0–1 probability scale; arrow marks this row's argmax when present.",
                )
            )
            expected.append(identifier)
    weights = data["normalized_effective_weights"]
    for start in range(0, len(weights), 2):
        stop = min(start + 2, len(weights))
        axis = _axes("Effective influence weights", "Agent", "Weight")
        for index in range(start, stop):
            style = semantic_style("adversarial" if index < adversaries else "honest")
            axis.bar(
                index,
                weights[index],
                color=style.color,
                edgecolor=style.keyline,
                hatch=style.hatch,
                width=0.6,
            )
            axis.text(index, float(weights[index]) + 0.04, f"{weights[index]:.3f}", fontsize=30, ha="center")
        axis.set_ylim(0, 1.15)
        axis.set_xticks(range(start, stop), labels[start:stop])
        axis.set_yticks([0, 1])
        identifier = f"weights-{start + 1}-{stop}"
        panels.append(
            PresentationPanel(
                identifier,
                cast(Figure, axis.figure),
                "Source-derived normalized heuristic influence weights; adversarial bars are "
                "crosshatched and agent labels preserve their roles.",
            )
        )
        expected.append(identifier)
    notes = (
        (
            "colony",
            f"Configured colony: {metadata['n_agents']} agents; {adversaries} adversarial, "
            f"{metadata['n_honest']} honest. Contamination: {metadata['contamination_pct']}%. "
            f"True state: {TRUE_STATE + 1}.",
        ),
        (
            "naive",
            f"Naive equal-weight log pool: true-state mass {metadata['naive_acc_pct']}%; argmax "
            f"state {int(np.argmax(data['naive'])) + 1}.",
        ),
        (
            "heuristic",
            f"Heuristic robust aggregation: true-state mass {metadata['robust_acc_pct']}%; "
            f"argmax state {int(np.argmax(data['robust'])) + 1}. Robustness {metadata['robustness']}.",
        ),
        (
            "key",
            "A / crosshatching: adversarial; H / plain: honest. All probability panels share a "
            "0–1 scale. The true state is green; other states are gray.",
        ),
        (
            "scope",
            "One deterministic configured colony, not a replicated benchmark. True-state mass "
            "percentages are not deployment accuracy estimates. No uncertainty interval applies.",
        ),
    )
    for identifier, text in notes:
        panels.append(_text_panel(identifier, text))
        expected.append(identifier)
    if network:
        for label in labels:
            identifier = f"network-{label.lower()}"
            panels.append(
                _edge_panel(
                    identifier, f"Agent {label}", "Fusion server", "Posterior message", "data_dependency"
                )
            )
            expected.append(identifier)
        panels.append(
            _text_panel(
                "variational-scope",
                "The variational server is a separate objective-backed route. The displayed "
                "naive-versus-heuristic colony values do not measure its performance.",
            )
        )
        expected.append("variational-scope")
    return save_presentation_panels(panels, canonical_path=path, expected_identifiers=expected)


def generative_presentation(path: Path) -> Path:
    """Keep sensing, all model factors, temporal depth and optional hierarchical context."""
    panels = [
        _grid_panel("hidden-location", "Hidden location s", 4),
        _grid_panel("outcome", "Private outcome o", 4),
    ]
    expected = ["hidden-location", "outcome"]
    relations = (
        ("sensor", "Hidden location s", "Private outcome o", "A[o,s] = P(o|s)", "data_dependency"),
        ("likelihood", "A: sensory likelihood", "Local posterior q(s)", "Model factor", "data_dependency"),
        ("dynamics", "B: state transition", "Local inference / action", "Model factor", "data_dependency"),
        (
            "preferences",
            "C: preferred outcomes",
            "Local inference / action",
            "Model factor",
            "data_dependency",
        ),
        ("prior", "D0: initial prior", "Local posterior q(s)", "Model factor", "data_dependency"),
        ("time-sensing", "Hidden state s(t)", "Private report o(t)", "Observation", "data_dependency"),
        ("time-inference", "Private report o(t)", "Posterior q(t)", "Inference", "data_dependency"),
        ("time-control", "Posterior q(t)", "Control u(t)", "EFE-guided action", "data_dependency"),
        ("time-transition", "Control u(t)", "Next state s(t+1)", "B = P(s′|s,u)", "data_dependency"),
        ("meta-context", "Meta-context s(L)", "Context s(2)", "Top-down context", "data_dependency"),
        ("context-location", "Context s(2)", "Location s(1)", "Conditional prior", "data_dependency"),
        ("location-posterior", "Location s(1)", "Local posterior q1", "Location evidence", "data_dependency"),
        (
            "context-posterior",
            "Context s(2)",
            "Local posterior q1",
            "Context conditions prior",
            "data_dependency",
        ),
        (
            "meta-posterior",
            "Meta-context s(L)",
            "Local posterior q1",
            "Optional higher context",
            "retained_warning",
        ),
    )
    for relation in relations:
        panels.append(_edge_panel(*relation))
        expected.append(relation[0])
    notes = (
        (
            "sensor-key",
            "Diagonal mass is sensor acuity; residual mass is categorical noise. Raw outcomes "
            "remain private; only posteriors are broadcast.",
        ),
        ("local-equation", r"$q(s)=\mathrm{softmax}(\ln D_0+\ln A[o,\cdot])$"),
        (
            "recovery",
            "Project recovery identity: robust_aggregate(0) = log_linear_pool. This limit does "
            "not reconstruct the full source protocol.",
        ),
        ("mixture-equation", r"$\bar q_1=\sum_kq_2[k]D_{1|k}$"),
        (
            "optional",
            "Flat studies stop after posterior sharing. Moving-world studies execute B and "
            "EFE-guided control. Hierarchy is an optional extension.",
        ),
        (
            "scope",
            "Schematic formalization, not an empirical estimate. Private sensory outcomes "
            "become posterior messages; fusion uses the shared categorical state.",
        ),
    )
    for identifier, text in notes:
        panels.append(_text_panel(identifier, text))
        expected.append(identifier)
    return save_presentation_panels(panels, canonical_path=path, expected_identifiers=expected)
