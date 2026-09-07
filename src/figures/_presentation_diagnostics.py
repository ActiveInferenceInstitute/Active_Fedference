"""Projection layouts for deterministic objective and weight diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MaxNLocator

from ._common import COLOR_DEEP, semantic_style
from ._presentation import PresentationPanel, save_presentation_panels
from ._presentation_flows import _text_panel
from ._presentation_robustness import _canvas


def _axes(title: str, xlabel: str, ylabel: str) -> Axes:
    fig = _canvas()
    axis = fig.add_axes((0.25, 0.32, 0.71, 0.42))
    axis.tick_params(labelsize=30)
    axis.xaxis.set_major_locator(MaxNLocator(3, integer=True))
    axis.yaxis.set_major_locator(MaxNLocator(2))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2g}"))
    axis.set_xlabel(xlabel, fontsize=30)
    fig.text(0.5, 0.98, title, ha="center", va="top", fontsize=30, color=COLOR_DEEP)
    fig.text(0.02, 0.55, ylabel, rotation=90, ha="left", va="center", fontsize=30, color=COLOR_DEEP)
    return axis


def _notes(text: str, *, prefix: str = "notes") -> list[PresentationPanel]:
    lines = text.splitlines()
    return [
        _text_panel(f"{prefix}-{index // 3 + 1}", "\n".join(lines[index : index + 3]))
        for index in range(0, len(lines), 3)
    ]


def _curve(axis: Axes, x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray, role: str) -> None:
    style = semantic_style(role)
    axis.plot(
        x,
        y,
        color=style.color,
        marker=style.marker,
        markerfacecolor="white",
        markeredgecolor=style.keyline,
        linestyle=style.dash,
        linewidth=3,
        markersize=8,
    )


def objective_presentation(
    path: Path,
    iterations: np.ndarray,
    history: np.ndarray,
    *,
    largest_step_index: int | None,
    stats_text: str,
) -> Path:
    """Keep the full trajectory, terminus, largest-step markers, and statistics."""
    axis = _axes("Variational objective descent", "Iteration", "F (nats)")
    _curve(axis, iterations, history, "variational")
    style = semantic_style("variational")
    axis.scatter(
        [iterations[-1]],
        [history[-1]],
        marker=style.marker,
        facecolor=style.color,
        edgecolor=style.keyline,
        s=140,
        zorder=5,
    )
    if largest_step_index is not None:
        selected = slice(largest_step_index, largest_step_index + 2)
        axis.scatter(
            iterations[selected],
            history[selected],
            marker=style.marker,
            facecolor="white",
            edgecolor=style.keyline,
            s=120,
            zorder=4,
        )
    reference = semantic_style("reference_rule")
    axis.axhline(float(history[-1]), color=reference.color, linestyle=reference.dash, linewidth=2)
    axis.margins(x=0.04, y=0.12)
    panels = [
        PresentationPanel(
            "trajectory",
            cast(Figure, axis.figure),
            "Variational objective in nats at every iteration; filled triangle is final iterate, "
            "large open triangles mark the largest observed step.",
        ),
        _text_panel(
            "encoding",
            "Triangles and dash-dot: variational objective.\nFilled triangle: final iterate.\n"
            "Large open triangles: largest observed step.",
        ),
        *_notes(stats_text),
        _text_panel(
            "scope",
            "One deterministic trajectory; no sampling interval.\n"
            "A final iterate is not a global-optimum certificate.",
        ),
    ]
    return save_presentation_panels(
        panels,
        canonical_path=path,
        expected_identifiers=(
            "trajectory",
            "encoding",
            *(f"notes-{index + 1}" for index in range((len(stats_text.splitlines()) + 2) // 3)),
            "scope",
        ),
    )


def influence_path_presentation(
    path: Path,
    drifts: np.ndarray,
    influence: np.ndarray,
    naive: float,
    *,
    crossing_index: int | None,
) -> Path:
    """Retain both weight paths, the deterministic gap, and the marked crossing."""
    axis = _axes("Normalized server weight", "Outlier drift", "Weight")
    axis.xaxis.set_major_locator(MaxNLocator(3))
    _curve(axis, drifts, influence, "variational")
    _curve(axis, drifts, np.full(drifts.shape, naive), "naive")
    style = semantic_style("variational")
    axis.fill_between(
        drifts, influence, naive, where=(influence <= naive).tolist(), color=style.color, alpha=0.12
    )
    if crossing_index is not None:
        attack = semantic_style("adversarial")
        axis.scatter(
            [drifts[crossing_index]],
            [influence[crossing_index]],
            marker=attack.marker,
            color=attack.color,
            edgecolor=attack.keyline,
            s=130,
            zorder=5,
        )
    axis.margins(x=0.04)
    axis.set_ylim(0, max(float(np.max(influence)), naive, 1e-6) * 1.18)
    panels = [
        PresentationPanel(
            "weight-path",
            cast(Figure, axis.figure),
            "Every configured drift: triangle/dash-dot variational weights, circle/solid naive weights; "
            "shaded gap is deterministic, not an interval.",
        ),
        _text_panel(
            "encoding",
            "Triangle/dash-dot: variational.\nCircle/solid: naive fixed weight.\n"
            "Shading: deterministic weight gap.",
        ),
        _text_panel(
            "drift-key",
            "Outlier drift: 0 is consensus; 1 is confidently wrong.\n"
            "This is a configured path, not estimator-level B-robustness.",
        ),
        _text_panel(
            "endpoints",
            f"Variational final weight: {influence[-1]:.3g}\nNaive fixed weight: 1/n = {naive:.3g}",
        ),
    ]
    if crossing_index is not None:
        panels.append(
            _text_panel(
                "crossing",
                f"Cross: first configured drift below 0.5 × naive.\nDrift = {drifts[crossing_index]:.2f}",
            )
        )
    return save_presentation_panels(
        panels,
        canonical_path=path,
        expected_identifiers=(
            "weight-path",
            "encoding",
            "drift-key",
            "endpoints",
            *(("crossing",) if crossing_index is not None else ()),
        ),
    )


def agent_weights_presentation(
    path: Path,
    weights: np.ndarray,
    contaminated: set[int],
    *,
    equal_weight: float,
    stats_text: str,
) -> Path:
    """Show all agent bars in ordered pairs with one shared vertical scale."""
    panels: list[PresentationPanel] = [
        _text_panel(
            "encoding",
            "Open fill: honest. Cross-hatch: adversary.\nDotted line: equal-weight pool.\n"
            "All panels share the same weight scale.",
        )
    ]
    for start in range(0, weights.size, 2):
        axis = _axes("Heuristic normalized weights", "Agent and role", "Weight")
        indices = list(range(start, min(start + 2, weights.size)))
        for index in indices:
            role = "adversarial" if index in contaminated else "honest"
            style = semantic_style(role)
            axis.bar(
                index,
                weights[index],
                width=0.6,
                color=style.color if index in contaminated else "white",
                edgecolor=style.keyline,
                hatch=style.hatch,
                linewidth=2,
            )
            axis.text(
                index,
                weights[index] + max(float(weights.max()) * 0.025, 0.003),
                f"{weights[index]:.3f}",
                ha="center",
                va="bottom",
                fontsize=30,
                color=COLOR_DEEP,
            )
            if index in contaminated and weights[index] < equal_weight:
                axis.annotate(
                    "",
                    xy=(index + 0.28, float(weights[index]) + 0.002),
                    xytext=(index + 0.28, equal_weight - 0.002),
                    arrowprops={"arrowstyle": "-|>", "color": style.keyline, "lw": 2},
                )
        reference = semantic_style("reference_rule")
        axis.axhline(equal_weight, color=reference.color, linestyle=reference.dash, linewidth=2)
        axis.set_ylim(0, max(float(weights.max()), equal_weight) * 1.55)
        axis.set_xlim(start - 0.6, indices[-1] + 0.6)
        axis.set_xticks(
            indices, [f"a{index}\n{'adversary' if index in contaminated else 'honest'}" for index in indices]
        )
        # The tick text itself supplies the agent/role axis name; the key panel
        # carries the shared encoding rather than duplicating a third line.
        axis.set_xlabel("")
        panels.append(
            PresentationPanel(
                f"agents-{start + 1}-{indices[-1] + 1}",
                cast(Figure, axis.figure),
                f"Agents {start} through {indices[-1]}: exact normalized heuristic weights; "
                "shared scale and equal-weight reference.",
            )
        )
    panels.extend(_notes(stats_text))
    panels.append(
        _text_panel(
            "scope",
            "Server heuristic weights support a recovery-limit diagnostic.\n"
            "They do not inherit client-loss robustness guarantees.",
        )
    )
    return save_presentation_panels(
        panels,
        canonical_path=path,
        expected_identifiers=(
            "encoding",
            *(f"agents-{start + 1}-{min(start + 2, weights.size)}" for start in range(0, weights.size, 2)),
            *(f"notes-{index + 1}" for index in range((len(stats_text.splitlines()) + 2) // 3)),
            "scope",
        ),
    )


def descent_comparison_presentation(
    path: Path,
    single: np.ndarray,
    multi: np.ndarray,
    stats_text: str,
) -> Path:
    """Show both complete observed initialization paths on common axes."""
    axis = _axes("Configured initializations", "Iteration", "F (nats)")
    maximum_iteration = max(single.size, multi.size)
    for values, role in ((single, "condition_reference"), (multi, "condition_comparison")):
        style = semantic_style(role)
        axis.plot(
            np.arange(1, values.size + 1),
            values,
            color=style.color,
            linestyle=style.dash,
            marker=style.marker,
            markerfacecolor=style.color if role == "condition_reference" else "white",
            markeredgecolor=style.keyline,
            linewidth=3,
            markersize=8,
        )
        axis.scatter(
            [values.size],
            [values[-1]],
            marker=style.marker,
            facecolor=style.color if role == "condition_reference" else "white",
            edgecolor=style.keyline,
            s=150,
            zorder=5,
        )
    reference = semantic_style("reference_rule")
    axis.axhline(multi[-1], color=reference.color, linestyle=reference.dash, linewidth=2)
    if single[-1] > multi[-1]:
        axis.annotate(
            "",
            xy=(maximum_iteration + 0.5, multi[-1]),
            xytext=(maximum_iteration + 0.5, single[-1]),
            arrowprops={"arrowstyle": "<->", "color": reference.color, "lw": 2},
        )
    axis.set_xlim(0.7, maximum_iteration + 1)
    axis.margins(y=0.18)
    panels = [
        PresentationPanel(
            "trajectories",
            cast(Figure, axis.figure),
            "Both complete objective trajectories, observed endpoints, lower final "
            "reference and terminal gap.",
        ),
        _text_panel(
            "key",
            "Filled circles / solid: single start. Open diamonds / dotted: multistart. "
            "Both use the same variational method.",
        ),
        *_notes(stats_text),
        _text_panel(
            "scope",
            "Two deterministic traces from one configured colony. No uncertainty "
            "interval applies. Observed basins do not certify a global optimum.",
        ),
    ]
    expected = (
        "trajectories",
        "key",
        *(f"notes-{i + 1}" for i in range((len(stats_text.splitlines()) + 2) // 3)),
        "scope",
    )
    return save_presentation_panels(panels, canonical_path=path, expected_identifiers=expected)


def free_energy_presentation(
    path: Path,
    incom: np.ndarray,
    comm: np.ndarray,
    differences: np.ndarray,
    jitter: np.ndarray,
    mean_difference: float,
    interval: tuple[float, float] | None,
    analysis_unit: str,
    replication_unit: str,
) -> Path:
    """Preserve every paired seed, descriptive envelope and report-owned interval."""
    axis = _axes("Matched seed conditions", "Condition", "F (nats)")
    reference = semantic_style("condition_reference")
    comparison = semantic_style("condition_comparison")
    for before, after in zip(incom, comm, strict=True):
        axis.plot((0, 1), (before, after), color=COLOR_DEEP, alpha=0.18, linewidth=1)
    for x, values, style, face in ((0, incom, reference, reference.color), (1, comm, comparison, "white")):
        axis.scatter(
            np.full(values.size, x),
            values,
            marker=style.marker,
            facecolor=face,
            edgecolor=style.keyline,
            s=35,
            zorder=3,
        )
    axis.plot(
        (0, 1),
        (incom.mean(), comm.mean()),
        color=COLOR_DEEP,
        marker="D",
        markerfacecolor="white",
        markersize=10,
        linewidth=3,
        zorder=4,
    )
    axis.set_xticks((0, 1), ("Isolated", "Sharing"))
    axis.set_xlim(-0.3, 1.3)
    panels = [
        PresentationPanel(
            "paired-seeds",
            cast(Figure, axis.figure),
            "Every seed connects its isolated and communicating colony mean; dark line "
            "with open diamonds marks condition means.",
        )
    ]
    contrast = _axes("Paired seed differences", "ΔF (nats)", "")
    violin = contrast.violinplot(
        differences,
        positions=[0.0],
        orientation="horizontal",
        widths=0.30,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in cast(list[PolyCollection], violin["bodies"]):
        body.set_facecolor(comparison.color)
        body.set_edgecolor(comparison.keyline)
        body.set_alpha(0.16)
    contrast.scatter(
        differences,
        jitter,
        marker=comparison.marker,
        facecolor="white",
        edgecolor=comparison.keyline,
        s=30,
        alpha=0.7,
        zorder=3,
    )
    zero = semantic_style("reference_rule")
    contrast.axvline(0, color=zero.color, linestyle=zero.dash, linewidth=2)
    if interval is None:
        contrast.scatter([mean_difference], [-0.26], marker="D", color=COLOR_DEEP, s=90)
        uncertainty = "Interval unavailable; no interval is inferred by the plotter."
    else:
        lo, hi = interval
        contrast.errorbar(
            [mean_difference],
            [-0.26],
            xerr=[[mean_difference - lo], [hi - mean_difference]],
            fmt="D",
            markersize=9,
            markerfacecolor="white",
            color=COLOR_DEEP,
            capsize=6,
            linewidth=2,
        )
        uncertainty = f"Report-owned 95% interval: [{lo:.4g}, {hi:.4g}] nats."
    contrast.set_ylim(-0.38, 0.21)
    contrast.set_yticks(())
    panels.extend(
        [
            PresentationPanel(
                "seed-differences",
                cast(Figure, contrast.figure),
                "All paired seed differences with jitter and descriptive violin, zero "
                "reference, and paired mean with the available report interval.",
            ),
            _text_panel(
                "direction",
                "ΔF = isolated minus communicating. Positive values indicate lower F with "
                "communication in this reduced categorical protocol.",
            ),
            _text_panel(
                "pair-key",
                "Filled circles: isolated. Open diamonds: sharing. Fine lines preserve "
                "pairing. Dark line / open diamonds: condition means.",
            ),
            _text_panel(
                "difference-key",
                "One open diamond per seed; vertical jitter carries no outcome meaning. "
                "Violin is descriptive. Dark dotted rule marks zero contrast.",
            ),
            _text_panel(
                "mean",
                f"Paired mean ΔF: {mean_difference:+.4g} nats. Independent unit: "
                f"{replication_unit}; n = {differences.size}.",
            ),
            _text_panel("interval", uncertainty),
            _text_panel(
                "unit", f"Analysis unit: {analysis_unit}. Agents and ordered steps remain nested within seed."
            ),
            _text_panel(
                "scope",
                "Reduced-protocol comparison. It does not establish a general communication "
                "benefit or reconstruct the complete source protocol.",
            ),
        ]
    )
    return save_presentation_panels(
        panels,
        canonical_path=path,
        expected_identifiers=(
            "paired-seeds",
            "seed-differences",
            "direction",
            "pair-key",
            "difference-key",
            "mean",
            "interval",
            "unit",
            "scope",
        ),
    )


def efe_presentation(path: Path, risk: float, ambiguity: float, pragmatic: float, epistemic: float) -> Path:
    """Retain the additive cost and signed value views on identical scales."""
    from ._common import COLOR_ACCENT, COLOR_MULTI_1, COLOR_MULTI_2, COLOR_MUTED

    total = risk + ambiguity
    axis = _axes("Cost and signed value views", "View", "Nats")
    axis.bar(0, risk, width=0.5, color=COLOR_MULTI_1, edgecolor=COLOR_DEEP)
    axis.bar(0, ambiguity, bottom=risk, width=0.5, color=COLOR_MULTI_2, edgecolor=COLOR_DEEP, hatch="//")
    axis.bar(1, -pragmatic, width=0.5, color=COLOR_ACCENT, edgecolor=COLOR_DEEP)
    axis.bar(1, -epistemic, bottom=-pragmatic, width=0.5, color=COLOR_MUTED, edgecolor=COLOR_DEEP, hatch="xx")
    axis.axhline(total, color=COLOR_DEEP, linestyle="--", linewidth=2)
    axis.scatter([0, 1], [total, total], marker="D", color=COLOR_DEEP, edgecolor="white", s=120, zorder=5)
    axis.plot([0.25, 0.75], [total, total], color=COLOR_DEEP, linestyle=":", linewidth=2)
    axis.set_xticks([0, 1], ["Cost", "Value"])
    axis.set_xlim(-0.6, 1.6)
    lower = min(0.0, risk, total, -pragmatic, -pragmatic - epistemic)
    upper = max(0.0, risk, total, -pragmatic, -pragmatic - epistemic)
    span = max(upper - lower, 1.0)
    axis.set_ylim(lower - 0.05 * span, upper + 0.18 * span)
    panels = [
        PresentationPanel(
            "identity",
            cast(Figure, axis.figure),
            "Risk plus ambiguity and signed pragmatic minus epistemic waterfall share the same terminal G.",
        ),
        _text_panel(
            "cost",
            f"Cost: risk {risk:.4g} plus ambiguity {ambiguity:.4g} nats. Hatched upper segment is ambiguity.",
        ),
        _text_panel(
            "value",
            f"Value: −pragmatic {-pragmatic:.4g} plus −epistemic {-epistemic:.4g} nats. "
            f"Crosshatching marks the signed epistemic correction.",
        ),
        _text_panel(
            "endpoint",
            f"Terminal diamond: G = {total:.4g} nats. The right endpoint equals G; the "
            f"intermediate top need not equal G.",
        ),
        _text_panel(
            "residual",
            f"Identity residual: {total + pragmatic + epistemic:.3e} nats. G = risk + "
            f"ambiguity = −(pragmatic + epistemic).",
        ),
        _text_panel(
            "scope",
            "Uniform diagnostic prior makes information gain visible. One categorical "
            "identity diagnostic; no stochastic replication or uncertainty interval.",
        ),
    ]
    return save_presentation_panels(
        panels,
        canonical_path=path,
        expected_identifiers=("identity", "cost", "value", "endpoint", "residual", "scope"),
    )


def belief_heatmap_presentation(path: Path, matrix: np.ndarray, labels: Sequence[str]) -> Path:
    """Show every posterior cell in bounded row segments with one fixed scale."""
    from matplotlib.colors import to_hex

    from ._common import contrasting_text_color

    panels = [
        _text_panel(
            "key",
            "Posterior probability mass on a common 0–1 scale. Rows retain agent order, "
            "followed by consensus. Location indices retain their original order.",
        )
    ]
    expected = ["key"]
    for row, label in enumerate(labels):
        for start in range(0, matrix.shape[1], 3):
            stop = min(start + 3, matrix.shape[1])
            axis = _axes(label, "Location", "")
            values = matrix[row : row + 1, start:stop]
            image = axis.imshow(values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
            axis.grid(False)
            axis.set_xticks(range(stop - start), [str(i) for i in range(start, stop)])
            axis.set_yticks(())
            for i, value in enumerate(values[0]):
                axis.text(
                    i,
                    0,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=30,
                    color=contrasting_text_color(to_hex(image.cmap(image.norm(value)))),
                )
            identifier = f"row-{row}-cells-{start}-{stop - 1}"
            expected.append(identifier)
            panels.append(
                PresentationPanel(
                    identifier,
                    cast(Figure, axis.figure),
                    f"{label}, locations {start} through {stop - 1}; every probability printed "
                    f"on the common 0–1 scale.",
                )
            )
    panels.append(
        _text_panel(
            "scope",
            f"{len(labels) - 1} agents and one consensus row. One configured colony "
            f"snapshot; no uncertainty interval. Categorical log-linear specialization, "
            f"not the full source protocol.",
        )
    )
    expected.append("scope")
    return save_presentation_panels(panels, canonical_path=path, expected_identifiers=expected)
