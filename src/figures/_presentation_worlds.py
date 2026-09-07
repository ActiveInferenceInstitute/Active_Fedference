"""Projection layouts for declared moving and hierarchical world diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
from matplotlib.figure import Figure

from ._common import COLOR_DEEP, COLOR_MUTED, COLOR_NAIVE, COLOR_ROBUST, COLOR_VARIATE
from ._presentation import PresentationPanel, save_presentation_panels
from ._presentation_diagnostics import _axes
from ._presentation_flows import _text_panel


def moving_presentation(path: Path, results: Mapping[str, Any]) -> Path:
    """Retain all three conditions in each distinct native-unit metric."""
    conditions = ("isolated", "communicating", "efe_guided")
    panels: list[PresentationPanel] = []
    expected: list[str] = []
    for metric, title, unit in (
        ("accuracy", "Accuracy", "Fraction"),
        ("free_energy_gap", "Free-energy gap", "Nats"),
        ("n_steps_to_consensus", "Time to consensus", "Steps"),
    ):
        values = [float(results[metric].get(c, 0)) for c in conditions]
        axis = _axes(title, "Condition", unit)
        axis.bar(
            [0, 1, 2],
            values,
            color=[COLOR_NAIVE, COLOR_ROBUST, COLOR_MUTED],
            edgecolor=COLOR_DEEP,
            hatch=["", "//", ".."],
            width=0.6,
        )
        axis.set_xticks([0, 1, 2], ["ISO", "COM", "EFE"])
        low = min(0, min(values))
        high = max(0, max(values))
        span = max(high - low, 0.01)
        axis.set_ylim(low - 0.12 * span, high + 0.15 * span)
        axis.axhline(0, color=COLOR_DEEP, linestyle=":", linewidth=2)
        identifier = metric.replace("_", "-")
        panels.append(
            PresentationPanel(
                identifier,
                cast(Figure, axis.figure),
                f"{title}: all isolated, communicating and EFE-guided values in {unit.lower()}, "
                f"retaining signed values and ties.",
            )
        )
        panels.append(
            _text_panel(
                f"{identifier}-values",
                f"{title} ({unit.lower()}): ISO {values[0]:+.4g}; COM {values[1]:+.4g}; EFE "
                f"{values[2]:+.4g}.",
            )
        )
        expected.extend((identifier, f"{identifier}-values"))
    acc = results["accuracy"]
    notes = (
        (
            "gains",
            f"Accuracy gains over isolated: COM "
            f"{float(acc.get('communicating', 0)) - float(acc.get('isolated', 0)):+.3f}; EFE "
            f"{float(acc.get('efe_guided', 0)) - float(acc.get('isolated', 0)):+.3f}.",
        ),
        (
            "key",
            "ISO / plain: isolated; COM / forward hatch: communicating; EFE / dots: EFE-guided. "
            "Different metric units have separate axes.",
        ),
        (
            "scope",
            "Declared moving-world summary. No interval is added to these scalar bars. Ordered "
            "steps and within-run observations do not become independent replications.",
        ),
    )
    for identifier, text in notes:
        panels.append(_text_panel(identifier, text))
        expected.append(identifier)
    return save_presentation_panels(panels, canonical_path=path, expected_identifiers=expected)


def disjoint_presentation(path: Path, multiseed: Mapping[str, Any]) -> Path:
    """Retain condition means and seed standard deviations, never rename SD as CI."""
    panels: list[PresentationPanel] = []
    expected: list[str] = []
    groups = (
        (
            "communication",
            ("isolated", "communicating"),
            ("ISO", "COM"),
            (COLOR_NAIVE, COLOR_ROBUST),
            ("", "//"),
        ),
        ("movement", ("efe_guided", "random"), ("EFE", "RND"), (COLOR_ROBUST, COLOR_MUTED), ("..", "xx")),
    )
    lower = min(
        0,
        min(
            float(row["mean"]) - float(row["std"])
            for row in multiseed.values()
            if isinstance(row, Mapping) and "mean" in row and "std" in row
        ),
    )
    upper = max(
        1,
        max(
            float(row["mean"]) + float(row["std"])
            for row in multiseed.values()
            if isinstance(row, Mapping) and "mean" in row and "std" in row
        ),
    )
    for name, conditions, labels, colors, hatches in groups:
        axis = _axes(name.title() + " contrast", "Condition", "Accuracy")
        means = [float(multiseed[c]["mean"]) for c in conditions]
        deviations = [float(multiseed[c]["std"]) for c in conditions]
        axis.bar(
            [0, 1],
            means,
            yerr=deviations,
            color=colors,
            hatch=hatches,
            edgecolor=COLOR_DEEP,
            width=0.5,
            capsize=5,
            error_kw={"elinewidth": 2, "ecolor": COLOR_DEEP},
        )
        axis.set_xticks([0, 1], labels)
        axis.set_ylim(lower - 0.03, upper + 0.05)
        panels.append(
            PresentationPanel(
                name,
                cast(Figure, axis.figure),
                f"{name}: both condition means with seed standard-deviation whiskers on a common "
                f"probability scale.",
            )
        )
        gain = means[1] - means[0] if name == "communication" else means[0] - means[1]
        panels.append(
            _text_panel(
                f"{name}-values",
                f"{labels[0]} mean {means[0]:.4g}, SD {deviations[0]:.4g}; {labels[1]} mean "
                f"{means[1]:.4g}, SD {deviations[1]:.4g}. "
                + ("COM−ISO" if name == "communication" else "EFE−RND")
                + f" = {gain:+.4g}.",
            )
        )
        expected.extend((name, f"{name}-values"))
    notes = (
        (
            "key",
            "ISO: isolated; COM: communicating; EFE: EFE-guided; RND: random movement. Hatches "
            "preserve condition identity.",
        ),
        (
            "uncertainty",
            "Whiskers show standard deviations across configured seeds, not confidence "
            "intervals. Accuracy is a fraction correct.",
        ),
        (
            "scope",
            "Separate communication and movement-policy contrasts in the declared disjoint-view "
            "worlds. A near-ceiling movement contrast does not imply a general navigation advantage.",
        ),
    )
    for identifier, text in notes:
        panels.append(_text_panel(identifier, text))
        expected.append(identifier)
    return save_presentation_panels(panels, canonical_path=path, expected_identifiers=expected)


def hierarchical_presentation(
    path: Path,
    *,
    posterior_flat: np.ndarray,
    posterior_two: np.ndarray,
    posterior_three: np.ndarray,
    consensus_flat: np.ndarray,
    consensus_two: np.ndarray,
    context_two: np.ndarray,
    context_three: np.ndarray,
    meta_three: np.ndarray,
    context_labels: list[str],
    gap_two: float,
    gap_three: float,
    n_trials: int,
    obs: int,
    true_state: int,
    acuity: float,
    n_agents: int,
    illustrative: bool,
) -> Path:
    """Preserve all six panel estimands and distinguish schematic from measured data."""
    panels: list[PresentationPanel] = []
    expected: list[str] = []
    pairs = (
        ("two-level posterior", posterior_flat, posterior_two, COLOR_ROBUST, obs),
        ("colony consensus", consensus_flat, consensus_two, COLOR_ROBUST, true_state),
        ("three-level posterior", posterior_flat, posterior_three, COLOR_VARIATE, obs),
    )
    for group, (title, flat, hier, color, reference) in enumerate(pairs):
        for start in range(0, len(flat), 3):
            stop = min(start + 3, len(flat))
            x = np.arange(start, stop)
            axis = _axes(title.title(), "State", "Probability")
            axis.bar(x - 0.2, flat[start:stop], width=0.4, color=COLOR_NAIVE, edgecolor=COLOR_DEEP)
            axis.bar(x + 0.2, hier[start:stop], width=0.4, color=color, edgecolor=COLOR_DEEP, hatch="//")
            if start <= reference < stop:
                axis.axvline(reference, color=COLOR_DEEP, linestyle="--", linewidth=2)
            axis.set_xticks(x, [str(v) for v in x])
            axis.set_ylim(0, 1.15)
            identifier = f"panel-{group + 1}-states-{start}-{stop - 1}"
            panels.append(
                PresentationPanel(
                    identifier,
                    cast(Figure, axis.figure),
                    f"{title}, states {start} to {stop - 1}: plain flat versus hatched "
                    f"hierarchical probabilities with the declared state reference.",
                )
            )
            expected.append(identifier)
        identifier = f"panel-{group + 1}-peaks"
        panels.append(
            _text_panel(
                identifier,
                f"{title.title()}: flat peak state {int(np.argmax(flat))}, mass "
                f"{float(flat.max()):.4g}; hierarchical peak state {int(np.argmax(hier))}, mass "
                f"{float(hier.max()):.4g}.",
            )
        )
        expected.append(identifier)
    curves = (
        (
            "Two-level context",
            [
                (context_two[:, 0], COLOR_NAIVE, "o", "-", context_labels[0]),
                (context_two[:, 1], COLOR_ROBUST, "s", "-", context_labels[1]),
            ],
        ),
        (
            "Three-level context",
            [
                (context_three[:, 1], COLOR_ROBUST, "s", "-", "L2 alert"),
                (meta_three[:, 1], COLOR_VARIATE, "^", "--", "L3 high threat"),
            ],
        ),
    )
    for index, (title, series) in enumerate(curves):
        axis = _axes(title, "Iteration", "Probability")
        for values, color, marker, dash, label in series:
            axis.plot(
                np.arange(1, len(values) + 1),
                values,
                color=color,
                marker=marker,
                linestyle=dash,
                linewidth=3,
                markersize=8,
            )
        axis.axhline(0.5, color=COLOR_DEEP, linestyle=":", linewidth=2)
        axis.set_ylim(-0.05, 1.05)
        axis.margins(x=0.05)
        identifier = f"context-{index + 1}"
        panels.append(
            PresentationPanel(
                identifier,
                cast(Figure, axis.figure),
                f"{title}: all alternating-minimization iterations with the half-probability reference.",
            )
        )
        panels.append(
            _text_panel(
                f"{identifier}-key",
                f"{title}: "
                + "; ".join(f"{label} ({marker}, {dash})" for _, _, marker, dash, label in series)
                + ". Ordered iterations are not independent replications.",
            )
        )
        expected.extend((identifier, f"{identifier}-key"))
    axis = _axes("Final accuracy contrast", "Hierarchy depth", "Fraction")
    axis.bar(
        [0, 1],
        [gap_two, gap_three],
        color=[COLOR_ROBUST, COLOR_VARIATE],
        hatch=["//", ".."],
        edgecolor=COLOR_DEEP,
    )
    axis.set_xticks([0, 1], ["2L", "3L"])
    axis.axhline(0, color=COLOR_DEEP, linestyle="--", linewidth=2)
    low = min(gap_two, gap_three, 0)
    high = max(gap_two, gap_three, 0)
    pad = max(0.01, 0.6 * (high - low))
    axis.set_ylim(low - pad, high + pad)
    panels.append(
        PresentationPanel(
            "gaps",
            cast(Figure, axis.figure),
            "Final signed hierarchical-minus-flat location accuracy for both hierarchy depths; "
            "zero reference retained.",
        )
    )
    expected.append("gaps")
    notes = (
        (
            "gap-values",
            f"Hierarchical minus flat: 2L {gap_two:+.4g}; 3L {gap_three:+.4g}. n = {n_trials} "
            f"nested trials. No per-trial trajectory or interval is inferred from these scalar gaps.",
        ),
        (
            "configuration",
            f"Acuity {acuity:.2f}; observed state {obs}; colony true state {true_state}; "
            f"{n_agents} agents. Plain left bars: flat; hatched right bars: hierarchical.",
        ),
        (
            "scope",
            "Explicitly requested illustrative synthetic gap calculation."
            if illustrative
            else (
                "Gap bars use the executed reports. Other panels are seeded diagnostic "
                "posterior calculations. No universal hierarchical benefit or calibrated "
                "uncertainty is established."
            ),
        ),
    )
    for identifier, text in notes:
        panels.append(_text_panel(identifier, text))
        expected.append(identifier)
    return save_presentation_panels(panels, canonical_path=path, expected_identifiers=expected)
