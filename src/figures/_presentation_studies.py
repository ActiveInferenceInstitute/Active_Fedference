"""Presentation panels for source-reported empirical study contrasts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from ._common import COLOR_PANEL_FAIL, SemanticStyle, semantic_style
from ._presentation import PresentationPanel, save_presentation_panels
from ._presentation_diagnostics import _axes
from ._presentation_flows import _text_panel, _text_panels


def sweep_presentation(
    path: Path,
    rates: Sequence[float],
    series: Mapping[str, Sequence[float]],
    intervals: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    styles: Mapping[str, SemanticStyle],
    labels: Mapping[str, str],
    threshold: float | None,
    n_trials: int | None,
    notes: str,
) -> Path:
    """Retain each configured preset and its own trial interval with shared axes."""
    panels: list[PresentationPanel] = []
    expected: list[str] = []
    for index, (method, values) in enumerate(series.items()):
        axis = _axes(labels[method], "Contamination rate", "q(true)")
        axis.xaxis.set_major_locator(MaxNLocator(3))
        style = styles[method]
        options: dict[str, Any] = dict(
            color=style.color,
            marker=style.marker,
            linestyle=style.dash,
            markerfacecolor=style.color if method == "KLD" else "white",
            markeredgecolor=style.keyline,
            linewidth=3,
            markersize=8,
        )
        if method in intervals:
            low, high = intervals[method]
            axis.errorbar(
                rates,
                values,
                yerr=[np.asarray(values) - low, np.asarray(high) - values],
                capsize=5,
                **options,
            )
        else:
            axis.plot(rates, values, **options)
        if threshold is not None:
            rule = semantic_style("reference_rule")
            axis.axhspan(0, threshold, color=COLOR_PANEL_FAIL, alpha=0.45)
            axis.axhline(threshold, color=rule.color, linestyle=rule.dash, linewidth=2)
        axis.set_ylim(-0.10, 1.10)
        span = max(max(rates) - min(rates), 0.1)
        axis.set_xlim(min(rates) - 0.04 * span, max(rates) + 0.04 * span)
        identifier = f"preset-{index + 1}"
        expected.append(identifier)
        panels.append(
            PresentationPanel(
                identifier,
                cast(Figure, axis.figure),
                f"{labels[method]}: all configured contamination rates and source-owned "
                f"uncertainty on a common probability scale.",
            )
        )
    cards = (
        (
            "uncertainty",
            f"{n_trials} matched trials per rate. Whiskers: each preset's own percentile-bootstrap interval."
            if n_trials is not None
            else (
                "Deterministic single-colony curves. No uncertainty interval supplied; "
                "comparative statistics remain in the tables."
            ),
        ),
        (
            "threshold",
            f"Predeclared floor: {threshold}. The pale band lies below this reference."
            if threshold is not None
            else "No predeclared accuracy floor supplied.",
        ),
        ("statistics", notes),
        (
            "scope",
            "Consensus accuracy here means q(true state). These configured server presets are "
            "not client-loss implementations or a complete source protocol.",
        ),
    )
    for identifier, text in cards:
        panels.append(_text_panel(identifier, text))
        expected.append(identifier)
    return save_presentation_panels(panels, canonical_path=path, expected_identifiers=expected)


def onset_presentation(path: Path, by_kind: Mapping[str, Any]) -> Path:
    """Preserve both displayed curves, distinct seed bands, onset and veto markers."""
    panels: list[PresentationPanel] = []
    expected: list[str] = []
    for index, (kind, cell) in enumerate(by_kind.items()):
        rates = np.asarray(cell["rates"], dtype=float)
        naive = np.asarray(cell["naive_curve"], dtype=float)
        robust = np.asarray(cell["robust_curve"], dtype=float)
        naive_ci = np.asarray(cell.get("naive_ci", []), dtype=float)
        robust_ci = np.asarray(cell.get("robust_ci", []), dtype=float)
        has_ci = naive_ci.shape == (rates.size, 2) and robust_ci.shape == (rates.size, 2)
        axis = _axes(kind.replace("_", " "), "Contamination rate", "q(true)")
        axis.xaxis.set_major_locator(MaxNLocator(3))
        for values, ci, role in ((naive, naive_ci, "naive"), (robust, robust_ci, "heuristic_robust")):
            style = semantic_style(role)
            axis.plot(
                rates,
                values,
                color=style.color,
                marker=style.marker,
                linestyle=style.dash,
                markerfacecolor=style.color if role == "naive" else "white",
                markeredgecolor=style.keyline,
                markersize=8,
                linewidth=3,
            )
            if has_ci:
                axis.fill_between(rates, ci[:, 0], ci[:, 1], color=style.color, alpha=0.10)
        onset = cell.get("onset_rate")
        if onset is not None:
            rule = semantic_style("reference_rule")
            axis.axvline(onset, color=rule.color, linestyle=rule.dash, linewidth=2)
            axis.axvspan(
                onset,
                min(float(rates.max()), float(onset) + 0.035),
                color=semantic_style("heuristic_robust").color,
                alpha=0.07,
            )
        if kind == "byzantine":
            axis.annotate(
                "",
                xy=(rates[-2], robust[-2]),
                xytext=(rates[-2] - 0.1, min(0.9, robust[-2] + 0.2)),
                arrowprops={"arrowstyle": "->", "color": semantic_style("reference_rule").color, "lw": 2},
            )
        axis.set_xlim(rates.min() - 0.03, rates.max() + 0.03)
        axis.set_ylim(-0.10, 1.10)
        identifier = f"mechanism-{index + 1}"
        panels.append(
            PresentationPanel(
                identifier,
                cast(Figure, axis.figure),
                f"{kind}: complete reference and pooled display-preset curves with their "
                f"source-supplied seed bands and onset marker.",
            )
        )
        expected.append(identifier)
        identifier = f"mechanism-{index + 1}-notes"
        uncertainty = "Each curve has its own 95% seed-bootstrap band." if has_ci else "No bands supplied."
        onset_text = f"Onset: {float(onset):g}." if onset is not None else "No reliable onset."
        panels.append(
            _text_panel(
                identifier,
                f"{kind.replace('_', ' ')}. {onset_text} Gap at maximum rate: "
                f"{robust[-1] - naive[-1]:+.3f}. {uncertainty}",
            )
        )
        expected.append(identifier)
        if kind == "byzantine":
            panels.append(
                _text_panel(
                    "veto-cliff",
                    "Arrow: declared byzantine veto cliff near the penultimate rate. The onset "
                    "window does not establish universal robustness.",
                )
            )
            expected.append("veto-cliff")
    panels.append(
        _text_panel(
            "key",
            "Filled circles / solid: reference log pool. Open squares / dashed: pooled display "
            "server preset. This selected display curve is not a fixed-preset confirmatory contrast.",
        )
    )
    expected.append("key")
    return save_presentation_panels(panels, canonical_path=path, expected_identifiers=expected)


def quality_presentation(path: Path, controls: Mapping[str, Any]) -> Path:
    """Keep all proper-score controls and finite reliability bins with fixed scales."""
    from .belief_quality import _CONTROL_ROLES

    names = ("oracle", "uniform", "confident_wrong")
    score_low = min(float(controls[name]["log_score_ci"][0]) for name in names)
    score_high = max(0.0, max(float(controls[name]["log_score_ci"][1]) for name in names))
    span = max(score_high - score_low, 1e-6)
    panels: list[PresentationPanel] = []
    expected: list[str] = []
    for name in names:
        title = name.replace("_", " ")
        identifier = name.replace("_", "-")
        style = semantic_style(_CONTROL_ROLES[name])
        score = float(controls[name]["mean_log_score"])
        low, high = map(float, controls[name]["log_score_ci"])
        axis = _axes(title, "Control", "Log score")
        axis.errorbar(
            [0],
            [score],
            yerr=[[score - low], [high - score]],
            fmt=style.marker,
            markerfacecolor=style.color if name == "confident_wrong" else "white",
            color=style.color,
            markeredgecolor=style.keyline,
            markersize=10,
            capsize=6,
            linewidth=3,
        )
        rule = semantic_style("reference_rule")
        axis.axhline(0, color=rule.color, linestyle=rule.dash, linewidth=2)
        axis.set_ylim(score_low - 0.08 * span, score_high + 0.08 * span)
        axis.set_xticks(())
        panels.append(
            PresentationPanel(
                f"{identifier}-score",
                cast(Figure, axis.figure),
                f"{title}: mean categorical log score and its seed-bootstrap interval on the "
                f"common nats scale.",
            )
        )
        expected.append(f"{identifier}-score")
        curve = controls[name]["reliability"]
        confidence = np.asarray(curve["mean_confidence"], dtype=float)
        accuracy = np.asarray(curve["accuracy"], dtype=float)
        mask = np.isfinite(confidence) & np.isfinite(accuracy)
        axis = _axes(title, "Mean confidence", "Accuracy")
        axis.xaxis.set_major_locator(MaxNLocator(3))
        axis.plot(
            confidence[mask],
            accuracy[mask],
            color=style.color,
            marker=style.marker,
            linestyle=style.dash,
            markerfacecolor=style.color if name == "confident_wrong" else "white",
            markeredgecolor=style.keyline,
            linewidth=3,
            markersize=9,
        )
        axis.plot([0, 1], [0, 1], color=rule.color, linestyle=rule.dash, linewidth=2)
        axis.set_xlim(-0.04, 1.04)
        axis.set_ylim(-0.10, 1.10)
        panels.append(
            PresentationPanel(
                f"{identifier}-reliability",
                cast(Figure, axis.figure),
                f"{title}: every finite reliability-bin confidence and accuracy, with the "
                f"perfect-calibration identity reference.",
            )
        )
        expected.append(f"{identifier}-reliability")
    notes = (
        (
            "score-key",
            "Categorical log score in nats: higher is better. Whiskers are 95% seed-bootstrap "
            "intervals; seed is the independent unit.",
        ),
        (
            "reliability-key",
            "Reliability is a descriptive control diagnostic. Every finite bin is drawn on "
            "common 0–1 axes. The dark dotted identity marks perfect calibration.",
        ),
        (
            "scope",
            "Oracle, uniform and confident-wrong controls check score and reliability behavior. "
            "They do not establish deployment calibration or universal robustness.",
        ),
    )
    for identifier, text in notes:
        panels.append(_text_panel(identifier, text))
        expected.append(identifier)
    return save_presentation_panels(panels, canonical_path=path, expected_identifiers=expected)


def gallery_presentation(path: Path, by_kind: Mapping[str, Any]) -> Path:
    """Preserve every mechanism, selected-preset flag and method-specific interval."""
    from .contamination_gallery import _robust_facecolors

    have_ci = all("naive_ci" in cell and "robust_ci" in cell for cell in by_kind.values())
    panels: list[PresentationPanel] = []
    expected: list[str] = []
    reference = semantic_style("naive")
    robust = semantic_style("heuristic_robust")
    for index, (kind, cell) in enumerate(by_kind.items()):
        flagged = bool(cell.get("reliably_beats", False))
        axis = _axes(kind.replace("_", " "), "Method", "q(true)")
        axis.bar(
            0,
            cell["naive_mean"],
            width=0.6,
            facecolor="white",
            edgecolor=reference.keyline,
            hatch=reference.hatch,
        )
        axis.bar(
            1,
            cell["robust_mean"],
            width=0.6,
            facecolor=_robust_facecolors([flagged], robust.color)[0],
            edgecolor=robust.keyline,
            hatch=robust.hatch if flagged else "..",
            linewidth=1.6 if flagged else 1,
        )
        if have_ci:
            for x, role in ((0, "naive"), (1, "robust")):
                mean = float(cell[f"{role}_mean"])
                low, high = map(float, cell[f"{role}_ci"])
                axis.errorbar(
                    [x],
                    [mean],
                    yerr=[[mean - low], [high - mean]],
                    fmt="none",
                    ecolor=reference.keyline,
                    capsize=5,
                    linewidth=2,
                )
        axis.set_ylim(0, 1.1)
        axis.set_xticks([0, 1], ["Reference", "Preset"])
        identifier = f"mechanism-{index + 1}"
        panels.append(
            PresentationPanel(
                identifier,
                cast(Figure, axis.figure),
                f"{kind}: reference versus selected server preset, its own source interval, and "
                f"the declared filled or dotted display flag.",
            )
        )
        expected.append(identifier)
        identifier = f"mechanism-{index + 1}-selection"
        mark = "flagged" if flagged else "below bar"
        win = float(cell.get("win_fraction", float("nan")))
        win_text = f"{win:.2f}" if np.isfinite(win) else "unavailable"
        panels.append(
            _text_panel(
                identifier,
                f"{kind.replace('_', ' ')}: selected preset "
                f"{cell.get('best_robust_method', 'robust')}; descriptive win fraction "
                f"{win_text}; {mark}.",
            )
        )
        expected.append(identifier)
    notes = (
        (
            "key",
            "Open reference bars retain their hatch. Selected presets use full fill when "
            "flagged and pale dotted fill below the display bar. Color is not the sole flag encoding.",
        ),
        (
            "uncertainty",
            "Each bar has its own 95% seed-bootstrap interval."
            if have_ci
            else "Legacy source supplies no complete interval pair; no error bars are inferred.",
        ),
        (
            "scope",
            f"Display flags: "
            f"{sum(bool(cell.get('reliably_beats', False)) for cell in by_kind.values())}/{len(by_kind)}. "
            "Win fractions are descriptive, not p-values. Selected presets do not define "
            "fixed-preset confirmatory contrasts.",
        ),
    )
    for identifier, text in notes:
        panels.append(_text_panel(identifier, text))
        expected.append(identifier)
    return save_presentation_panels(panels, canonical_path=path, expected_identifiers=expected)


def emergence_presentation(path: Path, redundant: float, supported: float, convergence: bool | None) -> Path:
    """Keep both signed model-reduction controls and their supplied disposition."""
    from ._common import COLOR_MUTED

    axis = _axes("Configured BMR sign control", "Pruned target", "ΔF (nats)")
    favored = semantic_style("favored")
    rejected = semantic_style("rejected")
    axis.bar(
        [0, 1],
        [redundant, supported],
        width=0.55,
        color=[favored.color, COLOR_MUTED],
        edgecolor=[favored.keyline, rejected.keyline],
        hatch=[favored.hatch, rejected.hatch],
    )
    reference = semantic_style("reference_rule")
    axis.axhline(0, color=reference.color, linestyle=reference.dash, linewidth=2)
    low = min(0, redundant, supported)
    high = max(0, redundant, supported)
    span = max(high - low, 1)
    axis.set_ylim(low - 0.15 * span, high + 0.15 * span)
    axis.set_xticks([0, 1], ["Redundant", "Supported"])
    panels = [
        PresentationPanel(
            "controls",
            cast(Figure, axis.figure),
            "Both signed model-reduction gains on one nats scale; hatches distinguish targets "
            "and a dotted rule marks zero.",
        ),
        _text_panel(
            "values",
            f"Redundant target ΔF: {redundant:.4g} nats. Supported target ΔF: {supported:.4g} "
            f"nats. Positive gain favors the reduction; negative gain rejects it.",
        ),
        _text_panel(
            "verdict",
            f"Source configured sign-control verdict: "
            f"{convergence if convergence is not None else 'not supplied'}. This is not an "
            f"optimizer convergence diagnostic.",
        ),
        _text_panel(
            "scope",
            "One fixed-posterior categorical diagnostic. No independent stochastic replication "
            "or uncertainty interval applies. Source-mechanism analogue, not full protocol "
            "reconstruction.",
        ),
    ]
    return save_presentation_panels(
        panels, canonical_path=path, expected_identifiers=("controls", "values", "verdict", "scope")
    )


def bnn_presentation(
    path: Path,
    curves: Mapping[str, Sequence[float]],
    levels: Sequence[float],
    intervals: Mapping[str, Sequence[Sequence[float]]] | None,
    disclosure: str,
    peak_note: str,
) -> Path:
    """Preserve every composite proxy curve and its supplied pointwise interval."""
    from .bnn_robustness import _configuration_style_role, _direct_condition_label

    panels: list[PresentationPanel] = []
    expected: list[str] = []
    comparison_index = 0
    for index, (label, curve) in enumerate(curves.items()):
        role, is_reference = _configuration_style_role(label, comparison_index)
        if not is_reference:
            comparison_index += 1
        style = semantic_style(role)
        axis = _axes(_direct_condition_label(label), "Label contamination", "Accuracy")
        axis.xaxis.set_major_locator(MaxNLocator(3))
        axis.plot(
            levels,
            curve,
            color=style.color,
            marker=style.marker,
            linestyle=style.dash,
            markeredgecolor=style.keyline,
            markerfacecolor=style.color if is_reference else "white",
            linewidth=3,
            markersize=8,
        )
        if intervals is not None and label in intervals:
            bounds = np.asarray(intervals[label], dtype=float)
            axis.fill_between(levels, bounds[:, 0], bounds[:, 1], color=style.color, alpha=0.14)
        axis.set_ylim(-0.03, 1.05)
        span = max(max(levels) - min(levels), 0.1)
        axis.set_xlim(min(levels) - 0.04 * span, max(levels) + 0.04 * span)
        identifier = f"configuration-{index + 1}"
        panels.append(
            PresentationPanel(
                identifier,
                cast(Figure, axis.figure),
                f"{label}: complete held-out accuracy curve and supplied interval band on shared axes.",
            )
        )
        panels.append(_text_panel(f"{identifier}-identity", label))
        expected.extend((identifier, f"{identifier}-identity"))
    notes = (
        (
            "composite",
            "Composite NLL/L2=0.05 versus RCCE/L2=0.10 point-estimate proxy. The joint change "
            "cannot identify an RCCE-only effect.",
        ),
        ("selection", disclosure),
        ("peak", peak_note),
        (
            "uncertainty",
            "Shaded bands retain the source-supplied intervals. They do not represent FedGVI "
            "posterior uncertainty."
            if intervals
            else "No intervals supplied; none are inferred.",
        ),
        (
            "scope",
            "Exploratory proxy only: no Alpha-Renyi objective, calibration, "
            "universal robustness, or exact source-protocol replication.",
        ),
    )
    for identifier, text in notes:
        note_panels = _text_panels(identifier, text)
        panels.extend(note_panels)
        expected.extend(panel.identifier for panel in note_panels)
    return save_presentation_panels(panels, canonical_path=path, expected_identifiers=expected)


def cross_study_presentation(
    path: Path, grouped: Mapping[str, Sequence[Mapping[str, Any]]], n_seeds: int
) -> Path:
    """Retain every study's native-unit estimand without a cross-unit ranking."""
    from ._common import COLOR_DEEP, COLOR_MUTED, COLOR_NAIVE, COLOR_ROBUST
    from .cross_study_summary import _UNIT_ORDER, _display_label

    panels: list[PresentationPanel] = []
    expected: list[str] = []
    for unit in _UNIT_ORDER:
        entries = grouped[unit]
        low = min(0.0, min(float(entry["ci_lo"]) for entry in entries))
        high = max(0.0, max(float(entry["ci_hi"]) for entry in entries))
        span = max(high - low, 1e-6)
        for index, entry in enumerate(entries):
            value = float(entry["mean"])
            lo = float(entry["ci_lo"])
            hi = float(entry["ci_hi"])
            axis = _axes(f"Study {entry['study']}", "R²" if unit == "R-sq" else f"Contrast ({unit})", "")
            axis.xaxis.set_major_locator(MaxNLocator(3))
            color = COLOR_ROBUST if value > 1e-3 else COLOR_NAIVE if value < -1e-3 else COLOR_MUTED
            hatch = "///" if value > 1e-3 else "xxx" if value < -1e-3 else ".."
            axis.barh(
                [0],
                [value],
                xerr=[[value - lo], [hi - value]],
                color=color,
                edgecolor=COLOR_DEEP,
                hatch=hatch,
                error_kw={"elinewidth": 2, "capsize": 5, "ecolor": COLOR_DEEP},
            )
            axis.axvline(0, color=COLOR_DEEP, linestyle=":", linewidth=2)
            axis.set_xlim(low - 0.1 * span, high + 0.1 * span)
            axis.set_ylim(-1, 1)
            axis.set_yticks(())
            identifier = f"{unit.lower().replace('-', '')}-study-{index + 1}"
            panels.append(
                PresentationPanel(
                    identifier,
                    cast(Figure, axis.figure),
                    f"{_display_label(dict(entry))}: mean and seed-bootstrap interval on the "
                    f"shared {unit} facet scale.",
                )
            )
            label = " ".join(_display_label(dict(entry)).splitlines())
            panels.append(
                _text_panel(
                    f"{identifier}-values",
                    f"{label}. Mean {value:+.4g}; 95% interval [{lo:+.4g}, {hi:+.4g}] {unit}.",
                )
            )
            expected.extend((identifier, f"{identifier}-values"))
    notes = (
        (
            "uncertainty",
            f"Separate harmonized seed-level rerun: n = {n_seeds} independent seeds. Whiskers "
            f"are 95% percentile-bootstrap intervals.",
        ),
        (
            "direction",
            "/// positive; xxx negative; dots within 0.001 of zero. Hatches describe the mean "
            "sign, not statistical significance. Direction remains metric-specific.",
        ),
        (
            "scope",
            "Native-unit facets stay separate. R² is a fit statistic, not an effect contrast. "
            "No cross-unit ranking; Study 4 is a within-run display-selected maximum.",
        ),
    )
    for identifier, text in notes:
        panels.append(_text_panel(identifier, text))
        expected.append(identifier)
    return save_presentation_panels(panels, canonical_path=path, expected_identifiers=expected)


def hierarchical_bmr_presentation(
    path: Path,
    degenerate: Mapping[str, Any],
    informative: Mapping[str, Any],
    tolerance: float,
    decision_rule: str | None,
) -> Path:
    """Retain each level, both configured worlds and source-owned prune flags."""
    levels_a = degenerate["levels"]
    levels_b = informative["levels"]
    upper = max([tolerance] + [max(float(row["bayesian_surprise"]), 0) for row in [*levels_a, *levels_b]])
    panels: list[PresentationPanel] = []
    expected: list[str] = []
    for index, (row_a, row_b) in enumerate(zip(levels_a, levels_b, strict=True)):
        axis = _axes(f"Level {row_b['level']}", "Surprise (nats)", "")
        axis.xaxis.set_major_locator(MaxNLocator(3))
        for y, row, role in ((0, row_a, "condition_reference"), (1, row_b, "condition_comparison")):
            style = semantic_style(role)
            axis.barh(
                y,
                max(float(row["bayesian_surprise"]), 0),
                height=0.5,
                color=style.color,
                edgecolor=style.keyline,
                hatch=style.hatch,
            )
        rule = semantic_style("reference_rule")
        axis.axvline(tolerance, color=rule.color, linestyle=rule.dash, linewidth=2)
        axis.set_xlim(0, upper * 1.2 if upper else 1)
        axis.set_ylim(-0.5, 1.5)
        axis.set_yticks(())
        identifier = f"level-{index + 1}"
        panels.append(
            PresentationPanel(
                identifier,
                cast(Figure, axis.figure),
                "Paired Bayesian-surprise values: lower reference-world bar and upper "
                "informative-world bar, with the configured threshold.",
            )
        )
        expected.append(identifier)
        for suffix, row in (("non-gating", row_a), ("informative", row_b)):
            identifier = f"level-{index + 1}-{suffix}"
            flag = "prunable" if row["prunable"] else "kept"
            panels.append(
                _text_panel(
                    identifier,
                    f"{suffix} world: L{row['level']} {row['label']}. Bayesian surprise "
                    f"{float(row['bayesian_surprise']):.4g} nats; source flag: {flag}.",
                )
            )
            expected.append(identifier)
    notes = (
        (
            "threshold",
            f"Declared surprise threshold: {tolerance:g} nats. "
            + (decision_rule or "Values at or below the threshold are flagged prunable."),
        ),
        (
            "key",
            "Lower bar: non-gating configured world. Upper hatched bar: informative configured "
            "world. All levels share the same surprise scale.",
        ),
        (
            "scope",
            "Bayesian surprise KL(posterior || empirical prior). This configured "
            "information-gating diagnostic has no stochastic interval and is distinct from "
            "exact BMR evidence gain.",
        ),
    )
    for identifier, text in notes:
        panels.append(_text_panel(identifier, text))
        expected.append(identifier)
    return save_presentation_panels(panels, canonical_path=path, expected_identifiers=expected)


def heuristic_presentation(path: Path, report: Mapping[str, Any]) -> Path:
    """Retain influence, finite-search capture, missing capture and complete attack coverage."""
    naive = report["influence_naive"]
    robust = report["influence_robust"]
    breakdown = report["breakdown"]
    axis = _axes("Numerical influence", "Perturbation fraction", "Weight")
    axis.xaxis.set_major_locator(MaxNLocator(3))
    arrays = []
    for row, role in ((naive, "naive"), (robust, "heuristic_robust")):
        values = row.get("normalized_effective_weights", row.get("agent_weight"))
        arrays.append(np.asarray(values, dtype=float))
        style = semantic_style(role)
        axis.plot(
            naive["eps_grid"],
            values,
            marker=style.marker,
            linestyle=style.dash,
            color=style.color,
            markeredgecolor=style.keyline,
            linewidth=3,
            markersize=8,
        )
    rule = semantic_style("reference_rule")
    axis.axhline(1 / naive["n_agents"], color=rule.color, linestyle=rule.dash, linewidth=2)
    axis.margins(0.05, 0.18)
    panels = [
        PresentationPanel(
            "influence",
            cast(Figure, axis.figure),
            "Complete numerical influence paths for the reference and heuristic with the "
            "equal-weight reference.",
        )
    ]
    expected = ["influence"]
    panels.append(
        _text_panel(
            "influence-drop",
            f"Endpoint weight drop: reference minus heuristic = "
            f"{arrays[0][-1] - arrays[1][-1]:.4g}. Empirical at these settings, not a guarantee.",
        )
    )
    expected.append("influence-drop")
    observed = [breakdown["robust_breakdown_k"], breakdown["variational_breakdown_k"]]
    upper = max([float(value) for value in observed if value is not None] + [0]) + 1.4
    for name, value, role in zip(
        ("Heuristic", "Variational"), observed, ("heuristic_robust", "variational"), strict=True
    ):
        identifier = name.lower()
        if value is None:
            panels.append(
                _text_panel(
                    identifier,
                    f"{name}: no argmax capture observed in the configured finite search. This "
                    f"is not a zero-adversary breakdown point.",
                )
            )
        else:
            axis = _axes(name, "Aggregator", "Count k")
            style = semantic_style(role)
            axis.bar(
                [0],
                [value],
                color=style.color,
                edgecolor=style.keyline,
                hatch="//" if name == "Heuristic" else "..",
            )
            axis.set_ylim(0, upper)
            axis.set_xticks(())
            panels.append(
                PresentationPanel(
                    identifier,
                    cast(Figure, axis.figure),
                    f"{name}: observed finite-search argmax capture at k={value} adversaries.",
                )
            )
        expected.append(identifier)
        panels.append(
            _text_panel(
                f"{identifier}-capture",
                f"{name}: "
                + (f"argmax captured at k={value}." if value is not None else "capture not observed."),
            )
        )
        expected.append(f"{identifier}-capture")
    grid = report.get("grid")
    if grid:
        for index, attack in enumerate(grid["parameter_grid"]["attacks"]):
            rows = [row for row in grid["rows"] if row["attack"] == attack]
            finite = sum(row["robust_breakdown_k"] is not None for row in rows)
            fraction = finite / len(rows) if rows else 0
            axis = _axes(attack.replace("_", " "), "Declared grid", "Fraction")
            axis.bar([0], [fraction], color=semantic_style("heuristic_robust").color, edgecolor=rule.color)
            axis.axhline(0.5, color=rule.color, linestyle=":", linewidth=2)
            axis.set_ylim(0, 1.12)
            axis.set_xticks(())
            identifier = f"attack-{index + 1}"
            panels.append(
                PresentationPanel(
                    identifier,
                    cast(Figure, axis.figure),
                    f"{attack}: {finite}/{len(rows)} declared rows have finite heuristic capture "
                    f"({fraction:.0%}).",
                )
            )
            panels.append(
                _text_panel(
                    f"{identifier}-count",
                    f"{attack.replace('_', ' ')}: {finite}/{len(rows)} rows with finite capture "
                    f"= {fraction:.0%}.",
                )
            )
            expected.extend((identifier, f"{identifier}-count"))
    panels.append(
        _text_panel(
            "scope",
            "Finite-search capture and grid frequencies describe the declared settings. They "
            "are not global breakdown probabilities or universal robustness guarantees.",
        )
    )
    expected.append("scope")
    return save_presentation_panels(panels, canonical_path=path, expected_identifiers=expected)
