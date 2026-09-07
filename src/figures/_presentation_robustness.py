"""Selection-free robustness panels with shared limits and source-owned CIs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Normalize, to_hex
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from analysis.figure_exact_values import conditional_display_summaries

from ._common import (
    COLOR_DEEP,
    COLOR_WHITE,
    apply_style,
    contrasting_text_color,
    semantic_style,
    signed_difference_colormap,
)
from ._presentation import PresentationPanel, save_presentation_panels
from ._presentation_flows import _text_panel


def _canvas() -> Figure:
    apply_style()
    fig = Figure(figsize=(7.4, 3.0), facecolor=COLOR_WHITE)
    FigureCanvasAgg(fig)
    fig.set_layout_engine("none")
    return fig


def generate_review_grid_presentation(report: Mapping[str, object], canonical_path: Path) -> Path:
    """Render every conditional cell and every predeclared method-rate series.

    Each method keeps its own seed-bootstrap interval. The limits are computed
    over all methods within a mechanism, matching the canonical plot's shared
    reference frame; no winner selection or interval pooling occurs here.
    """
    from analysis.report_schemas import validate_report

    from .robustness_review_grid import (
        _PRESET_STYLE_ROLES,
        _as_finite_number_list,
        _as_mapping,
        _as_string_list,
    )

    validate_report("robustness_review_grid", report)
    conditional = _as_mapping(report["conditional_world"], label="conditional_world")
    summaries = conditional_display_summaries(_as_mapping(conditional["by_scenario"], label="cells"))
    attacks = sorted({summary["attack"] for summary in summaries})
    weights = (0.5, 1.0)
    cells = {(summary["attack"], summary["adversary_weight"]): summary for summary in summaries}
    maximum = max(max(abs(summary["mean_contrast"]) for summary in summaries), 1e-6)
    cmap = signed_difference_colormap()
    norm = Normalize(-maximum, maximum)
    panels: list[PresentationPanel] = [
        _text_panel(
            "conditional-key",
            "Conditional preset-minus-reference means\n"
            "± half min–max span across declared cells; not a confidence interval.",
        )
    ]
    for attack in attacks:
        fig = _canvas()
        axis = fig.add_axes((0.06, 0.29, 0.90, 0.43))
        values = np.asarray([[cells[(attack, weight)]["mean_contrast"] for weight in weights]])
        axis.imshow(values, cmap=cmap, norm=norm, aspect="auto")
        axis.set_yticks([])
        axis.set_xticks([0, 1], ["Half weight", "Full weight"], fontsize=30)
        axis.grid(False)
        for col, weight in enumerate(weights):
            value = cells[(attack, weight)]["mean_contrast"]
            span = cells[(attack, weight)]["half_min_max_span"]
            background = to_hex(cmap(norm(value)))
            axis.text(
                col,
                0,
                f"{value:+.3f}\n±{span:.3f}",
                ha="center",
                va="center",
                fontsize=30,
                color=contrasting_text_color(background),
            )
        fig.text(0.5, 0.96, attack.replace("_", " "), ha="center", va="top", fontsize=30, color=COLOR_DEEP)
        panels.append(
            PresentationPanel(
                f"conditional-{attack}".replace("_", "-"),
                fig,
                f"{attack.replace('_', ' ')}: both adversary weights; "
                "signed mass means and half min–max spans; common color scale.",
            )
        )
    fig = _canvas()
    axis = fig.add_axes((0.1, 0.42, 0.8, 0.20))
    from matplotlib.cm import ScalarMappable

    fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        cax=axis,
        orientation="horizontal",
        ticks=[-maximum, 0, maximum],
        format="%+.3f",
    )
    axis.tick_params(labelsize=30)
    fig.text(
        0.5, 0.93, "Conditional mean mass contrast", ha="center", va="top", fontsize=30, color=COLOR_DEEP
    )
    panels.append(
        PresentationPanel(
            "conditional-scale",
            fig,
            "Common signed color scale for every conditional panel; zero is neutral.",
        )
    )
    panels.append(
        _text_panel(
            "directional-key",
            "Each preset retains its own 95% seed-bootstrap band.\n"
            "Y limits are shared within each mechanism. All predeclared presets are shown.",
        )
    )
    kinds = _as_string_list(report["directional_mechanisms"], label="mechanisms")
    methods = [
        method for method in _as_string_list(report["divergences"], label="methods") if method != "KLD"
    ]
    rates = np.asarray(_as_finite_number_list(report["rates"], label="rates"))
    statistics = _as_mapping(report["statistics"], label="statistics")
    by_kind = _as_mapping(statistics["by_mechanism"], label="mechanisms")
    for kind in kinds:
        by_rate = cast(Mapping[str, object], _as_mapping(by_kind[kind], label=kind)["by_rate"])
        series: list[tuple[list[float], list[float], list[float]]] = []
        for method in methods:
            means: list[float] = []
            lower: list[float] = []
            upper: list[float] = []
            for rate in rates:
                row = cast(dict, by_rate[f"{float(rate):g}"])["methods"][method]
                means.append(float(row["summary"]["mean"]))
                lower.append(float(row["contrast_ci"][0]))
                upper.append(float(row["contrast_ci"][1]))
            series.append((means, lower, upper))
        lo = min(0.0, *(value for _, lower, _ in series for value in lower))
        hi = max(0.0, *(value for _, _, upper in series for value in upper))
        span = max(hi - lo, 0.10)
        limits = (lo - 0.12 * span, hi + 0.12 * span)
        for index, (method, (means, lower, upper)) in enumerate(zip(methods, series, strict=True)):
            fig = _canvas()
            axis = fig.add_axes((0.25, 0.32, 0.71, 0.42))
            style = semantic_style(_PRESET_STYLE_ROLES[index % len(_PRESET_STYLE_ROLES)])
            axis.fill_between(rates, lower, upper, color=style.color, alpha=0.13, linewidth=0)
            axis.plot(
                rates,
                means,
                marker=style.marker,
                markerfacecolor="white",
                markeredgecolor=style.keyline,
                linestyle=style.dash,
                linewidth=3,
                markersize=8,
                color=style.color,
            )
            reference = semantic_style("reference_rule")
            axis.axhline(0, color=reference.color, linestyle=reference.dash, linewidth=2)
            axis.set_ylim(*limits)
            padding = max(float(np.ptp(rates)), 0.10) * 0.025
            axis.set_xlim(float(rates.min()) - padding, float(rates.max()) + padding)
            axis.xaxis.set_major_locator(MaxNLocator(3))
            axis.yaxis.set_major_locator(MaxNLocator(2))
            axis.yaxis.set_major_formatter(FormatStrFormatter("%+.2f"))
            axis.tick_params(labelsize=30)
            axis.set_xlabel("Contamination rate", fontsize=30)
            fig.text(
                0.5,
                0.98,
                f"{kind.replace('_', ' ')} · preset {index + 1}",
                ha="center",
                va="top",
                fontsize=30,
                color=COLOR_DEEP,
            )
            fig.text(0.02, 0.55, "Δ mass", rotation=90, ha="left", va="center", fontsize=30, color=COLOR_DEEP)
            panels.append(
                PresentationPanel(
                    f"directional-{kind}-{method}".lower().replace("_", "-"),
                    fig,
                    f"{kind.replace('_', ' ')}, preset {index + 1} ({method}): "
                    "signed mass contrast and its own 95% seed-bootstrap band; shared mechanism limits.",
                )
            )
    identifiers = (
        "conditional-key",
        *(f"conditional-{attack}".replace("_", "-") for attack in attacks),
        "conditional-scale",
        "directional-key",
        *(f"directional-{kind}-{method}".lower().replace("_", "-") for kind in kinds for method in methods),
    )
    return save_presentation_panels(panels, canonical_path=canonical_path, expected_identifiers=identifiers)
