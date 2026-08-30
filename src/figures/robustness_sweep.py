"""Robustness-sweep figure: consensus accuracy vs contamination rate.

Draws the contamination-rate sweep of the robustness experiment (ISC-27/30):
for each configured server operating point, consensus accuracy ``q(true_state)`` as a
function of the contamination rate. The naive ``KLD`` project pool
(:func:`fedference.aggregation.log_linear_pool`) is a qualified categorical
specialization of Friston et al. (2024) Eq. 7's message-combination term; it is
not the complete source protocol. It degrades monotonically under this declared
contamination setting, while the server-side ``robust_aggregate`` heuristic can
separate under declared mechanisms. The accuracy grid is computed by
:func:`fedference.experiments.run_robustness_sweep`; this module only draws.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ._common import (
    COLOR_GRID,
    COLOR_PANEL_FAIL,
    SemanticStyle,
    apply_style,
    figures_dir,
    plt,
    save_figure,
    semantic_style,
)

_ROBUST_STYLE_ROLES = (
    "heuristic_robust",
    "operating_point_1",
    "operating_point_2",
    "operating_point_3",
    "operating_point_4",
)


def generate_robustness_sweep(
    accuracy_by_method_and_rate: Mapping[str, Mapping[str, float]],
    rates: Sequence[float],
    *,
    accuracy_threshold: float | None = None,
    rate_summary: Mapping[str, Mapping[str, Any]] | None = None,
    server_robustness_by_label: Mapping[str, float] | None = None,
    project_root: Path | None = None,
    filename: str = "robustness_sweep.png",
) -> Path:
    """Render consensus-accuracy curves over the contamination-rate sweep.

    Args:
        accuracy_by_method_and_rate: Nested ``{method: {rate_key: accuracy}}``
            mapping, where ``rate_key`` is ``f"{rate:g}"`` (the experiment's
            JSON key convention).
        rates: The contamination rates, in sweep order.
        accuracy_threshold: Optional horizontal reference line.
        rate_summary: Optional trial-level summary from
            ``run_robustness_sweep['per_rate_summary']``. When supplied, the
            plotted curves are trial means with percentile-bootstrap intervals;
            otherwise deterministic single-colony mechanistic curves are drawn.
        server_robustness_by_label: Optional source-report mapping from legacy
            cross-reference labels to the actual server robustness constants.
            When present, visible labels use these constants rather than
            client-loss vocabulary.
        project_root: Project root override.
        filename: Output PNG name under ``output/figures``.

    Returns:
        Path to the written PNG.

    Raises:
        ValueError: If ``accuracy_by_method_and_rate`` or ``rates`` is empty.
    """
    if not accuracy_by_method_and_rate:
        raise ValueError("accuracy_by_method_and_rate must be non-empty")
    rate_vals = [float(r) for r in rates]
    if not rate_vals:
        raise ValueError("rates must be non-empty")

    apply_style()
    fig, ax = plt.subplots(figsize=(7.4, 5.1))
    fig.subplots_adjust(left=0.13, right=0.96, top=0.86, bottom=0.30)
    robust_idx = 0
    has_profile = rate_summary is not None
    if has_profile and not rate_summary:
        raise ValueError("rate_summary must be non-empty when provided")
    series: dict[str, list[float]] = {}
    intervals: dict[str, tuple[list[float], list[float]]] = {}
    styles: dict[str, SemanticStyle] = {}
    labels: dict[str, str] = {}
    for method, by_rate in accuracy_by_method_and_rate.items():
        if rate_summary is not None:
            profile_blocks = [rate_summary[f"{r:g}"] for r in rate_vals]
            profile_methods = [block["methods"] for block in profile_blocks]
            ys = [float(methods[method]["mean"]) for methods in profile_methods]
            lows = [float(methods[method]["ci_lo"]) for methods in profile_methods]
            highs = [float(methods[method]["ci_hi"]) for methods in profile_methods]
            intervals[method] = (lows, highs)
        else:
            ys = [float(by_rate[f"{r:g}"]) for r in rate_vals]
        series[method] = ys
        is_naive = method == "KLD"
        if is_naive:
            style = semantic_style("naive")
            label = "reference log pool (c=0)"
        else:
            style = semantic_style(_ROBUST_STYLE_ROLES[robust_idx % len(_ROBUST_STYLE_ROLES)])
            if server_robustness_by_label is None:
                label = f"server preset {robust_idx + 1}"
            else:
                if method not in server_robustness_by_label:
                    raise ValueError(f"missing server robustness for {method!r}")
                label = f"server preset c={float(server_robustness_by_label[method]):g}"
            robust_idx += 1
        styles[method] = style
        labels[method] = label
        if rate_summary is not None:
            lows, highs = intervals[method]
            ax.errorbar(
                rate_vals,
                ys,
                yerr=[
                    [mean - low for mean, low in zip(ys, lows)],
                    [high - mean for mean, high in zip(ys, highs)],
                ],
                fmt=style.marker,
                markersize=4.5,
                linewidth=style.linewidth,
                linestyle=style.dash,
                color=style.color,
                markeredgecolor=style.keyline,
                markerfacecolor="white" if not is_naive else style.color,
                capsize=2.5,
                elinewidth=0.9,
                label=label,
                zorder=3 if is_naive else 2,
            )
        else:
            ax.plot(
                rate_vals,
                ys,
                marker=style.marker,
                markersize=5,
                linewidth=style.linewidth,
                color=style.color,
                markeredgecolor=style.keyline,
                markerfacecolor="white" if not is_naive else style.color,
                linestyle=style.dash,
                label=label,
                zorder=3 if is_naive else 2,
            )
    if accuracy_threshold is not None:
        threshold = float(accuracy_threshold)
        ax.axhspan(0.0, threshold, color=COLOR_PANEL_FAIL, alpha=0.45, zorder=0)
        ax.axhline(
            threshold,
            color=semantic_style("reference_rule").color,
            linestyle=semantic_style("reference_rule").dash,
            linewidth=semantic_style("reference_rule").linewidth,
            label=f"predeclared floor = {threshold:g}",
        )
        ax.text(
            rate_vals[-1],
            threshold + 0.025,
            "floor",
            ha="right",
            va="bottom",
            fontsize=9.5,
            color=COLOR_GRID,
        )
    # Direct endpoint labels keep method roles readable in grayscale and avoid
    # forcing a legend-only lookup.  Vertically separate near-tied endpoints.
    endpoints = sorted(
        ((float(values[-1]), method) for method, values in series.items()),
        key=lambda item: item[0],
    )
    label_positions: dict[str, float] = {}
    previous = 0.42
    for endpoint, method in endpoints:
        placed = max(endpoint, previous + 0.042)
        label_positions[method] = min(placed, 1.005)
        previous = label_positions[method]
    x_span = max(rate_vals) - min(rate_vals) if len(rate_vals) > 1 else 1.0
    label_x = rate_vals[-1] + 0.035 * x_span
    for method, values in series.items():
        style = styles[method]
        endpoint = float(values[-1])
        placed = label_positions[method]
        ax.plot(
            [rate_vals[-1], label_x],
            [endpoint, placed],
            color=style.keyline,
            linewidth=0.8,
            clip_on=False,
        )
        ax.text(
            label_x,
            placed,
            labels[method],
            ha="left",
            va="center",
            fontsize=9.5,
            color=style.keyline,
            clip_on=False,
        )
    ax.set_xlabel("contamination rate")
    ax.set_ylabel("consensus accuracy  q(true state)")
    # Truncate just below the threshold band: the executed curves live well
    # above zero, so a full [0, 1] range wastes panel area; the floor stays visible.
    minimum_value = min(min(values) for values in series.values())
    ax.set_ylim(max(0.0, min(0.4, minimum_value - 0.05)), 1.02)
    ax.set_xlim(rate_vals[0], label_x + 0.23 * x_span)
    if rate_summary is not None:
        n_profile = int(next(iter(rate_summary.values())).get("n", 0))
        ax.set_title("Server-preset accuracy under contamination")
        final_rate = rate_vals[-1]
        naive_final = series.get("KLD", [float("nan")])[-1]
        best_final = max(
            (ys[-1] for method, ys in series.items() if method != "KLD"),
            default=float("nan"),
        )
        ax.text(
            0.02,
            0.02,
            f"n = {n_profile} matched trials/rate\n"
            f"largest max-rate preset-minus-reference = {best_final - naive_final:+.3f}\n"
            f"max rate = {final_rate:g}",
            transform=ax.transAxes,
            fontsize=9.5,
            ha="left",
            va="bottom",
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": COLOR_GRID, "alpha": 0.85},
        )
    else:
        ax.set_title("Deterministic server-preset contamination sweep")
    ax.legend(fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    if len(rates) > 1 and rate_summary is None:
        # Mid-panel dead space (between the legend and the KLD descent), so the
        # note no longer sits on the curves in the upper-right corner.
        ax.text(
            0.55,
            0.33,
            "single seeded curves\nlinear accuracy axis\ncomparative statistics in tables",
            transform=ax.transAxes,
            fontsize=9.5,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": COLOR_GRID, "alpha": 0.85},
        )

    return save_figure(fig, figures_dir(project_root) / filename)


__all__ = ["generate_robustness_sweep"]
