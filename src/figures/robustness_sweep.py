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
    COLOR_ACCENT,
    COLOR_GRID,
    COLOR_NAIVE,
    COLOR_PANEL_FAIL,
    apply_style,
    figures_dir,
    plt,
    robust_color,
    save_figure,
)


def generate_robustness_sweep(
    accuracy_by_method_and_rate: Mapping[str, Mapping[str, float]],
    rates: Sequence[float],
    *,
    accuracy_threshold: float | None = None,
    rate_summary: Mapping[str, Mapping[str, Any]] | None = None,
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
            color = COLOR_NAIVE
        else:
            color = robust_color(robust_idx)
            robust_idx += 1
        label = f"{method} (naive)" if is_naive else f"{method} (robust)"
        if rate_summary is not None:
            lows, highs = intervals[method]
            ax.errorbar(
                rate_vals,
                ys,
                yerr=[
                    [mean - low for mean, low in zip(ys, lows)],
                    [high - mean for mean, high in zip(ys, highs)],
                ],
                fmt="o-" if is_naive else "s--",
                markersize=4.5,
                linewidth=2.6 if is_naive else 1.7,
                color=color,
                capsize=2.5,
                elinewidth=0.9,
                label=label,
                zorder=3 if is_naive else 2,
            )
        else:
            ax.plot(
                rate_vals,
                ys,
                marker="o" if is_naive else "s",
                markersize=5,
                linewidth=2.6 if is_naive else 1.7,
                color=color,
                linestyle="-" if is_naive else "--",
                label=label,
                zorder=3 if is_naive else 2,
            )
    if accuracy_threshold is not None:
        threshold = float(accuracy_threshold)
        ax.axhspan(0.0, threshold, color=COLOR_PANEL_FAIL, alpha=0.45, zorder=0)
        ax.axhline(
            threshold,
            color=COLOR_GRID,
            linestyle="--",
            linewidth=1.2,
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
    if "KLD" in series:
        ax.annotate(
            "naive log-pool",
            xy=(rate_vals[-1], series["KLD"][-1]),
            xytext=(-72, 24),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "color": COLOR_NAIVE, "lw": 1.0},
            fontsize=9.5,
            color=COLOR_NAIVE,
            ha="right",
        )
    robust_endpoints = {
        method: ys[-1] for method, ys in series.items() if method != "KLD"
    }
    if robust_endpoints:
        best_method = max(robust_endpoints, key=robust_endpoints.__getitem__)
        ax.annotate(
            f"highest max-rate pooled mean (display): {best_method}",
            xy=(rate_vals[-1], series[best_method][-1]),
            xytext=(-120, -28),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "color": COLOR_ACCENT, "lw": 1.0},
            fontsize=9.5,
            color=COLOR_ACCENT,
            ha="right",
        )
    ax.set_xlabel("contamination rate")
    ax.set_ylabel("consensus accuracy  q(true state)")
    # Truncate just below the threshold band: the executed curves live well
    # above zero, so a full [0, 1] range wastes panel area; the floor stays visible.
    ax.set_ylim(0.4, 1.02)
    if rate_summary is not None:
        n_profile = int(next(iter(rate_summary.values())).get("n", 0))
        ax.set_title("Contamination robustness profile: mean ± 95% bootstrap CI")
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
            f"max-rate robust-minus-naive = {best_final - naive_final:+.3f}\n"
            f"max rate = {final_rate:g}",
            transform=ax.transAxes,
            fontsize=9.5,
            ha="left",
            va="bottom",
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": COLOR_GRID, "alpha": 0.85},
        )
    else:
        ax.set_title("Deterministic contamination sweep: statistics in tables")
    ax.legend(fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    if len(rates) > 1 and rate_summary is None:
        # Mid-panel dead space (between the legend and the KLD descent), so the
        # note no longer sits on the curves in the upper-right corner.
        ax.text(
            0.55, 0.33,
            "single seeded curves\nlinear y-axis: 0.4 to 1.0\nverdict statistics in tables",
            transform=ax.transAxes, fontsize=9.5, ha="center", va="center",
            bbox={"boxstyle": "round,pad=0.35", "fc": "white",
                  "ec": COLOR_GRID, "alpha": 0.85},
        )

    return save_figure(fig, figures_dir(project_root) / filename)


__all__ = ["generate_robustness_sweep"]
