"""Projection layouts for source-owned learning and recovery estimates."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from ._common import COLOR_ROBUST, semantic_style
from ._presentation import PresentationPanel, save_presentation_panels
from ._presentation_diagnostics import _axes, _notes
from ._presentation_flows import _text_panel


def language_presentation(
    path: Path,
    kl: np.ndarray,
    ci_lo: np.ndarray | None,
    ci_hi: np.ndarray | None,
    monotone: bool | None,
    n_seeds: int | None,
) -> Path:
    """Retain the full ordered mean trajectory and optional seed interval."""
    axis = _axes("Categorical learning", "Learning step", "KL (nats)")
    steps = np.arange(kl.size)
    axis.plot(steps, kl, color=COLOR_ROBUST, marker="o", markersize=8, linewidth=3)
    if ci_lo is not None and ci_hi is not None:
        axis.fill_between(steps, ci_lo, ci_hi, color=COLOR_ROBUST, alpha=0.16)
    axis.margins(y=0.10)
    panels = [
        PresentationPanel(
            "trajectory",
            cast(Figure, axis.figure),
            "Complete ordered seed-mean KL learning trajectory with the supplied pointwise seed interval.",
        ),
        _text_panel(
            "estimand",
            "KL(true A || learned A), in nats. Circles / solid: seed mean. The ordered "
            "x-axis is the Dirichlet count batch.",
        ),
        _text_panel(
            "uncertainty",
            (
                "Pointwise 95% percentile-bootstrap interval over independent configured seeds. "
                "Ordered steps are not independent replicates."
            )
            if ci_lo is not None
            else "No interval supplied. The figure does not infer one from the mean trajectory.",
        ),
        _text_panel(
            "endpoints",
            f"Initial mean: {kl[0]:.4g} nats. Final mean: {kl[-1]:.4g} nats. Steps: "
            f"{kl.size}. Seeds: {n_seeds if n_seeds is not None else 'not supplied'}.",
        ),
        _text_panel(
            "pattern",
            f"Source monotonicity check: {monotone if monotone is not None else 'not supplied'}. "
            "This is a source-mechanism analogue, not exact protocol replication.",
        ),
    ]
    return save_presentation_panels(
        panels,
        canonical_path=path,
        expected_identifiers=("trajectory", "estimand", "uncertainty", "endpoints", "pattern"),
    )


def recovery_presentation(
    path: Path,
    true: np.ndarray,
    recovered: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    errors: np.ndarray,
    mean_error: float | None,
    stats: str,
) -> Path:
    """Keep every recovery condition, empirical trial interval and error bar."""
    axis = _axes("Sensor acuity recovery", "True acuity", "Recovered")
    axis.xaxis.set_major_locator(MaxNLocator(3))
    estimate = semantic_style("estimate")
    reference = semantic_style("reference_rule")
    axis.errorbar(
        true,
        recovered,
        yerr=[recovered - low, high - recovered],
        fmt=estimate.marker,
        markerfacecolor="white",
        color=estimate.color,
        markeredgecolor=estimate.keyline,
        ecolor=estimate.keyline,
        capsize=5,
        markersize=9,
        linewidth=2,
    )
    bounds = (float(min(true.min(), low.min())), float(max(true.max(), high.max())))
    axis.plot(bounds, bounds, color=reference.color, linestyle=reference.dash, linewidth=2)
    panels = [
        PresentationPanel(
            "recovery",
            cast(Figure, axis.figure),
            "Every acuity condition's recovered mean and empirical trial interval, with identity reference.",
        )
    ]
    error = _axes("Absolute recovery error", "True acuity", "MAE")
    error.xaxis.set_major_locator(MaxNLocator(3))
    style = semantic_style("operating_point_1")
    error.bar(true, errors, width=0.04, facecolor="white", edgecolor=style.keyline, hatch=style.hatch)
    if mean_error is not None:
        error.axhline(mean_error, color=reference.color, linestyle=reference.dash, linewidth=2)
    panels.extend(
        [
            PresentationPanel(
                "errors",
                cast(Figure, error.figure),
                "Every acuity condition's mean absolute error with the supplied global mean error reference.",
            ),
            _text_panel(
                "uncertainty",
                "Recovery whiskers: 95% empirical percentile intervals over independent "
                "simulation trials within each acuity. Observations are nested within trial.",
            ),
            *_notes(stats),
            _text_panel(
                "scope",
                "Simulation-based recovery over the declared acuity grid. Identity "
                "reference is not an additional estimate; no uncertainty interval is inferred for MAE bars.",
            ),
        ]
    )
    expected = (
        "recovery",
        "errors",
        "uncertainty",
        *(f"notes-{i + 1}" for i in range((len(stats.splitlines()) + 2) // 3)),
        "scope",
    )
    return save_presentation_panels(panels, canonical_path=path, expected_identifiers=expected)


def complexity_presentation(path: Path, report: dict[str, object]) -> Path:
    """Preserve measured timing ranges and normalized guides on shared group scales."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    from .complexity_scaling import _METHOD_DIRECT_LABELS, _METHOD_ROLES, _measurement, _numbers

    groups = (
        ("agents", ("log_linear_pool", "robust_aggregate", "variational_aggregate")),
        ("agents", ("share_round_naive", "share_round_robust")),
        ("states", ("log_linear_pool", "robust_aggregate", "variational_aggregate")),
        ("modalities", ("infer_states",)),
    )
    panels: list[PresentationPanel] = []
    expected: list[str] = []
    for group_index, (dimension, methods) in enumerate(groups):
        rows = [_measurement(report, method=method, axis=dimension) for method in methods]
        all_min = min(float(_numbers(row, "min_seconds").min()) for row in rows)
        all_max = max(float(_numbers(row, "max_seconds").max()) for row in rows)
        for row in rows:
            sizes = _numbers(row, "sizes")
            medians = _numbers(row, "median_seconds")
            exponent = float(str(row["expected_exponent"]))
            guide = medians[0] * (sizes / sizes[0]) ** exponent
            all_min = min(all_min, float(guide.min()))
            all_max = max(all_max, float(guide.max()))
        for method, row in zip(methods, rows, strict=True):
            axis = _axes(_METHOD_DIRECT_LABELS[method], dimension.title(), "Seconds")
            sizes = _numbers(row, "sizes")
            medians = _numbers(row, "median_seconds")
            lower = _numbers(row, "min_seconds")
            upper = _numbers(row, "max_seconds")
            exponent = float(str(row["expected_exponent"]))
            style = semantic_style(_METHOD_ROLES[method])
            axis.errorbar(
                sizes,
                medians,
                yerr=[medians - lower, upper - medians],
                marker=style.marker,
                color=style.color,
                linestyle=style.dash,
                markerfacecolor=style.color if method == "log_linear_pool" else "white",
                markeredgecolor=style.keyline,
                linewidth=3,
                markersize=8,
                capsize=5,
            )
            reference = semantic_style("reference_rule")
            axis.plot(
                sizes,
                medians[0] * (sizes / sizes[0]) ** exponent,
                color=reference.color,
                linestyle=reference.dash,
                linewidth=2,
            )
            axis.set_xscale("log", base=2)
            axis.set_yscale("log")
            log_padding = max(float(np.log(all_max / all_min)) * 0.10, float(np.log(1.5)))
            axis.set_ylim(all_min * np.exp(-log_padding), all_max * np.exp(log_padding))
            axis.set_xlim(sizes.min() / 1.1, sizes.max() * 1.1)
            axis.xaxis.set_major_locator(FixedLocator([sizes[0], sizes[len(sizes) // 2], sizes[-1]]))
            axis.yaxis.set_major_locator(FixedLocator([all_min, all_max]))
            for dimension_axis in (axis.xaxis, axis.yaxis):
                dimension_axis.set_minor_locator(NullLocator())
                dimension_axis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2g}"))
            identifier = f"group-{group_index + 1}-{method.replace('_', '-')}"
            expected.extend((identifier, f"{identifier}-guide"))
            panels.extend(
                (
                    PresentationPanel(
                        identifier,
                        cast(Figure, axis.figure),
                        f"{method} by {dimension}: all measured medians, observed min-max ranges "
                        "and normalized expected-exponent guide on common group scales.",
                    ),
                    _text_panel(
                        f"{identifier}-guide",
                        f"{_METHOD_DIRECT_LABELS[method]} by {dimension}. Expected exponent: {exponent:g}. "
                        f"Observed log-log slope: {float(str(row['observed_log_log_slope'])):.4g}.",
                    ),
                )
            )
    benchmark = cast(dict[str, object], report["benchmark"])
    panels.append(
        _text_panel(
            "scope",
            f"Medians over {benchmark['repeats']} timing repeats; min–max bars are observed ranges, "
            "not confidence intervals. Normalized dotted guides are symbolic-order references. "
            f"Seed {report['seed']}.",
        )
    )
    expected.append("scope")
    return save_presentation_panels(panels, canonical_path=path, expected_identifiers=expected)


def conditional_presentation(
    path: Path,
    heatmap: np.ndarray,
    means: list[float],
    minima: list[float],
    maxima: list[float],
    attacks: tuple[str, ...],
    columns: list[str],
) -> Path:
    """Preserve the signed grid and asymmetric finite-grid min–max summaries."""
    from ._common import COLOR_DEEP

    limit = max(float(np.nanmax(np.abs(heatmap))), 1e-6)
    low = min(0.0, min(minima))
    high = max(0.0, max(maxima))
    span = max(high - low, 1e-6)
    panels: list[PresentationPanel] = []
    expected: list[str] = []
    for row, attack in enumerate(attacks):
        for start in range(0, len(columns), 2):
            stop = min(start + 2, len(columns))
            axis = _axes(attack.replace("_", " "), "World cell", "")
            axis.imshow(
                heatmap[row : row + 1, start:stop], cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto"
            )
            axis.grid(False)
            axis.set_yticks(())
            axis.set_xticks(range(stop - start), [columns[i].replace("_", " / ") for i in range(start, stop)])
            for index, value in enumerate(heatmap[row, start:stop]):
                axis.text(
                    index,
                    0,
                    f"{value:+.3f}",
                    fontsize=30,
                    ha="center",
                    va="center",
                    color=COLOR_DEEP,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5},
                )
            identifier = f"attack-{row + 1}-cells-{start + 1}-{stop}"
            panels.append(
                PresentationPanel(
                    identifier,
                    cast(Figure, axis.figure),
                    f"{attack}: signed seed-level contrasts for the two displayed world cells; "
                    f"fixed symmetric color scale.",
                )
            )
            expected.append(identifier)
        axis = _axes(attack.replace("_", " "), "Mass contrast", "")
        axis.errorbar(
            [means[row]],
            [0],
            xerr=[[means[row] - minima[row]], [maxima[row] - means[row]]],
            fmt="o",
            color=COLOR_ROBUST,
            capsize=7,
            markersize=10,
            linewidth=3,
        )
        reference = semantic_style("reference_rule")
        axis.axvline(0, color=reference.color, linestyle=reference.dash, linewidth=2)
        axis.set_xlim(low - 0.1 * span, high + 0.1 * span)
        axis.set_ylim(-1, 1)
        axis.set_yticks(())
        identifier = f"attack-{row + 1}-range"
        panels.append(
            PresentationPanel(
                identifier,
                cast(Figure, axis.figure),
                f"{attack}: finite-grid mean with asymmetric capped minimum-to-maximum range on "
                f"the common contrast scale.",
            )
        )
        expected.append(identifier)
    notes = (
        (
            "key",
            "Seed-level true-state-mass contrast: naive error minus robust error. Positive "
            "values favor the heuristic on the declared finite grid.",
        ),
        (
            "grid",
            f"Heatmap: adversary weight 1. State s0/s1; observability o45/o70. Common color "
            f"scale: [{-limit:.4g}, {limit:.4g}]. Every cell prints its signed mean.",
        ),
        (
            "ranges",
            "Summary circles: finite-grid means across configured cells per attack. Asymmetric "
            "capped whiskers reach the observed minimum and maximum; they are not confidence intervals.",
        ),
        (
            "scope",
            "Dark dotted rule: zero method contrast. This finite-grid summary does not support "
            "population, deployment or universal robustness claims.",
        ),
    )
    for identifier, text in notes:
        panels.append(_text_panel(identifier, text))
        expected.append(identifier)
    return save_presentation_panels(panels, canonical_path=path, expected_identifiers=expected)


def sensitivity_presentation(
    path: Path,
    sharing: np.ndarray,
    hierarchy: np.ndarray,
    acuities: list[str],
    colonies: list[str],
    band: float,
) -> Path:
    """Preserve every signed sensitivity cell and its descriptive display-band hatch."""
    from matplotlib.colors import to_hex
    from matplotlib.patches import Rectangle

    from ._common import contrasting_text_color, signed_difference_colormap

    limit = max(float(np.abs(sharing).max()), float(np.abs(hierarchy).max()), 1e-6)
    panels: list[PresentationPanel] = []
    expected: list[str] = []
    for group, data in (("Sharing", sharing), ("Hierarchy", hierarchy)):
        for row, acuity in enumerate(acuities):
            for start in range(0, len(colonies), 3):
                stop = min(start + 3, len(colonies))
                axis = _axes(f"{group}: acuity {acuity}", "Colony size", "")
                image = axis.imshow(
                    data[row : row + 1, start:stop],
                    aspect="auto",
                    cmap=signed_difference_colormap(),
                    vmin=-limit,
                    vmax=limit,
                )
                axis.grid(False)
                axis.set_yticks(())
                axis.set_xticks(range(stop - start), colonies[start:stop])
                for index, value in enumerate(data[row, start:stop]):
                    if abs(value) <= band:
                        style = semantic_style("display_band")
                        axis.add_patch(
                            Rectangle(
                                (index - 0.5, -0.5),
                                1,
                                1,
                                fill=False,
                                hatch=style.hatch,
                                edgecolor=style.keyline,
                                alpha=0.4,
                                linewidth=0,
                            )
                        )
                    background = to_hex(image.cmap(image.norm(value)))
                    axis.text(
                        index,
                        0,
                        f"{value:+.2f}",
                        fontsize=30,
                        ha="center",
                        va="center",
                        color=contrasting_text_color(background),
                        bbox={"facecolor": background, "edgecolor": "none", "pad": 0},
                    )
                identifier = f"{group.lower()}-acuity-{row + 1}-columns-{start + 1}-{stop}"
                panels.append(
                    PresentationPanel(
                        identifier,
                        cast(Figure, axis.figure),
                        f"{group}, acuity {acuity}: every signed accuracy gap in the displayed "
                        f"colony-size cells; common symmetric color scale and display-band hatch.",
                    )
                )
                expected.append(identifier)
        identifier = f"{group.lower()}-summary"
        panels.append(
            _text_panel(
                identifier,
                f"{group}: max |gap| = {float(np.abs(data).max()):.3f}; positive cells above the "
                f"display band = {int((data > band).sum())}/{data.size}.",
            )
        )
        expected.append(identifier)
    notes = (
        (
            "direction",
            "Sharing: communicating minus isolated accuracy. Hierarchy: hierarchical minus flat "
            "location accuracy. Positive and negative values retain their sign.",
        ),
        (
            "scale",
            f"Common signed color scale: [{-limit:.4g}, {limit:.4g}]. Every cell prints its "
            f"value; panels keep original acuity and colony ordering.",
        ),
        (
            "hatching",
            f"Hatching uses unrounded gaps: |gap| ≤ {band:.2f}. Printed values are rounded; "
            "equal printed values can have different hatching. It is not a CI, significance "
            f"test, unreliability flag or proof of zero effect.",
        ),
    )
    for identifier, text in notes:
        panels.append(_text_panel(identifier, text))
        expected.append(identifier)
    return save_presentation_panels(panels, canonical_path=path, expected_identifiers=expected)
