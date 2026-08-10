"""Manuscript variable submodule."""

from __future__ import annotations

import math
import warnings
from typing import Any

from .loaders import _fmt, _format_residual, _format_residual_math

_COMPLEXITY_KEYS = (
    "COMPLEXITY_AGENT_GRID",
    "COMPLEXITY_STATE_GRID",
    "COMPLEXITY_SHARING_AGENT_GRID",
    "COMPLEXITY_MODALITY_GRID",
    "COMPLEXITY_FIXED_AGENTS",
    "COMPLEXITY_FIXED_STATES",
    "COMPLEXITY_INFERENCE_STATES",
    "COMPLEXITY_REPEATS",
    "COMPLEXITY_WARMUPS",
    "COMPLEXITY_MAX_ITER",
    "COMPLEXITY_ROBUST_SHARING_ITERATIONS",
    "COMPLEXITY_SEED",
    "COMPLEXITY_MACHINE",
    "COMPLEXITY_PYTHON",
    "COMPLEXITY_NUMPY",
    "COMPLEXITY_LOG_TIME_ORDER",
    "COMPLEXITY_ROBUST_TIME_ORDER",
    "COMPLEXITY_VARIATIONAL_TIME_ORDER",
    "COMPLEXITY_SHARING_TIME_ORDER",
    "COMPLEXITY_ROBUST_SHARING_TIME_ORDER",
    "COMPLEXITY_INFER_TIME_ORDER",
    "COMPLEXITY_SERVER_TIME_ORDER",
    "COMPLEXITY_LOG_MEMORY_ORDER",
    "COMPLEXITY_ROBUST_MEMORY_ORDER",
    "COMPLEXITY_VARIATIONAL_MEMORY_ORDER",
    "COMPLEXITY_SHARING_MEMORY_ORDER",
    "COMPLEXITY_ROBUST_SHARING_MEMORY_ORDER",
    "COMPLEXITY_INFER_MEMORY_ORDER",
    "COMPLEXITY_SERVER_MEMORY_ORDER",
    "COMPLEXITY_VARIATIONAL_STARTS",
    "COMPLEXITY_AGENT_SLOPE_LOG",
    "COMPLEXITY_AGENT_SLOPE_ROBUST",
    "COMPLEXITY_AGENT_SLOPE_VARIATIONAL",
    "COMPLEXITY_AGENT_SLOPE_SHARING",
    "COMPLEXITY_AGENT_SLOPE_SHARING_ROBUST",
    "COMPLEXITY_STATE_SLOPE_LOG",
    "COMPLEXITY_STATE_SLOPE_ROBUST",
    "COMPLEXITY_STATE_SLOPE_VARIATIONAL",
    "COMPLEXITY_MODALITY_SLOPE",
)


def _complexity_variables(report: dict[str, Any]) -> dict[str, str]:
    """Flatten the source-bound complexity report into manuscript tokens."""
    out = {key: "N/A" for key in _COMPLEXITY_KEYS}
    if not report:
        return out
    benchmark = report.get("benchmark", {})
    machine = report.get("machine", {})
    specs = report.get("analytic_specs", [])
    measurements = report.get("measurements", [])
    if not isinstance(benchmark, dict) or not isinstance(machine, dict):
        return out
    if not isinstance(specs, list) or not isinstance(measurements, list):
        return out

    def _grid(key: str) -> str:
        values = benchmark.get(key, [])
        if not isinstance(values, list):
            return "N/A"
        return ", ".join(str(int(value)) for value in values)

    out["COMPLEXITY_AGENT_GRID"] = _grid("agent_sizes")
    out["COMPLEXITY_STATE_GRID"] = _grid("state_sizes")
    out["COMPLEXITY_SHARING_AGENT_GRID"] = _grid("sharing_agent_sizes")
    out["COMPLEXITY_MODALITY_GRID"] = _grid("modality_sizes")
    for token, field in (
        ("COMPLEXITY_FIXED_AGENTS", "fixed_agent_count"),
        ("COMPLEXITY_FIXED_STATES", "fixed_state_count"),
        ("COMPLEXITY_INFERENCE_STATES", "inference_state_count"),
        ("COMPLEXITY_REPEATS", "repeats"),
        ("COMPLEXITY_WARMUPS", "warmups"),
        ("COMPLEXITY_MAX_ITER", "max_iter"),
        ("COMPLEXITY_SEED", "seed"),
    ):
        if field in benchmark:
            out[token] = str(int(benchmark[field]))
    out["COMPLEXITY_MACHINE"] = str(machine.get("machine", "N/A"))
    out["COMPLEXITY_PYTHON"] = str(machine.get("python", "N/A"))
    out["COMPLEXITY_NUMPY"] = str(machine.get("numpy", "N/A"))

    order_tokens = {
        "log_linear_pool": "COMPLEXITY_LOG_TIME_ORDER",
        "robust_aggregate": "COMPLEXITY_ROBUST_TIME_ORDER",
        "variational_aggregate": "COMPLEXITY_VARIATIONAL_TIME_ORDER",
        "share_round_naive": "COMPLEXITY_SHARING_TIME_ORDER",
        "share_round_robust": "COMPLEXITY_ROBUST_SHARING_TIME_ORDER",
        "infer_states": "COMPLEXITY_INFER_TIME_ORDER",
        "federation_server_round": "COMPLEXITY_SERVER_TIME_ORDER",
    }
    memory_tokens = {
        "log_linear_pool": "COMPLEXITY_LOG_MEMORY_ORDER",
        "robust_aggregate": "COMPLEXITY_ROBUST_MEMORY_ORDER",
        "variational_aggregate": "COMPLEXITY_VARIATIONAL_MEMORY_ORDER",
        "share_round_naive": "COMPLEXITY_SHARING_MEMORY_ORDER",
        "share_round_robust": "COMPLEXITY_ROBUST_SHARING_MEMORY_ORDER",
        "infer_states": "COMPLEXITY_INFER_MEMORY_ORDER",
        "federation_server_round": "COMPLEXITY_SERVER_MEMORY_ORDER",
    }
    for row in specs:
        if not isinstance(row, dict):
            continue
        operation = str(row.get("operation", ""))
        order_token = order_tokens.get(operation)
        if order_token is None:
            continue
        out[order_token] = str(row.get("time_order", "N/A")).replace("Theta", "\\Theta")
        memory_token = memory_tokens.get(operation)
        if memory_token is not None:
            out[memory_token] = str(row.get("memory_order", "N/A")).replace("Theta", "\\Theta")

    if "variational_starts" in benchmark:
        out["COMPLEXITY_VARIATIONAL_STARTS"] = str(int(benchmark["variational_starts"]))

    for row in measurements:
        if not isinstance(row, dict):
            continue
        if row.get("method") != "share_round_robust" or row.get("axis") != "agents":
            continue
        parameters = row.get("parameters")
        if not isinstance(parameters, dict):
            continue
        iterations = parameters.get("I")
        if isinstance(iterations, (int, float, str)):
            out["COMPLEXITY_ROBUST_SHARING_ITERATIONS"] = str(int(iterations))
        break

    slope_tokens = {
        ("log_linear_pool", "agents"): "COMPLEXITY_AGENT_SLOPE_LOG",
        ("robust_aggregate", "agents"): "COMPLEXITY_AGENT_SLOPE_ROBUST",
        ("variational_aggregate", "agents"): "COMPLEXITY_AGENT_SLOPE_VARIATIONAL",
        ("share_round_naive", "agents"): "COMPLEXITY_AGENT_SLOPE_SHARING",
        ("share_round_robust", "agents"): "COMPLEXITY_AGENT_SLOPE_SHARING_ROBUST",
        ("log_linear_pool", "states"): "COMPLEXITY_STATE_SLOPE_LOG",
        ("robust_aggregate", "states"): "COMPLEXITY_STATE_SLOPE_ROBUST",
        ("variational_aggregate", "states"): "COMPLEXITY_STATE_SLOPE_VARIATIONAL",
        ("infer_states", "modalities"): "COMPLEXITY_MODALITY_SLOPE",
    }
    for row in measurements:
        if not isinstance(row, dict):
            continue
        slope_token = slope_tokens.get((str(row.get("method", "")), str(row.get("axis", ""))))
        if slope_token is None:
            continue
        slope = row["observed_log_log_slope"]
        if not isinstance(slope, (int, float, str)):
            continue
        out[slope_token] = _fmt(float(slope), 2)
    return out


def _onset_variables(report: dict[str, Any]) -> dict[str, str]:
    """Flatten the robustness-onset report into manuscript tokens.

    Surfaces, per directional mechanism, the onset rate (the smallest rate at
    which the robust win fraction clears the reliability bar) alongside the
    naive and best-robust accuracy at the worst (highest) swept rate — the
    rate-resolved companion to the fixed-strength gallery.
    """
    out: dict[str, str] = {}
    out["ONSET_N_SEEDS"] = str(report["n_seeds"])
    out["ONSET_N_TRIALS"] = str(report["n_trials"])
    out["ONSET_WIN_FRACTION"] = _fmt(report["onset_win_fraction"], 2)
    rows = []
    for kind, cell in report["by_kind"].items():
        onset = cell["onset_rate"]
        onset_str = f"{onset:g}" if onset is not None else "none"
        rows.append(
            f"| {kind.replace('_', ' ')} | {onset_str} | "
            f"{_fmt(cell['naive_curve'][-1])} | {_fmt(cell['robust_curve'][-1])} | "
            f"{cell['best_robust_method_by_rate'][-1]} |"
        )
    out["ONSET_TABLE_ROWS"] = "\n".join(rows)
    return out


def _review_grid_variables(
    report: dict[str, Any],
    *,
    configured_target_max_mcse: float | None = None,
    require_reported_target: bool = False,
) -> dict[str, str]:
    """Flatten the source-bound review-grid design and ownership metadata.

    The MCSE target is registered in the experiment configuration before the
    analysis runs.  A smoke workflow intentionally leaves the report's target
    ``null`` because its bounded sample size is not allowed to claim it met the
    publication stopping rule.  In that diagnostic case, use the registered
    configuration value for the manuscript-design token while retaining the
    observed MCSE from the report.  Release-facing hydration sets
    ``require_reported_target`` and therefore refuses a report that does not
    bind that target back to the executed payload.
    """
    if not report:
        return {key: "N/A" for key in _REVIEW_GRID_KEYS}
    statistics = report.get("statistics", {})
    precision_plan = report.get("precision_plan", {})

    def _precision_number(field: str) -> str:
        if not isinstance(precision_plan, dict):
            return "N/A"
        value = precision_plan.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return "N/A"
        numeric = float(value)
        return _fmt(numeric, 4) if math.isfinite(numeric) else "N/A"

    def _precision_count(field: str) -> str:
        if not isinstance(precision_plan, dict):
            return "N/A"
        value = precision_plan.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return "N/A"
        return str(value)

    def _finite_positive_number(value: object) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        numeric = float(value)
        return numeric if math.isfinite(numeric) and numeric > 0.0 else None

    configured_target = _finite_positive_number(configured_target_max_mcse)
    reported_target_state = "missing"
    reported_target: float | None = None
    if isinstance(precision_plan, dict) and "target_max_mcse" in precision_plan:
        raw_reported_target = precision_plan["target_max_mcse"]
        if raw_reported_target is None:
            reported_target_state = "unregistered"
        else:
            reported_target = _finite_positive_number(raw_reported_target)
            reported_target_state = "recorded" if reported_target is not None else "invalid"

    if reported_target_state == "recorded":
        assert reported_target is not None
        if configured_target is not None and not math.isclose(
            reported_target,
            configured_target,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "review-grid report target_max_mcse disagrees with the registered experiment configuration"
            )
        effective_target = reported_target
    elif reported_target_state == "unregistered" and configured_target is not None:
        if require_reported_target:
            raise ValueError(
                "validation-backed hydration requires robustness_review_grid.json "
                "to record its registered target_max_mcse"
            )
        effective_target = configured_target
    else:
        if require_reported_target:
            raise ValueError(
                "validation-backed hydration requires a finite positive review-grid target_max_mcse"
            )
        effective_target = None

    out = {
        "REVIEW_GRID_N_SEEDS": str(report.get("n_seeds", "N/A")),
        "REVIEW_GRID_N_TRIALS": str(report.get("n_trials", "N/A")),
        "REVIEW_GRID_RATES": ", ".join(f"{float(rate):g}" for rate in report.get("rates", [])) or "N/A",
        "REVIEW_GRID_ATTACKS": ", ".join(
            str(attack).replace("_", " ") for attack in report.get("attack_mechanisms", [])
        )
        or "N/A",
        "REVIEW_GRID_DIRECTIONAL": ", ".join(
            str(attack).replace("_", " ") for attack in report.get("directional_mechanisms", [])
        )
        or "N/A",
        "REVIEW_GRID_ENTROPY_CONTROLS": ", ".join(
            str(attack).replace("_", " ") for attack in report.get("entropy_controls", [])
        )
        or "N/A",
        "REVIEW_GRID_SELECTION_STATUS": str(report.get("selection_status", "N/A")),
        "REVIEW_GRID_INDEPENDENT_UNIT": str(report.get("independent_unit", "N/A")),
        "REVIEW_GRID_TRIAL_STRUCTURE": str(report.get("trial_structure", "N/A")),
        "REVIEW_GRID_STATS_STATUS": (
            "selection-free"
            if isinstance(statistics, dict) and bool(statistics.get("selection_free"))
            else "N/A"
        ),
        "REVIEW_GRID_BH_OWNERSHIP": (
            str(statistics.get("bh_family_ownership", "N/A")) if isinstance(statistics, dict) else "N/A"
        ),
        "REVIEW_GRID_TARGET_MAX_MCSE": (_fmt(effective_target, 4) if effective_target is not None else "N/A"),
        "REVIEW_GRID_OBSERVED_MAX_MCSE": _precision_number("observed_max_mcse"),
        "REVIEW_GRID_SIGNED_CELLS": _precision_count("n_signed_method_rate_cells"),
    }
    return out


def _gallery_variables(report: dict[str, Any]) -> dict[str, str]:
    """Flatten the seed-aggregated contamination-gallery report into tokens.

    Surfaces the honest, seed-robust verdict (replacing a fragile single-seed
    boolean): the per-mechanism table of naive vs best-robust accuracy with the
    mean robust-minus-naive difference, its bootstrap CI, the across-seed win
    fraction, and whether the robust advantage is *reliable* (win fraction high
    and CI excludes zero). ``GALLERY_RELIABLE_KINDS`` lists exactly the mechanisms
    that earned a reliable robust advantage — no categorical "every directional"
    claim.
    """
    out: dict[str, str] = {}
    out["GALLERY_RATE"] = _fmt(report["rate"], 2)
    out["GALLERY_N_TRIALS"] = str(report["n_trials"])
    out["GALLERY_N_SEEDS"] = str(report["n_seeds"])
    out["GALLERY_RELIABLE_WIN_FRACTION"] = _fmt(report["reliable_win_fraction"], 2)
    reliable = report["reliable_kinds"]
    out["GALLERY_RELIABLE_KINDS"] = ", ".join(k.replace("_", " ") for k in reliable) if reliable else "none"
    out["GALLERY_ENTROPY_NAIVE_ROBUST"] = "Yes" if report["entropy_naive_robust"] else "No"
    out["GALLERY_DIRECTIONAL_KINDS"] = ", ".join(k.replace("_", " ") for k in report["directional_kinds"])
    out["GALLERY_ENTROPY_KINDS"] = ", ".join(k.replace("_", " ") for k in report["entropy_kinds"])
    rows = []
    for kind, cell in report["by_kind"].items():
        lo, hi = cell["diff_ci"]
        rows.append(
            f"| {kind.replace('_', ' ')} | "
            f"{'directional' if cell['directional'] else 'entropy'} | "
            f"{_fmt(cell['naive_mean'])} | "
            f"{_fmt(cell['robust_mean'])} ({cell['best_robust_method']}) | "
            f"{_fmt(cell['mean_diff'])} | "
            f"[{_fmt(lo)}, {_fmt(hi)}] | "
            f"{_fmt(cell['win_fraction'], 2)} | "
            f"{'Yes' if cell['reliably_beats'] else 'No'} |"
        )
    out["GALLERY_TABLE_ROWS"] = "\n".join(rows)
    return out


def _variational_variables(report: dict[str, Any]) -> dict[str, str]:
    """Flatten the variational-aggregation diagnostics into manuscript tokens.

    Surfaces the descent (initial/final F, the monotone drop ``DELTA_F``, and the
    largest single-step ascent — ~0, the numerical witness of monotonicity) and
    the redescending-weight sweep (a probed agent's normalized influence when clean
    vs fully diverged, against the naive pool's fixed ``1/n``). Honesty contract:
    these certify objective descent and a tested normalized-weight path, not
    estimator-level B-robustness or a peak-accuracy win over the naive pool.
    """
    out: dict[str, str] = {}
    out["VARIATIONAL_ROBUSTNESS"] = _fmt(report["robustness"], 2)
    out["VARIATIONAL_ITERATIONS"] = str(report["iterations"])
    out["VARIATIONAL_CONVERGED"] = "Yes" if report["converged"] else "No"

    fe = [float(x) for x in report["free_energy_history"]]
    if fe:
        ascents = [fe[i + 1] - fe[i] for i in range(len(fe) - 1)]
        out["VARIATIONAL_F_INITIAL"] = _fmt(fe[0])
        out["VARIATIONAL_F_FINAL"] = _fmt(fe[-1])
        out["VARIATIONAL_DELTA_F"] = _fmt(fe[0] - fe[-1])
        max_ascent = max(ascents) if ascents else 0.0
        out["VARIATIONAL_MAX_ASCENT"] = _format_residual(max_ascent)
        out["VARIATIONAL_MAX_ASCENT_MATH"] = _format_residual_math(max_ascent)
    else:  # pragma: no cover - history is populated for robustness > 0
        for key in (
            "VARIATIONAL_F_INITIAL",
            "VARIATIONAL_F_FINAL",
            "VARIATIONAL_DELTA_F",
            "VARIATIONAL_MAX_ASCENT",
            "VARIATIONAL_MAX_ASCENT_MATH",
        ):
            out[key] = "N/A"

    influence = [float(x) for x in report["variational_influence"]]
    naive = float(report["naive_influence"])
    clean, diverged = influence[0], influence[-1]
    out["VARIATIONAL_INFLUENCE_CLEAN"] = _fmt(clean, 3)
    out["VARIATIONAL_INFLUENCE_DIVERGED"] = _fmt(diverged, 3)
    out["VARIATIONAL_NAIVE_INFLUENCE"] = _fmt(naive, 3)
    # How many times smaller the diverged outlier's influence is vs the naive pool
    # (which would hold it at the fixed 1/n): the bounded-influence headline number.
    out["VARIATIONAL_INFLUENCE_DROP_FACTOR"] = _fmt(naive / diverged, 1) if diverged > 0 else ">1000"
    # Descent comparison: single-start capture vs multi-start escape (near-vertex).
    if "single_start_final_f" in report:
        out["VARIATIONAL_SINGLE_START_F"] = _fmt(report["single_start_final_f"])
        out["VARIATIONAL_MULTI_START_F"] = _fmt(report["multi_start_final_f"])
        out["VARIATIONAL_CAPTURE_GAP"] = _fmt(report["capture_gap"])
    else:  # pragma: no cover - report always carries the comparison after a run
        for key in ("VARIATIONAL_SINGLE_START_F", "VARIATIONAL_MULTI_START_F", "VARIATIONAL_CAPTURE_GAP"):
            out[key] = "N/A"
    return out


_SWEEP_KEYS = (
    "SWEEP_NAIVE_ACCURACY",
    "SWEEP_BEST_ROBUST_ACCURACY",
    "SWEEP_PROFILE_NAIVE_ACCURACY",
    "SWEEP_PROFILE_BEST_ROBUST_ACCURACY",
    "SWEEP_BEST_ROBUST_METHOD",
    "SWEEP_BEST_EFFECT_SIZE",
    "SWEEP_BEST_QVALUE",
    "SWEEP_BEST_QVALUE_MATH",
    "SWEEP_ANY_ROBUST_WINS",
    "SWEEP_NAIVE_DEGRADES",
    "SWEEP_ROBUST_ABOVE_THRESHOLD",
    "SWEEP_ACCURACY_THRESHOLD",
    "SWEEP_N_AGENTS",
    "SWEEP_N_CONTAMINATED",
    "SWEEP_N_TRIALS",
    "SWEEP_RATES",
    "SWEEP_SERVER_OPERATING_POINTS",
    "SWEEP_WORST_RATE",
    "SWEEP_VERDICT_RATE",
    "SWEEP_RATE_TABLE_ROWS",
    "SWEEP_VERDICT_TABLE_ROWS",
    # --- statistics enrichment (added) ---
    "SWEEP_N",
    "SWEEP_NAIVE_VERDICT_RATE_MEAN",
    "SWEEP_HEADLINE_SELECTION_RULE",
    "SWEEP_HEADLINE_TIE_SET",
    "SWEEP_HEADLINE_TIE_BREAK",
    "SWEEP_LARGEST_MEAN_DIFFERENCE_METHOD",
    "SWEEP_WORST_RATE_BEST_METHOD",
    "SWEEP_BEST_D_EQUIVALENT",
    "SWEEP_BEST_COHENS_D",
    "SWEEP_BEST_EFFECT_LABEL",
    "SWEEP_BEST_RAW_PVALUE",
    "SWEEP_BEST_RAW_PVALUE_MATH",
    "SWEEP_BEST_MEAN_ACC_DIFF",
    "SWEEP_BEST_MEAN_ACC_DIFF_CI_LO",
    "SWEEP_BEST_MEAN_ACC_DIFF_CI_HI",
    "SWEEP_NAIVE_VERDICT_ACCURACY_MEAN",
    "SWEEP_NAIVE_VERDICT_ACCURACY_CI_LO",
    "SWEEP_NAIVE_VERDICT_ACCURACY_CI_HI",
    "SWEEP_BEST_VERDICT_ACCURACY_MEAN",
    "SWEEP_BEST_VERDICT_ACCURACY_CI_LO",
    "SWEEP_BEST_VERDICT_ACCURACY_CI_HI",
    "SWEEP_ACCURACY_AT_VERDICT_TABLE_ROWS",
    "SWEEP_VERDICT_EFFECT_TABLE_ROWS",
    "SWEEP_PAIRED_BY_RATE_TABLE_ROWS",
    # --- power analysis (added) ---
    "SWEEP_POWER_ALPHA",
    "SWEEP_POWER_ALTERNATIVE",
    "SWEEP_TARGET_POWER",
    "SWEEP_HEADLINE_POWER",
    "SWEEP_HEADLINE_METHOD",
    "SWEEP_PROSPECTIVE_N",
    "SWEEP_HEADLINE_N_FOR_TARGET_POWER",
    "SWEEP_BEST_POWER",
    "SWEEP_BEST_N_FOR_TARGET_POWER",
)

_REVIEW_GRID_KEYS = (
    "REVIEW_GRID_N_SEEDS",
    "REVIEW_GRID_N_TRIALS",
    "REVIEW_GRID_RATES",
    "REVIEW_GRID_ATTACKS",
    "REVIEW_GRID_DIRECTIONAL",
    "REVIEW_GRID_ENTROPY_CONTROLS",
    "REVIEW_GRID_SELECTION_STATUS",
    "REVIEW_GRID_INDEPENDENT_UNIT",
    "REVIEW_GRID_TRIAL_STRUCTURE",
    "REVIEW_GRID_STATS_STATUS",
    "REVIEW_GRID_BH_OWNERSHIP",
    "REVIEW_GRID_TARGET_MAX_MCSE",
    "REVIEW_GRID_OBSERVED_MAX_MCSE",
    "REVIEW_GRID_SIGNED_CELLS",
)


#: Magnitude at which the JSON-safe d-equivalent sentinel is treated as
#: unbounded. The primary effect remains rank-biserial r.
_D_EQUIVALENT_SENTINEL = 1e6


def _format_d_equivalent(value: Any) -> str:
    """Format a rank-biserial-derived d-equivalent and its saturation boundary.

    The robustness report caps the secondary transform at ±1e6 so written JSON
    remains strict-standards (no ``Infinity``). A finite value prints with two
    decimals; the sentinel prints as a signed saturation marker rather than a
    misleading literal.
    """
    if value is None:
        return "N/A"
    d = float(value)
    if abs(d) >= _D_EQUIVALENT_SENTINEL:
        # d = 2r/sqrt(1-r^2) diverges as rank-biserial r saturates at +-1.
        return "saturated (r=+1)" if d > 0 else "saturated (r=-1)"
    return f"{d:.2f}"


def _format_cohens_d(value: Any) -> str:
    """Deprecated compatibility alias for :func:`_format_d_equivalent`."""
    warnings.warn(
        "_format_cohens_d is deprecated; use _format_d_equivalent",
        DeprecationWarning,
        stacklevel=2,
    )
    return _format_d_equivalent(value)


def _sweep_variables(sweep: dict[str, Any]) -> dict[str, str]:
    """Flatten the robustness-sweep report into manuscript tokens."""
    out: dict[str, str] = {}
    verdict: dict[str, dict] = sweep.get("verdict", {})
    worst_key = f"{float(sweep['worst_rate']):g}"
    accuracy = sweep["accuracy_by_method_and_rate"]

    out["SWEEP_NAIVE_ACCURACY"] = _fmt(accuracy["KLD"][worst_key])
    operating_points = sweep.get("server_robustness_by_label", {})
    labels = sweep.get("divergences", operating_points)
    out["SWEEP_SERVER_OPERATING_POINTS"] = (
        ", ".join(
            f"{label} (c={_fmt(operating_points[label], 2)})" for label in labels if label in operating_points
        )
        or "N/A"
    )

    # The robustness figure is now the uncertainty-aware trial profile, not the
    # deterministic mechanistic curve used by the table above. Keep distinct
    # tokens so its caption cannot silently quote the wrong estimand.
    profile = sweep.get("per_rate_summary", {})
    worst_profile = profile.get(worst_key, {})
    profile_methods = worst_profile.get("methods", {})
    out["SWEEP_PROFILE_NAIVE_ACCURACY"] = (
        _fmt(profile_methods["KLD"]["mean"]) if "KLD" in profile_methods else "N/A"
    )
    robust_profile = {method: cell for method, cell in profile_methods.items() if method != "KLD"}
    if robust_profile:
        profile_best = max(robust_profile.values(), key=lambda cell: cell["mean"])
        out["SWEEP_PROFILE_BEST_ROBUST_ACCURACY"] = _fmt(profile_best["mean"])
    else:
        out["SWEEP_PROFILE_BEST_ROBUST_ACCURACY"] = "N/A"

    # Headline method = largest positive rank-biserial effect in the verdict
    # panel. Ties and the stable display tie-break are explicit report fields.
    best_method, best = "", None
    for method, stats in verdict.items():
        if best is None or stats["effect_size"] > best["effect_size"]:
            best_method, best = method, stats
    if best is not None:
        out["SWEEP_BEST_ROBUST_METHOD"] = best_method
        out["SWEEP_BEST_ROBUST_ACCURACY"] = _fmt(accuracy[best_method][worst_key])
        out["SWEEP_BEST_EFFECT_SIZE"] = _fmt(best["effect_size"])
        out["SWEEP_BEST_QVALUE"] = _format_residual(float(best["qvalue"]))
        out["SWEEP_BEST_QVALUE_MATH"] = _format_residual_math(float(best["qvalue"]))
        # Secondary d-equivalent display, its magnitude label, and both raw +
        # BH-deflated p-values. Keep the old token as a deprecated compatibility
        # alias until downstream manuscript consumers migrate.
        d_value = best.get("d_equivalent", best.get("cohens_d"))
        out["SWEEP_BEST_D_EQUIVALENT"] = _format_d_equivalent(d_value)
        out["SWEEP_BEST_COHENS_D"] = out["SWEEP_BEST_D_EQUIVALENT"]
        out["SWEEP_BEST_EFFECT_LABEL"] = str(best.get("effect_label", "N/A"))
        out["SWEEP_BEST_RAW_PVALUE"] = _format_residual(float(best["raw_pvalue"]))
        out["SWEEP_BEST_RAW_PVALUE_MATH"] = _format_residual_math(float(best["raw_pvalue"]))
        # Mean naive-minus-robust accuracy difference and its 95% bootstrap CI.
        out["SWEEP_BEST_MEAN_ACC_DIFF"] = _fmt(best["mean_accuracy_diff"])
        diff_lo, diff_hi = best["mean_accuracy_diff_ci"]
        out["SWEEP_BEST_MEAN_ACC_DIFF_CI_LO"] = _fmt(diff_lo)
        out["SWEEP_BEST_MEAN_ACC_DIFF_CI_HI"] = _fmt(diff_hi)
    else:  # pragma: no cover - verdict always populated when robust methods exist
        for key in (
            "SWEEP_BEST_ROBUST_METHOD",
            "SWEEP_BEST_ROBUST_ACCURACY",
            "SWEEP_BEST_EFFECT_SIZE",
            "SWEEP_BEST_QVALUE",
            "SWEEP_BEST_QVALUE_MATH",
            "SWEEP_BEST_D_EQUIVALENT",
            "SWEEP_BEST_COHENS_D",
            "SWEEP_BEST_EFFECT_LABEL",
            "SWEEP_BEST_RAW_PVALUE",
            "SWEEP_BEST_RAW_PVALUE_MATH",
            "SWEEP_BEST_MEAN_ACC_DIFF",
            "SWEEP_BEST_MEAN_ACC_DIFF_CI_LO",
            "SWEEP_BEST_MEAN_ACC_DIFF_CI_HI",
        ):
            out[key] = "N/A"

    out["SWEEP_ANY_ROBUST_WINS"] = "Yes" if sweep["any_robust_wins"] else "No"
    out["SWEEP_NAIVE_DEGRADES"] = "Yes" if sweep["naive_degrades_with_rate"] else "No"
    out["SWEEP_ROBUST_ABOVE_THRESHOLD"] = "Yes" if sweep["robust_above_threshold_at_worst_rate"] else "No"
    out["SWEEP_ACCURACY_THRESHOLD"] = _fmt(sweep["accuracy_threshold"], 2)
    out["SWEEP_N_AGENTS"] = str(sweep["n_agents"])
    out["SWEEP_N_TRIALS"] = str(sweep["n_trials"])
    out["SWEEP_WORST_RATE"] = _fmt(sweep["worst_rate"], 3)
    out["SWEEP_VERDICT_RATE"] = _fmt(sweep["verdict_rate"], 3)
    out["SWEEP_RATES"] = ", ".join(f"{r:g}" for r in sweep["rates"])
    out["SWEEP_N_CONTAMINATED"] = str(sweep["n_contaminated"])
    # Top-level sample size behind every paired contrast (trials per condition).
    out["SWEEP_N"] = str(sweep["n"])
    out["SWEEP_NAIVE_VERDICT_RATE_MEAN"] = _fmt(sweep["naive_verdict_rate_mean"])

    # --- Power analysis of the headline robust-vs-naive verdict --------------
    # Observed-effect design power of the best robust method's paired Wilcoxon
    # at the run's n, and the prospective n a confirmatory replication should
    # budget to reach the target power at the headline observed effect. Honesty:
    # these quantify the SERVER-SIDE aggregation heuristic's contrast, not the
    # beta/rcce per-agent FedGVI guarantee.
    out["SWEEP_POWER_ALPHA"] = _fmt(sweep["power_alpha"], 2)
    out["SWEEP_POWER_ALTERNATIVE"] = str(sweep["power_alternative"])
    out["SWEEP_TARGET_POWER"] = _fmt(sweep["target_power"], 2)
    out["SWEEP_HEADLINE_POWER"] = _fmt(sweep["headline_power"], 4)
    out["SWEEP_HEADLINE_METHOD"] = str(sweep.get("headline_method", ""))
    out["SWEEP_HEADLINE_SELECTION_RULE"] = str(sweep.get("headline_selection_rule", "N/A"))
    tie_set = sweep.get("headline_tie_set", [])
    out["SWEEP_HEADLINE_TIE_SET"] = ", ".join(str(item) for item in tie_set) or "none"
    out["SWEEP_HEADLINE_TIE_BREAK"] = str(sweep.get("headline_tie_break", "N/A"))
    out["SWEEP_LARGEST_MEAN_DIFFERENCE_METHOD"] = str(sweep.get("largest_mean_difference_method", "N/A"))
    out["SWEEP_WORST_RATE_BEST_METHOD"] = str(sweep.get("worst_rate_best_method", "N/A"))
    out["SWEEP_PROSPECTIVE_N"] = str(sweep["prospective_n_for_target_power"])
    out["SWEEP_HEADLINE_N_FOR_TARGET_POWER"] = str(sweep["headline_n_for_target_power"])
    # Headline display method's own observed-effect design power / prospective n.
    if best is not None:
        out["SWEEP_BEST_POWER"] = _fmt(float(best.get("power", 0.0)), 4)
        out["SWEEP_BEST_N_FOR_TARGET_POWER"] = str(best.get("n_for_target_power", "N/A"))
    else:  # pragma: no cover - verdict always populated when robust methods exist
        out["SWEEP_BEST_POWER"] = "N/A"
        out["SWEEP_BEST_N_FOR_TARGET_POWER"] = "N/A"

    # Accuracy at the verdict (worst) rate, per method, with bootstrap CIs. The
    # The naive baseline is the project KLD log-linear pool. Under the explicit
    # bridge assumptions it specializes Friston Eq. 7's message-combination
    # term; the headline robust method is the largest-effect display choice.
    at_verdict = sweep.get("accuracy_at_verdict_rate", {})
    naive_av = at_verdict.get("KLD")
    if naive_av is not None:
        out["SWEEP_NAIVE_VERDICT_ACCURACY_MEAN"] = _fmt(naive_av["mean"])
        n_lo, n_hi = naive_av["ci"]
        out["SWEEP_NAIVE_VERDICT_ACCURACY_CI_LO"] = _fmt(n_lo)
        out["SWEEP_NAIVE_VERDICT_ACCURACY_CI_HI"] = _fmt(n_hi)
    else:  # pragma: no cover - KLD baseline always present in the sweep
        out["SWEEP_NAIVE_VERDICT_ACCURACY_MEAN"] = "N/A"
        out["SWEEP_NAIVE_VERDICT_ACCURACY_CI_LO"] = "N/A"
        out["SWEEP_NAIVE_VERDICT_ACCURACY_CI_HI"] = "N/A"
    best_av = at_verdict.get(best_method) if best_method else None
    if best_av is not None:
        out["SWEEP_BEST_VERDICT_ACCURACY_MEAN"] = _fmt(best_av["mean"])
        b_lo, b_hi = best_av["ci"]
        out["SWEEP_BEST_VERDICT_ACCURACY_CI_LO"] = _fmt(b_lo)
        out["SWEEP_BEST_VERDICT_ACCURACY_CI_HI"] = _fmt(b_hi)
    else:  # pragma: no cover - winning robust method always has a verdict entry
        out["SWEEP_BEST_VERDICT_ACCURACY_MEAN"] = "N/A"
        out["SWEEP_BEST_VERDICT_ACCURACY_CI_LO"] = "N/A"
        out["SWEEP_BEST_VERDICT_ACCURACY_CI_HI"] = "N/A"

    # | method | n | mean accuracy @ verdict rate | 95% CI |
    av_rows = []
    for method in sweep["divergences"]:
        cell = at_verdict.get(method)
        if cell is None:  # pragma: no cover - all divergences carry a cell
            continue
        lo, hi = cell["ci"]
        av_rows.append(f"| {method} | {cell['n']} | {_fmt(cell['mean'])} | [{_fmt(lo)}, {_fmt(hi)}] |")
    out["SWEEP_ACCURACY_AT_VERDICT_TABLE_ROWS"] = "\n".join(av_rows)

    rate_rows = []
    for rate in sweep["rates"]:
        key = f"{float(rate):g}"
        cells = " | ".join(_fmt(accuracy[m][key], 3) for m in sweep["divergences"])
        rate_rows.append(f"| {key} | {cells} |")
    out["SWEEP_RATE_TABLE_ROWS"] = "\n".join(rate_rows)

    verdict_rows = []
    for method, stats in verdict.items():
        verdict_rows.append(
            f"| {method} | {_fmt(stats['effect_size'])} | "
            f"{_format_residual(float(stats['pvalue']))} | "
            f"{_format_residual(float(stats['qvalue']))} | "
            f"{'Yes' if stats['wins'] else 'No'} |"
        )
    out["SWEEP_VERDICT_TABLE_ROWS"] = "\n".join(verdict_rows)

    # Standardized-effect verdict table (with planning power and prospective n):
    # | method | d-equivalent | label | mean acc. diff | 95% CI | raw p | q | power |
    # | n for target power | reject |
    effect_rows = []
    for method, stats in verdict.items():
        diff_lo, diff_hi = stats["mean_accuracy_diff_ci"]
        effect_rows.append(
            f"| {method} | {_format_d_equivalent(stats.get('d_equivalent', stats.get('cohens_d')))} | "
            f"{stats.get('effect_label', 'N/A')} | "
            f"{_fmt(stats['mean_accuracy_diff'])} | "
            f"[{_fmt(diff_lo)}, {_fmt(diff_hi)}] | "
            f"{_format_residual(float(stats['raw_pvalue']))} | "
            f"{_format_residual(float(stats['qvalue']))} | "
            f"{_fmt(float(stats.get('power', 0.0)), 4)} | "
            f"{stats.get('n_for_target_power', 'N/A')} | "
            f"{'Yes' if stats['rejected'] else 'No'} |"
        )
    out["SWEEP_VERDICT_EFFECT_TABLE_ROWS"] = "\n".join(effect_rows)

    # Per-contamination-rate naive-vs-robust paired tests (BH-deflated per
    # method): | method | rate | d-equivalent | label | raw p | q | reject |.
    paired = sweep.get("paired_tests_by_rate", {})
    paired_rows = []
    for method in sweep["divergences"]:
        by_rate = paired.get(method)
        if by_rate is None:  # KLD is the naive baseline — no self-contrast.
            continue
        for rate in sweep["rates"]:
            cell = by_rate.get(f"{float(rate):g}")
            if cell is None:  # pragma: no cover - every rate carries a cell
                continue
            paired_rows.append(
                f"| {method} | {float(rate):g} | "
                f"{_format_d_equivalent(cell.get('d_equivalent', cell.get('cohens_d')))} | "
                f"{cell.get('effect_label', 'N/A')} | "
                f"{_format_residual(float(cell['raw_pvalue']))} | "
                f"{_format_residual(float(cell['qvalue']))} | "
                f"{'Yes' if cell['rejected'] else 'No'} |"
            )
    out["SWEEP_PAIRED_BY_RATE_TABLE_ROWS"] = "\n".join(paired_rows)
    return out
