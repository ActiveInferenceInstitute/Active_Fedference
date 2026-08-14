"""Manuscript variable submodule."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
from pathlib import Path

import numpy as np
import scipy

from experiment_config import (
    DEFAULT_SENSITIVITY_ACUITY,
    DEFAULT_SENSITIVITY_COLONY_SIZES,
    load_experiment_config,
    load_manuscript_config,
)
from fedference.experiments import run_belief_sharing, run_emergence
from publication.identifiers import doi_url, normalize_doi

from .loaders import (
    _BOOTSTRAP_ALPHA,
    _CROSS_STUDY_SENS_N_TRIALS,
    _N_BOOT,
    _OFFSWITCH_ALPHA,
    _OFFSWITCH_BETA,
    _OFFSWITCH_Q,
    _SENS_N_ACUITY_LEVELS,
    _SENS_N_COLONY_SIZES,
    _SENS_N_TRIALS,
    _SENS_NOISE_FLOOR,
    _SENS_SEED_BASE,
    _TOKEN_BLOCK_EXPECTED,
    _bnn_robustness_variables,
    _bnn_torch_variables,
    _build_timestamp,
    _count_isc,
    _count_tests,
    _coverage_percent,
    _disjoint_fov_variables,
    _federation_variables,
    _fmt,
    _format_residual,
    _format_residual_math,
    _hierarchical_variables,
    _load_report,
    _moving_world_variables,
    _nlevel3_variables,
    _project_root,
    _recovery_residuals,
    _tempered_variables,
    _validation_receipt_variables,
)
from .tokens import (
    _COMPLEXITY_KEYS,
    _REVIEW_GRID_KEYS,
    _SWEEP_KEYS,
    _complexity_variables,
    _gallery_variables,
    _onset_variables,
    _review_grid_variables,
    _sweep_variables,
    _variational_variables,
)

_log = logging.getLogger(__name__)


def _apply_report_tokens(
    variables: dict[str, str],
    root: Path,
    allow_draft: bool,
    report_name: str,
    keys: tuple[str, ...],
    loader,
    *,
    strict: bool = False,
) -> None:
    """Load tokens from a JSON report or degrade explicitly when absent."""
    try:
        variables.update(loader(root))
    except FileNotFoundError:
        if strict and not allow_draft:
            raise
        _log.warning("Report %s missing — degrading %d tokens to N/A", report_name, len(keys))
        for key in keys:
            variables[key] = "N/A"


def generate_variables(
    project_root: Path | None = None,
    allow_draft: bool = False,
    *,
    require_validation_receipt: bool = False,
) -> dict[str, str]:
    """Resolve every manuscript token placeholder to a string value.

    Reads the JSON reports produced by the analysis workflow and computes the
    deterministic FedGVI recovery / emergence / statistics constants from the
    locked core. Missing reports degrade to ``"N/A"`` sentinels so a draft
    manuscript still renders.

    Args:
        project_root: Directory containing ``manuscript/`` and ``output/``;
            defaults to the project root.
        allow_draft: When ``False`` (default), missing core analysis reports
            raise :class:`FileNotFoundError`. When ``True``, missing reports
            degrade affected tokens to ``"N/A"`` so draft manuscripts still render.
        require_validation_receipt: When ``True``, load test-count, coverage,
            and core environment tokens only from a fresh successful
            test-and-coverage receipt. The final hydration CLI enables this;
            library callers and provisional renders retain the bounded legacy
            fallbacks so isolated analysis fixtures remain usable.

    Returns:
        Flat mapping of UPPERCASE_KEY -> string (no surrounding braces).
    """
    if allow_draft and require_validation_receipt:
        raise ValueError("draft generation cannot require a validation receipt")
    root = _project_root(project_root)
    reports = root / "output" / "reports"
    if not allow_draft:
        required = (
            "belief_sharing.json",
            "language_acquisition.json",
            "robustness_sweep.json",
            "complexity_scaling.json",
            "conditional_world.json",
            "belief_quality.json",
            "robustness_review_grid.json",
        )
        missing = [name for name in required if not (reports / name).exists()]
        if missing:
            raise FileNotFoundError(
                "Missing required analysis reports (pass --allow-draft to degrade): " + ", ".join(missing)
            )
    cfg = load_experiment_config(root)

    bs = _load_report(reports, "belief_sharing.json")
    lang = _load_report(reports, "language_acquisition.json")
    sweep = _load_report(reports, "robustness_sweep.json")
    variational = _load_report(reports, "variational_aggregation.json")
    gallery = _load_report(reports, "contamination_gallery.json")
    onset = _load_report(reports, "robustness_onset.json")
    complexity = _load_report(reports, "complexity_scaling.json")
    conditional = _load_report(reports, "conditional_world.json")
    quality = _load_report(reports, "belief_quality.json")
    review_grid = _load_report(reports, "robustness_review_grid.json")

    v: dict[str, str] = {}

    # ---- Generic / provenance ----
    _isc_total, _isc_passed = _count_isc(root)
    v["ISC_TOTAL"] = _isc_total
    v["ISC_PASSED"] = _isc_passed
    if require_validation_receipt:
        v.update(_validation_receipt_variables(root))
    else:
        v["TEST_COUNT"] = _count_tests(root)
        v["COVERAGE_PERCENT"] = _coverage_percent(root)
    v["EXPERIMENT_SEED"] = str(cfg.seeds[0])
    # Every frozen experiment-config value as a token — no hardcoded numbers
    # downstream. (Some are echoed from the sweep report below as the *executed*
    # value; these are the configured source-of-truth values.)
    v["CONFIG_N_AGENTS"] = str(cfg.n_agents)
    v["CONFIG_N_LOCATIONS"] = str(cfg.n_locations)
    v["CONFIG_N_SEEDS"] = str(cfg.n_seeds)
    v["CONFIG_REPLICATE_SEEDS"] = str(cfg.replicate_seeds)
    v["CONFIG_CROSS_STUDY_N_TRIALS"] = str(cfg.cross_study_n_trials)
    v["CONFIG_CONDITIONAL_WORLD_N_SEEDS"] = str(cfg.conditional_world_n_seeds)
    v["CONFIG_CONDITIONAL_WORLD_N_TRIALS"] = str(cfg.conditional_world_n_trials)
    # Number of creature control actions (still / left / right) — derived from
    # the CONTROL_LABELS constant in pomdp.py so prose never hardcodes the count.
    from fedference.pomdp import CONTROL_LABELS as _CONTROL_LABELS

    v["CONFIG_N_ACTIONS"] = str(len(_CONTROL_LABELS))
    v["CONFIG_N_TRIALS"] = str(cfg.n_trials)
    v["CONFIG_DIVERGENCES"] = ", ".join(cfg.divergences)
    v["CONFIG_ROBUST_DIVERGENCES"] = ", ".join(cfg.robust_divergences)
    v["CONFIG_CONTAMINATION_RATES"] = ", ".join(f"{r:g}" for r in cfg.contamination_rates)
    v["CONFIG_N_RATES"] = str(len(cfg.contamination_rates))
    v["CONFIG_POWER_ALPHA"] = _fmt(cfg.power_alpha, 2)
    v["CONFIG_POWER_ALTERNATIVE"] = cfg.power_alternative
    v["CONFIG_TARGET_POWER"] = _fmt(cfg.target_power, 2)
    v["GENERATION_TIMESTAMP"] = _build_timestamp(os.environ.get("SOURCE_DATE_EPOCH"))
    if not require_validation_receipt:
        v["PYTHON_VERSION"] = platform.python_version()
        v["NUMPY_VERSION"] = np.__version__
        v["SCIPY_VERSION"] = scipy.__version__
        v["PLATFORM"] = f"{platform.system()} {platform.machine()}"
    # Executed BH-FDR level: strict-read from the sweep report the verdict
    # actually ran at (a pre-fdr_alpha report must be regenerated — never
    # papered over by a constant that can silently desynchronize from the run).
    v["STATISTICS_FDR_ALPHA"] = _fmt(float(sweep["fdr_alpha"]), 2) if sweep else "N/A"
    # EFE identity pin tolerance — bound to the shared EFE_IDENTITY_ATOL
    # constant (ISC-19) so the formalism prose tracks the code, never a
    # re-typed literal.
    from fedference.expected_free_energy import EFE_IDENTITY_ATOL

    _efe_log = float(np.log10(EFE_IDENTITY_ATOL))
    if abs(_efe_log - round(_efe_log)) < 1e-9:
        v["ISC_EFE_TOLERANCE"] = f"10^{{{int(round(_efe_log))}}}"
    else:
        # Non-power-of-ten tolerance: render the literal value rather than
        # silently rounding to the nearest decade.
        v["ISC_EFE_TOLERANCE"] = f"{EFE_IDENTITY_ATOL:.0e}"
    # CI confidence level (percent) = 100*(1 - bootstrap alpha); tracks the analysis
    # choice so prose never hardcodes "95%".
    v["CI_PERCENT"] = str(int(round((1.0 - _BOOTSTRAP_ALPHA) * 100)))

    # ---- Sensitivity sweep parameters (Study 8 / S13-S14) ----
    # Grid definitions must match run_belief_sharing_sensitivity defaults.
    _sens_acuity_values = DEFAULT_SENSITIVITY_ACUITY
    _sens_colony_sizes = DEFAULT_SENSITIVITY_COLONY_SIZES
    v["SENS_N_TRIALS"] = str(_SENS_N_TRIALS)
    v["SENS_SEED_BASE"] = str(_SENS_SEED_BASE)
    v["SENS_N_ACUITY_LEVELS"] = str(_SENS_N_ACUITY_LEVELS)
    v["SENS_N_COLONY_SIZES"] = str(_SENS_N_COLONY_SIZES)
    v["SENS_N_CELLS"] = str(_SENS_N_ACUITY_LEVELS * _SENS_N_COLONY_SIZES)
    v["SENS_NOISE_FLOOR"] = f"{_SENS_NOISE_FLOOR:.2f}"
    # Grid display strings for prose citations (e.g. "\\{0.40, 0.55, ...\\}")
    _acuity_str = ", ".join(f"{a:.2f}" for a in _sens_acuity_values)
    _colony_str = ", ".join(str(n) for n in _sens_colony_sizes)
    v["SENS_ACUITY_GRID"] = "\\{" + _acuity_str + "\\}"
    v["SENS_COLONY_SIZE_GRID"] = "\\{" + _colony_str + "\\}"
    v["SENS_ACUITY_MIN"] = f"{min(_sens_acuity_values):.2f}"
    v["SENS_ACUITY_MAX"] = f"{max(_sens_acuity_values):.2f}"
    v["SENS_COLONY_SIZE_MIN"] = str(min(_sens_colony_sizes))
    v["SENS_COLONY_SIZE_MAX"] = str(max(_sens_colony_sizes))
    v["BOOTSTRAP_N_BOOT"] = str(_N_BOOT)

    # ---- Config / authorship (from manuscript/config.yaml) ----
    config_path = root / "manuscript" / "config.yaml"
    paper_version, first_author, keywords = "1.0", "Unknown", ""
    publication_doi: str | None = None
    config_hash = "N/A"
    if config_path.exists():
        config_bytes = config_path.read_bytes()
        config_hash = hashlib.sha256(config_bytes).hexdigest()[:16]
        data = load_manuscript_config(root)
        paper_version = str(data.get("paper", {}).get("version", "1.0"))
        authors = data.get("authors", [])
        first_author = authors[0].get("name", "Unknown") if authors else "Unknown"
        keywords = ", ".join(data.get("keywords", []))
        publication_doi = normalize_doi(
            data.get("publication", {}).get("doi"),
            allow_placeholder=True,
        )
    v["CONFIG_VERSION"] = paper_version
    v["CONFIG_FIRST_AUTHOR"] = first_author
    v["CONFIG_KEYWORDS"] = keywords
    v["CONFIG_HASH"] = config_hash
    v["PUBLICATION_DOI"] = publication_doi or "N/A"
    v["PUBLICATION_DOI_URL"] = doi_url(publication_doi) or "N/A"

    # ---- Artifact counts ----
    counts = {"figures": 0, "data": 0, "reports": 0}
    for sub in counts:
        d = root / "output" / sub
        counts[sub] = sum(1 for f in d.iterdir() if f.is_file()) if d.exists() else 0
    v["ARTIFACT_FIGURES"] = str(counts["figures"])
    v["ARTIFACT_DATA_FILES"] = str(counts["data"])
    v["ARTIFACT_REPORTS"] = str(counts["reports"])
    v["ARTIFACT_TOTAL"] = str(sum(counts.values()))

    # ---- Belief sharing (Fig. 5) ----
    if bs:
        v["BELIEF_SHARING_MEAN_F_COMMUNICATE"] = _fmt(bs["communicating_mean"])
        v["BELIEF_SHARING_MEAN_F_INCOMMUNICADO"] = _fmt(bs["incommunicado_mean"])
        v["BELIEF_SHARING_DELTA_F"] = _fmt(bs["free_energy_gap"])
        v["BELIEF_SHARING_N_AGENTS"] = str(bs["n_agents"])
        # Sample size behind the across-seed colony means (one F per seed).
        v["BELIEF_SHARING_N_SEEDS"] = str(bs["n_seeds"])
        # Across-seed bootstrap CI of the headline seed-mean F — the interval
        # that belongs to BELIEF_SHARING_MEAN_F_COMMUNICATE (one sample per
        # seed). The per-agent seed-0 CI below is a different quantity and
        # must never be attached to the seed mean (review finding: the two
        # were conflated, producing a CI excluding its own point estimate).
        from fedference.statistics import bootstrap_ci as _bootstrap_ci

        _seed_ci_lo, _seed_ci_hi = _bootstrap_ci(
            bs["communicating_free_energy"],
            alpha=_BOOTSTRAP_ALPHA,
            n_boot=_N_BOOT,
            rng=np.random.default_rng(cfg.seeds[0]),
        )
        v["BELIEF_SHARING_MEAN_F_SEED_CI_LO"] = _fmt(_seed_ci_lo)
        v["BELIEF_SHARING_MEAN_F_SEED_CI_HI"] = _fmt(_seed_ci_hi)
    else:
        for key in (
            "BELIEF_SHARING_MEAN_F_COMMUNICATE",
            "BELIEF_SHARING_MEAN_F_INCOMMUNICADO",
            "BELIEF_SHARING_DELTA_F",
            "BELIEF_SHARING_N_AGENTS",
            "BELIEF_SHARING_N_SEEDS",
            "BELIEF_SHARING_MEAN_F_SEED_CI_LO",
            "BELIEF_SHARING_MEAN_F_SEED_CI_HI",
        ):
            v[key] = "N/A"

    # Mean surprise / accuracy / acuity come from a fresh seeded share round so
    # the figure-3 prose resolves even without re-reading per-agent detail. The
    # same round carries the per-agent sample size n and the 95% bootstrap CI of
    # the colony-mean free energy (statistics enrichment).
    bs_detail = run_belief_sharing(cfg.seeds[0], communicate=True, n_agents=cfg.n_agents)
    v["BELIEF_SHARING_MEAN_SURPRISE"] = _fmt(bs_detail["mean_surprise"])
    v["BELIEF_SHARING_MEAN_ACCURACY"] = _fmt(bs_detail["mean_accuracy"])
    # Acuity comes from the experiment's own default constant — one definition.
    from fedference.experiments.belief_sharing import DEFAULT_ACUITY

    v["BELIEF_SHARING_ACUITY"] = _fmt(DEFAULT_ACUITY, 2)
    v["BELIEF_SHARING_N"] = str(bs_detail["n"])
    # The illustrative seed-0 run's own colony-mean F — the point estimate the
    # per-agent CI below actually belongs to (never attach that CI to the
    # across-seed mean; review finding: the interval excluded the seed mean).
    v["BELIEF_SHARING_SEED0_MEAN_F"] = _fmt(bs_detail["mean_free_energy"])
    fe_ci_lo, fe_ci_hi = bs_detail["mean_free_energy_ci"]
    v["BELIEF_SHARING_MEAN_F_CI_LO"] = _fmt(fe_ci_lo)
    v["BELIEF_SHARING_MEAN_F_CI_HI"] = _fmt(fe_ci_hi)
    # Seed and hidden-state grid size used for the heatmap figure caption.
    v["BELIEF_SHARING_SEED"] = str(cfg.seeds[0])
    v["BELIEF_SHARING_N_STATES"] = str(cfg.n_locations)

    # ---- Emergence (Fig. 9, Bayesian model reduction) ----
    # Single deterministic BMR evidence comparison: n is the candidate-state
    # count it ranges over. By design there is no resampled sample here, hence
    # no CI / paired test (documented in run_emergence).
    emergence = run_emergence(cfg.seeds[0])
    v["EMERGENCE_CONVERGENCE"] = "Yes" if emergence["convergence"] else "No"
    v["EMERGENCE_DELTA_F_REDUNDANT"] = _fmt(emergence["delta_F_redundant"], 2)
    v["EMERGENCE_DELTA_F_SUPPORTED"] = _fmt(emergence["delta_F_supported"], 2)
    v["EMERGENCE_N"] = str(emergence["n"])

    # ---- Language acquisition (Fig. 7) ----
    if lang:
        v["LANGUAGE_INITIAL_KL"] = _fmt(lang["initial_kl"])
        v["LANGUAGE_FINAL_KL"] = _fmt(lang["final_kl"])
        v["LANGUAGE_KL_REDUCTION"] = _fmt(lang["initial_kl"] - lang["final_kl"])
        v["LANGUAGE_MONOTONE"] = "Yes" if lang["monotone_decreasing"] else "No"
        v["LANGUAGE_NUM_STEPS"] = str(lang["num_steps"])
        # The ordered curve has points, while independent configured seeds are
        # the replication unit for the pointwise bootstrap interval.
        v["LANGUAGE_N_POINTS"] = str(lang["n_points"])
        v["LANGUAGE_N_SEEDS"] = str(lang["n_seeds"])
    else:
        for key in (
            "LANGUAGE_INITIAL_KL",
            "LANGUAGE_FINAL_KL",
            "LANGUAGE_KL_REDUCTION",
            "LANGUAGE_MONOTONE",
            "LANGUAGE_NUM_STEPS",
            "LANGUAGE_N_POINTS",
            "LANGUAGE_N_SEEDS",
        ):
            v[key] = "N/A"

    # ---- Robustness sweep (ISC-27/30) ----
    if sweep:
        v.update(_sweep_variables(sweep))
    else:
        for key in _SWEEP_KEYS:
            v[key] = "N/A"

    # ---- Variational aggregation (axis-2 made rigorous) ----
    if variational:
        v.update(_variational_variables(variational))
    else:
        for key in _VARIATIONAL_KEYS:
            v[key] = "N/A"

    # ---- Contamination gallery (robustness across attack mechanisms) ----
    if gallery:
        v.update(_gallery_variables(gallery))
    else:
        for key in _GALLERY_KEYS:
            v[key] = "N/A"

    # ---- Robustness onset (rate where robust overtakes naive, per mechanism) ----
    if onset:
        v.update(_onset_variables(onset))
    else:
        for key in _ONSET_KEYS:
            v[key] = "N/A"

    # ---- Source-bound robustness review grid -------------------------------
    if review_grid:
        v.update(
            _review_grid_variables(
                review_grid,
                configured_target_max_mcse=cfg.review_grid_target_max_mcse,
                require_reported_target=require_validation_receipt,
            )
        )
    else:
        for key in _REVIEW_GRID_KEYS:
            v[key] = "N/A"

    # ---- FedGVI recovery residuals (deterministic constants) ----
    # Each residual is emitted twice: the plain ``.2e`` form for prose/tables,
    # and a ``*_MATH`` sibling in LaTeX scientific notation for $...$ spans.
    for key, val in _recovery_residuals().items():
        v[key] = _format_residual(val)
        v[f"{key}_MATH"] = _format_residual_math(val)

    # ---- M2: off-switch-point parameter values (the constants themselves,
    # not residuals) so the manuscript cites the exact q_loss/beta/alpha used ----
    v["RECOVERY_OFFSWITCH_Q"] = str(_OFFSWITCH_Q)
    v["RECOVERY_OFFSWITCH_Q_MATH"] = _format_residual_math(_OFFSWITCH_Q)
    v["RECOVERY_OFFSWITCH_BETA"] = str(_OFFSWITCH_BETA)
    v["RECOVERY_OFFSWITCH_ALPHA"] = str(_OFFSWITCH_ALPHA)

    # ---- Parameter recovery (generative-model identifiability) ----
    param_rec = _load_report(reports, "parameter_recovery.json")
    if param_rec:
        v["PARAM_RECOVERY_MEAN_ABS_ERROR"] = _fmt(param_rec["mean_abs_error"], 4)
        v["PARAM_RECOVERY_R_SQUARED"] = _fmt(param_rec["r_squared"], 4)
        v["PARAM_RECOVERY_N_TRIALS"] = str(int(param_rec["n_trials"]))
        v["PARAM_RECOVERY_N_OBSERVATIONS"] = str(int(param_rec["n_observations"]))
        v["PARAM_RECOVERY_INTERVAL_PERCENT"] = str(int(param_rec.get("interval_percent", 95)))
        v["PARAM_RECOVERY_ACUITY_GRID"] = ", ".join(f"{float(a):.2f}" for a in param_rec["acuity_grid"])
    else:
        for key in (
            "PARAM_RECOVERY_MEAN_ABS_ERROR",
            "PARAM_RECOVERY_R_SQUARED",
            "PARAM_RECOVERY_N_TRIALS",
            "PARAM_RECOVERY_N_OBSERVATIONS",
            "PARAM_RECOVERY_INTERVAL_PERCENT",
            "PARAM_RECOVERY_ACUITY_GRID",
        ):
            v[key] = "N/A"

    # ---- V1/V3/V4 token additions (optional import-only blocks) ----
    try:
        v.update(_tempered_variables(root))
    except _TOKEN_BLOCK_EXPECTED as exc:
        _log.warning("Tempered-aggregation tokens omitted (fallback taken): %r", exc)
    try:
        v.update(_federation_variables(root))
    except _TOKEN_BLOCK_EXPECTED as exc:
        _log.warning("Federation-transport tokens omitted (fallback taken): %r", exc)

    _apply_report_tokens(
        v,
        root,
        allow_draft,
        "moving_world.json",
        _MOVING_KEYS,
        _moving_world_variables,
    )
    _apply_report_tokens(
        v,
        root,
        allow_draft,
        "disjoint_fov_world.json",
        _DISJOINT_KEYS,
        _disjoint_fov_variables,
    )
    _apply_report_tokens(
        v,
        root,
        allow_draft,
        "hierarchical_world.json",
        _HIER_KEYS,
        _hierarchical_variables,
    )
    _apply_report_tokens(
        v,
        root,
        allow_draft,
        "nlevel3_world.json",
        _NLEVEL3_KEYS,
        _nlevel3_variables,
    )

    # ---- PyTorch point-mass MLP FedGVI complement tokens ----
    # Sourced from the EXECUTED report output/reports/bnn_torch.json when the
    # analysis pipeline ran with PyTorch installed. Absent that report (PyTorch
    # not installed), the configuration tokens degrade to documented defaults and
    # the executed tokens read a clearly labeled "N/A (PyTorch not run)". The
    # generator itself never imports torch — it only reads the JSON the pipeline
    # wrote.
    v.update(_bnn_torch_variables(root))

    # ---- Federated logistic-regression baseline tokens (fig:bnn-robustness) ----
    v.update(_bnn_robustness_variables(root))

    # ---- System overview / cover schematic tokens (intro + abstract captions).
    # SYSTEM_OVERVIEW_METADATA percentages are DERIVED inside the figure module
    # from its own pooled beliefs (never typed); these tokens re-export them.
    # The schematic is a single deterministic construction — it has no seeds,
    # trials, or CI, so no such tokens exist for it.
    from figures.system_overview import (
        N_STATES as _SO_N_STATES,
    )
    from figures.system_overview import (
        SYSTEM_OVERVIEW_METADATA,
    )
    from figures.system_overview import (
        TRUE_STATE as _SO_TRUE_STATE,
    )

    v["SYSTEM_OVERVIEW_N_AGENTS"] = str(SYSTEM_OVERVIEW_METADATA["n_agents"])
    v["SYSTEM_OVERVIEW_N_ADVERSARIAL"] = str(SYSTEM_OVERVIEW_METADATA["n_adversarial"])
    v["SYSTEM_OVERVIEW_N_HONEST"] = str(SYSTEM_OVERVIEW_METADATA["n_honest"])
    v["SYSTEM_OVERVIEW_CONTAMINATION_PCT"] = str(SYSTEM_OVERVIEW_METADATA["contamination_pct"])
    v["SYSTEM_OVERVIEW_NAIVE_ACC_PCT"] = str(SYSTEM_OVERVIEW_METADATA["naive_acc_pct"])
    v["SYSTEM_OVERVIEW_ROBUST_ACC_PCT"] = str(SYSTEM_OVERVIEW_METADATA["robust_acc_pct"])
    v["SYSTEM_OVERVIEW_N_STATES"] = str(_SO_N_STATES)
    v["SYSTEM_OVERVIEW_TRUE_STATE_DISPLAY"] = str(_SO_TRUE_STATE + 1)

    # ---- Hierarchical BMR / structure-learning tokens (Study 10, MAJ-7) ----
    hbmr = _load_report(reports, "hierarchical_bmr.json")
    if hbmr:
        v["HBMR_DEGEN_TOP_SURPRISE"] = _fmt(float(hbmr["degenerate_top_surprise"]), 3)
        v["HBMR_INFORM_TOP_SURPRISE"] = _fmt(float(hbmr["informative_top_surprise"]), 3)
        v["HBMR_DEGEN_PRUNES_TOP"] = "Yes" if hbmr["degenerate_recommends_prune_top"] else "No"
        v["HBMR_INFORM_KEEPS_TOP"] = "Yes" if hbmr["informative_keeps_top"] else "No"
        v["HBMR_N_LEVELS"] = str(int(hbmr["n_levels"]))
    else:
        for _k in (
            "HBMR_DEGEN_TOP_SURPRISE",
            "HBMR_INFORM_TOP_SURPRISE",
            "HBMR_DEGEN_PRUNES_TOP",
            "HBMR_INFORM_KEEPS_TOP",
            "HBMR_N_LEVELS",
        ):
            v[_k] = "N/A"

    # ---- Heuristic characterization tokens (MAJ-1: breakdown witness) ----
    hchar = _load_report(reports, "heuristic_characterization.json")
    if hchar:
        bd = hchar["breakdown"]
        v["HCHAR_ROBUST_BREAKDOWN_K"] = str(int(bd["robust_breakdown_k"]))
        v["HCHAR_VARIATIONAL_BREAKDOWN_K"] = str(int(bd["variational_breakdown_k"]))
        v["HCHAR_N_HONEST"] = str(int(bd["n_honest"]))
        v["HCHAR_HAS_FINITE_BREAKDOWN"] = "Yes" if bd["robust_has_finite_breakdown"] else "No"
    else:
        for _k in (
            "HCHAR_ROBUST_BREAKDOWN_K",
            "HCHAR_VARIATIONAL_BREAKDOWN_K",
            "HCHAR_N_HONEST",
            "HCHAR_HAS_FINITE_BREAKDOWN",
        ):
            v[_k] = "N/A"

    # ---- Cross-study summary tokens (study count + executed seed count) ----
    cross = _load_report(reports, "cross_study_summary.json")
    if cross:
        v["N_STUDIES"] = str(len(cross["studies"]))
        v["CROSS_STUDY_N_SEEDS"] = str(int(cross["n_seeds"]))
        v["CROSS_STUDY_N_TRIALS"] = str(int(cross.get("n_trials", cfg.cross_study_n_trials)))
    else:
        v["N_STUDIES"] = "N/A"
        v["CROSS_STUDY_N_SEEDS"] = "N/A"
        v["CROSS_STUDY_N_TRIALS"] = "N/A"
    v["CROSS_STUDY_SENS_N_TRIALS"] = str(_CROSS_STUDY_SENS_N_TRIALS)

    # ---- Implementation complexity and measured scaling diagnostics ----
    if complexity:
        v.update(_complexity_variables(complexity))
    else:
        for key in _COMPLEXITY_KEYS:
            v[key] = "N/A"

    # ---- Conditional-world and proper-score extensions -------------------
    if conditional:
        grid = conditional.get("grid", [])
        controls = conditional.get("controls", {})
        v["CONDITIONAL_N_SCENARIOS"] = str(len(grid)) if isinstance(grid, list) else "N/A"
        v["CONDITIONAL_N_SEEDS"] = str(conditional.get("n_seeds", "N/A"))
        v["CONDITIONAL_N_TRIALS"] = str(conditional.get("n_trials", "N/A"))
        v["CONDITIONAL_ZERO_CONTROL"] = (
            "pass"
            if isinstance(controls, dict) and controls.get("robustness_zero_recovers_log_pool")
            else "fail"
        )
        v["CONDITIONAL_CLAIM_STATUS"] = str(conditional.get("claim_status", "N/A"))
    else:
        for key in (
            "CONDITIONAL_N_SCENARIOS",
            "CONDITIONAL_N_SEEDS",
            "CONDITIONAL_N_TRIALS",
            "CONDITIONAL_ZERO_CONTROL",
            "CONDITIONAL_CLAIM_STATUS",
        ):
            v[key] = "N/A"
    if quality:
        controls = quality.get("controls", {})
        v["QUALITY_N_SEEDS"] = str(quality.get("n_seeds", "N/A"))
        v["QUALITY_N_TRIALS"] = str(quality.get("n_trials", "N/A"))
        v["QUALITY_CONTROL_ORDER"] = (
            "pass" if isinstance(controls, dict) and controls.get("oracle_best_log_score") else "fail"
        )
        v["QUALITY_CONFIDENT_WRONG_CONTROL"] = (
            "pass"
            if isinstance(controls, dict) and controls.get("confident_wrong_worse_than_uniform")
            else "fail"
        )
    else:
        for key in (
            "QUALITY_N_SEEDS",
            "QUALITY_N_TRIALS",
            "QUALITY_CONTROL_ORDER",
            "QUALITY_CONFIDENT_WRONG_CONTROL",
        ):
            v[key] = "N/A"

    # ---- Stage timings ----
    # Read-only: the pipeline (run_analysis_pipeline) is the ONLY writer of
    # stage_timings.json. A missing file degrades the tokens to honest "N/A" —
    # never a fabricated 0.0 rendered as a measured duration. (A prior version
    # wrote an all-zeros file here; review flagged it as result fabrication.)
    timings_path = root / "output" / "data" / "stage_timings.json"
    try:
        timings = json.loads(timings_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        timings = {}
    if not isinstance(timings, dict):
        timings = {}

    def _duration(key: str) -> str:
        value = timings.get(key)
        if value is None:
            return "N/A"
        try:
            duration = float(value)
        except (TypeError, ValueError):
            return "N/A"
        return f"{duration:.1f}" if np.isfinite(duration) and duration >= 0.0 else "N/A"

    v["STAGE_ANALYSIS_TOTAL_DURATION"] = _duration("total")
    v["STAGE_MOVING_WORLD_DURATION"] = _duration("moving_world")
    v["STAGE_HIERARCHICAL_DURATION"] = _duration("hierarchical")
    v["STAGE_SENSITIVITY_DURATION"] = _duration("sensitivity")
    v["STAGE_PARAMETER_RECOVERY_DURATION"] = _duration("parameter_recovery")
    v["STAGE_COMPLEXITY_DURATION"] = _duration("complexity")

    return v


_VARIATIONAL_KEYS = (
    "VARIATIONAL_ROBUSTNESS",
    "VARIATIONAL_ITERATIONS",
    "VARIATIONAL_CONVERGED",
    "VARIATIONAL_F_INITIAL",
    "VARIATIONAL_F_FINAL",
    "VARIATIONAL_DELTA_F",
    "VARIATIONAL_MAX_ASCENT",
    "VARIATIONAL_MAX_ASCENT_MATH",
    "VARIATIONAL_INFLUENCE_CLEAN",
    "VARIATIONAL_INFLUENCE_DIVERGED",
    "VARIATIONAL_NAIVE_INFLUENCE",
    "VARIATIONAL_INFLUENCE_DROP_FACTOR",
    "VARIATIONAL_SINGLE_START_F",
    "VARIATIONAL_MULTI_START_F",
    "VARIATIONAL_CAPTURE_GAP",
)


_GALLERY_KEYS = (
    "GALLERY_RATE",
    "GALLERY_N_TRIALS",
    "GALLERY_N_SEEDS",
    "GALLERY_RELIABLE_WIN_FRACTION",
    "GALLERY_RELIABLE_KINDS",
    "GALLERY_ENTROPY_NAIVE_ROBUST",
    "GALLERY_DIRECTIONAL_KINDS",
    "GALLERY_ENTROPY_KINDS",
    "GALLERY_TABLE_ROWS",
)


_ONSET_KEYS = (
    "ONSET_N_SEEDS",
    "ONSET_N_TRIALS",
    "ONSET_WIN_FRACTION",
    "ONSET_TABLE_ROWS",
)

_MOVING_KEYS = (
    "MOVING_ACC_ISOLATED",
    "MOVING_ACC_COMMUNICATING",
    "MOVING_ACC_EFE",
    "MOVING_FE_GAP_COMMUNICATING",
    "MOVING_N_TRIALS",
    "MOVING_N_STEPS",
    "MOVING_N_POSITIONS",
    "MOVING_N_AGENTS",
    "MOVING_N_SEEDS",
    "MOVING_ACC_ISO_MEAN",
    "MOVING_ACC_ISO_CI_LO",
    "MOVING_ACC_ISO_CI_HI",
    "MOVING_ACC_COMM_MEAN",
    "MOVING_ACC_COMM_CI_LO",
    "MOVING_ACC_COMM_CI_HI",
    "MOVING_ACC_EFE_MEAN",
    "MOVING_ACC_EFE_CI_LO",
    "MOVING_ACC_EFE_CI_HI",
    "MOVING_FE_GAP_EFE_MEAN",
    "MOVING_FE_GAP_EFE_CI_LO",
    "MOVING_FE_GAP_EFE_CI_HI",
    "MOVING_WILCOX_PVALUE",
    "MOVING_EFFECT_SIZE",
    "MOVING_EFFECT_LABEL",
)

_DISJOINT_KEYS = (
    "V4_ISOLATED_ACCURACY",
    "V4_COMMUNICATING_ACCURACY",
    "V4_ACCURACY_GAP",
    "V4_N_AGENTS",
    "V4_FOV_WIDTH",
    "V4_N_POSITIONS",
    "V4_CHANCE_BASELINE",
    "V4_EFE_N_AGENTS",
    "V4_EFE_N_POSITIONS",
    "V4_N_SEEDS",
    "V4_ISO_MEAN",
    "V4_ISO_CI_LO",
    "V4_ISO_CI_HI",
    "V4_COMM_MEAN",
    "V4_COMM_CI_LO",
    "V4_COMM_CI_HI",
    "V4_WILCOX_PVALUE",
    "V4_EFFECT_SIZE",
    "V4_EFFECT_LABEL",
    "V4_EFE_ACC_MEAN",
    "V4_RANDOM_ACC_MEAN",
    "V4_EFE_WILCOX_PVALUE",
    "V4_EFE_EFFECT_SIZE",
    "V4_EFE_EFFECT_LABEL",
)

_HIER_KEYS = (
    "HIER_N_LOCATIONS",
    "HIER_N_CONTEXTS",
    "HIER_N_AGENTS",
    "HIER_N_TRIALS",
    "HIER_ACUITY",
    "HIER_N_ITERS",
    "HIER_SEED",
    "HIER_ALERT_CENTER_MASS",
    "HIER_CTX_PERSIST",
    "HIER_LOC_ACC_FLAT",
    "HIER_LOC_ACC_HIER",
    "HIER_LOC_ACC_GAP",
    "HIER_CTX_ACC",
    "HIER_N_SEEDS",
    "HIER_LOC_ACC_HIER_MEAN",
    "HIER_LOC_ACC_HIER_STD",
    "HIER_LOC_ACC_HIER_CI_LO",
    "HIER_LOC_ACC_HIER_CI_HI",
    "HIER_LOC_ACC_FLAT_MEAN",
    "HIER_LOC_ACC_FLAT_CI_LO",
    "HIER_LOC_ACC_FLAT_CI_HI",
    "HIER_LOC_ACC_GAP_MEAN",
    "HIER_LOC_ACC_GAP_CI_LO",
    "HIER_LOC_ACC_GAP_CI_HI",
    "HIER_WILCOX_PVALUE",
    "HIER_EFFECT_SIZE",
    "HIER_EFFECT_LABEL",
)

_NLEVEL3_KEYS = (
    "NLEVEL3_N_LOCATIONS",
    "NLEVEL3_N_CONTEXTS",
    "NLEVEL3_N_META_CONTEXTS",
    "NLEVEL3_N_AGENTS",
    "NLEVEL3_N_TRIALS",
    "NLEVEL3_ACUITY",
    "NLEVEL3_N_ITERS",
    "NLEVEL3_SEED",
    "NLEVEL3_N_LEVELS",
    "NLEVEL3_LOC_ACC_FLAT",
    "NLEVEL3_LOC_ACC_3LEVEL",
    "NLEVEL3_LOC_ACC_GAP",
    "NLEVEL3_CTX_ACC",
    "NLEVEL3_META_CTX_ACC",
    "NLEVEL3_N_SEEDS",
    "NLEVEL3_LOC_ACC_3LEVEL_MEAN",
    "NLEVEL3_LOC_ACC_3LEVEL_STD",
    "NLEVEL3_LOC_ACC_3LEVEL_CI_LO",
    "NLEVEL3_LOC_ACC_3LEVEL_CI_HI",
    "NLEVEL3_LOC_ACC_FLAT_MEAN",
    "NLEVEL3_LOC_ACC_FLAT_CI_LO",
    "NLEVEL3_LOC_ACC_FLAT_CI_HI",
    "NLEVEL3_LOC_ACC_GAP_MEAN",
    "NLEVEL3_LOC_ACC_GAP_CI_LO",
    "NLEVEL3_LOC_ACC_GAP_CI_HI",
    "NLEVEL3_WILCOX_PVALUE",
    "NLEVEL3_EFFECT_SIZE",
    "NLEVEL3_EFFECT_LABEL",
    "NLEVEL3_LOW_THREAT_QUIET_PRIOR",
    "NLEVEL3_LOW_THREAT_ALERT_PRIOR",
    "NLEVEL3_HIGH_THREAT_QUIET_PRIOR",
    "NLEVEL3_HIGH_THREAT_ALERT_PRIOR",
    "NLEVEL3_ALERT_CENTER_MASS",
    "NLEVEL3_CENTER_CELL_INDEX",
)
