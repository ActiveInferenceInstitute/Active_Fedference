"""Tests for manuscript variable generation.

No mocks: the analysis pipeline runs for real into ``tmp_path``, then
``generate_variables`` reads the produced JSON reports. The headline test
cross-checks that every fedference ``{{TOKEN}}`` used in the shipped manuscript
prose is resolved by ``generate_variables`` — the prose and the code share one
source of truth.
"""

from __future__ import annotations

import inspect
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from analysis.workflow import run_analysis_pipeline
from fedference.experiments.cross_study import CROSS_STUDY_SENS_N_TRIALS
from figures.system_overview import SYSTEM_OVERVIEW_METADATA
from manuscript_variables import (
    _count_tests,
    _recovery_residuals,
    generate_variables,
    render_manuscript_tree,
    save_variables,
)
from manuscript_vars.loaders import _build_timestamp, _coverage_percent

# Real project root (three levels up from tests/): the shipped manuscript prose.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TOKEN_RE = re.compile(r"\{\{([A-Z0-9_]+)\}\}")
# Reserved example names in the syntax guide / prose — not emitted tokens.
_RESERVED_EXAMPLE_NAMES = {"TOKEN", "VARIABLE", "VARIABLE_NAME", "MY_TOKEN"}
# Every shipped IMRAD section file: the token-coverage guarantee must hold across
# the WHOLE manuscript, not a fixed subset, so this globs the live section set.
_FEDFERENCE_SECTIONS = tuple(
    sorted(
        p.name
        for p in (_PROJECT_ROOT / "manuscript").glob("*.md")
        if p.name not in ("AGENTS.md", "README.md", "SYNTAX.md", "preamble.md")
        and not p.name.startswith("99_")  # references: no tokens
    )
)


def _make_project(root: Path) -> None:
    manuscript = root / "manuscript"
    manuscript.mkdir(parents=True, exist_ok=True)
    config = {
        "paper": {"version": "0.2"},
        "authors": [{"name": "Test Author"}],
        "keywords": ["fedgvi", "belief sharing"],
        "publication": {"doi": "10.5281/zenodo.12345"},
        "experiment": {
            "analysis_profile": "smoke",
            "n_agents": 5,
            "n_seeds": 4,
            "n_trials": 6,
            "contamination_rates": [0.0, 0.45, 0.9],
            "divergences": ["KLD", "RKL", "beta"],
            "review_grid_target_max_mcse": 0.02,
            "bnn_torch": {
                "n_clients": 2,
                "n_per": 8,
                "hidden_dim": 4,
                "n_steps": 3,
                "contamination_levels": [0.0, 0.5],
            },
            "statistics": {
                "power_alpha": 0.05,
                "power_alternative": "greater",
                "target_power": 0.80,
            },
        },
    }
    (manuscript / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    # A real project carries an ISA (live ISC tally) and a tests/ tree (live
    # TEST_COUNT); provide minimal versions so the provenance tokens resolve to
    # real values rather than the "N/A" sentinel.
    (root / "ISA.md").write_text(
        "## Criteria\n- [x] ISC-1: done.\n- [x] ISC-2: done.\n- [ ] ISC-3: pending.\n",
        encoding="utf-8",
    )
    tests_dir = root / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "test_fixture.py").write_text(
        "def test_a():\n    assert True\n\n\ndef test_b():\n    assert True\n",
        encoding="utf-8",
    )


def _manuscript_tokens(section_files: tuple[str, ...]) -> set[str]:
    tokens: set[str] = set()
    manuscript = _PROJECT_ROOT / "manuscript"
    for name in section_files:
        path = manuscript / name
        if path.exists():
            tokens.update(_TOKEN_RE.findall(path.read_text(encoding="utf-8")))
    return tokens - _RESERVED_EXAMPLE_NAMES


def test_generate_variables_returns_strings(tmp_path: Path) -> None:
    _make_project(tmp_path)
    run_analysis_pipeline(project_root=tmp_path)
    variables = generate_variables(tmp_path)
    assert variables
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in variables.items())


def test_build_timestamp_is_reproducible_and_never_uses_wall_clock() -> None:
    assert _build_timestamp(None) == "omitted (unreleased reproducible build)"
    assert _build_timestamp("0") == "1970-01-01T00:00:00Z"
    assert _build_timestamp("1785205200") == "2026-07-28T02:20:00Z"
    with pytest.raises(ValueError, match="SOURCE_DATE_EPOCH"):
        _build_timestamp("not-an-epoch")


def test_count_tests_uses_pytest_collected_parametrizations(tmp_path: Path) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_parametrized.py").write_text(
        "import pytest\n\n"
        "@pytest.mark.parametrize('value', [1, 2, 3])\n"
        "def test_parametrized(value):\n"
        "    assert value > 0\n\n"
        "def test_plain():\n"
        "    assert True\n",
        encoding="utf-8",
    )

    assert _count_tests(tmp_path) == "4"


def test_covers_every_fedference_manuscript_token(tmp_path: Path) -> None:
    _make_project(tmp_path)
    run_analysis_pipeline(project_root=tmp_path)
    variables = generate_variables(tmp_path)

    prose_tokens = _manuscript_tokens(_FEDFERENCE_SECTIONS)
    assert prose_tokens, "expected fedference tokens in the shipped manuscript prose"
    missing = sorted(prose_tokens - set(variables))
    assert not missing, f"manuscript tokens not produced by generate_variables: {missing}"


def test_headline_numbers_are_resolved_not_na(tmp_path: Path) -> None:
    _make_project(tmp_path)
    run_analysis_pipeline(project_root=tmp_path)
    variables = generate_variables(tmp_path)
    # With reports present, the headline results must be concrete, not sentinels.
    for key in (
        "BELIEF_SHARING_DELTA_F",
        "LANGUAGE_FINAL_KL",
        "SWEEP_NAIVE_ACCURACY",
        "SWEEP_BEST_ROBUST_METHOD",
    ):
        assert key in variables
        assert variables[key] != "N/A"


#: New statistics tokens emitted for the backend-phase stats/keys/reports,
#: grouped by study. Every one must be present and concrete after a real run.
_NEW_BELIEF_SHARING_TOKENS = (
    "BELIEF_SHARING_N",
    "BELIEF_SHARING_N_SEEDS",
    "BELIEF_SHARING_MEAN_F_CI_LO",
    "BELIEF_SHARING_MEAN_F_CI_HI",
)
_NEW_LANGUAGE_TOKENS = (
    "LANGUAGE_N_POINTS",
    "LANGUAGE_N_SEEDS",
)
_NEW_EMERGENCE_TOKENS = ("EMERGENCE_N",)
# Config-completeness tokens: every frozen experiment-config value as a token.
_NEW_CONFIG_TOKENS = (
    "CONFIG_N_AGENTS",
    "CONFIG_N_LOCATIONS",
    "CONFIG_N_SEEDS",
    "CONFIG_N_TRIALS",
    "CONFIG_DIVERGENCES",
    "CONFIG_ROBUST_DIVERGENCES",
    "CONFIG_CONTAMINATION_RATES",
    "CONFIG_N_RATES",
    "CONFIG_POWER_ALPHA",
    "CONFIG_POWER_ALTERNATIVE",
    "CONFIG_TARGET_POWER",
)
# Power-analysis tokens for the headline robust-vs-naive Wilcoxon.
_NEW_POWER_TOKENS = (
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
_NEW_SWEEP_TOKENS = (
    "SWEEP_N",
    "SWEEP_NAIVE_VERDICT_RATE_MEAN",
    "SWEEP_BEST_COHENS_D",
    "SWEEP_BEST_EFFECT_LABEL",
    "SWEEP_BEST_RAW_PVALUE",
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
)
_ALL_NEW_STATS_TOKENS = (
    _NEW_BELIEF_SHARING_TOKENS
    + _NEW_LANGUAGE_TOKENS
    + _NEW_EMERGENCE_TOKENS
    + _NEW_CONFIG_TOKENS
    + _NEW_POWER_TOKENS
    + _NEW_SWEEP_TOKENS
)


def test_new_statistics_tokens_present_and_concrete(tmp_path: Path) -> None:
    # Every new per-condition n / CI / p-value / q-value / effect-size token
    # exists and resolves to a concrete (non-N/A, non-empty) value after a real
    # seeded pipeline run.
    _make_project(tmp_path)
    run_analysis_pipeline(project_root=tmp_path)
    variables = generate_variables(tmp_path)
    for key in _ALL_NEW_STATS_TOKENS:
        assert key in variables, f"missing new statistics token: {key}"
        assert variables[key] != "N/A", f"token left as N/A after real run: {key}"
        assert variables[key].strip() != "", f"token resolved to empty: {key}"


def test_config_tokens_reflect_the_project_config(tmp_path: Path) -> None:
    # Every config value is a token; they mirror the written experiment block,
    # so no number is hand-typed downstream.
    _make_project(tmp_path)
    run_analysis_pipeline(project_root=tmp_path)
    variables = generate_variables(tmp_path)
    assert variables["CONFIG_N_AGENTS"] == "5"
    assert variables["CONFIG_N_SEEDS"] == "4"
    assert variables["CONFIG_N_TRIALS"] == "6"
    assert variables["CONFIG_N_RATES"] == "3"
    assert "RKL" in variables["CONFIG_ROBUST_DIVERGENCES"]
    assert "KLD" not in variables["CONFIG_ROBUST_DIVERGENCES"]
    assert "KLD" in variables["CONFIG_DIVERGENCES"]
    assert variables["CONFIG_POWER_ALPHA"] == "0.05"
    assert variables["CONFIG_POWER_ALTERNATIVE"] == "greater"
    assert variables["CONFIG_TARGET_POWER"] == "0.80"


def test_cross_study_sensitivity_trial_token_matches_source_constant(tmp_path: Path) -> None:
    _make_project(tmp_path)
    run_analysis_pipeline(project_root=tmp_path)
    variables = generate_variables(tmp_path)
    assert variables["CROSS_STUDY_SENS_N_TRIALS"] == str(CROSS_STUDY_SENS_N_TRIALS)


def test_count_isc_deferred_rows_stay_in_the_denominator(tmp_path: Path) -> None:
    """A [DEFERRED-VERIFY] ISC counts toward total but never passed — otherwise
    the manuscript renders "N of N verified" while a criterion is open."""
    from manuscript_vars.loaders import _count_isc

    (tmp_path / "ISA.md").write_text(
        "- [x] ISC-1: done thing. Probe: x.\n"
        "- [ ] ISC-2: open thing. Probe: y.\n"
        "- [DEFERRED-VERIFY] ISC-3: deferred thing. Probe: z.\n",
        encoding="utf-8",
    )
    total, passed = _count_isc(tmp_path)
    assert (total, passed) == ("3", "1")


def test_default_project_root_is_repo_root_not_src() -> None:
    """Regression: _project_root(None) once resolved to src/, silently growing a
    stray src/output/ tree (and a fabricated stage_timings.json) on any
    default-root call."""
    from manuscript_vars.loaders import _project_root

    root = _project_root(None)
    assert root.name != "src"
    assert (root / "src" / "manuscript_vars").is_dir()


def test_missing_stage_timings_degrades_to_na_and_writes_nothing(tmp_path: Path) -> None:
    """A missing timings artifact must yield N/A tokens, not a fabricated file."""
    _make_project(tmp_path)
    run_analysis_pipeline(project_root=tmp_path)
    timings = tmp_path / "output" / "data" / "stage_timings.json"
    assert timings.exists()  # pipeline writes real timings
    timings.unlink()
    variables = generate_variables(tmp_path)
    assert variables["STAGE_ANALYSIS_TOTAL_DURATION"] == "N/A"
    assert not timings.exists()  # generator must not re-create it


def test_malformed_stage_timings_degrade_to_na(tmp_path: Path) -> None:
    """Invalid timing metadata must not become a fabricated rendered duration."""
    _make_project(tmp_path)
    run_analysis_pipeline(project_root=tmp_path)
    timings = tmp_path / "output" / "data" / "stage_timings.json"
    timings.write_text("{not valid json", encoding="utf-8")
    variables = generate_variables(tmp_path)
    assert variables["STAGE_ANALYSIS_TOTAL_DURATION"] == "N/A"
    assert timings.read_text(encoding="utf-8") == "{not valid json"


def test_system_overview_tokens_reflect_figure_metadata(tmp_path: Path) -> None:
    _make_project(tmp_path)
    run_analysis_pipeline(project_root=tmp_path)
    variables = generate_variables(tmp_path)
    assert variables["SYSTEM_OVERVIEW_N_AGENTS"] == str(SYSTEM_OVERVIEW_METADATA["n_agents"])
    assert variables["SYSTEM_OVERVIEW_N_ADVERSARIAL"] == str(SYSTEM_OVERVIEW_METADATA["n_adversarial"])
    assert variables["SYSTEM_OVERVIEW_N_HONEST"] == str(SYSTEM_OVERVIEW_METADATA["n_honest"])
    assert variables["SYSTEM_OVERVIEW_CONTAMINATION_PCT"] == str(
        SYSTEM_OVERVIEW_METADATA["contamination_pct"]
    )
    assert variables["SYSTEM_OVERVIEW_NAIVE_ACC_PCT"] == str(SYSTEM_OVERVIEW_METADATA["naive_acc_pct"])
    assert variables["SYSTEM_OVERVIEW_ROBUST_ACC_PCT"] == str(SYSTEM_OVERVIEW_METADATA["robust_acc_pct"])
    assert variables["SYSTEM_OVERVIEW_N_STATES"] == "8"
    assert variables["SYSTEM_OVERVIEW_TRUE_STATE_DISPLAY"] == "3"


def test_system_overview_metadata_is_derived_from_pooled_beliefs() -> None:
    """The schematic's exported percentages must equal quantities recomputed
    from the very beliefs the figure draws (no hand-typed number survives a
    colony/concentration change), and the schematic must actually exhibit the
    story the panels claim: the equal-weight pool flips its argmax off the
    true state while robust aggregation recovers it."""
    import numpy as np

    from figures.system_overview import (
        SYSTEM_OVERVIEW_METADATA,
        TRUE_STATE,
        build_data,
        naive_pool,
        robust_pool,
    )

    data = build_data()
    beliefs = data["beliefs"]
    naive = naive_pool(beliefs)
    robust, _weights = robust_pool(beliefs)

    assert SYSTEM_OVERVIEW_METADATA["naive_acc_pct"] == round(100.0 * float(naive[TRUE_STATE]))
    assert SYSTEM_OVERVIEW_METADATA["robust_acc_pct"] == round(100.0 * float(robust[TRUE_STATE]))
    assert SYSTEM_OVERVIEW_METADATA["contamination_pct"] == round(
        100.0 * SYSTEM_OVERVIEW_METADATA["n_adversarial"] / SYSTEM_OVERVIEW_METADATA["n_agents"]
    )
    assert SYSTEM_OVERVIEW_METADATA["n_agents"] == len(beliefs)
    assert SYSTEM_OVERVIEW_METADATA["n_adversarial"] == 2
    assert (
        SYSTEM_OVERVIEW_METADATA["n_honest"]
        == SYSTEM_OVERVIEW_METADATA["n_agents"] - SYSTEM_OVERVIEW_METADATA["n_adversarial"]
    )
    assert int(np.argmax(data["naive"])) != TRUE_STATE
    assert int(np.argmax(data["robust"])) == TRUE_STATE
    adversary_mean = float(np.mean(data["weights"][:2]))
    honest_mean = float(np.mean(data["weights"][2:]))
    assert adversary_mean < honest_mean


def test_power_tokens_are_concrete_and_bounded(tmp_path: Path) -> None:
    # The headline design power is a probability in [0,1]; the prospective n is
    # a positive integer. These decorate the SERVER-SIDE aggregation heuristic's
    # contrast — never the beta/rcce per-agent FedGVI guarantee.
    _make_project(tmp_path)
    run_analysis_pipeline(project_root=tmp_path)
    variables = generate_variables(tmp_path)
    power = float(variables["SWEEP_HEADLINE_POWER"])
    assert 0.0 <= power <= 1.0
    assert int(variables["SWEEP_PROSPECTIVE_N"]) >= 1
    assert int(variables["SWEEP_HEADLINE_N_FOR_TARGET_POWER"]) >= 1
    assert variables["SWEEP_HEADLINE_METHOD"] != "KLD"
    assert variables["SWEEP_HEADLINE_METHOD"].strip() != ""
    # The standardized-effect verdict table now carries a power column: each
    # robust row has the extra cell (10 pipe-delimited fields).
    rows = variables["SWEEP_VERDICT_EFFECT_TABLE_ROWS"].splitlines()
    assert rows
    for line in rows:
        assert line.count("|") == 11  # 10 columns -> 11 pipe separators


def test_paired_by_rate_table_has_rows(tmp_path: Path) -> None:
    # The per-contamination-rate paired-test table emits a markdown row per
    # (robust method x rate); the naive KLD baseline is excluded (no self-test).
    _make_project(tmp_path)
    run_analysis_pipeline(project_root=tmp_path)
    variables = generate_variables(tmp_path)
    rows = variables["SWEEP_PAIRED_BY_RATE_TABLE_ROWS"].splitlines()
    assert rows, "expected at least one paired-by-rate row"
    assert all(line.startswith("| ") and line.endswith(" |") for line in rows)
    # KLD is the naive baseline and must not appear as its own contrast row.
    assert not any(line.startswith("| KLD |") for line in rows)


def test_paired_by_rate_table_covers_every_robust_method_and_rate(tmp_path: Path) -> None:
    _make_project(tmp_path)
    run_analysis_pipeline(project_root=tmp_path)
    variables = generate_variables(tmp_path)
    rows = variables["SWEEP_PAIRED_BY_RATE_TABLE_ROWS"].splitlines()
    robust_methods = [method for method in variables["CONFIG_DIVERGENCES"].split(", ") if method != "KLD"]
    rates = [f"{float(rate):g}" for rate in variables["CONFIG_CONTAMINATION_RATES"].split(", ")]
    seen = set()
    for row in rows:
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        assert len(cells) == 7
        method, rate, d_equivalent, label, raw_p, qvalue, reject = cells
        assert method in robust_methods
        assert rate in rates
        assert d_equivalent
        assert label in {"negligible", "small", "medium", "large"}
        assert raw_p
        assert qvalue
        assert reject in {"Yes", "No"}
        seen.add((method, rate))
    assert seen == {(method, rate) for method in robust_methods for rate in rates}


def test_no_token_left_unresolved_on_real_run(tmp_path: Path) -> None:
    # Contract: given a real pipeline run, NO emitted token is an unresolved
    # sentinel — every value is a non-empty string and none is "N/A".
    _make_project(tmp_path)
    run_analysis_pipeline(project_root=tmp_path)
    variables = generate_variables(tmp_path)
    unresolved = sorted(k for k, val in variables.items() if val == "N/A" or val.strip() == "")
    assert not unresolved, f"tokens unresolved after real run: {unresolved}"
    # The smoke review grid deliberately does not assert that its small sample
    # meets the publication stopping rule, but its source-registered design
    # target must still hydrate the manuscript rather than leak an N/A token.
    assert variables["REVIEW_GRID_TARGET_MAX_MCSE"] == "0.0200"


def test_d_equivalent_sentinel_renders_saturated() -> None:
    from manuscript_variables import _format_d_equivalent

    assert _format_d_equivalent(None) == "N/A"
    assert _format_d_equivalent(0.83) == "0.83"
    assert _format_d_equivalent(-0.46) == "-0.46"
    assert _format_d_equivalent(1e6) == "saturated (r=+1)"
    assert _format_d_equivalent(-1e6) == "saturated (r=-1)"


def test_old_d_equivalent_formatter_name_is_warned_adapter() -> None:
    from manuscript_variables import _format_cohens_d

    with pytest.warns(DeprecationWarning):
        assert _format_cohens_d(0.83) == "0.83"


def test_coverage_percent_ignores_stale_project_artifact(tmp_path: Path) -> None:
    """A live coverage database must not hydrate prose from an older JSON file."""
    coverage_json = tmp_path / "coverage_project.json"
    coverage_json.write_text(json.dumps({"totals": {"percent_covered": 99.99}}), encoding="utf-8")
    database = tmp_path / ".coverage"
    database.write_bytes(b"current coverage database")
    # Explicitly establish the ordering instead of relying on filesystem
    # timestamp granularity.
    import os

    os.utime(coverage_json, (100.0, 100.0))
    os.utime(database, (200.0, 200.0))
    assert _coverage_percent(tmp_path) == "≥90"


def test_bnn_torch_variables_read_executed_report(tmp_path: Path) -> None:
    """BNN tokens come from the executed report when present, else labeled N/A."""
    import json as _json

    from manuscript_variables import _bnn_torch_variables

    # No report -> documented-default config tokens + explicit "not run" markers.
    absent = _bnn_torch_variables(tmp_path)
    assert absent["BNN_HIDDEN_DIM"] == "16"
    assert "N/A" in absent["BNN_TORCH_STD_ACC"]
    assert absent["BNN_DETERMINISTIC"] == "N/A"
    assert absent["PYTORCH_VERSION"] == "not installed"

    # A skipped report degrades the same way (no fabricated numbers).
    reports = tmp_path / "output" / "reports"
    reports.mkdir(parents=True)
    (reports / "bnn_torch.json").write_text(
        _json.dumps({"status": "skipped: PyTorch unavailable"}), encoding="utf-8"
    )
    assert _bnn_torch_variables(tmp_path)["BNN_TORCH_ROBUST_ACC"].startswith("N/A")

    # An executed report surfaces its real numbers as tokens.
    (reports / "bnn_torch.json").write_text(
        _json.dumps(
            {
                "status": "ok",
                "torch_version": "2.3.1",
                "hidden_dim": 16,
                "n_steps": 200,
                "beta": 0.5,
                "robustness": 0.5,
                "n_clients": 5,
                "standard_accuracy": 0.812,
                "robust_accuracy": 0.844,
                "reported_contamination": 0.4,
                "consensus_max_simplex_deviation": 0.0,
                "deterministic": True,
            }
        ),
        encoding="utf-8",
    )
    ok = _bnn_torch_variables(tmp_path)
    assert ok["BNN_TORCH_STD_ACC"] == "0.812"
    assert ok["BNN_TORCH_ROBUST_ACC"] == "0.844"
    assert ok["BNN_TORCH_CONTAM"] == "0.40"
    assert ok["BNN_DETERMINISTIC"] == "Yes"
    assert ok["PYTORCH_VERSION"] == "2.3.1"


def test_bnn_robustness_variables_read_generated_report(tmp_path: Path) -> None:
    """fig:bnn-robustness caption tokens (seed count, n_per, loss_param, max
    contamination) come from the generated report, never hand-typed (C3 fix)."""
    import json as _json

    from manuscript_variables import _bnn_robustness_variables

    # No report -> explicit N/A, never a fabricated seed count.
    absent = _bnn_robustness_variables(tmp_path)
    assert absent["BNN_ROBUSTNESS_N_SEEDS"] == "N/A"

    reports = tmp_path / "output" / "reports"
    reports.mkdir(parents=True)
    # Peak-then-reconverge shape (advisor fix): the last level (0.4) is kept
    # in the sweep even though it does NOT favor the robust client, so
    # BNN_ROBUSTNESS_PEAK_CONTAM must resolve to 0.35 (the actual max-gap
    # level), not 0.4 (the last/max-contamination level).
    (reports / "bnn_robustness.json").write_text(
        _json.dumps(
            {
                "contamination_levels": [0.0, 0.1, 0.2, 0.3, 0.35, 0.4],
                "accuracy_by_config": {
                    "nll / KLD (standard)": [0.86, 0.86, 0.86, 0.85, 0.58, 0.15],
                    "rcce / AR (robust)": [0.86, 0.86, 0.87, 0.86, 0.60, 0.14],
                },
                "peak_margin": 0.0387,
                "peak_margin_contamination": 0.35,
                "n_seeds": 20,
                "n_per": 200,
                "robust_loss_param": 1.0,
                "seed": 0,
            }
        ),
        encoding="utf-8",
    )
    ok = _bnn_robustness_variables(tmp_path)
    assert ok["BNN_ROBUSTNESS_N_SEEDS"] == "20"
    assert ok["BNN_ROBUSTNESS_N_PER"] == "200"
    assert ok["BNN_ROBUSTNESS_LOSS_PARAM"] == "1.00"
    assert ok["BNN_ROBUSTNESS_MAX_CONTAM"] == "0.40"
    assert ok["BNN_ROBUSTNESS_PEAK_CONTAM"] == "0.35"
    assert ok["BNN_ROBUSTNESS_PEAK_GAP"] == "0.039"


def test_disjoint_fov_variables_earn_the_necessity_claim(tmp_path: Path) -> None:
    """C1 fix: the necessity claim must be backed by a powered, significant
    paired test and an explicit chance baseline — not a small-sample point
    estimate. The EFE-navigation contrast must be reported honestly even
    though it turns out to be a null result (near-ceiling, not significant)."""
    from fedference.experiments import disjoint_fov_report
    from manuscript_variables import _disjoint_fov_variables

    reports = tmp_path / "output" / "reports"
    reports.mkdir(parents=True)
    report = disjoint_fov_report(0)
    (reports / "disjoint_fov_world.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )

    v = _disjoint_fov_variables(tmp_path)
    assert float(v["V4_WILCOX_PVALUE"]) < 0.05, "necessity contrast must be significant"
    assert float(v["V4_EFFECT_SIZE"]) > 0.0
    assert float(v["V4_COMM_MEAN"]) > float(v["V4_ISO_MEAN"])
    chance = 1.0 / int(v["V4_N_POSITIONS"])
    assert abs(float(v["V4_CHANCE_BASELINE"]) - chance) < 1e-3
    assert float(v["V4_ISO_MEAN"]) > float(v["V4_CHANCE_BASELINE"])
    # The EFE-vs-random navigation contrast is honestly a null result here
    # (near-ceiling task) — assert it is reported as such, not overclaimed.
    assert float(v["V4_EFE_WILCOX_PVALUE"]) > 0.05
    assert v["V4_EFE_EFFECT_LABEL"] == "negligible"


def test_recovery_residuals_are_near_zero(tmp_path: Path) -> None:
    _make_project(tmp_path)
    run_analysis_pipeline(project_root=tmp_path)
    variables = generate_variables(tmp_path)
    for key in (
        "RECOVERY_RCCE_MAXDIFF",
        "RECOVERY_BETA_MAXDIFF",
        "RECOVERY_RENYI_MAXDIFF",
        "RECOVERY_AGGREGATE_MAXDIFF",
        "RECOVERY_POSTERIOR_MAXDIFF",
    ):
        val = variables[key]
        # Either exactly "0" or scientific notation with a tiny exponent.
        if val != "0":
            assert float(val) < 1e-6


def test_recovery_offswitch_residuals_are_nonzero_and_tiny(tmp_path: Path) -> None:
    """M2 fix: the off-switch-point residuals must be genuinely nonzero
    (unlike the exact-branch residuals, which are exactly 0/near machine
    epsilon) — proof the general formula converges, not just the branch."""
    _make_project(tmp_path)
    run_analysis_pipeline(project_root=tmp_path)
    variables = generate_variables(tmp_path)
    for key in (
        "RECOVERY_RCCE_OFFSWITCH_MAXDIFF",
        "RECOVERY_BETA_OFFSWITCH_MAXDIFF",
        "RECOVERY_RENYI_OFFSWITCH_MAXDIFF",
    ):
        val = float(variables[key])
        assert val > 0.0, f"{key} must be a genuine nonzero residual, not a branch-exact 0"
        assert val < 1e-3, f"{key} should still be a small convergence residual"
    assert variables["RECOVERY_OFFSWITCH_Q"] == "1e-06"
    assert variables["RECOVERY_OFFSWITCH_BETA"] == "1e-06"
    assert variables["RECOVERY_OFFSWITCH_ALPHA"] == "1.00001"


def test_recovery_residuals_have_claim_specific_semantics() -> None:
    source = inspect.getsource(_recovery_residuals)
    required_primitives = (
        "rcce(",
        "beta_loss(",
        "renyi_divergence(",
        "robust_aggregate(",
        "log_linear_pool(",
        "generalized_posterior(",
        "analytic_bayes",
    )
    for primitive in required_primitives:
        assert primitive in source

    residuals = _recovery_residuals()
    assert set(residuals) >= {
        "RECOVERY_RCCE_MAXDIFF",
        "RECOVERY_BETA_MAXDIFF",
        "RECOVERY_RENYI_MAXDIFF",
        "RECOVERY_AGGREGATE_MAXDIFF",
        "RECOVERY_POSTERIOR_MAXDIFF",
    }
    assert all(np.isfinite(value) and value >= 0.0 for value in residuals.values())


def test_draft_mode_degrades_to_na(tmp_path: Path) -> None:
    # No analysis reports written → headline result tokens fall back to N/A,
    # but generation still succeeds (draft manuscript renders).
    _make_project(tmp_path)
    variables = generate_variables(tmp_path, allow_draft=True)
    assert variables["BELIEF_SHARING_DELTA_F"] == "N/A"
    assert variables["LANGUAGE_FINAL_KL"] == "N/A"
    assert variables["SWEEP_NAIVE_ACCURACY"] == "N/A"
    # Deterministic constants are still computed even without reports.
    assert variables["EMERGENCE_CONVERGENCE"] in {"Yes", "No"}
    # Draft mode has no executed sweep report, so the executed-FDR-level token
    # honestly degrades to N/A (it is read from the report the verdict ran at,
    # never from a constant that can desynchronize).
    assert variables["STATISTICS_FDR_ALPHA"] == "N/A"


def test_config_metadata_tokens(tmp_path: Path) -> None:
    _make_project(tmp_path)
    variables = generate_variables(tmp_path, allow_draft=True)
    assert variables["CONFIG_VERSION"] == "0.2"
    assert variables["CONFIG_FIRST_AUTHOR"] == "Test Author"
    assert "fedgvi" in variables["CONFIG_KEYWORDS"]
    assert variables["PUBLICATION_DOI"] == "10.5281/zenodo.12345"
    assert variables["PUBLICATION_DOI_URL"] == "https://doi.org/10.5281/zenodo.12345"


def test_assigned_doi_metadata_tokens_are_canonical(tmp_path: Path) -> None:
    _make_project(tmp_path)
    config_path = tmp_path / "manuscript" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["publication"] = {"doi": "https://doi.org/10.5281/zenodo.12345"}
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    variables = generate_variables(tmp_path, allow_draft=True)
    assert variables["PUBLICATION_DOI"] == "10.5281/zenodo.12345"
    assert variables["PUBLICATION_DOI_URL"] == "https://doi.org/10.5281/zenodo.12345"


def test_save_variables_round_trip(tmp_path: Path) -> None:
    _make_project(tmp_path)
    variables = generate_variables(tmp_path, allow_draft=True)
    out = tmp_path / "output" / "data" / "manuscript_variables.json"
    saved = save_variables(variables, out)
    assert saved.exists()
    loaded = json.loads(saved.read_text(encoding="utf-8"))
    assert loaded == variables


def test_render_manuscript_tree_is_standalone(tmp_path: Path) -> None:
    _make_project(tmp_path)
    manuscript = tmp_path / "manuscript"
    (manuscript / "01_intro.md").write_text("Value: {{VALUE}}\n", encoding="utf-8")
    (manuscript / "README.md").write_text("Example: {{VALUE}}\n", encoding="utf-8")
    (manuscript / "refs.bib").write_text("@misc{x, title={X}}\n", encoding="utf-8")

    out_dir = render_manuscript_tree(tmp_path, {"VALUE": "42"})

    assert (out_dir / "01_intro.md").read_text(encoding="utf-8") == "Value: 42\n"
    assert not (out_dir / "README.md").exists()
    assert (out_dir / "config.yaml").exists()
    assert (out_dir / "refs.bib").exists()
    assert "infrastructure" not in inspect.getsource(render_manuscript_tree)


def test_full_manuscript_tree_is_exactly_hydrated_without_tokens(tmp_path: Path) -> None:
    """The shipped manuscript tree, not only a toy fixture, must fully inject."""
    variables = generate_variables(_PROJECT_ROOT, allow_draft=True)
    shutil.copytree(_PROJECT_ROOT / "manuscript", tmp_path / "manuscript")

    output = render_manuscript_tree(tmp_path, variables)

    source_names = {
        path.name
        for path in (tmp_path / "manuscript").glob("*.md")
        if path.name not in {"AGENTS.md", "README.md", "SYNTAX.md"}
    }
    assert {path.name for path in output.glob("*.md")} == source_names
    unresolved = {
        path.relative_to(output).as_posix(): _TOKEN_RE.findall(path.read_text(encoding="utf-8"))
        for path in output.rglob("*.md")
    }
    assert not {name: tokens for name, tokens in unresolved.items() if tokens}, (
        "full manuscript hydration left unresolved {{TOKEN}} markers"
    )


def test_render_manuscript_tree_refuses_symlinked_output_dir(tmp_path: Path) -> None:
    _make_project(tmp_path)
    manuscript = tmp_path / "manuscript"
    (manuscript / "01_intro.md").write_text("Value: {{VALUE}}\n", encoding="utf-8")
    external = tmp_path / "external"
    external.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    (output / "manuscript").symlink_to(external, target_is_directory=True)

    with pytest.raises(RuntimeError, match="symlinked output path"):
        render_manuscript_tree(tmp_path, {"VALUE": "42"})
    assert not (external / "01_intro.md").exists()


def test_uses_real_project_root_by_default() -> None:
    # Smoke: default project_root reads the shipped config without raising.
    variables = generate_variables(allow_draft=True)
    assert "ISC_TOTAL" in variables


def test_missing_config_uses_metadata_defaults(tmp_path: Path) -> None:
    # No manuscript/config.yaml at all → config-derived tokens fall back to
    # documented defaults but generation still succeeds.
    variables = generate_variables(tmp_path, allow_draft=True)
    assert variables["CONFIG_VERSION"] == "1.0"
    assert variables["CONFIG_FIRST_AUTHOR"] == "Unknown"
    assert variables["CONFIG_KEYWORDS"] == ""
    assert variables["ARTIFACT_TOTAL"] == "0"


def test_format_residual_handles_zero() -> None:
    from manuscript_variables import _format_residual

    assert _format_residual(0.0) == "0"
    assert _format_residual(-1.0) == "0"
    assert _format_residual(1e-12).startswith("1.00e")


def test_format_residual_math_renders_latex_scientific_notation() -> None:
    from manuscript_variables import _format_residual_math

    # Zero / non-positive: same "0" sentinel as the prose formatter.
    assert _format_residual_math(0.0) == "0"
    assert _format_residual_math(-1.0) == "0"
    # Exact power of ten: exponent normalized to int, no leading zeros.
    assert _format_residual_math(1e-06) == "1.00 \\times 10^{-6}"
    # Tiny negative exponent (the class of value that breaks $...$ as .2e).
    assert _format_residual_math(2.33e-80) == "2.33 \\times 10^{-80}"
    # Values >= 1 still round-trip through the same .2e representation.
    assert _format_residual_math(3.5) == "3.50 \\times 10^{0}"
    assert _format_residual_math(1234.0) == "1.23 \\times 10^{3}"
    # No raw exponent-marker leaks: the math form never contains "e-"/"e+".
    for value in (1e-06, 2.33e-80, 3.5, 1234.0):
        assert "e" not in _format_residual_math(value).replace("\\times", "")


def test_math_sibling_tokens_resolve_alongside_their_prose_twins(
    variables,
) -> None:
    """Every *_MATH token used in manuscript $...$ spans must be emitted.

    Contract: a _MATH sibling exists for each math-consumed scientific-notation
    token, its prose twin is still emitted unchanged, and the sibling is either
    the "0" sentinel or valid LaTeX scientific notation (never bare ".2e").
    """
    math_tokens = sorted(t for t in _manuscript_tokens(_FEDFERENCE_SECTIONS) if t.endswith("_MATH"))
    assert math_tokens, "expected *_MATH tokens in the shipped manuscript prose"
    latex_sci = re.compile(r"-?\d\.\d{2} \\times 10\^\{-?\d+\}")
    for token in math_tokens:
        assert token in variables, f"{token} not emitted by generate_variables"
        prose_twin = token[: -len("_MATH")]
        assert prose_twin in variables, f"{prose_twin} missing for {token}"
        value = variables[token]
        assert value == "0" or latex_sci.fullmatch(value) or value == "N/A", (
            f"{token} value {value!r} is neither the zero sentinel nor LaTeX scientific notation"
        )
        # A resolved sibling never leaks the raw .2e exponent marker.
        assert not re.search(r"\de[+-]\d", value), f"{token} leaked .2e: {value!r}"


@pytest.fixture
def variables(tmp_path: Path):
    _make_project(tmp_path)
    run_analysis_pipeline(project_root=tmp_path)
    return generate_variables(tmp_path)


def test_sensitivity_and_bootstrap_token_values(variables):
    """SENS_* and BOOTSTRAP_N_BOOT constants must match experiment defaults."""
    assert variables["SENS_N_TRIALS"] == "20"
    assert variables["SENS_SEED_BASE"] == "0"
    assert variables["SENS_N_ACUITY_LEVELS"] == "5"
    assert variables["SENS_N_COLONY_SIZES"] == "5"
    assert variables["SENS_N_CELLS"] == "25"
    assert variables["SENS_NOISE_FLOOR"] == "0.05"
    assert variables["BOOTSTRAP_N_BOOT"] == "5000"
