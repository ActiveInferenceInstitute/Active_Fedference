"""Tests for the frozen Active Fedference experiment configuration.

No mocks: real YAML round-trips through ``tmp_path`` and explicit numeric
expectations on the loaded dataclass.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from experiment_config import ExperimentConfig, load_experiment_config


def _write_config(root: Path, experiment: dict) -> Path:
    manuscript = root / "manuscript"
    manuscript.mkdir(parents=True, exist_ok=True)
    path = manuscript / "config.yaml"
    path.write_text(yaml.safe_dump({"experiment": experiment}), encoding="utf-8")
    return path


def test_defaults_match_canonical_sentinel_world() -> None:
    cfg = ExperimentConfig()
    assert cfg.n_agents == 7
    assert cfg.n_locations == 9
    assert cfg.divergences[0] == "KLD"
    assert cfg.contamination_rates[0] == 0.0
    # The dataclass fallback is the canonical publication contract, not the
    # older diagnostic grid that predates the current source configuration.
    assert cfg.n_seeds == 480
    assert cfg.replicate_seeds == 128
    assert cfg.n_trials == 960
    assert cfg.cross_study_n_trials == 40
    assert cfg.conditional_world_n_seeds == 64
    assert cfg.conditional_world_n_trials == 24
    assert cfg.review_grid_n_seeds == 160
    assert cfg.review_grid_n_trials == 24
    assert cfg.review_grid_target_max_mcse == 0.01
    assert cfg.gallery_n_seeds == 64
    assert cfg.gallery_n_trials == 24
    assert cfg.onset_n_seeds == 64
    assert cfg.onset_n_trials == 24
    assert cfg.bnn_n_seeds == 64
    assert cfg.bnn_n_per == 200
    assert cfg.robustness == 0.0
    # Power-analysis defaults for the headline robust-vs-naive Wilcoxon.
    assert cfg.fdr_alpha == 0.05
    assert cfg.power_alpha == 0.05
    assert cfg.power_alternative == "greater"
    assert cfg.target_power == 0.80


def test_seeds_and_robust_divergences_properties() -> None:
    cfg = ExperimentConfig(n_seeds=5, divergences=("KLD", "RKL", "beta"))
    assert cfg.seeds == (0, 1, 2, 3, 4)
    assert cfg.robust_divergences == ("RKL", "beta")


def test_missing_config_returns_defaults(tmp_path: Path) -> None:
    cfg = load_experiment_config(tmp_path)
    assert cfg == ExperimentConfig()


def test_loads_experiment_block(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        {
            "n_agents": 5,
            "n_locations": 4,
            "contamination_rates": [0.0, 0.5, 1.0],
            "divergences": ["KLD", "AR"],
            "n_seeds": 8,
            "replicate_seeds": 12,
            "n_trials": 30,
            "cross_study_n_trials": 7,
            "gallery_n_seeds": 10,
            "gallery_n_trials": 8,
            "onset_n_seeds": 9,
            "onset_n_trials": 7,
            "bnn_n_seeds": 12,
            "bnn_n_per": 40,
            "review_grid_n_seeds": 8,
            "review_grid_n_trials": 4,
            "review_grid_rates": [0.0, 0.5],
            "review_grid_target_max_mcse": 0.03,
            "complexity": {
                "agent_sizes": [2, 4, 8],
                "state_sizes": [4, 8, 16],
                "sharing_agent_sizes": [2, 4, 8],
                "modality_sizes": [1, 2, 4],
                "repeats": 2,
            },
            "robustness": 1.5,
            "statistics": {
                "fdr_alpha": 0.10,
                "power_alpha": 0.01,
                "power_alternative": "two-sided",
                "target_power": 0.90,
            },
        },
    )
    cfg = load_experiment_config(tmp_path)
    assert cfg.n_agents == 5
    assert cfg.n_locations == 4
    assert cfg.contamination_rates == (0.0, 0.5, 1.0)
    assert cfg.divergences == ("KLD", "AR")
    assert cfg.n_seeds == 8
    assert cfg.replicate_seeds == 12
    assert cfg.n_trials == 30
    assert cfg.cross_study_n_trials == 7
    assert cfg.gallery_n_seeds == 10
    assert cfg.gallery_n_trials == 8
    assert cfg.onset_n_seeds == 9
    assert cfg.onset_n_trials == 7
    assert cfg.bnn_n_seeds == 12
    assert cfg.bnn_n_per == 40
    assert cfg.review_grid_target_max_mcse == 0.03
    assert cfg.robustness == 1.5
    # Statistics fields are read from the statistics sub-block.
    assert cfg.fdr_alpha == 0.10
    assert cfg.power_alpha == 0.01
    assert cfg.power_alternative == "two-sided"
    assert cfg.target_power == 0.90
    assert cfg.complexity.agent_sizes == (2, 4, 8)
    assert cfg.complexity.repeats == 2


def test_power_fields_default_when_statistics_block_absent(tmp_path: Path) -> None:
    _write_config(tmp_path, {"n_agents": 4})
    cfg = load_experiment_config(tmp_path)
    assert cfg.fdr_alpha == 0.05
    assert cfg.power_alpha == 0.05
    assert cfg.power_alternative == "greater"
    assert cfg.target_power == 0.80
    assert cfg.n_trials == 960


def test_empty_experiment_block_uses_defaults(tmp_path: Path) -> None:
    (tmp_path / "manuscript").mkdir()
    (tmp_path / "manuscript" / "config.yaml").write_text(
        yaml.safe_dump({"paper": {"version": "9"}}), encoding="utf-8"
    )
    cfg = load_experiment_config(tmp_path)
    assert cfg == ExperimentConfig()


@pytest.mark.parametrize(
    ("experiment", "message"),
    [
        ([], "experiment block must be a mapping"),
        ("publication", "experiment block must be a mapping"),
        ({"statistics": []}, "statistics block must be a mapping"),
    ],
)
def test_malformed_yaml_blocks_fail_closed(
    tmp_path: Path,
    experiment: object,
    message: str,
) -> None:
    manuscript = tmp_path / "manuscript"
    manuscript.mkdir()
    (manuscript / "config.yaml").write_text(
        yaml.safe_dump({"experiment": experiment}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=message):
        load_experiment_config(tmp_path)


def test_nested_seed_fallback_never_uses_robustness_trial_count(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        {
            "belief_sharing": {"n_seeds": 7},
            "robustness_sweep": {"n_trials": 999},
        },
    )
    cfg = load_experiment_config(tmp_path)
    assert cfg.n_seeds == 7
    assert cfg.n_trials == 999


def test_scalar_coercion_for_single_value_lists(tmp_path: Path) -> None:
    _write_config(tmp_path, {"contamination_rates": 0.3, "divergences": "KLD"})
    cfg = load_experiment_config(tmp_path)
    assert cfg.contamination_rates == (0.3,)
    assert cfg.divergences == ("KLD",)


@pytest.mark.parametrize(
    ("experiment", "message"),
    [
        ({"n_agents": 2.5}, "experiment.n_agents"),
        ({"n_seeds": "8"}, "experiment.n_seeds"),
        ({"review_grid_target_max_mcse": "0.01"}, "experiment.review_grid_target_max_mcse"),
        ({"contamination_rates": ["0.3"]}, r"experiment.contamination_rates\[\]"),
        ({"statistics": {"power_alternative": 1}}, "power_alternative must be a string"),
    ],
)
def test_numeric_yaml_values_are_not_silently_truncated_or_coerced(
    tmp_path: Path,
    experiment: dict[str, object],
    message: str,
) -> None:
    _write_config(tmp_path, experiment)
    with pytest.raises(ValueError, match=message):
        load_experiment_config(tmp_path)


def test_frozen_dataclass_is_immutable() -> None:
    cfg = ExperimentConfig()
    with pytest.raises(AttributeError):
        cfg.n_agents = 99  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_agents": 2.5},
        {"n_agents": True},
        {"n_agents": 1},
        {"n_locations": 1},
        {"contamination_rates": ()},
        {"contamination_rates": (1.5,)},
        {"divergences": ("RKL",)},
        {"divergences": ("KLD", "KLD")},
        {"n_seeds": 1},
        {"replicate_seeds": 1},
        {"n_trials": 1},
        {"cross_study_n_trials": 1},
        {"review_grid_n_trials": 1},
        {"review_grid_n_seeds": 1},
        {"review_grid_rates": (0.5, 0.0)},
        {"review_grid_rates": (0.0, 0.0)},
        {"review_grid_target_max_mcse": 0.0},
        {"review_grid_target_max_mcse": "0.01"},
        {"gallery_n_seeds": 1},
        {"gallery_n_trials": 1},
        {"onset_n_seeds": 1},
        {"onset_n_trials": 1},
        {"bnn_n_seeds": 1},
        {"bnn_n_per": 0},
        {"robustness": -0.1},
        {"power_alpha": 0.0},
        {"power_alpha": 1.0},
        {"power_alternative": "bogus"},
        {"target_power": 0.0},
        {"target_power": 1.0},
    ],
)
def test_invalid_parameters_raise(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(**kwargs)


def test_default_root_resolves_real_project_config() -> None:
    # Smoke: loading with no override reads the shipped manuscript/config.yaml
    # (or defaults) without raising.
    cfg = load_experiment_config()
    assert cfg.n_agents >= 2
    assert "KLD" in cfg.divergences


def test_shipped_publication_config_matches_the_fallback_contract() -> None:
    """Keep a missing/minimal config from silently selecting an older budget."""
    assert load_experiment_config() == ExperimentConfig()
