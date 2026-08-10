"""Regression tests for the documented 3-level YAML defaults."""

from __future__ import annotations

import inspect
from pathlib import Path

import yaml

from fedference.pomdp import build_3level_world


def test_hierarchical_layers_yaml_matches_build_3level_world_defaults() -> None:
    """Guard the pre-fix YAML acuity drift (0.85) from code default 0.9."""
    config_path = (
        Path(__file__).resolve().parent.parent.parent
        / "src"
        / "fedference"
        / "config"
        / "hierarchical_layers.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    params = inspect.signature(build_3level_world).parameters

    sensor = payload["sensor"]
    assert sensor["acuity"] == params["acuity"].default
    assert sensor["goal_bonus"] == params["goal_bonus"].default

    l3_spec = payload["layers"][0]
    assert l3_spec["default_prior"] == list(params["l3_prior"].default)

