"""Regression tests for the documented 3-level YAML defaults."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from fedference.pomdp import build_3level_world


def test_hierarchical_layers_yaml_matches_build_3level_world_defaults() -> None:
    """Numeric YAML reference agrees with source-derived public defaults."""
    config_path = (
        Path(__file__).resolve().parent.parent.parent
        / "src"
        / "fedference"
        / "config"
        / "hierarchical_layers.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    world = build_3level_world()

    sensor = payload["sensor"]
    np.testing.assert_allclose(np.diag(world["L1"]["A"][0]), sensor["acuity"])
    assert float(np.max(world["L1"]["C"][0])) == sensor["goal_bonus"]

    l3_spec = payload["layers"][0]
    np.testing.assert_array_equal(l3_spec["default_prior"], world["L3_prior"])
    np.testing.assert_array_equal(payload["layers"][-1]["default_prior"], world["level_priors"][-1])

