"""Tests for the logistic-regression robustness figure generator.

No mocks: real accuracy curves and optional seed-level intervals are rendered to
``tmp_path``; we assert the PNG exists and error paths raise.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from figures import generate_bnn_robustness
from figures._common import semantic_style
from figures.bnn_robustness import _configuration_style_role, _direct_condition_label

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_bnn_configuration_styles_are_neutral_condition_roles() -> None:
    reference_role = _configuration_style_role("nll / L2=0.05 (standard proxy)", 0)
    comparison_role = _configuration_style_role(
        "rcce / L2=0.10 (exploratory proxy)", 0
    )

    assert reference_role == (
        "condition_reference",
        True,
    )
    assert comparison_role == (
        "condition_comparison",
        False,
    )
    reference_style = semantic_style(reference_role[0])
    comparison_style = semantic_style(comparison_role[0])
    assert (reference_style.marker, reference_style.dash, reference_style.hatch) != (
        comparison_style.marker,
        comparison_style.dash,
        comparison_style.hatch,
    )


def test_bnn_direct_labels_are_compact_and_neutral() -> None:
    assert _direct_condition_label("nll / L2=0.05 (standard proxy)") == "standard proxy"
    assert _direct_condition_label("rcce / L2=0.10 (exploratory proxy)") == "exploratory proxy"
    assert _direct_condition_label("additional operating point") == "additional operating point"


def test_bnn_robustness_happy_path(tmp_path: Path) -> None:
    levels = [0.0, 0.1, 0.2, 0.3]
    accuracy = {
        "nll / L2=0.05 (standard proxy)": [0.95, 0.88, 0.78, 0.70],
        "rcce / L2=0.10 (exploratory proxy)": [0.95, 0.93, 0.91, 0.90],
    }
    intervals = {
        "nll / L2=0.05 (standard proxy)": [
            [0.94, 0.96],
            [0.87, 0.89],
            [0.76, 0.80],
            [0.68, 0.72],
        ],
        "rcce / L2=0.10 (exploratory proxy)": [
            [0.94, 0.96],
            [0.92, 0.94],
            [0.90, 0.92],
            [0.89, 0.91],
        ],
    }
    path = generate_bnn_robustness(
        accuracy,
        levels,
        accuracy_ci_by_config=intervals,
        project_root=tmp_path,
    )
    assert path.exists()
    assert path.read_bytes()[:8] == _PNG_MAGIC
    assert path.stat().st_size > 0


def test_bnn_robustness_single_config(tmp_path: Path) -> None:
    path = generate_bnn_robustness({"robust": [0.9, 0.9]}, [0.0, 0.5], project_root=tmp_path)
    assert path.exists()
    assert path.read_bytes()[:8] == _PNG_MAGIC


def test_bnn_robustness_rejects_empty_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        generate_bnn_robustness({}, [0.0, 0.5], project_root=tmp_path)


def test_bnn_robustness_rejects_empty_levels(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        generate_bnn_robustness({"x": []}, [], project_root=tmp_path)


def test_bnn_robustness_rejects_length_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        generate_bnn_robustness({"x": [0.9]}, [0.0, 0.5], project_root=tmp_path)


def test_bnn_robustness_rejects_bad_ci(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        generate_bnn_robustness(
            {"x": [0.9, 0.8]},
            [0.0, 0.5],
            accuracy_ci_by_config={"x": [[0.8, 1.0]]},
            project_root=tmp_path,
        )
