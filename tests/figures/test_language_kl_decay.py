"""Tests for the language-acquisition KL-decay figure generator.

No mocks: a real declining KL curve is rendered to ``tmp_path`` (headless Agg);
we assert the PNG exists, is non-empty PNG bytes, and the error path raises.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from figures import generate_language_kl_decay

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_language_kl_decay_happy_path(tmp_path: Path) -> None:
    kl = [1.2, 0.9, 0.7, 0.5, 0.4, 0.3]
    path = generate_language_kl_decay(
        kl,
        trajectory_ci=([1.0, 0.7, 0.5, 0.35, 0.3, 0.2],
                       [1.4, 1.1, 0.9, 0.7, 0.6, 0.5]),
        monotone_decreasing=True,
        n_seeds=4,
        project_root=tmp_path,
    )
    assert path.exists()
    assert path.read_bytes()[:8] == _PNG_MAGIC
    assert path.stat().st_size > 0


def test_language_kl_decay_without_ci(tmp_path: Path) -> None:
    path = generate_language_kl_decay([2.0, 1.0, 0.5], project_root=tmp_path)
    assert path.exists()
    assert path.read_bytes()[:8] == _PNG_MAGIC


def test_language_kl_decay_rejects_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        generate_language_kl_decay([], project_root=tmp_path)


def test_language_kl_decay_rejects_bad_ci(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        generate_language_kl_decay(
            [1.0, 0.5], trajectory_ci=([0.5], [0.8]), project_root=tmp_path
        )


def test_language_kl_decay_rejects_nonfinite_and_negative_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        generate_language_kl_decay([1.0, float("nan")], project_root=tmp_path)
    with pytest.raises(ValueError):
        generate_language_kl_decay([-0.1, 0.0], project_root=tmp_path)


def test_language_kl_decay_rejects_bad_pointwise_bounds(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        generate_language_kl_decay(
            [1.0, 0.5], trajectory_ci=([0.9, 0.6], [1.1, 0.55]), project_root=tmp_path
        )
