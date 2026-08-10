"""Tests for figures.disjoint_fov_world.generate_disjoint_fov_figure."""

from __future__ import annotations

from pathlib import Path

from figures.disjoint_fov_world import generate_disjoint_fov_figure

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_MIN_SIZE_BYTES = 10 * 1024  # 10 KB


def test_disjoint_fov_figure_happy_path(tmp_path: Path) -> None:
    """generate_disjoint_fov_figure runs without error and produces a PNG."""
    result = generate_disjoint_fov_figure(project_root=tmp_path)
    path = Path(result)
    assert path.exists(), "output file was not created"
    assert path.read_bytes()[:8] == _PNG_MAGIC, "file is not a valid PNG"


def test_disjoint_fov_figure_size(tmp_path: Path) -> None:
    """Output PNG is larger than 10 KB (non-trivial content)."""
    result = generate_disjoint_fov_figure(project_root=tmp_path)
    path = Path(result)
    assert path.stat().st_size > _MIN_SIZE_BYTES, (
        f"PNG is unexpectedly small ({path.stat().st_size} bytes < {_MIN_SIZE_BYTES}): "
        "figure may be empty or degenerate"
    )


def test_disjoint_fov_figure_filename(tmp_path: Path) -> None:
    """Output file is named disjoint_fov_world.png."""
    result = generate_disjoint_fov_figure(project_root=tmp_path)
    assert Path(result).name == "disjoint_fov_world.png"


def test_disjoint_fov_figure_creates_output_dir(tmp_path: Path) -> None:
    """generate_disjoint_fov_figure creates the output directory if it does not exist."""
    nested = tmp_path / "sub" / "figures"
    result = generate_disjoint_fov_figure(project_root=nested)
    assert nested.exists(), "output directory was not created"
    assert Path(result).exists(), "output file was not created in nested dir"
