"""Source/resource and runtime identities for independent analysis workspaces."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import sys
import sysconfig
from pathlib import Path
from typing import Any

from analysis.stage_io import StageError, confined, digest


def runtime_identity() -> dict[str, Any]:
    """Bind actual Python, numerical packages and thread selectors, without host paths."""
    packages = sorted((item.metadata["Name"], item.version) for item in importlib.metadata.distributions())
    return {
        "python": platform.python_version(),
        "implementation": sys.implementation.name,
        "interpreter_sha256": digest(Path(sys.executable).resolve().read_bytes()),
        "abi": sys.implementation.cache_tag,
        "gil_disabled": sysconfig.get_config_var("Py_GIL_DISABLED"),
        "platform": sys.platform,
        "machine": platform.machine(),
        "packages": packages,
        "threads": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
    }


def source_names(project: Path, snapshot: dict[str, Any]) -> set[str]:
    """Bind the full admitted source/resource roster, including additions."""
    names = set(snapshot["input_hashes"])
    names.add("scripts/analysis_stages.py")
    for path in (project / "src").rglob("*"):
        if path.is_symlink():
            raise StageError("source resource tree contains a symlink")
        if path.suffix in {".json", ".yaml", ".csv", ".md"}:
            names.add(path.relative_to(project).as_posix())
    return names


def validate_source_tree(project: Path) -> None:
    """Reject source links before fingerprinting can read through them."""
    for name in ("src", "manuscript", "scripts"):
        folder = confined(project, name)
        for path in folder.rglob("*"):
            if path.is_symlink():
                raise StageError("source input tree contains a symlink")
    for name in ("experiment_plan.yaml", "pyproject.toml", "uv.lock"):
        confined(project, name)


def output_baseline(project: Path) -> dict[str, str]:
    """Bind replaceable outputs so later peer edits cannot be clobbered by import."""
    from publication.pipeline_freshness import PIPELINE_STAGES

    spec = next(stage for stage in PIPELINE_STAGES if stage.name == "analysis")
    return {
        path.relative_to(project).as_posix(): digest(
            confined(project, path.relative_to(project).as_posix()).read_bytes()
        )
        for pattern in spec.output_patterns
        for path in project.glob(pattern)
        if path.is_file()
    }
