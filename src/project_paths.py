"""Resolve project root and standard output paths."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_DEFAULT_ROOT = Path(__file__).resolve().parent.parent

#: Environment override for the effective project root. Subprocess tests set
#: this to a temporary scaffold so pipeline scripts never write into the real
#: committed ``output/`` tree (the smoke-clobbering incident class).
ENV_PROJECT_ROOT_VAR = "ACTIVE_FEDFERENCE_PROJECT_ROOT"


def resolve_env_project_root(default: Path) -> Path:
    """Return the effective project root, honoring ``ACTIVE_FEDFERENCE_PROJECT_ROOT``.

    When the environment variable is unset, *default* is returned unchanged —
    default behavior is byte-for-byte identical to the pre-override wiring.
    When it is set, the value must name an existing directory that contains
    ``manuscript/config.yaml``; anything else raises :class:`RuntimeError`
    loudly. There is deliberately no silent fallback to *default* on an invalid
    value (masking discipline): a typo'd override must fail the run, not quietly
    redirect writes back into the real project tree.
    """
    raw = os.environ.get(ENV_PROJECT_ROOT_VAR)
    if raw is None:
        return default
    root = Path(raw)
    if not root.is_dir():
        raise RuntimeError(
            f"{ENV_PROJECT_ROOT_VAR}={raw!r} is not an existing directory"
        )
    if not (root / "manuscript" / "config.yaml").is_file():
        raise RuntimeError(
            f"{ENV_PROJECT_ROOT_VAR}={raw!r} does not contain manuscript/config.yaml "
            "— refusing to treat it as a project root"
        )
    return root.resolve()


def resolve_script_project_root(default: Path, explicit: Path | None = None) -> Path:
    """Resolve a script's effective checkout root.

    A command-line ``--project-root`` takes precedence over the validated
    ``ACTIVE_FEDFERENCE_PROJECT_ROOT`` test/review override. Keeping this
    precedence in one source-owned helper makes scripts composable from CI, a
    sibling checkout, or a real subprocess test without copying path-selection
    rules into each entry point. The helper deliberately only requires an
    existing directory: validators may target incomplete fixture trees and
    should report their own findings.
    """
    if explicit is None:
        return resolve_env_project_root(default)
    root = Path(explicit).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"explicit project root is not an existing directory: {root}")
    return root


def resolve_project_root(package_name: str) -> Path:
    """Resolve the project root directory from a loaded package or default to parent.

    Checks whether *package_name* is already imported and has a ``project_root``
    attribute; if so, returns that. Otherwise returns the default root — two
    levels above this file (``src/project_paths.py`` -> ``src/`` -> project root).
    """
    mod = sys.modules.get(package_name)
    if mod is not None and hasattr(mod, "project_root"):
        return Path(mod.project_root)
    return _DEFAULT_ROOT


def project_output_dirs(project_root: Path | None = None) -> dict[str, Path]:
    """Return common output directories for Active Fedference."""
    root = project_root or _DEFAULT_ROOT
    output = root / "output"
    return {
        "output": output,
        "figures": output / "figures",
        "data": output / "data",
        "reports": output / "reports",
        "web": output / "web",
    }
