"""Pytest configuration and child-process isolation for Active Fedference."""

from __future__ import annotations

import os
import shlex
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


@pytest.fixture(scope="session", autouse=True)
def isolated_test_processes(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Bound fixture discovery and use this populated interpreter in child builds."""
    temporary_root = tmp_path_factory.getbasetemp().resolve()
    if temporary_root.is_relative_to(ROOT):
        raise pytest.UsageError(
            "Test temporary storage must be outside the checkout; set TMPDIR or --basetemp"
        )
    ceilings = os.environ.get("GIT_CEILING_DIRECTORIES", "")
    environment = {
        "GIT_CEILING_DIRECTORIES": os.pathsep.join(filter(None, (ceilings, str(temporary_root)))),
        # Child suites use the known project configuration, but their own cwd
        # for node IDs. An explicit child -c/--rootdir still takes precedence.
        "PYTEST_ADDOPTS": shlex.join(["-c", str(ROOT / "pyproject.toml"), "--rootdir=."]),
        "UV_PYTHON": sys.executable,
    }
    previous = {key: os.environ.get(key) for key in environment}
    os.environ.update(environment)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
