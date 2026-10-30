"""Explicit test profiles; required upstream tests never become optional skips."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

import pytest

GNN_MODULES: Final = frozenset({
    "tests/integration/test_gnn_model_cli.py",
    "tests/integration/test_gnn_pymdp.py",
    "tests/test_gnn_interop.py",
    "tests/fedference/test_gnn_sentinel.py",
    "tests/fedference/test_gnn_scientific.py",
    "tests/fedference/test_gnn_structured.py",
})


def node_profile(node: str) -> str:
    """Assign every current node to exactly one required execution profile."""
    return "gnn" if node.split("::", 1)[0] in GNN_MODULES else "native"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Expose native default and explicit complete/upstream validation."""
    parser.addoption("--validation-profile", choices=("native", "gnn", "full"),
                     default="native", help="native excludes external GNN modules; full includes all")


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    """Avoid importing external-backend tests in the default native profile."""
    if not collection_path.is_file() or not collection_path.name.startswith("test_"):
        return None
    if not collection_path.is_relative_to(config.rootpath):
        return None
    relative = collection_path.relative_to(config.rootpath).as_posix()
    profile = config.getoption("--validation-profile")
    if profile == "full":
        return None
    return node_profile(relative) != profile


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Also enforce profiles for explicitly named modules and individual nodes."""
    profile = config.getoption("--validation-profile")
    if profile != "full":
        rejected = [item for item in items if node_profile(item.nodeid) != profile]
        items[:] = [item for item in items if node_profile(item.nodeid) == profile]
        config.hook.pytest_deselected(items=rejected)


def pytest_runtestloop(session: pytest.Session) -> None:
    """Refuse a required external profile before fixtures can skip or KeyError."""
    if session.config.option.collectonly:
        return
    if session.config.getoption("--validation-profile") in ("gnn", "full"):
        required = ("GNN_ROOT", "GNN_PARSER_DEPENDENCIES")
        missing = [name for name in required if not os.environ.get(name)]
        if missing:
            raise pytest.UsageError("GNN_PROFILE_RUNTIME_REQUIRED: " + ", ".join(missing))
        root = Path(os.environ["GNN_ROOT"])
        dependencies = Path(os.environ["GNN_PARSER_DEPENDENCIES"])
        python = Path(os.environ.get("GNN_PYTHON", str(root / ".venv/bin/python")))
        if (not (root / "src/gnn").is_dir()
                or not python.is_file() or not dependencies.is_dir()):
            raise pytest.UsageError(
                "GNN_PROFILE_RUNTIME_REQUIRED: invalid source, runtime or parser dependencies"
            )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Required execution profiles cannot report success with skipped tests."""
    if session.config.getoption("--validation-profile") in ("gnn", "full"):
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None and reporter.stats.get("skipped") and exitstatus == 0:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
