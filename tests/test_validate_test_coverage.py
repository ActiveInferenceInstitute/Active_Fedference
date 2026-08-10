"""Regression tests for the authoritative test-coverage receipt wrapper."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_PATH = _ROOT / "scripts" / "validate_test_coverage.py"
_SPEC = importlib.util.spec_from_file_location("validate_test_coverage", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_SCRIPT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SCRIPT)


def _write_junit(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")


def test_junit_summary_reads_a_direct_testsuite(tmp_path: Path) -> None:
    path = tmp_path / "direct.xml"
    _write_junit(
        path,
        '<testsuite tests="5" failures="1" errors="1" skipped="1" />',
    )

    assert _SCRIPT._junit_summary(path) == {
        "collected": 5,
        "passed": 2,
        "failed": 2,
        "skipped": 1,
    }


def test_junit_summary_reads_pytest_testsuites_wrapper(tmp_path: Path) -> None:
    """pytest 9 puts the actual totals on a child, not its wrapper root."""
    path = tmp_path / "pytest9.xml"
    _write_junit(
        path,
        '<testsuites name="pytest tests">'
        '<testsuite tests="3" failures="0" errors="0" skipped="1" />'
        '<testsuite tests="5" failures="1" errors="0" skipped="0" />'
        "</testsuites>",
    )

    assert _SCRIPT._junit_summary(path) == {
        "collected": 8,
        "passed": 6,
        "failed": 1,
        "skipped": 1,
    }


@pytest.mark.parametrize(
    "contents",
    (
        "<testsuites />",
        "<testsuite />",
        '<report tests="1" />',
        '<testsuite tests="1" failures="2" />',
    ),
)
def test_junit_summary_fails_closed_on_invalid_totals(tmp_path: Path, contents: str) -> None:
    path = tmp_path / "invalid.xml"
    _write_junit(path, contents)

    with pytest.raises(RuntimeError, match="invalid pytest JUnit report|invalid pytest JUnit totals"):
        _SCRIPT._junit_summary(path)
