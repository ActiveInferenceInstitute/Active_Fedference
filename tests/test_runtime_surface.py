"""Static contract for a complete runtime surface.

The source tree must not silently retain retired placeholder implementations or
test-double APIs. The explicit real-computation policy remains documented in the
project instructions; this test checks the executable Python surfaces themselves.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from _runtime import save_figure_data

ROOT = Path(__file__).resolve().parent.parent
# Scan the complete executable surface.  Keeping this at selected package
# directories would allow retired APIs to survive in root-level helpers or
# future source packages without tripping the policy gate.
RUNTIME_DIRS = (ROOT / "src", ROOT / "scripts")
TEST_DIR = ROOT / "tests"
# Compatibility adapters and deprecation warnings are part of the supported
# runtime surface; this gate only rejects placeholder implementations. The
# checker file necessarily spells those marker names in its own regex literal.
_SELF = Path(__file__).resolve()
_RETIRED_MARKER = re.compile(r"\b(?:stub|fake|dummy)\b", re.IGNORECASE)
_TEST_DOUBLE_API = re.compile(
    r"(?:unittest\.mock|MagicMock|\bMock\s*\(|@patch\b|create_autospec|pytest-mock)"
)
# The tests/ scan additionally forbids pytest's monkeypatch fixture. The
# patterns are deliberately API-shaped (imports, calls, decorators, the fixture
# name) rather than marker words, so prose in docstrings/comments describing
# the no-mock policy or negative-control names cannot false-positive.
# ``monkeypatch.setenv``/``delenv`` manipulate the REAL process environment that
# the code under test genuinely reads (e.g. the ACTIVE_FEDFERENCE_PROJECT_ROOT
# override) — that is a real execution path, not a test double. Only
# ``monkeypatch.setattr`` (behavior patching) counts as a double here.
_TEST_DOUBLE_API_TESTS = re.compile(
    r"(?:unittest\.mock|MagicMock|\bMock\s*\(|@patch\b|create_autospec"
    r"|pytest-mock|monkeypatch\.setattr)"
)


def _python_files() -> list[Path]:
    return [path for directory in RUNTIME_DIRS for path in directory.rglob("*.py")]


def _test_python_files() -> list[Path]:
    return [
        path for path in sorted(TEST_DIR.rglob("*.py"))
        if path.resolve() != _SELF and "__pycache__" not in path.parts
    ]


def test_runtime_python_has_no_retired_placeholder_markers() -> None:
    offenders = [
        f"{path.relative_to(ROOT)}:{line_number}"
        for path in _python_files()
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if _RETIRED_MARKER.search(line)
    ]
    assert offenders == []


def test_runtime_python_has_no_test_double_apis() -> None:
    offenders = [
        str(path.relative_to(ROOT))
        for path in _python_files()
        if _TEST_DOUBLE_API.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_test_suite_has_no_test_double_apis() -> None:
    """The no-mock policy binds the test suite itself (SYN-8).

    Only the test-double API scan is extended to tests/ — the retired-marker
    word scan is not, because test prose legitimately names negative controls
    and the no-mock policy in docstrings/comments.
    """
    offenders = [
        f"{path.relative_to(ROOT)}:{line_number}"
        for path in _test_python_files()
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        )
        if _TEST_DOUBLE_API_TESTS.search(line)
    ]
    assert offenders == []


def test_save_figure_data_writes_nested_json_payload(tmp_path: Path) -> None:
    """The shared figure-data boundary writes a portable, nested payload."""
    payload = {"series": [1, 2, 3], "metadata": {"unit": "nats"}}

    path = save_figure_data(payload, "runtime_surface", tmp_path)

    assert path == tmp_path / "data" / "runtime_surface.json"
    assert json.loads(path.read_text(encoding="utf-8")) == payload


def test_save_figure_data_rejects_nonfinite_payload_before_replacing_file(tmp_path: Path) -> None:
    path = tmp_path / "data" / "runtime_surface.json"
    path.parent.mkdir()
    path.write_text("original\n", encoding="utf-8")

    try:
        save_figure_data({"value": float("nan")}, "runtime_surface", tmp_path)
        raise AssertionError("expected strict JSON serialization to fail")
    except ValueError:
        pass
    assert path.read_text(encoding="utf-8") == "original\n"
