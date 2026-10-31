"""Real runtime inventories ignore caller metadata and bind installed versions."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
import venv
import zipfile
from pathlib import Path

import pytest

from analysis.stage_workspace import prepare_workspace
from tests.analysis.stage_fixtures import SOURCE, run
from tests.analysis.stage_fixtures import prepared as prepared

PROBE = (
    "import json,sys; sys.path[:0] = json.loads(sys.argv[1]); "
    "from analysis.stage_identity import runtime_identity; "
    "print(json.dumps(runtime_identity(), sort_keys=True))"
)


def _identity(python: Path, *extra: Path, environment: dict[str, str] | None = None) -> dict:
    process = subprocess.run(
        [str(python), "-IB", "-c", PROBE, json.dumps([*(str(p) for p in extra), str(SOURCE / "src")])],
        capture_output=True, text=True, check=True, env=environment, timeout=15,
    )
    return json.loads(process.stdout)


def _wheel(directory: Path, version: str) -> Path:
    """Build a complete local wheel for an actual offline uv installation."""
    metadata = f"native_stage_probe-{version}.dist-info"
    contents = {
        "native_stage_probe.py": b"VALUE = 42\n",
        f"{metadata}/METADATA": (
            f"Metadata-Version: 2.1\nName: native-stage-probe\nVersion: {version}\n".encode()
        ),
        f"{metadata}/WHEEL": (
            b"Wheel-Version: 1.0\nGenerator: stage-runtime-test\n"
            b"Root-Is-Purelib: true\nTag: py3-none-any\n"
        ),
    }
    record = []
    for name, data in contents.items():
        encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
        record.append(f"{name},sha256={encoded},{len(data)}\n")
    contents[f"{metadata}/RECORD"] = ("".join(record) + f"{metadata}/RECORD,,\n").encode()
    path = directory / f"native_stage_probe-{version}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
    return path


@pytest.fixture
def runtime_python(tmp_path: Path) -> Path:
    environment = tmp_path / "runtime"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    python = environment / "bin/python"
    check = subprocess.run(
        [str(python), "-IB", "-c", "import json,sys;print(json.dumps([sys.prefix,sys.base_prefix]))"],
        capture_output=True, text=True, check=True, timeout=15,
    )
    prefix, base_prefix = json.loads(check.stdout)
    assert Path(prefix).resolve() == environment.resolve()
    assert Path(base_prefix).resolve() == Path(sys.base_prefix).resolve()
    return python


def _install(python: Path, wheel: Path) -> None:
    result = subprocess.run(
        ["uv", "pip", "install", "--python", str(python), "--no-deps", "--no-index", str(wheel)],
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)


def test_runtime_inventory_ignores_caller_metadata_and_duplicate_search_paths(
    runtime_python: Path, tmp_path: Path,
) -> None:
    _install(runtime_python, _wheel(tmp_path, "1.0"))
    expected = _identity(runtime_python)
    assert expected["packages"] == [["native-stage-probe", "1.0"]]
    ambient = tmp_path / "caller"
    metadata = ambient / "caller_only-9.0.dist-info"
    metadata.mkdir(parents=True)
    (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: caller-only\nVersion: 9.0\n")
    assert _identity(runtime_python, ambient, ambient) == expected


def test_runtime_inventory_changes_when_a_real_installed_distribution_changes(
    runtime_python: Path, tmp_path: Path,
) -> None:
    _install(runtime_python, _wheel(tmp_path, "1.0"))
    before = _identity(runtime_python)
    _install(runtime_python, _wheel(tmp_path, "2.0"))
    after = _identity(runtime_python)
    assert before["packages"] == [["native-stage-probe", "1.0"]]
    assert after["packages"] == [["native-stage-probe", "2.0"]]
    assert {k: v for k, v in before.items() if k != "packages"} == {
        k: v for k, v in after.items() if k != "packages"
    }
    changed_environment = dict(os.environ, OMP_NUM_THREADS="37")
    changed = _identity(runtime_python, environment=changed_environment)
    assert changed["threads"]["OMP_NUM_THREADS"] == "37"
    assert changed != after
    assert changed["packages"] == after["packages"]


def test_stage_executes_with_unchanged_installed_runtime_from_caller_metadata_context(
    prepared: tuple[Path, Path], tmp_path: Path,
) -> None:
    project, _ = prepared
    ambient = tmp_path / "caller"
    metadata = ambient / "caller_only-9.0.dist-info"
    metadata.mkdir(parents=True)
    (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: caller-only\nVersion: 9.0\n")
    workspace = tmp_path / "context-workspace"
    before = sys.path.copy()
    try:
        sys.path.insert(0, str(ambient))
        manifest = prepare_workspace(project, workspace, profile="smoke")
    finally:
        sys.path[:] = before
    assert run(workspace)["status"] == "complete"
    assert "caller-only" not in {name for name, _ in manifest["runtime"]["packages"]}
