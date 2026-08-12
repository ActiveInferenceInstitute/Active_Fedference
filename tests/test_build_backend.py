"""Real-archive tests for the reproducible PEP 517 build wrapper."""

from __future__ import annotations

import gzip
import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_BACKEND_PATH = _ROOT / "_fedference_build_backend.py"
_SPEC = importlib.util.spec_from_file_location("_fedference_build_backend", _BACKEND_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_BACKEND = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BACKEND)


def _write_sdist(path: Path, *, mtime: int, reverse: bool) -> None:
    members = [("pkg/alpha.txt", b"alpha\n"), ("pkg/beta.txt", b"beta\n")]
    if reverse:
        members.reverse()
    with path.open("wb") as raw:
        with gzip.GzipFile(filename=path.name, mode="wb", fileobj=raw, mtime=mtime) as zipped:
            with tarfile.open(fileobj=zipped, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                for name, payload in members:
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    info.mtime = mtime
                    info.uid = 501
                    info.gid = 20
                    info.uname = "developer"
                    info.gname = "staff"
                    archive.addfile(info, io.BytesIO(payload))


def _write_wheel(path: Path, *, timestamp: tuple[int, int, int, int, int, int], reverse: bool) -> None:
    members = [("pkg/__init__.py", b"VALUE = 1\n"), ("pkg/data.txt", b"data\n")]
    if reverse:
        members.reverse()
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members:
            info = zipfile.ZipInfo(name, date_time=timestamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload)


@pytest.mark.parametrize("value", ("", "-1", "1.5", "tomorrow", "９", "9" * 21))
def test_source_date_epoch_rejects_noncanonical_values(value: str) -> None:
    with pytest.raises(ValueError, match="SOURCE_DATE_EPOCH"):
        _BACKEND._source_date_epoch(value)


def test_source_date_epoch_accepts_absence_and_nonnegative_integer() -> None:
    assert _BACKEND._source_date_epoch("1785205200") == 1785205200
    assert _BACKEND._source_date_epoch(" 0 ") == 0


def test_sdist_normalization_removes_order_owner_and_time_drift(tmp_path: Path) -> None:
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    _write_sdist(first, mtime=1_700_000_000, reverse=False)
    _write_sdist(second, mtime=1_800_000_000, reverse=True)

    _BACKEND._normalize_sdist_archive(first, 1_785_205_200)
    _BACKEND._normalize_sdist_archive(second, 1_785_205_200)

    assert first.read_bytes() == second.read_bytes()
    with tarfile.open(first, mode="r:gz") as archive:
        members = archive.getmembers()
    assert [member.name for member in members] == sorted(member.name for member in members)
    assert all(member.mtime == 1_785_205_200 for member in members)
    assert all((member.uid, member.gid, member.uname, member.gname) == (0, 0, "", "") for member in members)


def test_wheel_normalization_removes_order_and_timestamp_drift(tmp_path: Path) -> None:
    first = tmp_path / "first.whl"
    second = tmp_path / "second.whl"
    _write_wheel(first, timestamp=(2024, 1, 1, 0, 0, 0), reverse=False)
    _write_wheel(second, timestamp=(2026, 1, 1, 0, 0, 0), reverse=True)

    _BACKEND._normalize_wheel_archive(first, 1_785_205_200)
    _BACKEND._normalize_wheel_archive(second, 1_785_205_200)

    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        members = archive.infolist()
    assert [member.filename for member in members] == sorted(
        member.filename for member in members
    )
    assert {member.date_time for member in members} == {(2026, 7, 28, 2, 20, 0)}


def test_build_configuration_pins_and_ships_the_reproducible_backend() -> None:
    pyproject = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    manifest = (_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert 'requires = ["setuptools==81.0.0"]' in pyproject
    assert 'build-backend = "_fedference_build_backend"' in pyproject
    assert 'backend-path = ["."]' in pyproject
    assert "include _fedference_build_backend.py" in manifest


def test_package_metadata_and_source_manifest_are_release_complete() -> None:
    pyproject = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    manifest = (_ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert (_ROOT / "LICENSE").is_file()
    assert 'readme = {file = "README.md", content-type = "text/markdown"}' in pyproject
    assert 'license = "MIT"' in pyproject
    assert 'license-files = ["LICENSE"]' in pyproject
    assert '{name = "Daniel Ari Friedman", email = "daniel@activeinference.institute"}' in pyproject
    assert 'Repository = "https://github.com/ActiveInferenceInstitute/Active_Fedference"' in pyproject
    assert 'DOI = "https://doi.org/10.5281/zenodo.21864004"' in pyproject
    assert "recursive-include docs *.md" in manifest
    assert "recursive-include manuscript *.bib *.md *.png *.yaml" in manifest
    assert "recursive-include scripts *.py *.md" in manifest
    assert "recursive-include tests *.md *.py" in manifest
