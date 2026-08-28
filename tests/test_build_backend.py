"""Real-archive tests for the reproducible PEP 517 build wrapper."""

from __future__ import annotations

import gzip
import importlib.util
import io
import os
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest
import tomllib
import yaml

from publication.metadata import validate_publication_lifecycle, write_metadata

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


def _make_lifecycle_build_project(
    root: Path,
    *,
    version: str,
    doi: str | None,
    date_released: str | None,
) -> Path:
    """Create a real, isolated PEP 517 project for one lifecycle state."""
    root.mkdir()
    (root / "manuscript").mkdir()
    package = root / "src" / "lifecycle_fixture"
    package.mkdir(parents=True)
    package.joinpath("__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    root.joinpath("README.md").write_text("# Lifecycle fixture\n", encoding="utf-8")
    root.joinpath("LICENSE").write_text("MIT fixture\n", encoding="utf-8")
    root.joinpath("_fedference_build_backend.py").write_bytes(_BACKEND_PATH.read_bytes())
    root.joinpath("MANIFEST.in").write_text(
        "include _fedference_build_backend.py\n"
        "include LICENSE README.md CITATION.cff .zenodo.json codemeta.json uv.lock\n"
        "recursive-include manuscript *.yaml\n",
        encoding="utf-8",
    )
    doi_url_line = f'DOI = "https://doi.org/{doi}"\n' if doi is not None else ""
    root.joinpath("pyproject.toml").write_text(
        "[build-system]\n"
        'requires = ["setuptools==81.0.0"]\n'
        'build-backend = "_fedference_build_backend"\n'
        'backend-path = ["."]\n'
        "[project]\n"
        'name = "lifecycle-fixture"\n'
        f'version = "{version}"\n'
        'description = "Lifecycle fixture"\n'
        'readme = {file = "README.md", content-type = "text/markdown"}\n'
        'license = "MIT"\n'
        'license-files = ["LICENSE"]\n'
        'requires-python = ">=3.10"\n'
        "[project.urls]\n"
        f"{doi_url_line}"
        "[tool.setuptools.packages.find]\n"
        'where = ["src"]\n',
        encoding="utf-8",
    )
    root.joinpath("uv.lock").write_text(
        "version = 1\n"
        "[[package]]\n"
        'name = "lifecycle-fixture"\n'
        f'version = "{version}"\n'
        'source = { editable = "." }\n',
        encoding="utf-8",
    )
    publication: dict[str, object] = {
        "doi": doi or "",
        "software_name": "Lifecycle fixture",
        "github_repository": "https://example.invalid/lifecycle-fixture",
        "date_created": "2026-08-01",
        "date_released": date_released,
        "abstract": "Lifecycle fixture abstract.",
        "description": "Lifecycle fixture description.",
    }
    if doi is None:
        publication["doi_status"] = "(forthcoming)"
    root.joinpath("manuscript", "config.yaml").write_text(
        yaml.safe_dump(
            {
                "paper": {"title": "Lifecycle fixture", "version": version},
                "authors": [
                    {
                        "name": "Ada Lovelace",
                        "orcid": "0000-0000-0000-0001",
                        "affiliation": "Analytical Engine Society",
                    }
                ],
                "publication": publication,
                "metadata": {"license": "MIT"},
            }
        ),
        encoding="utf-8",
    )
    write_metadata(root)
    lifecycle = validate_publication_lifecycle(
        root,
        require_generated_metadata=True,
        require_canonical_pdf=False,
    )
    if lifecycle.canonical_pdf is not None:
        root.joinpath(lifecycle.canonical_pdf).write_bytes(b"canonical PDF fixture\n")
    validate_publication_lifecycle(root)
    return root


@pytest.mark.publication
@pytest.mark.parametrize(
    ("version", "doi", "date_released"),
    (
        ("3.2.1.dev0", None, None),
        ("3.2.1", "10.5281/zenodo.12345", "2026-08-27"),
    ),
)
def test_real_temporary_build_is_lifecycle_aware(
    tmp_path: Path,
    version: str,
    doi: str | None,
    date_released: str | None,
) -> None:
    project = _make_lifecycle_build_project(
        tmp_path / "project",
        version=version,
        doi=doi,
        date_released=date_released,
    )
    distribution = tmp_path / "dist"
    environment = dict(os.environ)
    environment.update({"SOURCE_DATE_EPOCH": "1785205200", "UV_NO_PROGRESS": "1"})
    completed = subprocess.run(
        [
            "uv",
            "build",
            "--force-pep517",
            "--no-build-isolation",
            "--out-dir",
            str(distribution),
            str(project),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout

    wheel = next(distribution.glob("*.whl"))
    sdist = next(distribution.glob("*.tar.gz"))
    with zipfile.ZipFile(wheel) as archive:
        metadata_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        wheel_metadata = archive.read(metadata_name).decode("utf-8")
    with tarfile.open(sdist, mode="r:gz") as archive:
        pkg_info_name = next(name for name in archive.getnames() if name.endswith("/PKG-INFO"))
        extracted = archive.extractfile(pkg_info_name)
        assert extracted is not None
        sdist_metadata = extracted.read().decode("utf-8")

    for metadata in (wheel_metadata, sdist_metadata):
        assert f"Version: {version}\n" in metadata
        if doi is None:
            assert "Project-URL: DOI" not in metadata
        else:
            assert f"Project-URL: DOI, https://doi.org/{doi}\n" in metadata


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
    config = yaml.safe_load((_ROOT / "manuscript/config.yaml").read_text(encoding="utf-8"))
    package_metadata = tomllib.loads(pyproject)["project"]
    lifecycle = validate_publication_lifecycle(
        _ROOT,
        require_generated_metadata=False,
        require_canonical_pdf=False,
    )
    assert package_metadata["version"] == config["paper"]["version"] == lifecycle.version
    if lifecycle.state == "development":
        assert lifecycle.version.endswith(".dev0")
        assert config["publication"]["doi"] == ""
        assert config["publication"]["doi_status"] == "(forthcoming)"
        assert config["publication"]["date_released"] is None
        assert lifecycle.canonical_pdf is None
        assert "DOI" not in package_metadata["urls"]
    else:
        assert ".dev" not in lifecycle.version
        assert lifecycle.doi is not None
        assert lifecycle.date_released is not None
        assert "doi_status" not in config["publication"]
        assert package_metadata["urls"]["DOI"] == f"https://doi.org/{lifecycle.doi}"
        assert lifecycle.canonical_pdf is not None
    assert (_ROOT / "manuscript" / "config.yaml.example").is_file()
    assert "recursive-include docs *.md" in manifest
    assert "recursive-include examples *.json *.md *.py" in manifest
    assert "recursive-include src *.md" in manifest
    assert "include src/fedference/py.typed" in manifest
    assert '"py.typed"' in pyproject
    assert "recursive-include manuscript *.bib *.md *.png *.yaml *.yaml.example" in manifest
    assert "recursive-include scripts *.py *.md" in manifest
    assert "recursive-include tests *.md *.py" in manifest


def test_ci_installed_artifact_gate_executes_the_labeled_application_outside_checkout() -> None:
    workflow = (_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    for required_member in (
        "docs/application-guide.md",
        "examples/05_labeled_application.py",
        "examples/data/labeled_aggregation_request.json",
        "src/fedference/py.typed",
    ):
        assert required_member in workflow
    assert 'cd "$installed_smoke_root"' in workflow
    assert "LabeledAggregationRequest.from_json" in workflow
    assert "aggregate_labeled(request).aggregation.solver_status == 'nominal'" in workflow
    assert workflow.count('aggregate --input request.json --output-dir') == 2
    assert workflow.count('verify wheel-run/receipt.json --require-nominal-solver') == 1
    assert workflow.count('verify sdist-run/receipt.json --require-nominal-solver') == 1
    assert workflow.count("assert 'torch' not in") >= 4
    assert workflow.count("joinpath('py.typed').is_file()") == 2
    assert "version('active_fedference')" in workflow
    assert "== '1.1.0.dev0'" not in workflow
    assert workflow.count("- name: Release bundle verify") == 1
