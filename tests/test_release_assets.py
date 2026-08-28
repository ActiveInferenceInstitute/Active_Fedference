"""External release-asset staging and GitHub download verification tests."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from publication.release_assets import (
    CHECKSUM_FILENAME,
    GitHubReleaseAsset,
    ReleaseAssetError,
    ReleaseAssetInput,
    _claim_new_destination,
    _destination_path,
    expected_release_asset_names,
    load_github_release_assets,
    stage_release_assets,
    verify_github_release_downloads,
)

_VERSION = "1.1.0"
_DOI = "10.5281/zenodo.12345678"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _release_project(root: Path) -> tuple[ReleaseAssetInput, ...]:
    names = expected_release_asset_names(_VERSION, _DOI)
    locations = {
        "manifest": Path("output/release") / names["manifest"],
        "pdf": Path(names["pdf"]),
        "sdist": Path("dist") / names["sdist"],
        "wheel": Path("dist") / names["wheel"],
    }
    payloads = {
        "manifest": b'{"fingerprint":"fixture"}\n',
        "pdf": b"%PDF-1.7 fixture\n",
        "sdist": b"sdist fixture bytes\n",
        "wheel": b"wheel fixture bytes\n",
    }
    for role, relative in locations.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payloads[role])
    return tuple(
        ReleaseAssetInput(role=role, source=locations[role], filename=names[role])
        for role in ("wheel", "manifest", "sdist", "pdf")
    )


def _stage(root: Path, name: str = "staged") -> Path:
    stage_release_assets(
        root,
        Path(name),
        _release_project(root),
        version=_VERSION,
        doi=_DOI,
    )
    return root / name


def _download_copy(staged: Path, downloaded: Path) -> Path:
    downloaded.mkdir()
    for path in staged.iterdir():
        shutil.copyfile(path, downloaded / path.name)
    return downloaded


def _api_assets(directory: Path) -> tuple[GitHubReleaseAsset, ...]:
    return tuple(
        GitHubReleaseAsset(
            name=path.name,
            size=path.stat().st_size,
            digest=f"sha256:{_sha256(path)}",
        )
        for path in sorted(directory.iterdir(), key=lambda value: value.name)
    )


def test_stage_release_assets_writes_exact_sorted_set(tmp_path: Path) -> None:
    inputs = _release_project(tmp_path)
    result = stage_release_assets(
        tmp_path,
        Path("stage"),
        inputs,
        version=_VERSION,
        doi=_DOI,
    )

    expected_names = set(expected_release_asset_names(_VERSION, _DOI).values())
    assert {path.name for path in result.destination.iterdir()} == expected_names | {
        CHECKSUM_FILENAME
    }
    assert result.filenames == tuple(sorted(expected_names))
    assert result.checksum_filename == CHECKSUM_FILENAME
    assert result.checksum_sha256 == _sha256(result.destination / CHECKSUM_FILENAME)
    lines = (result.destination / CHECKSUM_FILENAME).read_text(encoding="utf-8").splitlines()
    assert [line.split("  ", 1)[1] for line in lines] == sorted(expected_names)
    assert CHECKSUM_FILENAME not in "\n".join(lines)
    for record in result.assets:
        assert record.sha256 == _sha256(result.destination / record.filename)
        assert record.size == (result.destination / record.filename).stat().st_size
    json.dumps(result.as_dict(), sort_keys=True)


def test_reordered_inputs_and_repeated_staging_are_byte_identical(tmp_path: Path) -> None:
    inputs = _release_project(tmp_path)
    first = stage_release_assets(
        tmp_path,
        Path("first"),
        inputs,
        version=_VERSION,
        doi=_DOI,
    )
    second = stage_release_assets(
        tmp_path,
        Path("second"),
        reversed(inputs),
        version=_VERSION,
        doi=_DOI,
    )

    assert first.as_dict()["assets"] == second.as_dict()["assets"]
    assert first.checksum_sha256 == second.checksum_sha256
    assert {
        path.name: path.read_bytes() for path in first.destination.iterdir()
    } == {path.name: path.read_bytes() for path in second.destination.iterdir()}


def test_stage_rejects_missing_input_without_creating_destination(tmp_path: Path) -> None:
    inputs = _release_project(tmp_path)
    missing = tmp_path / inputs[0].source
    missing.unlink()

    with pytest.raises(ReleaseAssetError, match="does not exist"):
        stage_release_assets(
            tmp_path,
            Path("stage"),
            inputs,
            version=_VERSION,
            doi=_DOI,
        )
    assert not (tmp_path / "stage").exists()


def test_stage_rejects_duplicate_role_and_filename(tmp_path: Path) -> None:
    inputs = list(_release_project(tmp_path))
    duplicate_role = [*inputs[:3], inputs[0]]
    with pytest.raises(ReleaseAssetError, match="duplicate release asset role"):
        stage_release_assets(
            tmp_path,
            Path("duplicate-role"),
            duplicate_role,
            version=_VERSION,
            doi=_DOI,
        )

    duplicate_name = [
        inputs[0],
        inputs[1],
        inputs[2],
        ReleaseAssetInput(
            role=inputs[3].role,
            source=inputs[3].source,
            filename=inputs[0].filename,
        ),
    ]
    with pytest.raises(ReleaseAssetError, match="duplicate release asset filename"):
        stage_release_assets(
            tmp_path,
            Path("duplicate-name"),
            duplicate_name,
            version=_VERSION,
            doi=_DOI,
        )


def test_stage_rejects_missing_and_unexpected_roles(tmp_path: Path) -> None:
    inputs = list(_release_project(tmp_path))
    with pytest.raises(ReleaseAssetError, match="exactly four"):
        stage_release_assets(
            tmp_path,
            Path("missing"),
            inputs[:-1],
            version=_VERSION,
            doi=_DOI,
        )

    inputs[-1] = ReleaseAssetInput(
        role="notes",
        source=inputs[-1].source,
        filename=inputs[-1].filename,
    )
    with pytest.raises(ReleaseAssetError, match="unexpected release asset role"):
        stage_release_assets(
            tmp_path,
            Path("unexpected"),
            inputs,
            version=_VERSION,
            doi=_DOI,
        )


def test_stage_rejects_symlinked_and_escaping_inputs(tmp_path: Path) -> None:
    inputs = list(_release_project(tmp_path))
    source = tmp_path / inputs[0].source
    target = source.with_suffix(source.suffix + ".real")
    source.rename(target)
    source.symlink_to(target.name)
    with pytest.raises(ReleaseAssetError, match="symlink"):
        stage_release_assets(
            tmp_path,
            Path("symlinked"),
            inputs,
            version=_VERSION,
            doi=_DOI,
        )

    inputs = list(_release_project(tmp_path))
    inputs[0] = ReleaseAssetInput(
        role=inputs[0].role,
        source=Path("..") / inputs[0].filename,
        filename=inputs[0].filename,
    )
    with pytest.raises(ReleaseAssetError, match="safe relative path"):
        stage_release_assets(
            tmp_path,
            Path("escaping"),
            inputs,
            version=_VERSION,
            doi=_DOI,
        )


def test_stage_rejects_reused_or_symlinked_destination(tmp_path: Path) -> None:
    inputs = _release_project(tmp_path)
    (tmp_path / "existing").mkdir()
    with pytest.raises(ReleaseAssetError, match="new and absent"):
        stage_release_assets(
            tmp_path,
            Path("existing"),
            inputs,
            version=_VERSION,
            doi=_DOI,
        )

    (tmp_path / "real-parent").mkdir()
    (tmp_path / "linked-parent").symlink_to(tmp_path / "real-parent", target_is_directory=True)
    with pytest.raises(ReleaseAssetError, match="must not contain symlinks"):
        stage_release_assets(
            tmp_path,
            Path("linked-parent/stage"),
            inputs,
            version=_VERSION,
            doi=_DOI,
        )


def test_destination_claim_never_replaces_concurrently_created_empty_directory(
    tmp_path: Path,
) -> None:
    candidate = _destination_path(tmp_path, Path("stage"))
    candidate.mkdir()
    inode_before = candidate.stat().st_ino

    with pytest.raises(ReleaseAssetError, match="claimed concurrently"):
        _claim_new_destination(candidate)

    assert candidate.is_dir()
    assert candidate.stat().st_ino == inode_before
    assert list(candidate.iterdir()) == []


def test_stage_rejects_symlink_in_earlier_destination_ancestor(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    inputs = _release_project(project)
    outside = tmp_path / "outside"
    (outside / "deep").mkdir(parents=True)
    (project / "safe").mkdir()
    (project / "safe" / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ReleaseAssetError, match="must not contain symlinks"):
        stage_release_assets(
            project,
            Path("safe/linked/deep/stage"),
            inputs,
            version=_VERSION,
            doi=_DOI,
        )
    assert not (outside / "deep" / "stage").exists()


def test_stage_rejects_symlinked_project_root(tmp_path: Path) -> None:
    real_root = tmp_path / "real-project"
    real_root.mkdir()
    inputs = _release_project(real_root)
    linked_root = tmp_path / "linked-project"
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(ReleaseAssetError, match="project root path must not contain symlinks"):
        stage_release_assets(
            linked_root,
            Path("stage"),
            inputs,
            version=_VERSION,
            doi=_DOI,
        )
    assert not (real_root / "stage").exists()


def test_stage_preserves_safe_absolute_destination_policy(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    inputs = _release_project(project)
    outside = tmp_path / "outside"
    outside.mkdir()
    absolute_destination = outside / "stage"

    result = stage_release_assets(
        project,
        absolute_destination,
        inputs,
        version=_VERSION,
        doi=_DOI,
    )

    assert result.destination == absolute_destination.resolve()
    assert result.destination.is_dir()


@pytest.mark.parametrize(
    ("version", "doi", "match"),
    [
        ("1.1.0.dev0", _DOI, "final X.Y.Z"),
        (_VERSION, "(forthcoming)", "DOI is not assigned"),
    ],
)
def test_stage_requires_final_release_identity(
    tmp_path: Path, version: str, doi: str, match: str
) -> None:
    with pytest.raises(ReleaseAssetError, match=match):
        stage_release_assets(
            tmp_path,
            Path("stage"),
            (),
            version=version,
            doi=doi,
        )


def test_verify_github_release_downloads_binds_api_and_downloaded_bytes(
    tmp_path: Path,
) -> None:
    staged = _stage(tmp_path)
    downloaded = _download_copy(staged, tmp_path / "downloaded")
    api_assets = _api_assets(downloaded)

    result = verify_github_release_downloads(
        staged,
        downloaded,
        api_assets,
        version=_VERSION,
        doi=_DOI,
    )

    assert result.api_digest_verified is True
    assert result.staged_bytes_verified is True
    assert result.checksum_sha256 == _sha256(downloaded / CHECKSUM_FILENAME)
    assert {record.filename for record in result.assets} == {
        path.name for path in downloaded.iterdir()
    }
    assert next(record for record in result.assets if record.role == "checksums").sha256 == (
        result.checksum_sha256
    )
    json.dumps(result.as_dict(), sort_keys=True)


@pytest.mark.parametrize("symlinked_side", ["staged", "downloaded"])
def test_verify_rejects_earlier_symlink_component(
    tmp_path: Path, symlinked_side: str
) -> None:
    staged = _stage(tmp_path)
    downloaded = _download_copy(staged, tmp_path / "downloaded")
    api_assets = _api_assets(downloaded)
    view = tmp_path / "view"
    view.mkdir()
    (view / "linked-root").symlink_to(tmp_path, target_is_directory=True)
    staged_argument = staged
    downloaded_argument = downloaded
    if symlinked_side == "staged":
        staged_argument = view / "linked-root" / staged.name
    else:
        downloaded_argument = view / "linked-root" / downloaded.name

    with pytest.raises(ReleaseAssetError, match="path must not contain symlinks"):
        verify_github_release_downloads(
            staged_argument,
            downloaded_argument,
            api_assets,
            version=_VERSION,
            doi=_DOI,
        )


@pytest.mark.parametrize("tampered_name", ["manifest.json", CHECKSUM_FILENAME])
def test_verify_rejects_tampered_downloaded_bytes(tmp_path: Path, tampered_name: str) -> None:
    staged = _stage(tmp_path)
    downloaded = _download_copy(staged, tmp_path / "downloaded")
    api_assets = _api_assets(downloaded)
    (downloaded / tampered_name).write_bytes(b"tampered\n")

    with pytest.raises(ReleaseAssetError, match="differs from staged bytes"):
        verify_github_release_downloads(
            staged,
            downloaded,
            api_assets,
            version=_VERSION,
            doi=_DOI,
        )


def test_verify_rejects_github_api_digest_or_size_mismatch(tmp_path: Path) -> None:
    staged = _stage(tmp_path)
    downloaded = _download_copy(staged, tmp_path / "downloaded")
    api_assets = list(_api_assets(downloaded))
    first = api_assets[0]
    api_assets[0] = GitHubReleaseAsset(
        name=first.name,
        size=first.size,
        digest="sha256:" + "0" * 64,
    )
    with pytest.raises(ReleaseAssetError, match="API digest mismatch"):
        verify_github_release_downloads(
            staged,
            downloaded,
            api_assets,
            version=_VERSION,
            doi=_DOI,
        )

    api_assets = list(_api_assets(downloaded))
    first = api_assets[0]
    api_assets[0] = GitHubReleaseAsset(
        name=first.name,
        size=first.size + 1,
        digest=first.digest,
    )
    with pytest.raises(ReleaseAssetError, match="API size mismatch"):
        verify_github_release_downloads(
            staged,
            downloaded,
            api_assets,
            version=_VERSION,
            doi=_DOI,
        )


def test_verify_rejects_missing_duplicate_and_unexpected_assets(tmp_path: Path) -> None:
    staged = _stage(tmp_path)
    downloaded = _download_copy(staged, tmp_path / "downloaded")
    api_assets = list(_api_assets(downloaded))

    with pytest.raises(ReleaseAssetError, match="missing release assets"):
        verify_github_release_downloads(
            staged,
            downloaded,
            api_assets[:-1],
            version=_VERSION,
            doi=_DOI,
        )

    with pytest.raises(ReleaseAssetError, match="duplicate release asset"):
        verify_github_release_downloads(
            staged,
            downloaded,
            [*api_assets, api_assets[0]],
            version=_VERSION,
            doi=_DOI,
        )

    extra = downloaded / "unexpected.txt"
    extra.write_text("not approved\n", encoding="utf-8")
    with pytest.raises(ReleaseAssetError, match="unexpected assets"):
        verify_github_release_downloads(
            staged,
            downloaded,
            api_assets,
            version=_VERSION,
            doi=_DOI,
        )


def test_verify_rejects_tampered_staged_checksum_entry(tmp_path: Path) -> None:
    staged = _stage(tmp_path)
    downloaded = _download_copy(staged, tmp_path / "downloaded")
    checksum = staged / CHECKSUM_FILENAME
    lines = checksum.read_text(encoding="utf-8").splitlines()
    digest, filename = lines[0].split("  ", 1)
    lines[0] = f"{'0' * len(digest)}  {filename}"
    checksum.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ReleaseAssetError, match="differs from staged bytes|checksum-file mismatch"):
        verify_github_release_downloads(
            staged,
            downloaded,
            _api_assets(downloaded),
            version=_VERSION,
            doi=_DOI,
        )


def test_verify_rejects_non_utf8_checksum_as_typed_error(tmp_path: Path) -> None:
    staged = _stage(tmp_path)
    downloaded = _download_copy(staged, tmp_path / "downloaded")
    api_assets = _api_assets(downloaded)
    (staged / CHECKSUM_FILENAME).write_bytes(b"\xff\xfe\x00\x80")

    with pytest.raises(ReleaseAssetError, match="could not be read as UTF-8"):
        verify_github_release_downloads(
            staged,
            downloaded,
            api_assets,
            version=_VERSION,
            doi=_DOI,
        )


def test_github_api_records_and_saved_json_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ReleaseAssetError, match="invalid SHA-256"):
        GitHubReleaseAsset.from_api_dict(
            {"name": "manifest.json", "size": 1, "digest": "sha256:not-a-digest"}
        )
    with pytest.raises(ReleaseAssetError, match="invalid size"):
        GitHubReleaseAsset.from_api_dict(
            {"name": "manifest.json", "size": True, "digest": "sha256:" + "0" * 64}
        )

    payload_path = tmp_path / "assets.json"
    payload_path.write_text(
        json.dumps(
            {
                "assets": [
                    {
                        "name": "manifest.json",
                        "size": 1,
                        "digest": "sha256:" + "0" * 64,
                        "browser_download_url": "https://example.invalid/manifest.json",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    loaded = load_github_release_assets(payload_path)
    assert loaded == (
        GitHubReleaseAsset(
            name="manifest.json", size=1, digest="sha256:" + "0" * 64
        ),
    )

    payload_path.write_bytes(b"\xff\xfe\x00\x80")
    with pytest.raises(ReleaseAssetError, match="could not read GitHub release asset JSON"):
        load_github_release_assets(payload_path)
