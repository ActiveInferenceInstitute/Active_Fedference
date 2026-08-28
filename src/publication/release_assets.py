"""Deterministic staging and post-download verification of release assets.

The repository's :mod:`publication.release_manifest` module binds the
reviewer snapshot.  This module owns the narrower, external publication
boundary: copying the four explicitly approved release inputs into a new
directory and producing the checksum file uploaded alongside them.

The destination name is claimed atomically with an exclusive directory
creation, so a concurrently created directory is never replaced.  Population
then occurs file by file with exclusive creation.  Consequently this boundary
does not claim that the complete directory becomes visible atomically; callers
must consume it only after :func:`stage_release_assets` returns successfully.

``SHA256SUMS.txt`` intentionally lists only the PDF, wheel, source
distribution, and manifest.  A checksum file cannot include its own digest.
After GitHub publication, :func:`verify_github_release_downloads` instead
binds that fifth file to the digest reported by the GitHub release-asset API
and to an exact downloaded-byte comparison with the staged copy.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from publication.identifiers import manuscript_pdf_filename, normalize_doi

CHECKSUM_FILENAME = "SHA256SUMS.txt"
RELEASE_ASSET_ROLES: tuple[str, ...] = ("manifest", "pdf", "sdist", "wheel")

_FINAL_VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ReleaseAssetError(ValueError):
    """Raised when release staging or downloaded verification fails closed."""


@dataclass(frozen=True)
class ReleaseAssetInput:
    """One explicitly named, project-root-relative release input."""

    role: str
    source: Path
    filename: str


@dataclass(frozen=True)
class ReleaseAssetRecord:
    """Byte identity of one staged or downloaded release asset."""

    role: str
    filename: str
    size: int
    sha256: str

    def as_dict(self) -> dict[str, str | int]:
        """Return a stable JSON-compatible record."""
        return {
            "role": self.role,
            "filename": self.filename,
            "size": self.size,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class StagedReleaseAssets:
    """Canonical external asset set written to one new directory."""

    destination: Path
    assets: tuple[ReleaseAssetRecord, ...]
    checksum_filename: str
    checksum_size: int
    checksum_sha256: str

    @property
    def filenames(self) -> tuple[str, ...]:
        """Return the four non-checksum asset names in deterministic order."""
        return tuple(record.filename for record in self.assets)

    def as_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible staging summary."""
        return {
            "destination": str(self.destination),
            "assets": [record.as_dict() for record in self.assets],
            "checksum": {
                "filename": self.checksum_filename,
                "size": self.checksum_size,
                "sha256": self.checksum_sha256,
            },
        }


@dataclass(frozen=True)
class GitHubReleaseAsset:
    """The integrity fields exposed by one GitHub release-asset API record."""

    name: str
    size: int
    digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _is_safe_filename(self.name):
            raise ReleaseAssetError("GitHub asset record has an unsafe name")
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 0:
            raise ReleaseAssetError(f"GitHub asset {self.name!r} has an invalid size")
        if not isinstance(self.digest, str) or not self.digest.startswith("sha256:"):
            raise ReleaseAssetError(f"GitHub asset {self.name!r} has no SHA-256 API digest")
        raw_digest = self.digest.removeprefix("sha256:")
        if not _SHA256_RE.fullmatch(raw_digest):
            raise ReleaseAssetError(
                f"GitHub asset {self.name!r} has an invalid SHA-256 API digest"
            )

    @classmethod
    def from_api_dict(cls, payload: Mapping[str, object]) -> GitHubReleaseAsset:
        """Parse the required fields from a GitHub release-asset response.

        GitHub API responses contain additional transport and URL fields.  They
        are intentionally ignored here; only the name, byte count, and API
        digest participate in the publication-integrity boundary.
        """
        try:
            name = payload["name"]
            size = payload["size"]
            digest = payload["digest"]
        except KeyError as exc:
            raise ReleaseAssetError("GitHub asset record is missing name, size, or digest") from exc
        if not isinstance(name, str):
            raise ReleaseAssetError("GitHub asset record has an invalid name field type")
        if isinstance(size, bool) or not isinstance(size, int):
            raise ReleaseAssetError(f"GitHub asset {name!r} has an invalid size")
        if not isinstance(digest, str):
            raise ReleaseAssetError("GitHub asset record has an invalid digest field type")
        return cls(name=name, size=size, digest=digest)

    @property
    def sha256(self) -> str:
        """Return the bare SHA-256 value from GitHub's algorithm-prefixed digest."""
        return self.digest.removeprefix("sha256:")


@dataclass(frozen=True)
class GitHubReleaseVerification:
    """Verified downloaded bytes for one exact GitHub release asset set."""

    assets: tuple[ReleaseAssetRecord, ...]
    checksum_sha256: str
    api_digest_verified: bool
    staged_bytes_verified: bool

    def as_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible verification summary."""
        return {
            "assets": [record.as_dict() for record in self.assets],
            "checksum_sha256": self.checksum_sha256,
            "api_digest_verified": self.api_digest_verified,
            "staged_bytes_verified": self.staged_bytes_verified,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_safe_filename(value: str) -> bool:
    return (
        bool(value)
        and value not in {".", ".."}
        and Path(value).name == value
        and not any(character in value for character in "\x00\r\n")
        and len(value.encode("utf-8")) <= 255
    )


def _final_version(value: object) -> str:
    version = str(value)
    if not _FINAL_VERSION_RE.fullmatch(version):
        raise ReleaseAssetError("external release staging requires a final X.Y.Z version")
    return version


def expected_release_asset_names(version: object, doi: object) -> dict[str, str]:
    """Return the exact four filenames accepted by the staging boundary."""
    final_version = _final_version(version)
    try:
        normalized_doi = normalize_doi(doi)
    except ValueError as exc:
        raise ReleaseAssetError(str(exc)) from exc
    assert normalized_doi is not None
    names = {
        "manifest": "manifest.json",
        "pdf": manuscript_pdf_filename(final_version, normalized_doi),
        "sdist": f"active_fedference-{final_version}.tar.gz",
        "wheel": f"active_fedference-{final_version}-py3-none-any.whl",
    }
    if len(set(names.values())) != len(names):
        raise ReleaseAssetError("canonical release asset names are not unique")
    return names


def _relative_source(value: Path) -> Path:
    if not isinstance(value, Path):
        raise ReleaseAssetError("release asset source must be a pathlib.Path")
    if value.is_absolute() or not value.parts or any(part in {"", ".", ".."} for part in value.parts):
        raise ReleaseAssetError(f"release asset source must be a safe relative path: {value}")
    return value


def _absolute_lexical_path(value: Path, *, label: str) -> Path:
    """Return an absolute path without resolving away symlink evidence."""
    if any(part == ".." for part in value.parts):
        raise ReleaseAssetError(f"{label} must not contain parent traversal")
    return value if value.is_absolute() else Path.cwd() / value


def _ensure_no_existing_ancestor_symlinks(path: Path, *, label: str) -> None:
    """Reject a symlink at any existing component of an absolute path."""
    if not path.is_absolute():
        raise ReleaseAssetError(f"{label} path validation requires an absolute path")
    current = Path(path.anchor)
    components = path.parts[1:] if path.anchor else path.parts
    candidates = [current]
    for component in components:
        current = current / component
        candidates.append(current)
    for candidate in candidates:
        try:
            mode = os.lstat(candidate).st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ReleaseAssetError(f"{label} path must not contain symlinks: {candidate}")


def _project_root_path(value: Path) -> Path:
    if not isinstance(value, Path):
        raise ReleaseAssetError("project root must be a pathlib.Path")
    absolute = _absolute_lexical_path(value, label="project root")
    _ensure_no_existing_ancestor_symlinks(absolute, label="project root")
    try:
        mode = os.lstat(absolute).st_mode
    except FileNotFoundError as exc:
        raise ReleaseAssetError("project root must be an existing non-symlink directory") from exc
    if not stat.S_ISDIR(mode):
        raise ReleaseAssetError("project root must be an existing non-symlink directory")
    return absolute.resolve(strict=True)


def _ensure_no_symlink_chain(root: Path, relative: Path) -> None:
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ReleaseAssetError(f"release asset path must not contain symlinks: {relative}")


def _regular_input(root: Path, relative: Path) -> Path:
    _ensure_no_symlink_chain(root, relative)
    path = root / relative
    try:
        mode = path.stat(follow_symlinks=False).st_mode
    except FileNotFoundError as exc:
        raise ReleaseAssetError(f"release asset does not exist: {relative}") from exc
    if not stat.S_ISREG(mode):
        raise ReleaseAssetError(f"release asset is not a regular file: {relative}")
    try:
        path.resolve(strict=True).relative_to(root)
    except ValueError as exc:
        raise ReleaseAssetError(f"release asset escapes the project root: {relative}") from exc
    return path


def _destination_path(project_root: Path, destination: Path) -> Path:
    if not isinstance(destination, Path):
        raise ReleaseAssetError("release destination must be a pathlib.Path")
    path = _absolute_lexical_path(
        destination if destination.is_absolute() else project_root / destination,
        label="release destination",
    )
    _ensure_no_existing_ancestor_symlinks(path, label="release destination")
    parent = path.parent
    try:
        parent_mode = os.lstat(parent).st_mode
    except FileNotFoundError as exc:
        raise ReleaseAssetError(
            "release destination parent must be an existing non-symlink directory"
        ) from exc
    if not stat.S_ISDIR(parent_mode):
        raise ReleaseAssetError("release destination parent must be an existing non-symlink directory")
    if os.path.lexists(path):
        raise ReleaseAssetError("release destination must be new and absent")
    return path


def _claim_new_destination(path: Path) -> None:
    """Atomically claim an absent destination without replacing any entry."""
    _ensure_no_existing_ancestor_symlinks(path, label="release destination")
    try:
        path.mkdir(mode=0o755, exist_ok=False)
    except FileExistsError as exc:
        raise ReleaseAssetError(
            "release destination was claimed concurrently and must not be reused"
        ) from exc
    except OSError as exc:
        raise ReleaseAssetError(f"release destination could not be claimed: {path}") from exc


def _validated_inputs(
    project_root: Path,
    inputs: Iterable[ReleaseAssetInput],
    *,
    version: object,
    doi: object,
) -> tuple[tuple[str, Path, str], ...]:
    expected_names = expected_release_asset_names(version, doi)
    items = tuple(inputs)
    if len(items) != len(RELEASE_ASSET_ROLES):
        raise ReleaseAssetError("release staging requires exactly four explicitly named inputs")

    roles: set[str] = set()
    filenames: set[str] = set()
    sources: set[Path] = set()
    validated: list[tuple[str, Path, str]] = []
    for item in items:
        if not isinstance(item, ReleaseAssetInput):
            raise ReleaseAssetError("release inputs must be ReleaseAssetInput values")
        if item.role not in RELEASE_ASSET_ROLES:
            raise ReleaseAssetError(f"unexpected release asset role: {item.role!r}")
        if item.role in roles:
            raise ReleaseAssetError(f"duplicate release asset role: {item.role}")
        roles.add(item.role)
        if not _is_safe_filename(item.filename):
            raise ReleaseAssetError(f"unsafe release asset filename: {item.filename!r}")
        if item.filename in filenames:
            raise ReleaseAssetError(f"duplicate release asset filename: {item.filename}")
        filenames.add(item.filename)
        expected_name = expected_names[item.role]
        if item.filename != expected_name:
            raise ReleaseAssetError(
                f"release asset {item.role!r} must be named {expected_name!r}, got {item.filename!r}"
            )
        relative = _relative_source(item.source)
        source = _regular_input(project_root, relative)
        if source.name != item.filename:
            raise ReleaseAssetError(
                f"release asset source name does not match declared filename: {relative}"
            )
        resolved_source = source.resolve(strict=True)
        if resolved_source in sources:
            raise ReleaseAssetError(f"duplicate release asset source: {relative}")
        sources.add(resolved_source)
        validated.append((item.role, source, item.filename))

    missing = set(RELEASE_ASSET_ROLES) - roles
    if missing:
        raise ReleaseAssetError("missing release asset roles: " + ", ".join(sorted(missing)))
    return tuple(sorted(validated, key=lambda item: item[2]))


def _copy_bytes(source: Path, destination: Path) -> None:
    created = False
    try:
        with source.open("rb") as input_handle:
            with destination.open("xb") as output_handle:
                created = True
                shutil.copyfileobj(input_handle, output_handle, length=1 << 20)
    except Exception:
        if created:
            destination.unlink(missing_ok=True)
        raise


def _files_equal(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as left_handle, right.open("rb") as right_handle:
        while True:
            left_chunk = left_handle.read(1 << 20)
            right_chunk = right_handle.read(1 << 20)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def _stage_summary(destination: Path, role_by_name: Mapping[str, str]) -> StagedReleaseAssets:
    checksum_path = destination / CHECKSUM_FILENAME
    assets = tuple(
        ReleaseAssetRecord(
            role=role_by_name[path.name],
            filename=path.name,
            size=path.stat().st_size,
            sha256=_sha256(path),
        )
        for path in sorted(destination.iterdir(), key=lambda value: value.name)
        if path.name != CHECKSUM_FILENAME
    )
    return StagedReleaseAssets(
        destination=destination.resolve(),
        assets=assets,
        checksum_filename=CHECKSUM_FILENAME,
        checksum_size=checksum_path.stat().st_size,
        checksum_sha256=_sha256(checksum_path),
    )


def stage_release_assets(
    project_root: Path,
    destination: Path,
    inputs: Iterable[ReleaseAssetInput],
    *,
    version: object,
    doi: object,
) -> StagedReleaseAssets:
    """Stage the exact external release set into a new destination.

    All inputs, identity, path containment, and destination safety are checked
    before the destination is created.  The caller must provide exactly one
    explicitly named manifest, PDF, source distribution, and wheel.  No file
    discovery or glob selection occurs at this boundary.  The destination name
    is claimed without clobbering a concurrent entry; complete directory
    visibility is not atomic, so consumers must wait for a successful return.
    """
    root = _project_root_path(project_root)
    validated = _validated_inputs(root, inputs, version=version, doi=doi)
    output = _destination_path(root, destination)
    role_by_name = {filename: role for role, _, filename in validated}

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    claimed = False
    populated: list[Path] = []
    try:
        for _, source, filename in validated:
            _copy_bytes(source, temporary / filename)
        records = [
            (filename, _sha256(temporary / filename))
            for filename in sorted(role_by_name)
        ]
        (temporary / CHECKSUM_FILENAME).write_text(
            "".join(f"{digest}  {filename}\n" for filename, digest in records),
            encoding="utf-8",
        )
        _claim_new_destination(output)
        claimed = True
        for prepared in sorted(temporary.iterdir(), key=lambda path: path.name):
            target = output / prepared.name
            try:
                _copy_bytes(prepared, target)
            except FileExistsError as exc:
                raise ReleaseAssetError(
                    f"release destination changed during population: {prepared.name}"
                ) from exc
            populated.append(target)
        _directory_files(
            output,
            set(role_by_name) | {CHECKSUM_FILENAME},
            label="staged release directory",
        )
        return _stage_summary(output, role_by_name)
    except Exception:
        if claimed:
            for path in reversed(populated):
                path.unlink(missing_ok=True)
            try:
                output.rmdir()
            except OSError:
                # Never recursively remove a directory that another process
                # populated after the exclusive claim. Only our named files
                # are eligible for cleanup.
                pass
        raise
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _directory_files(directory: Path, expected: set[str], *, label: str) -> dict[str, Path]:
    if not isinstance(directory, Path):
        raise ReleaseAssetError(f"{label} must be a pathlib.Path")
    absolute = _absolute_lexical_path(directory, label=label)
    _ensure_no_existing_ancestor_symlinks(absolute, label=label)
    try:
        directory_mode = os.lstat(absolute).st_mode
    except FileNotFoundError as exc:
        raise ReleaseAssetError(f"{label} must be an existing non-symlink directory") from exc
    if not stat.S_ISDIR(directory_mode):
        raise ReleaseAssetError(f"{label} must be an existing non-symlink directory")
    files: dict[str, Path] = {}
    for path in absolute.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ReleaseAssetError(f"{label} contains a non-regular entry: {path.name}")
        if not _is_safe_filename(path.name):
            raise ReleaseAssetError(f"{label} contains an unsafe filename: {path.name!r}")
        files[path.name] = path
    missing = expected - set(files)
    unexpected = set(files) - expected
    if missing:
        raise ReleaseAssetError(f"{label} is missing assets: " + ", ".join(sorted(missing)))
    if unexpected:
        raise ReleaseAssetError(f"{label} contains unexpected assets: " + ", ".join(sorted(unexpected)))
    return files


def _parse_checksums(path: Path, expected_names: set[str]) -> dict[str, str]:
    entries: dict[str, str] = {}
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ReleaseAssetError(f"{CHECKSUM_FILENAME} could not be read as UTF-8") from exc
    lines = content.splitlines(keepends=True)
    if not lines or any(not line.endswith("\n") for line in lines):
        raise ReleaseAssetError(f"{CHECKSUM_FILENAME} must be non-empty and newline terminated")
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\x00\r\n]+)\n", line)
        if match is None:
            raise ReleaseAssetError(f"{CHECKSUM_FILENAME} contains a malformed line")
        digest, filename = match.groups()
        if not _is_safe_filename(filename):
            raise ReleaseAssetError(f"{CHECKSUM_FILENAME} contains an unsafe filename")
        if filename in entries:
            raise ReleaseAssetError(f"{CHECKSUM_FILENAME} contains a duplicate filename: {filename}")
        entries[filename] = digest
    if list(entries) != sorted(entries):
        raise ReleaseAssetError(f"{CHECKSUM_FILENAME} entries are not sorted")
    if set(entries) != expected_names:
        raise ReleaseAssetError(
            f"{CHECKSUM_FILENAME} must cover exactly the four non-checksum release assets"
        )
    if CHECKSUM_FILENAME in entries:
        raise ReleaseAssetError(f"{CHECKSUM_FILENAME} must not contain a self-reference")
    return entries


def _api_asset_map(
    assets: Iterable[GitHubReleaseAsset | Mapping[str, object]],
    expected_names: set[str],
) -> dict[str, GitHubReleaseAsset]:
    result: dict[str, GitHubReleaseAsset] = {}
    for raw in assets:
        asset = raw if isinstance(raw, GitHubReleaseAsset) else GitHubReleaseAsset.from_api_dict(raw)
        if asset.name in result:
            raise ReleaseAssetError(f"GitHub API contains a duplicate release asset: {asset.name}")
        result[asset.name] = asset
    missing = expected_names - set(result)
    unexpected = set(result) - expected_names
    if missing:
        raise ReleaseAssetError("GitHub API is missing release assets: " + ", ".join(sorted(missing)))
    if unexpected:
        raise ReleaseAssetError(
            "GitHub API contains unexpected release assets: " + ", ".join(sorted(unexpected))
        )
    return result


def verify_github_release_downloads(
    staged_directory: Path,
    downloaded_directory: Path,
    github_assets: Iterable[GitHubReleaseAsset | Mapping[str, object]],
    *,
    version: object,
    doi: object,
) -> GitHubReleaseVerification:
    """Verify GitHub API digests and downloaded bytes against the staged set.

    The GitHub API must report exactly the four release assets plus
    ``SHA256SUMS.txt``.  The downloaded directory must contain that same exact
    set.  Every downloaded file is byte-identical to its staged source and its
    API SHA-256 digest; the four non-checksum downloads additionally match the
    sorted checksum file.  This is how the checksum file itself is verified
    without an impossible self-reference.
    """
    expected_non_checksum = set(expected_release_asset_names(version, doi).values())
    expected_all = expected_non_checksum | {CHECKSUM_FILENAME}
    staged = _directory_files(Path(staged_directory), expected_all, label="staged release directory")
    downloaded = _directory_files(
        Path(downloaded_directory), expected_all, label="downloaded release directory"
    )
    api = _api_asset_map(github_assets, expected_all)
    checksums = _parse_checksums(staged[CHECKSUM_FILENAME], expected_non_checksum)

    records: list[ReleaseAssetRecord] = []
    role_by_name = {
        name: role for role, name in expected_release_asset_names(version, doi).items()
    }
    role_by_name[CHECKSUM_FILENAME] = "checksums"
    for filename in sorted(expected_all):
        staged_path = staged[filename]
        downloaded_path = downloaded[filename]
        downloaded_digest = _sha256(downloaded_path)
        if not _files_equal(staged_path, downloaded_path):
            raise ReleaseAssetError(f"downloaded release asset differs from staged bytes: {filename}")
        api_asset = api[filename]
        if downloaded_path.stat().st_size != api_asset.size:
            raise ReleaseAssetError(f"GitHub API size mismatch for release asset: {filename}")
        if downloaded_digest != api_asset.sha256:
            raise ReleaseAssetError(f"GitHub API digest mismatch for release asset: {filename}")
        if filename != CHECKSUM_FILENAME and downloaded_digest != checksums[filename]:
            raise ReleaseAssetError(f"checksum-file mismatch for release asset: {filename}")
        records.append(
            ReleaseAssetRecord(
                role=role_by_name[filename],
                filename=filename,
                size=downloaded_path.stat().st_size,
                sha256=downloaded_digest,
            )
        )

    return GitHubReleaseVerification(
        assets=tuple(records),
        checksum_sha256=_sha256(downloaded[CHECKSUM_FILENAME]),
        api_digest_verified=True,
        staged_bytes_verified=True,
    )


def load_github_release_assets(path: Path) -> tuple[GitHubReleaseAsset, ...]:
    """Load release assets from a saved GitHub API response without network I/O."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseAssetError(f"could not read GitHub release asset JSON: {path}") from exc
    if isinstance(payload, Mapping):
        payload = payload.get("assets")
    if not isinstance(payload, list):
        raise ReleaseAssetError("GitHub release asset JSON must be a list or contain an assets list")
    if any(not isinstance(item, Mapping) for item in payload):
        raise ReleaseAssetError("GitHub release asset JSON contains a malformed asset record")
    return tuple(GitHubReleaseAsset.from_api_dict(item) for item in payload)


__all__ = [
    "CHECKSUM_FILENAME",
    "GitHubReleaseAsset",
    "GitHubReleaseVerification",
    "RELEASE_ASSET_ROLES",
    "ReleaseAssetError",
    "ReleaseAssetInput",
    "ReleaseAssetRecord",
    "StagedReleaseAssets",
    "expected_release_asset_names",
    "load_github_release_assets",
    "stage_release_assets",
    "verify_github_release_downloads",
]
