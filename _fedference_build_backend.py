"""PEP 517 backend wrapper for reproducible Active Fedference distributions.

Setuptools already honors ``SOURCE_DATE_EPOCH`` for wheel member timestamps,
but its source distributions retain checkout and build-time metadata. This
wrapper delegates all build semantics to the exactly pinned setuptools backend
and normalizes only archive metadata after a successful build. File contents
are never rewritten.
"""

from __future__ import annotations

import copy
import gzip
import os
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from setuptools import build_meta as _setuptools

# Editable and metadata hooks retain setuptools semantics.
build_editable = _setuptools.build_editable
get_requires_for_build_editable = _setuptools.get_requires_for_build_editable
get_requires_for_build_sdist = _setuptools.get_requires_for_build_sdist
get_requires_for_build_wheel = _setuptools.get_requires_for_build_wheel
prepare_metadata_for_build_editable = _setuptools.prepare_metadata_for_build_editable
prepare_metadata_for_build_wheel = _setuptools.prepare_metadata_for_build_wheel

_ZIP_EPOCH_FLOOR = 315532800  # 1980-01-01T00:00:00Z, the ZIP format floor.


def _source_date_epoch(value: str | None = None) -> int | None:
    """Return a validated reproducible-build epoch, or ``None`` when unset."""
    raw = os.environ.get("SOURCE_DATE_EPOCH") if value is None else value
    if raw is None:
        return None
    normalized = raw.strip()
    if not normalized or not normalized.isascii() or not normalized.isdecimal():
        raise ValueError("SOURCE_DATE_EPOCH must be a nonnegative integer")
    if len(normalized) > 20:
        raise ValueError("SOURCE_DATE_EPOCH is outside the supported range")
    epoch = int(normalized)
    if epoch > 253402300799:  # 9999-12-31T23:59:59Z
        raise ValueError("SOURCE_DATE_EPOCH is outside the supported range")
    return epoch


def _temporary_archive_path(archive: Path) -> Path:
    """Allocate a sibling temporary path for an atomic archive replacement."""
    with tempfile.NamedTemporaryFile(
        prefix=f".{archive.name}.",
        suffix=".tmp",
        dir=archive.parent,
        delete=False,
    ) as handle:
        return Path(handle.name)


def _normalize_sdist_archive(archive: Path, epoch: int) -> None:
    """Normalize tar, gzip, owner, and timestamp metadata in one sdist."""
    temporary = _temporary_archive_path(archive)
    try:
        with tarfile.open(archive, mode="r:gz") as source:
            members = source.getmembers()
            with temporary.open("wb") as raw_output:
                with gzip.GzipFile(
                    filename="",
                    mode="wb",
                    fileobj=raw_output,
                    compresslevel=9,
                    mtime=epoch,
                ) as compressed:
                    with tarfile.open(
                        fileobj=compressed,
                        mode="w|",
                        format=tarfile.PAX_FORMAT,
                    ) as destination:
                        for member in sorted(members, key=lambda item: item.name):
                            normalized = copy.copy(member)
                            normalized.uid = 0
                            normalized.gid = 0
                            normalized.uname = ""
                            normalized.gname = ""
                            normalized.mtime = epoch
                            normalized.pax_headers = {}
                            if member.isfile():
                                extracted = source.extractfile(member)
                                if extracted is None:
                                    raise ValueError(
                                        f"sdist member could not be read: {member.name}"
                                    )
                                with extracted:
                                    destination.addfile(normalized, extracted)
                            else:
                                destination.addfile(normalized)
        os.replace(temporary, archive)
    finally:
        temporary.unlink(missing_ok=True)


def _zip_datetime(epoch: int) -> tuple[int, int, int, int, int, int]:
    """Convert an epoch into a UTC ZIP timestamp at the format's lower bound."""
    import time

    bounded = max(epoch, _ZIP_EPOCH_FLOOR)
    value = time.gmtime(bounded)
    return (
        value.tm_year,
        value.tm_mon,
        value.tm_mday,
        value.tm_hour,
        value.tm_min,
        value.tm_sec,
    )


def _normalize_wheel_archive(archive: Path, epoch: int) -> None:
    """Normalize member order and ZIP metadata without changing wheel payloads."""
    temporary = _temporary_archive_path(archive)
    try:
        with zipfile.ZipFile(archive, mode="r") as source:
            members = [(item, source.read(item.filename)) for item in source.infolist()]
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as destination:
            for original, payload in sorted(members, key=lambda item: item[0].filename):
                normalized = zipfile.ZipInfo(
                    filename=original.filename,
                    date_time=_zip_datetime(epoch),
                )
                normalized.compress_type = zipfile.ZIP_DEFLATED
                normalized.create_system = original.create_system
                normalized.create_version = original.create_version
                normalized.extract_version = original.extract_version
                normalized.external_attr = original.external_attr
                normalized.internal_attr = original.internal_attr
                normalized.flag_bits = original.flag_bits
                normalized.comment = original.comment
                destination.writestr(
                    normalized,
                    payload,
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
        os.replace(temporary, archive)
    finally:
        temporary.unlink(missing_ok=True)


def build_sdist(
    sdist_directory: str,
    config_settings: dict[str, Any] | None = None,
) -> str:
    """Build with setuptools and normalize archive metadata when time is pinned."""
    filename = _setuptools.build_sdist(sdist_directory, config_settings)
    epoch = _source_date_epoch()
    if epoch is not None:
        _normalize_sdist_archive(Path(sdist_directory) / filename, epoch)
    return filename


def build_wheel(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    """Build with setuptools and normalize wheel archive metadata when pinned."""
    filename = _setuptools.build_wheel(
        wheel_directory,
        config_settings,
        metadata_directory,
    )
    epoch = _source_date_epoch()
    if epoch is not None:
        _normalize_wheel_archive(Path(wheel_directory) / filename, epoch)
    return filename
