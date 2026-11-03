"""Transaction files for source-preserving HTML publication."""
from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path

from publication.web.models import _HTMLDocument, _StagedHTMLWrite


def _stage_bytes(path: Path, payload: bytes, mode: int, purpose: str) -> Path:
    """Durably stage bytes beside their destination for an atomic replacement."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.xref-{purpose}-",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    descriptor_open = True
    try:
        os.fchmod(descriptor, stat.S_IMODE(mode))
        with os.fdopen(descriptor, "wb") as handle:
            descriptor_open = False
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if descriptor_open:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _document_is_unchanged(document: _HTMLDocument) -> bool:
    """Check identity and bytes immediately before replacing a captured file."""
    try:
        current = document.path.lstat()
    except OSError:
        return False
    expected = document.source_stat
    return (
        stat.S_ISREG(current.st_mode)
        and not stat.S_ISLNK(current.st_mode)
        and current.st_dev == expected.st_dev
        and current.st_ino == expected.st_ino
        and current.st_size == expected.st_size
        and current.st_mtime_ns == expected.st_mtime_ns
        and document.path.read_bytes() == document.original
    )


def _path_is_staged_replacement(item: _StagedHTMLWrite, path: Path) -> bool:
    """Check that ``path`` still holds the exact file installed by this batch."""
    try:
        current = path.lstat()
        payload = path.read_bytes()
    except OSError:
        return False
    expected = item.replacement_stat
    return (
        stat.S_ISREG(current.st_mode)
        and not stat.S_ISLNK(current.st_mode)
        and current.st_dev == expected.st_dev
        and current.st_ino == expected.st_ino
        and stat.S_IMODE(current.st_mode) == stat.S_IMODE(expected.st_mode)
        and current.st_size == expected.st_size
        and current.st_mtime_ns == expected.st_mtime_ns
        and payload == item.replacement_bytes
    )


def _path_is_captured_original(item: _StagedHTMLWrite, path: Path) -> bool:
    """Check that a commit capture still contains the caller's source file."""
    try:
        current = path.lstat()
        payload = path.read_bytes()
    except OSError:
        return False
    expected = item.document.source_stat
    return (
        stat.S_ISREG(current.st_mode)
        and not stat.S_ISLNK(current.st_mode)
        and current.st_dev == expected.st_dev
        and current.st_ino == expected.st_ino
        and stat.S_IMODE(current.st_mode) == stat.S_IMODE(expected.st_mode)
        and current.st_size == expected.st_size
        and current.st_mtime_ns == expected.st_mtime_ns
        and payload == item.document.original
    )


def _paths_share_regular_file(left: Path, right: Path) -> bool:
    """Return whether two paths are hard links to the same regular file."""
    try:
        left_stat = left.lstat()
        right_stat = right.lstat()
    except OSError:
        return False
    return (
        stat.S_ISREG(left_stat.st_mode)
        and stat.S_ISREG(right_stat.st_mode)
        and not stat.S_ISLNK(left_stat.st_mode)
        and not stat.S_ISLNK(right_stat.st_mode)
        and left_stat.st_dev == right_stat.st_dev
        and left_stat.st_ino == right_stat.st_ino
    )


def _link_no_clobber(source: Path, destination: Path) -> tuple[bool, BaseException | None]:
    """Link only into an absent destination and observe post-syscall interrupts."""
    try:
        os.link(source, destination, follow_symlinks=False)
    except BaseException as exc:
        return _paths_share_regular_file(source, destination), exc
    return True, None
