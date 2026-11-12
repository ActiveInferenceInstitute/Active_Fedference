"""Private, descriptor-bound write-once publication (not crash recovery).

The hard link is the commit point. A later failure can lose acknowledgement,
but cannot roll back that committed name. SIGKILL requires an external reaper.
Directory descriptors retain authority even if a caller renames an ancestor;
identity checks refuse observed substitutions, without claiming an atomic
whole-ancestry check on every platform.
"""

from __future__ import annotations

import json
import os
import secrets
import signal
import stat
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path


@dataclass(eq=False)
class PathCustodyError(ValueError):
    """The selected pathname no longer denotes the admitted authority."""

    path: Path
    reason: str

    def __str__(self) -> str:
        return f"path custody refused for {self.path}: {self.reason}"


def serialize_json(payload: object) -> bytes:
    """Finish finite canonical serialization before allocating staging."""
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


@contextmanager
def _defer_python_signals() -> Iterator[None]:
    """Defer calling-thread Python signal handlers through resource registration.

    This uses the POSIX thread mask without installing process signal handlers.
    It does not promise protection from arbitrary asynchronous Python exceptions.
    """
    if not hasattr(signal, "pthread_sigmask"):
        raise OSError("descriptor ownership requires POSIX signal masking")
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, set())
    handled = {item for item in signal.valid_signals() if callable(signal.getsignal(item))}
    try:
        signal.pthread_sigmask(signal.SIG_BLOCK, handled)
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


@contextmanager
def _directory_fd(name: str, parent: int | None = None) -> Iterator[int]:
    descriptor = -1
    try:
        descriptor = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        yield descriptor
    finally:
        if descriptor >= 0:
            os.close(descriptor)


@dataclass(frozen=True)
class _DirectoryEdge:
    parent: int
    name: str
    device: int
    inode: int


@dataclass(frozen=True)
class DirectoryAuthority:
    """A lexical directory selection and its retained no-follow descriptors.

    Instances are valid only inside ``open_directory``. All child I/O uses the
    descriptor; ``path`` is a display locator, never a replacement authority.
    """

    path: Path
    descriptor: int
    edges: tuple[_DirectoryEdge, ...]

    def verify(self) -> None:
        """Refuse any observed renamed, replaced, or symlinked directory edge."""
        for edge in self.edges:
            try:
                current = os.stat(edge.name, dir_fd=edge.parent, follow_symlinks=False)
            except FileNotFoundError as exc:
                raise PathCustodyError(self.path, "directory ancestry disappeared") from exc
            if (current.st_dev, current.st_ino) != (edge.device, edge.inode) or not stat.S_ISDIR(
                current.st_mode
            ):
                raise PathCustodyError(self.path, "directory ancestry changed")

    def publish(self, name: str, data: bytes) -> Path:
        """Publish complete bytes once, or raise without replacing a winner."""
        _child_name(name)
        self.verify()
        temporary = f".{name}.{secrets.token_hex(16)}.tmp"
        # Arm exact-name cleanup before exclusive creation. Open's file object
        # is context-owned before its local binding, so a catchable interrupt
        # at that binding closes the stream as well as unlinking its stage.
        cleanup = True

        def create_stage(filename: str, flags: int) -> int:
            return os.open(filename, flags | os.O_NOFOLLOW, 0o600, dir_fd=self.descriptor)

        try:
            try:
                with ExitStack() as streams:
                    with _defer_python_signals():
                        opened = streams.enter_context(
                            open(temporary, "xb", buffering=0, opener=create_stage)
                        )
                    stream = opened
                    remaining = memoryview(data)
                    while remaining:
                        written = stream.write(remaining)
                        if written is None or written == 0:
                            raise OSError("staging write made no progress")
                        remaining = remaining[written:]
                    os.fsync(stream.fileno())
            except FileExistsError:
                cleanup = False
                raise
            self.verify()
            os.link(
                temporary, name, src_dir_fd=self.descriptor, dst_dir_fd=self.descriptor, follow_symlinks=False
            )
            os.fsync(self.descriptor)
        finally:
            if cleanup:
                try:
                    os.unlink(temporary, dir_fd=self.descriptor)
                except FileNotFoundError:
                    # Creation may have been interrupted before its syscall.
                    pass
        return self.path / name

    def read_bytes(self, name: str) -> bytes:
        """Read one regular child through the retained directory authority."""
        _child_name(name)
        self.verify()
        descriptor = -1
        try:
            with _defer_python_signals():
                descriptor = os.open(
                    name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self.descriptor
                )
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise PathCustodyError(self.path / name, "artifact is not a regular file")
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                data = stream.read()
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        self.verify()
        return data


def _child_name(name: str) -> None:
    if not name or name in (".", "..") or Path(name).name != name:
        raise PathCustodyError(Path(name), "expected one child name")


@contextmanager
def open_directory(path: str | Path) -> Iterator[DirectoryAuthority]:
    """Walk/create raw path components without following directory symlinks.

    POSIX dirfd, O_NOFOLLOW, hard-link and directory-fsync support are required;
    unsupported operations propagate explicitly and have no weaker fallback.
    """
    selected = Path(path).absolute()
    if ".." in selected.parts:
        raise PathCustodyError(selected, "parent traversal is not admitted")
    if not all(hasattr(os, flag) for flag in ("O_DIRECTORY", "O_NOFOLLOW")):
        raise PathCustodyError(selected, "no-follow directory operations are unsupported")
    with ExitStack() as stack:
        with _defer_python_signals():
            descriptor = stack.enter_context(_directory_fd(selected.anchor))
        edges: list[_DirectoryEdge] = []
        for name in selected.parts[1:]:
            parent = descriptor
            try:
                os.mkdir(name, dir_fd=parent)
            except FileExistsError:
                pass  # Admission below opens the existing component without following links.
            with _defer_python_signals():
                descriptor = stack.enter_context(_directory_fd(name, parent))
            identity = os.fstat(descriptor)
            edges.append(_DirectoryEdge(parent, name, identity.st_dev, identity.st_ino))
        authority = DirectoryAuthority(selected, descriptor, tuple(edges))
        authority.verify()
        yield authority


def publish_json(path: str | Path, payload: object) -> Path:
    """Serialize finite JSON first, then publish through pinned parent authority."""
    data = serialize_json(payload)
    destination = Path(path)
    _child_name(destination.name)
    with open_directory(destination.parent) as directory:
        directory.publish(destination.name, data)
    return destination
