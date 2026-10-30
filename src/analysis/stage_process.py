"""External workspace lock ownership, cancellation and process observation."""

from __future__ import annotations

import fcntl
import os
import signal
import stat
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from analysis.stage_io import StageError, atomic_bytes, confined, encode, read_json


@contextmanager
def workspace_lock(workspace: Path) -> Iterator[int]:
    """Serialize this external workspace only; never acquire a collection lease."""
    path = confined(workspace, "writer.lock")
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise StageError("workspace lock must be a private regular file")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise StageError("workspace already has an active writer") from exc
        yield fd
    finally:
        os.close(fd)


def request_cancel(workspace: Path) -> Path:
    """Request cooperative cancellation of one named attempt, without signalling others."""
    active = read_json(confined(workspace, "active.json"))
    token = active.get("attempt")
    if active.get("status") != "running" or not isinstance(token, str):
        raise StageError("no running attempt is recorded")
    if len(token) != 32 or any(c not in "0123456789abcdef" for c in token):
        raise StageError("invalid attempt token")
    target = confined(workspace, f"cancel-{token}.json")
    atomic_bytes(target, encode({"attempt": token}))
    return target


def _stop_owned(process: subprocess.Popen[bytes]) -> None:
    """Reap the actual child; escalate only within its newly owned process group."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass  # The owned group exited between the poll and signal.
        process.wait()


def _process_start(pid: int) -> str | None:
    """Read a narrow OS identity; observation errors cannot mean an absent worker."""
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "lstart="], capture_output=True, text=True, timeout=2, check=False
    )
    if result.returncode == 1 and not result.stdout.strip():
        return None
    if result.returncode != 0 or not result.stdout.strip():
        raise StageError("cannot establish previous worker identity")
    return result.stdout.strip()


def _refuse_live_previous_worker(workspace: Path) -> None:
    """A supervisor crash must not permit a second producer beside its orphan."""
    path = confined(workspace, "active.json")
    if not path.exists():
        return
    previous = read_json(path)
    if previous.get("status") != "running":
        return
    pid = previous.get("pid")
    if type(pid) is not int or pid <= 0:
        raise StageError("invalid previous worker identity")
    actual = _process_start(pid)
    if actual is not None and (previous.get("process_start") is None or actual == previous["process_start"]):
        raise StageError("previous worker is still live; refuse concurrent resume")


class StageCancelled(BaseException):
    """A Python producer observed its owned supervisor's termination request."""


class StageTimedOut(StageCancelled):
    """The worker's own budget expired, independently of its supervisor."""


def inherit_workspace_lock(workspace: Path, descriptor: int) -> None:
    """Keep the same kernel lock alive across supervisor death, before recording a PID."""
    expected = confined(workspace, "writer.lock").stat()
    actual = os.fstat(descriptor)
    if (
        not stat.S_ISREG(actual.st_mode)
        or actual.st_nlink != 1
        or (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino)
    ):
        raise StageError("worker did not inherit its workspace lock")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise StageError("worker workspace lock is held by another writer") from exc


def writer_held(workspace: Path) -> bool:
    """Observe the kernel lock without creating files or trusting an active record."""
    path = confined(workspace, "writer.lock")
    if not path.exists():
        return False
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise StageError("workspace lock must be a private regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False
    finally:
        os.close(descriptor)
