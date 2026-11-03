"""Strict JSON, path admission and durable atomic artifact writes."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


class StageError(ValueError):
    """A staged run's identity, artifact or operation is inadmissible."""


def digest(data: bytes) -> str:
    """Hash exact bytes rather than a path or modification time."""
    return hashlib.sha256(data).hexdigest()


def encode(value: object) -> bytes:
    """Produce one finite canonical JSON representation."""
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def read_json(path: Path) -> dict[str, Any]:
    """Reject ambiguous JSON before interpreting a marker or report."""

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise StageError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value: str) -> None:
        raise StageError(f"nonfinite JSON constant: {value}")

    try:
        value = json.loads(path.read_text(), object_pairs_hook=pairs, parse_constant=constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StageError(f"unreadable staged JSON: {path.name}") from exc
    if not isinstance(value, dict):
        raise StageError(f"staged JSON must be an object: {path.name}")
    return value


def confined(root: Path, relative: str) -> Path:
    """Refuse noncanonical names, symlinks and escapes beneath an owned root."""
    if root.is_symlink() or root.resolve() != root.absolute():
        raise StageError("staged root must be canonical, without symlinks")
    name = Path(relative)
    if name.is_absolute() or name.as_posix() != relative or ".." in name.parts or not name.parts:
        raise StageError(f"unsafe staged path: {relative}")
    current = root
    for part in name.parts:
        current = current / part
        if current.is_symlink():
            raise StageError(f"symlink staged path: {relative}")
    if not current.resolve().is_relative_to(root.resolve()):
        raise StageError(f"escaping staged path: {relative}")
    return current


def atomic_bytes(path: Path, data: bytes) -> None:
    """Replace one validated artifact atomically; retain interrupted temporaries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(name, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
