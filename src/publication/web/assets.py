"""Assets for source-preserving HTML publication."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from publication.web.paths import _root, _web_dir

_PUBLICATION_TEXT_SUFFIXES = frozenset(
    {".csv", ".html", ".json", ".jsonl", ".log", ".md", ".svg", ".tex", ".txt", ".yaml", ".yml"}
)


_MACHINE_PATH_RE = re.compile(r"(?P<prefix>/private/tmp|/tmp|/Users|/home|/Volumes)/[^/\s\"'<>]+")


def sanitize_machine_paths(project_root: str | Path | None = None) -> tuple[Path, ...]:
    """Replace local home, temporary, and volume prefixes in text artifacts.

    TeX and renderer logs can contain absolute paths even though the published
    PDF/HTML does not depend on them.  Sanitising the committed text surfaces
    keeps reviewer snapshots clone-independent and prevents local workspace
    names from leaking into a release.  Binary figures and PDFs are untouched.
    """
    root = _root(project_root)
    output_dir = root / "output"
    if not output_dir.is_dir() or output_dir.is_symlink():
        return ()
    replacements = {
        "/private/tmp": "<tmp>",
        "/tmp": "<tmp>",
        "/Users": "<home>",
        "/home": "<home>",
        "/Volumes": "<volume>",
    }
    changed: list[Path] = []
    for path in sorted(output_dir.rglob("*")):
        if (
            path.is_symlink()
            or not path.is_file()
            or path.suffix.casefold() not in _PUBLICATION_TEXT_SUFFIXES
        ):
            continue
        try:
            original = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        sanitized = _MACHINE_PATH_RE.sub(lambda match: replacements[match.group("prefix")], original)
        if path.suffix.casefold() == ".log":
            sanitized = sanitized.rstrip("\n") + "\n"
        if sanitized == original:
            continue
        path.write_text(sanitized, encoding="utf-8")
        changed.append(path)
    return tuple(changed)


def mirror_web_figures(project_root: str | Path | None = None) -> tuple[Path, ...]:
    """Mirror every generated figure into the web package, removing stale files."""
    root = _root(project_root)
    source_dir = root / "output" / "figures"
    destination_dir = _web_dir(root) / "figures"
    if not source_dir.exists():
        raise FileNotFoundError(f"missing source figures directory: {source_dir}")

    destination_dir.mkdir(parents=True, exist_ok=True)
    source_files = {
        path.relative_to(source_dir): path for path in sorted(source_dir.rglob("*")) if path.is_file()
    }

    for existing in sorted(destination_dir.rglob("*"), reverse=True):
        if existing.is_file() and existing.relative_to(destination_dir) not in source_files:
            existing.unlink()
        elif existing.is_dir():
            try:
                existing.rmdir()
            except OSError:
                pass

    copied: list[Path] = []
    for relative_path, source_path in source_files.items():
        destination_path = destination_dir / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
        copied.append(destination_path)
    return tuple(copied)
