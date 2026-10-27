"""Paths for source-preserving HTML publication."""
from __future__ import annotations

import posixpath
import stat
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _root(project_root: str | Path | None = None) -> Path:
    return Path(project_root) if project_root is not None else _PROJECT_ROOT


def _web_dir(project_root: Path) -> Path:
    return project_root / "output" / "web"


def _safe_html_paths(project_root: Path, web_dir: Path) -> tuple[tuple[Path, ...], tuple[str, ...]]:
    """Enumerate regular, non-symlinked HTML files confined to the project."""
    issues: list[str] = []
    if web_dir.is_symlink():
        return (), (f"{web_dir}: web output directory must not be a symlink",)
    try:
        project_resolved = project_root.resolve(strict=True)
        web_resolved = web_dir.resolve(strict=True)
        web_resolved.relative_to(project_resolved)
    except (OSError, ValueError) as exc:
        return (), (f"{web_dir}: web output directory is not confined to the project: {exc}",)

    paths: list[Path] = []
    try:
        candidates = sorted(web_dir.rglob("*.html"))
    except OSError as exc:
        return (), (f"{web_dir}: cannot enumerate HTML files: {exc}",)
    for path in candidates:
        try:
            relative = path.relative_to(web_dir)
            source_stat = path.lstat()
        except (OSError, ValueError) as exc:
            issues.append(f"{path}: cannot inspect HTML input: {exc}")
            continue
        if stat.S_ISLNK(source_stat.st_mode):
            issues.append(f"{path}: HTML input must not be a symlink")
            continue
        if not stat.S_ISREG(source_stat.st_mode):
            issues.append(f"{path}: HTML input must be a regular file")
            continue
        parent = web_dir
        parent_symlink = False
        for part in relative.parts[:-1]:
            parent /= part
            if parent.is_symlink():
                issues.append(f"{path}: HTML input has a symlinked parent directory")
                parent_symlink = True
                break
        if parent_symlink:
            continue
        try:
            path.resolve(strict=True).relative_to(web_resolved)
        except (OSError, ValueError):
            issues.append(f"{path}: HTML input resolves outside the web output directory")
            continue
        paths.append(path)
    return tuple(paths), tuple(sorted(set(issues)))


def _xref_target_prefix(web_dir: Path, html_path: Path) -> str:
    """Return the correct POSIX path from one page to the combined index."""
    root_index = web_dir / "index.html"
    if html_path == root_index:
        return ""
    relative_parent = html_path.relative_to(web_dir).parent.as_posix()
    return posixpath.relpath("index.html", start=relative_parent or ".")
