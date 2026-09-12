"""Persist and render manuscript {{TOKEN}} substitutions."""

from __future__ import annotations

import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

import yaml

from .loaders import _EXCLUDED_MANUSCRIPT_DOCS

_TOKEN_RE = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")


def _resolve_config_value(value: Any, variables: dict[str, str]) -> Any:
    """Substitute YAML string values without interpreting replacement syntax."""
    if isinstance(value, str):
        return _TOKEN_RE.sub(lambda match: variables.get(match.group(1), match.group(0)), value)
    if isinstance(value, list):
        return [_resolve_config_value(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_config_value(item, variables) for key, item in value.items()}
    return value


def _hydrate_config(source: Path, destination: Path, variables: dict[str, str]) -> None:
    """Hydrate a token-bearing configuration, rejecting unresolved placeholders."""
    text = source.read_text(encoding="utf-8")
    if not _TOKEN_RE.search(text):
        shutil.copy2(source, destination)
        return
    config = yaml.safe_load(text)
    if not isinstance(config, dict):
        raise ValueError("manuscript config.yaml must contain a mapping")
    resolved = yaml.safe_dump(
        _resolve_config_value(config, variables), sort_keys=False, allow_unicode=True,
    )
    unresolved = sorted(set(_TOKEN_RE.findall(resolved)))
    if unresolved:
        raise ValueError(f"unresolved manuscript config.yaml tokens: {', '.join(unresolved)}")
    destination.write_text(resolved, encoding="utf-8")


def save_variables(variables: dict[str, str], output_path: Path) -> Path:
    """Atomically persist *variables* as JSON for rendering and debugging."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_symlink():
        raise RuntimeError(f"Refusing to write variables through symlink: {output_path}")
    temporary = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(variables, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return output_path


def render_manuscript_tree(project_root: Path, variables: dict[str, str]) -> Path:
    """Hydrate manuscript tokens into a guarded ``output/manuscript`` tree.

    Markdown sections and configuration string values are hydrated before the
    replacement becomes visible. YAML serialization keeps quotes, colons,
    newlines, and non-string values intact. Unresolved config tokens reject the
    transaction. Preamble and BibTeX files remain source-owned auxiliaries for
    the renderer. The symlink and project-root checks prevent a render from
    writing through an unintended checkout boundary.
    """
    root = Path(project_root)
    manuscript_dir = root / "manuscript"
    output_root = root / "output"
    out_dir = output_root / "manuscript"
    if output_root.is_symlink() or out_dir.is_symlink():
        raise RuntimeError(f"Refusing to render through symlinked output path: {out_dir}")
    output_root.mkdir(parents=True, exist_ok=True)
    if not out_dir.resolve().is_relative_to(root.resolve()):
        raise RuntimeError(f"Refusing to render outside project root: {out_dir}")

    # Assemble an entire replacement tree before moving it into place.  The
    # freshness preflight in the CLI catches stale inputs; this transaction
    # additionally prevents a filesystem or token-render failure from leaving
    # a mixture of old and new manuscript sections behind.
    staging_dir = output_root / f".manuscript-staging-{uuid.uuid4().hex}"
    backup_dir = output_root / f".manuscript-backup-{uuid.uuid4().hex}"
    staging_dir.mkdir()

    def replace_token(match: re.Match[str]) -> str:
        key = match.group(1)
        return variables.get(key, match.group(0))

    try:
        for md_file in sorted(manuscript_dir.glob("*.md")):
            if md_file.name in _EXCLUDED_MANUSCRIPT_DOCS or md_file.name == "preamble.md":
                continue
            text = md_file.read_text(encoding="utf-8")
            resolved = _TOKEN_RE.sub(replace_token, text)
            (staging_dir / md_file.name).write_text(resolved, encoding="utf-8")

        config = manuscript_dir / "config.yaml"
        if config.is_file():
            _hydrate_config(config, staging_dir / config.name, variables)
        preamble = manuscript_dir / "preamble.md"
        if preamble.is_file():
            shutil.copy2(preamble, staging_dir / preamble.name)
        for bib_file in sorted(manuscript_dir.glob("*.bib")):
            shutil.copy2(bib_file, staging_dir / bib_file.name)

        if out_dir.exists():
            out_dir.replace(backup_dir)
        try:
            staging_dir.replace(out_dir)
        except BaseException:
            if backup_dir.exists():
                backup_dir.replace(out_dir)
            raise
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
    except BaseException:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise

    return out_dir


__all__ = ["render_manuscript_tree", "save_variables"]
