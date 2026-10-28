"""Explicit token sources for the Zenodo boundary; no logging or implicit loading."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from publication.zenodo_errors import ZenodoError

DEFAULT_TOKEN_ENV_NAMES: tuple[str, ...] = (
    "ZENODO_PROD_TOKEN",
    "ZENODO_TOKEN",
    "ZENODO_API_TOKEN",
)


def _parse_env_value(value: str) -> str:
    """Parse the small dotenv subset needed for a bearer token."""
    normalized = value.strip()
    if len(normalized) >= 2 and normalized[0] == normalized[-1] in {"'", '"'}:
        normalized = normalized[1:-1]
    return normalized


def token_from_env_file(
    env_file: str | Path,
    *,
    names: tuple[str, ...] = DEFAULT_TOKEN_ENV_NAMES,
) -> tuple[str, str]:
    """Read the first configured Zenodo token from a dotenv-style file.

    The returned tuple is ``(token, variable_name)`` so a caller can record
    which credential slot was selected without ever logging the secret.
    """
    path = Path(env_file)
    if not path.is_file():
        raise ZenodoError("Zenodo env file does not exist")
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ZenodoError("Zenodo env file could not be read") from exc
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, raw_value = line.partition("=")
        if separator and key.strip() in names:
            values[key.strip()] = _parse_env_value(raw_value)
    for name in names:
        if values.get(name):
            return values[name], name
    raise ZenodoError(f"Zenodo env file has none of: {', '.join(names)}")


def token_from_environment(
    environ: Mapping[str, str] | None = None,
    *,
    names: tuple[str, ...] = DEFAULT_TOKEN_ENV_NAMES,
) -> tuple[str, str]:
    """Read the first configured Zenodo token from process environment."""
    source = os.environ if environ is None else environ
    for name in names:
        value = source.get(name, "").strip()
        if value:
            return value, name
    raise ZenodoError(f"environment has none of: {', '.join(names)}")
