"""Strict JSON decoding for evidence boundaries."""

from __future__ import annotations

import json
from typing import Any


def _reject_json_constant(value: str) -> Any:
    """Reject non-standard JSON constants such as NaN and Infinity."""
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")



def _loads_json_strict(value: str) -> Any:
    """Decode standards-compliant JSON without Python's NaN extensions."""
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key is not allowed: {key!r}")
            result[key] = item
        return result

    return json.loads(
        value,
        parse_constant=_reject_json_constant,
        object_pairs_hook=reject_duplicate_keys,
    )

