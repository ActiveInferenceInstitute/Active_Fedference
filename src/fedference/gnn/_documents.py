"""Shared source-document JSON and packaged resource locations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Final

from .admission import GNNAdmissionError
from .types import JSONValue

MODEL_DIR: Final = Path(__file__).resolve().parents[1] / "models"



UPSTREAM_REVISION: Final = "8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1"



def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()



def _pairs(pairs: list[tuple[str, JSONValue]]) -> dict[str, JSONValue]:
    result: dict[str, JSONValue] = {}
    for key, value in pairs:
        if key in result:
            raise GNNAdmissionError("GNN_DUPLICATE", key)
        result[key] = value
    return result



def _json(text: str) -> JSONValue:
    try:
        value: JSONValue = json.loads(text, object_pairs_hook=_pairs)
    except GNNAdmissionError:
        raise
    except (ValueError, RecursionError) as error:
        raise GNNAdmissionError("GNN_JSON", "document") from error
    return value



def _mapping(value: JSONValue, location: str) -> dict[str, JSONValue]:
    if not isinstance(value, dict):
        raise GNNAdmissionError("GNN_FIELDS", location)
    return value
