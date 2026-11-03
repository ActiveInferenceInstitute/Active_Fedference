"""Offline structured cache integrity through canonical source re-admission."""
from __future__ import annotations

import json
from pathlib import Path

from ._documents import _json, _mapping, _sha
from .admission import GNNAdmissionError, dumps, loads
from .source import EXTENSION_MANIFEST, _sections
from .structured_admission import StructuredProgram, _admit_sections, _native, structured_tasks
from .types import HierarchicalModel, MovingObserverModel


def cached_structured(path: Path) -> StructuredProgram:
    """Load a generated IR only when bound source/parser/full payload hashes agree."""
    source_bytes = path.read_bytes()
    cache = _mapping(_json(path.with_suffix(".ir.json").read_text()), "cache")
    if set(cache) != {"model", "native", "digest"}:
        raise GNNAdmissionError("GNN_CACHE", "fields")
    model = loads(json.dumps(cache["model"], allow_nan=False))
    if not isinstance(model, (MovingObserverModel, HierarchicalModel)):
        raise GNNAdmissionError("GNN_KIND", "cache")
    manifest = _json(EXTENSION_MANIFEST.read_text())
    parser_digest = _sha(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())
    if (
        model.metadata.provenance.source != _sha(source_bytes)
        or model.metadata.provenance.asset != _sha(source_bytes)
        or model.metadata.provenance.parser != parser_digest
    ):
        raise GNNAdmissionError("GNN_CACHE", "source/parser binding")
    program = StructuredProgram(model, json.dumps(cache["native"], sort_keys=True, allow_nan=False))
    if program.digest != cache["digest"]:
        raise GNNAdmissionError("GNN_CACHE", "payload binding")
    # Offline integrity reuses typed admission; it does not claim parser execution.
    admitted = _admit_sections(
        _sections(source_bytes.decode("utf-8"), ("FedferenceStructured",)),
        _sha(source_bytes),
        parser_digest,
    )
    if dumps(admitted.model) != dumps(program.model) or admitted.native_json != program.native_json:
        raise GNNAdmissionError("GNN_CACHE", "source payload")
    _native(program)
    if isinstance(program.model, HierarchicalModel):
        structured_tasks(program)
    return program
