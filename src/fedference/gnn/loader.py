"""Explicit closed Sentinel extraction and source-bound offline loading.

The upstream extractor runs in isolation; warning-only validation is not admission.
Friston et al. (2024) categorical subset, not protocol reconstruction.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Final

import numpy as np

from ._documents import MODEL_DIR as MODEL_DIR
from ._documents import UPSTREAM_REVISION as UPSTREAM_REVISION
from ._documents import _json, _mapping, _sha
from .admission import GNNAdmissionError, dumps, loads
from .sentinel_admission import _AXES, _model, _raw
from .types import CategoricalModel, JSONValue

SENTINEL_ASSET: Final = MODEL_DIR / "sentinel.gnn.md"



PARSER_MANIFEST: Final = MODEL_DIR / "sentinel.parser.json"



SENTINEL_CACHE: Final = MODEL_DIR / "sentinel.ir.json"



def _manifest() -> tuple[dict[str, JSONValue], str]:
    manifest = _mapping(_json(PARSER_MANIFEST.read_text(encoding="utf-8")), "manifest")
    if set(manifest) != {"revision", "files"} or manifest["revision"] != UPSTREAM_REVISION:
        raise GNNAdmissionError("GNN_PARSER", "manifest.revision")
    files = _mapping(manifest["files"], "manifest.files")
    if not files:
        raise GNNAdmissionError("GNN_PARSER", "manifest.files")
    return files, _sha(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())



def _verify_parser(root: Path) -> str:
    files, digest = _manifest()
    for relative, expected in files.items():
        path = root / relative
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise GNNAdmissionError("GNN_PARSER", relative)
        if not path.is_file() or _sha(path.read_bytes()) != expected:
            raise GNNAdmissionError("GNN_PARSER_DRIFT", relative)
    return digest



def load_gnn(path: Path, *, upstream_root: Path, python: Path | None = None) -> CategoricalModel:
    """Run the pinned real extractor against an immutable caller-owned snapshot.

    Explicit upstream_root authorizes local source execution, never installation
    or network access. No extraction failure falls back to the packaged cache.
    """
    source_bytes = path.read_bytes()
    document = source_bytes.decode("utf-8")
    _raw(document)
    parser_digest = _verify_parser(upstream_root)
    with tempfile.TemporaryDirectory(prefix="fedference-gnn-") as directory:
        snapshot = Path(directory) / "model.gnn.md"
        snapshot.write_bytes(source_bytes)
        # Execute only verified closure bytes. The supplied source root may
        # contain unrelated modules that could otherwise shadow the stdlib.
        parser_source = Path(directory) / "parser"
        for relative, expected in _manifest()[0].items():
            payload = (upstream_root / relative).read_bytes()
            if _sha(payload) != expected:
                raise GNNAdmissionError("GNN_PARSER_DRIFT", relative)
            destination = parser_source / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        command = [
            str(python or sys.executable),
            "-I",
            "-S",
            "-c",
            "import sys,json; sys.path.insert(0,sys.argv[1]); "
            "from gnn.extract import extract_to_json; "
            "result=extract_to_json(sys.argv[2],on_error='raise',strict_validation=True); "
            "print(result); sys.exit(json.loads(result).get('status')=='error')",
            str(parser_source / "src"),
            str(snapshot),
        ]
        try:
            result = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise GNNAdmissionError("GNN_PROCESS", str(path)) from error
    if result.returncode != 0:
        raise GNNAdmissionError("GNN_EXTRACTION", result.stdout + result.stderr)
    if _verify_parser(upstream_root) != parser_digest:
        raise GNNAdmissionError("GNN_PARSER_DRIFT", "post-extraction")
    extracted = _mapping(_json(result.stdout), "extraction")
    _, schema, _ = _raw(document)
    for label, key in (
        ("states", "num_states"),
        ("observations", "num_observations"),
        ("actions", "num_actions"),
    ):
        labels = schema[label]
        if (
            not isinstance(labels, list)
            or type(extracted.get(key)) is not int
            or extracted[key] != len(labels)
        ):
            raise GNNAdmissionError("GNN_EXTRACTION_DIMENSION", key)
    provenance = _mapping(extracted.get("matrix_provenance"), "matrix_provenance")
    b = _mapping(provenance.get("B"), "matrix_provenance.B")
    if b.get("declared_order_explicit") is not True or b.get("detected_order") != _AXES["B"]:
        raise GNNAdmissionError("GNN_EXTRACTION_AXIS", "B")
    return _model(document, parser_digest, extracted, _sha(source_bytes))



def load_sentinel(*, asset: Path = SENTINEL_ASSET, cache: Path = SENTINEL_CACHE) -> CategoricalModel:
    """Load the compiled, source-bound sentinel offline; never invoke upstream."""
    model = loads(cache.read_text(encoding="utf-8"))
    if not isinstance(model, CategoricalModel):
        raise GNNAdmissionError("GNN_KIND", "cache")
    source_bytes = asset.read_bytes()
    document = source_bytes.decode("utf-8")
    _, parser_digest = _manifest()
    if model.metadata.provenance.asset != _sha(source_bytes):
        raise GNNAdmissionError("GNN_STALE_CACHE", "asset")
    c = model.categorical
    extracted: dict[str, JSONValue] = {"extraction_schema_version": "1.0.0"}
    for name, tensor in (("A_matrix", c.A), ("B_matrix", c.B), ("C_vector", c.C), ("D_vector", c.D)):
        extracted[name] = np.array(tensor.values).reshape(tensor.shape).tolist()
    expected = _model(document, parser_digest, extracted, _sha(source_bytes))
    if dumps(model) != dumps(expected):
        raise GNNAdmissionError("GNN_STALE_CACHE", "content")
    return model



def verify_sentinel_cache(*, upstream_root: Path) -> CategoricalModel:
    """Build-time real-parser conformance gate; compare without overwriting."""
    parsed = load_gnn(SENTINEL_ASSET, upstream_root=upstream_root)
    if dumps(parsed) != dumps(load_sentinel()):
        raise GNNAdmissionError("GNN_STALE_CACHE", "regeneration")
    return parsed
