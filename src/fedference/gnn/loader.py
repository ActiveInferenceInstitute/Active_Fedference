"""Explicit GNN extraction boundary and source-bound offline sentinel loading.

Only the documented categorical subset is admitted. The upstream extractor is
executed in an isolated process; its warning-only validation is not admission.
Friston et al. (2024) categorical specialization, not protocol reconstruction.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Final

import numpy as np

from .admission import GNNAdmissionError, dumps, loads
from .types import CategoricalModel, JSONValue

MODEL_DIR: Final = Path(__file__).resolve().parents[1] / "models"
SENTINEL_ASSET: Final = MODEL_DIR / "sentinel.gnn.md"
PARSER_MANIFEST: Final = MODEL_DIR / "sentinel.parser.json"
SENTINEL_CACHE: Final = MODEL_DIR / "sentinel.ir.json"
UPSTREAM_REVISION: Final = "f0dfb68368280f3a43ff91ac28db058c968002d8"
_SECTIONS: Final = {
    "GNNSection",
    "GNNVersionAndFlags",
    "ModelName",
    "ModelAnnotation",
    "StateSpaceBlock",
    "Connections",
    "InitialParameterization",
    "ModelParameters",
    "FedferenceSchema",
}
_AXES: Final = {
    "A": ["observation", "state"],
    "B": ["next_state", "previous_state", "action"],
    "C": ["observation"],
    "D": ["state"],
}


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


def _raw(document: str) -> tuple[dict[str, str], dict[str, JSONValue], dict[str, JSONValue]]:
    document = document.replace("\r\n", "\n").replace("\r", "\n")
    sections: dict[str, str] = {}
    parts = re.split(r"(?m)^## ([^\n]+)\n", document)
    if not parts[0].startswith("# ") or len(parts) != 1 + 2 * len(_SECTIONS):
        raise GNNAdmissionError("GNN_SECTIONS", "document")
    for name, body in zip(parts[1::2], parts[2::2]):
        if name in sections or name not in _SECTIONS:
            raise GNNAdmissionError("GNN_SECTIONS", name)
        sections[name] = body.strip()
    if set(sections) != _SECTIONS:
        raise GNNAdmissionError("GNN_SECTIONS", "document")
    if sections["GNNSection"] != "ActInfPOMDP" or sections["GNNVersionAndFlags"] != "GNN v1.0":
        raise GNNAdmissionError("GNN_SCHEMA", "GNNVersionAndFlags")
    schema = _mapping(_json(sections["FedferenceSchema"]), "FedferenceSchema")
    if set(schema) != {"schema", "states", "observations", "actions", "axes"}:
        raise GNNAdmissionError("GNN_FIELDS", "FedferenceSchema")
    if schema["schema"] != "fedference.gnn/1":
        raise GNNAdmissionError("GNN_SCHEMA", "FedferenceSchema")
    if schema["axes"] != _AXES:
        raise GNNAdmissionError("GNN_AXIS", "FedferenceSchema.axes")
    parameters: dict[str, JSONValue] = {}
    for line in sections["InitialParameterization"].splitlines():
        key, sep, value = line.partition("=")
        if not sep or key not in _AXES or key in parameters:
            raise GNNAdmissionError("GNN_PARAMETER", key)
        parameters[key] = _json(value)
    if set(parameters) != set(_AXES):
        raise GNNAdmissionError("GNN_PARAMETER", "InitialParameterization")
    dimensions = []
    for label in ("states", "observations", "actions"):
        values = schema[label]
        if not isinstance(values, list) or not values:
            raise GNNAdmissionError("GNN_LABEL", label)
        dimensions.append(len(values))
    s, o, a = dimensions
    expected = f"num_states: {s}\nnum_observations: {o}\nnum_actions: {a}"
    if sections["ModelParameters"] != expected:
        raise GNNAdmissionError("GNN_DIMENSION", "ModelParameters")
    state_space = (
        f"A[{o},{s},type=float]\n# B[next_state,previous_state,action]\n"
        f"B[{s},{s},{a},type=float]\nC[{o},type=float]\nD[{s},type=float]\n"
        f"s[{s},type=float]\no[{o},type=int]\nu[1,type=int]"
    )
    if sections["StateSpaceBlock"] != state_space:
        raise GNNAdmissionError("GNN_AXIS", "StateSpaceBlock")
    if sections["Connections"] != "D>s\ns>A\nA>o\ns>B\nu>B":
        raise GNNAdmissionError("GNN_CONNECTIONS", "Connections")
    return sections, schema, parameters


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


def _model(
    document: str, parser_digest: str, extracted: dict[str, JSONValue], source_digest: str
) -> CategoricalModel:
    sections, schema, parameters = _raw(document)
    if extracted.get("extraction_schema_version") != "1.0.0" or "status" in extracted:
        raise GNNAdmissionError("GNN_EXTRACTION_SCHEMA", "extraction")
    categorical: dict[str, JSONValue] = {key: schema[key] for key in ("states", "observations", "actions")}
    for name, key in (("A", "A_matrix"), ("B", "B_matrix"), ("C", "C_vector"), ("D", "D_vector")):
        if key not in extracted or extracted[key] != parameters[name]:
            raise GNNAdmissionError("GNN_EXTRACTION_MISMATCH", name)

        def numeric(value: JSONValue) -> bool:
            if isinstance(value, list):
                return all(numeric(item) for item in value)
            return type(value) in (int, float)

        if not numeric(extracted[key]):
            raise GNNAdmissionError("GNN_NUMBER", name)
        try:
            array = np.asarray(extracted[key], dtype=np.float64)
        except (ValueError, TypeError) as error:
            raise GNNAdmissionError("GNN_SHAPE", name) from error
        if not np.all(np.isfinite(array)):
            raise GNNAdmissionError("GNN_NONFINITE", name)
        categorical[name] = {
            "axes": [axis for axis in _AXES[name]],
            "shape": list(array.shape),
            "values": [float(value) for value in array.ravel()],
        }
    digest = source_digest
    payload = {
        "kind": "categorical",
        "categorical": categorical,
        "metadata": {
            "name": sections["ModelName"],
            "schema": schema["schema"],
            "precision": "float64-atol-1e-12-rtol-0",
            "capabilities": ["fedference.gnn/categorical/1"],
            "schedule": ["predict", "observe"],
            "provenance": {
                "source": digest,
                "asset": digest,
                "parser": parser_digest,
                "backend": None,
                "native_result": None,
            },
        },
    }
    model = loads(json.dumps(payload, allow_nan=False))
    if not isinstance(model, CategoricalModel):
        raise GNNAdmissionError("GNN_KIND", "model")
    return model


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
