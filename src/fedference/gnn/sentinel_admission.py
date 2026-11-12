"""Closed Sentinel source/extraction agreement (Friston et al., 2024 subset)."""
from __future__ import annotations

import json
import re
from typing import Final

import numpy as np

from ._documents import _json, _mapping
from .admission import GNNAdmissionError, loads
from .types import CategoricalModel, JSONValue

_SECTIONS: Final = {
    "GNNSection",
    "GNNVersionAndFlags",
    "ModelName",
    "ModelAnnotation",
    "StateSpaceBlock",
    "Connections",
    "InitialParameterization",
    "Time",
    "ModelParameters",
    "Footer",
    "FedferenceSchema",
}



_AXES: Final = {
    "A": ["observation", "state"],
    "B": ["next_state", "previous_state", "action"],
    "C": ["observation"],
    "D": ["state"],
}



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
