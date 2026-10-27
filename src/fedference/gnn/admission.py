"""Strict, pure admission of canonical JSON IR, not a raw GNN language parser.

Duplicate keys are rejected before JSON becomes a dictionary. Raw GNN duplicate
parameters must be rejected by the upstream loader before constructing this IR.
PMF sums use float64, absolute tolerance 1e-12 and zero relative tolerance;
values are never normalized. This does not establish backend/scientific parity.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from typing import NoReturn, cast

import numpy as np

from .types import (
    Categorical,
    CategoricalModel,
    HierarchicalModel,
    HybridGaussianModel,
    JSONValue,
    Layer,
    Metadata,
    Model,
    ModelKind,
    MovingObserverModel,
    Provenance,
    SiteFactorModel,
    Tensor,
)


class GNNAdmissionError(ValueError):
    """Stable machine-readable refusal and exact IR location."""

    def __init__(self, code: str, location: str) -> None:
        self.code = code
        self.location = location
        super().__init__(f"{code}: {location}")


def _fail(code: str, location: str) -> NoReturn:
    raise GNNAdmissionError("GNN_" + code, location)


def _unhandled_kind(kind: NoReturn) -> NoReturn:
    _fail("KIND", "kind")


def _record(value: JSONValue, keys: set[str], location: str) -> dict[str, JSONValue]:
    if not isinstance(value, dict) or set(value) != keys:
        _fail("FIELDS", location)
    return value


def _text(value: JSONValue, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail("LABEL", location)
    return value


def _labels(value: JSONValue, location: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        _fail("LABEL", location)
    result = tuple(_text(item, location) for item in value)
    if len(set(result)) != len(result):
        _fail("DUPLICATE", location)
    return result


def _tensor(
    value: JSONValue, axes: tuple[str, ...], shape: tuple[int, ...], location: str, pmf: bool = False
) -> Tensor:
    data = _record(value, {"axes", "shape", "values"}, location)
    if data["axes"] != list(axes):
        _fail("AXIS", location)
    declared = data["shape"]
    if not isinstance(declared, list) or any(type(n) is not int for n in declared) or declared != list(shape):
        _fail("SHAPE", location)
    values = data["values"]
    if not isinstance(values, list) or len(values) != int(np.prod(shape)):
        _fail("SHAPE", location)
    numbers: list[float] = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            _fail("NUMBER", location)
        try:
            number = float(item)
        except OverflowError:
            _fail("NONFINITE", location)
        if not np.isfinite(number):
            _fail("NONFINITE", location)
        numbers.append(number)
    array = np.array(numbers).reshape(shape)
    if pmf and (np.any(array < 0) or not np.allclose(array.sum(axis=0), 1, atol=1e-12, rtol=0)):
        _fail("PMF", location)
    return Tensor(axes, shape, tuple(numbers))


def _categorical(value: JSONValue, location: str) -> Categorical:
    data = _record(value, {"states", "observations", "actions", "A", "B", "C", "D"}, location)
    states = _labels(data["states"], location + ".states")
    observations = _labels(data["observations"], location + ".observations")
    actions = _labels(data["actions"], location + ".actions")
    s, o, a = len(states), len(observations), len(actions)
    return Categorical(
        states,
        observations,
        actions,
        _tensor(data["A"], ("observation", "state"), (o, s), location + ".A", True),
        _tensor(data["B"], ("next_state", "previous_state", "action"), (s, s, a), location + ".B", True),
        _tensor(data["C"], ("observation",), (o,), location + ".C"),
        _tensor(data["D"], ("state",), (s,), location + ".D", True),
    )


def _digest(value: JSONValue, location: str) -> str:
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        _fail("DIGEST", location)
    return value


def _metadata(value: JSONValue, kind: str) -> Metadata:
    data = _record(
        value, {"name", "schema", "precision", "capabilities", "schedule", "provenance"}, "metadata"
    )
    if data["schema"] != "fedference.gnn/1":
        _fail("SCHEMA", "metadata.schema")
    if data["precision"] != "float64-atol-1e-12-rtol-0":
        _fail("PRECISION", "metadata.precision")
    capabilities = _labels(data["capabilities"], "metadata.capabilities")
    # Versioned namespace is mandatory; capability support means representation only.
    if capabilities != (f"fedference.gnn/{kind}/1",):
        _fail("CAPABILITY", "metadata.capabilities")
    schedule = _labels(data["schedule"], "metadata.schedule")
    expected = ("site-add",) if kind == "site-factor" else ("predict", "observe")
    if schedule != expected:
        _fail("SCHEDULE", "metadata.schedule")
    p = _record(data["provenance"], {"source", "asset", "parser", "backend", "native_result"}, "provenance")
    provenance = Provenance(
        _digest(p["source"], "provenance.source"),
        _digest(p["asset"], "provenance.asset"),
        _digest(p["parser"], "provenance.parser"),
        None if p["backend"] is None else _digest(p["backend"], "provenance.backend"),
        None if p["native_result"] is None else _digest(p["native_result"], "provenance.native_result"),
    )
    return Metadata(_text(data["name"], "metadata.name"), capabilities, schedule, provenance)


def _symmetric(tensor: Tensor, location: str, positive: bool) -> None:
    array = np.array(tensor.values).reshape(tensor.shape)
    if not np.array_equal(array, array.T):
        _fail("SYMMETRY", location)
    if positive:
        try:
            np.linalg.cholesky(array)
        except np.linalg.LinAlgError:
            _fail("COVARIANCE", location)


def admit(value: JSONValue) -> Model:
    """Admit canonical IR with explicit family semantics; never execute a model."""
    if not isinstance(value, dict):
        _fail("FIELDS", "model")
    raw_kind = _text(value.get("kind"), "kind")
    family_fields: dict[ModelKind, set[str]] = {
        "categorical": {"categorical"},
        "moving-observer": {
            "threat",
            "positions",
            "position_transition",
            "position_prior",
            "observation_map",
        },
        "hierarchical": {"layers"},
        "hybrid-gaussian": {"categorical", "coordinates", "units", "mean", "covariance"},
        "site-factor": {"coordinates", "units", "precision_increment", "information_increment", "protocol"},
    }
    if raw_kind not in family_fields:
        _fail("KIND", "kind")
    kind = cast(ModelKind, raw_kind)
    data = _record(value, {"kind", "metadata"} | family_fields[kind], "model")
    metadata = _metadata(data["metadata"], kind)
    if kind == "categorical":
        return CategoricalModel(metadata, _categorical(data["categorical"], "categorical"))
    if kind == "moving-observer":
        threat = _categorical(data["threat"], "threat")
        positions = _labels(data["positions"], "positions")
        p, a, s, o = len(positions), len(threat.actions), len(threat.states), len(threat.observations)
        return MovingObserverModel(
            metadata,
            threat,
            positions,
            _tensor(
                data["position_transition"],
                ("next_position", "previous_position", "action"),
                (p, p, a),
                "position_transition",
                True,
            ),
            _tensor(data["position_prior"], ("position",), (p,), "position_prior", True),
            _tensor(
                data["observation_map"],
                ("observation", "threat", "position"),
                (o, s, p),
                "observation_map",
                True,
            ),
        )
    if kind == "hierarchical":
        raw_layers = data["layers"]
        if not isinstance(raw_layers, list) or len(raw_layers) < 2:
            _fail("LAYERS", "layers")
        layers: list[Layer] = []
        for raw in raw_layers:
            item = _record(raw, {"name", "categorical", "parent", "conditioned_prior"}, "layer")
            name = _text(item["name"], "layer.name")
            if any(layer.name == name for layer in layers):
                _fail("DUPLICATE", name)
            categorical = _categorical(item["categorical"], name)
            parent_name = item["parent"]
            if not layers:
                if parent_name is not None or item["conditioned_prior"] is not None:
                    _fail("PARENT", name)
                layers.append(Layer(name, categorical, None, None))
            else:
                parents = [layer for layer in layers if layer.name == parent_name]
                if len(parents) != 1:
                    _fail("PARENT", name)
                parent = parents[0]
                prior = _tensor(
                    item["conditioned_prior"],
                    ("child_state", "parent_state"),
                    (len(categorical.states), len(parent.categorical.states)),
                    name + ".conditioned_prior",
                    True,
                )
                layers.append(Layer(name, categorical, parent.name, prior))
        return HierarchicalModel(metadata, tuple(layers))
    coordinates = _labels(data["coordinates"], "coordinates")
    raw_units = data["units"]
    if not isinstance(raw_units, list) or len(raw_units) != len(coordinates):
        _fail("UNIT", "units")
    units = tuple(_text(unit, "units") for unit in raw_units)
    if any(unit not in {"m", "m/s", "rad", "s", "1"} for unit in units):
        _fail("UNIT", "units")
    n = len(coordinates)
    if kind == "hybrid-gaussian":
        categorical = _categorical(data["categorical"], "categorical")
        mean = _tensor(data["mean"], ("coordinate",), (n,), "mean")
        covariance = _tensor(data["covariance"], ("coordinate", "coordinate_dual"), (n, n), "covariance")
        _symmetric(covariance, "covariance", True)
        return HybridGaussianModel(metadata, categorical, coordinates, units, mean, covariance)
    if kind != "site-factor":
        _unhandled_kind(kind)
    if data["protocol"] != "gaussian-natural-additive/1":
        _fail("PROTOCOL", "protocol")
    precision = _tensor(
        data["precision_increment"], ("coordinate", "coordinate_dual"), (n, n), "precision_increment"
    )
    _symmetric(precision, "precision_increment", False)
    information = _tensor(data["information_increment"], ("coordinate",), (n,), "information_increment")
    return SiteFactorModel(metadata, coordinates, units, precision, information)


def _pairs(pairs: list[tuple[str, JSONValue]]) -> dict[str, JSONValue]:
    result: dict[str, JSONValue] = {}
    for key, value in pairs:
        if key in result:
            _fail("DUPLICATE", key)
        result[key] = value
    return result


def loads(document: str) -> Model:
    """Lossless duplicate-aware JSON admission, not raw GNN duplicate detection."""
    try:
        value: JSONValue = json.loads(document, object_pairs_hook=_pairs)
    except GNNAdmissionError:
        raise
    except (ValueError, RecursionError) as exc:
        raise GNNAdmissionError("GNN_JSON", "document") from exc
    return admit(value)


def dumps(model: Model) -> str:
    """Serialize the immutable IR deterministically; re-admit to reject forged values."""
    document = json.dumps(asdict(model), sort_keys=True, separators=(",", ":"), allow_nan=False)
    loads(document)
    return document


def content_digest(model: Model) -> str:
    """SHA-256 binds full admitted content and declared provenance, not its truth."""
    return hashlib.sha256(dumps(model).encode("utf-8")).hexdigest()
