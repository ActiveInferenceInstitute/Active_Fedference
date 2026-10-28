"""Strict pure admission of canonical JSON IR; never execute a family.

Duplicate-aware serialization preserves float64 PMF tolerance and provenance.
Raw GNN language parsing belongs to the source loaders.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import NoReturn, cast

from .admission_values import GNNAdmissionError as GNNAdmissionError
from .admission_values import _categorical, _fail, _labels, _metadata, _record, _symmetric, _tensor, _text
from .types import (
    CategoricalModel,
    HierarchicalModel,
    HybridGaussianModel,
    JSONValue,
    Layer,
    Model,
    ModelKind,
    MovingObserverModel,
    SiteFactorModel,
)


def _unhandled_kind(kind: NoReturn) -> NoReturn:
    _fail("KIND", "kind")



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
