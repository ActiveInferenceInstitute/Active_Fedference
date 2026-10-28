"""Pure canonical IR field admission; never parse source or execute a family."""
from __future__ import annotations

import re
from typing import NoReturn

import numpy as np

from .types import Categorical, JSONValue, Metadata, Provenance, Tensor


class GNNAdmissionError(ValueError):
    """Stable machine-readable refusal and exact IR location."""

    def __init__(self, code: str, location: str) -> None:
        self.code = code
        self.location = location
        super().__init__(f"{code}: {location}")



def _fail(code: str, location: str) -> NoReturn:
    raise GNNAdmissionError("GNN_" + code, location)



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
