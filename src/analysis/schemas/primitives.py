"""Primitives report contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, NoReturn

TypeTag = Literal["bool", "dict", "int", "list", "number", "str"]


class ReportSchemaError(ValueError):
    """Raised when a report or figure-registry payload violates its schema."""


@dataclass(frozen=True)
class SchemaDefinition:
    """Shallow top-level schema definition."""

    required: Mapping[str, TypeTag]
    optional: Mapping[str, TypeTag] = field(default_factory=dict)


@dataclass(frozen=True)
class FigureDependencyContract:
    """Declared top-level report fields consumed by one figure generator."""

    report_name: str
    required_fields: Mapping[str, TypeTag]
    optional_fields: Mapping[str, TypeTag] = field(default_factory=dict)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _check_finite_json(value: object, *, path: str = "payload") -> None:
    """Reject non-standard JSON numbers at every nested report position."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise ReportSchemaError(f"{path} contains a non-finite JSON number")
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _check_finite_json(nested, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, nested in enumerate(value):
            _check_finite_json(nested, path=f"{path}[{index}]")


def _matches_type(tag: TypeTag, value: object) -> bool:
    if tag == "bool":
        return isinstance(value, bool)
    if tag == "dict":
        return isinstance(value, Mapping)
    if tag == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if tag == "list":
        return isinstance(value, list)
    if tag == "number":
        return _is_number(value)
    if tag == "str":
        return isinstance(value, str)
    raise AssertionError(f"Unhandled type tag: {tag}")


def _describe_value(value: object) -> str:
    return type(value).__name__


def _raise_missing(schema_name: str, field_name: str, tag: TypeTag) -> NoReturn:
    raise ReportSchemaError(f"{schema_name} payload missing required field {field_name!r} (expected {tag})")


def _raise_type_error(schema_name: str, field_name: str, tag: TypeTag, value: object) -> NoReturn:
    raise ReportSchemaError(
        f"{schema_name} payload field {field_name!r} expected {tag}, got {_describe_value(value)}"
    )


def _check(
    schema_name: str,
    payload: Mapping[str, object],
    spec: SchemaDefinition,
) -> None:
    for field_name, tag in spec.required.items():
        if field_name not in payload:
            _raise_missing(schema_name, field_name, tag)
        value = payload[field_name]
        if not _matches_type(tag, value):
            _raise_type_error(schema_name, field_name, tag, value)
    for field_name, tag in spec.optional.items():
        if field_name in payload and not _matches_type(tag, payload[field_name]):
            _raise_type_error(schema_name, field_name, tag, payload[field_name])


def _number_list(value: object, *, schema: str, field: str) -> list[float]:
    """Return a finite numeric list with a field-specific diagnostic."""
    if not isinstance(value, list) or any(not _is_number(item) for item in value):
        raise ReportSchemaError(f"{schema} payload field {field!r} must be a list of finite numbers")
    return [float(item) for item in value]


def _number(value: object, *, schema: str, field: str) -> float:
    """Return one finite non-Boolean number with a field-specific diagnostic."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportSchemaError(f"{schema} payload field {field!r} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ReportSchemaError(f"{schema} payload field {field!r} must be a finite number")
    return number


def _mapping(value: object, *, schema: str, field: str) -> Mapping[str, object]:
    """Return a nested mapping with a field-specific diagnostic."""
    if not isinstance(value, Mapping):
        raise ReportSchemaError(f"{schema} payload field {field!r} must be a mapping")
    return value
