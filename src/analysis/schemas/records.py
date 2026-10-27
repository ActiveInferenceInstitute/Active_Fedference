"""Records report contracts."""

from __future__ import annotations

from collections.abc import Mapping

from analysis.schemas.primitives import ReportSchemaError


def _record_list(value: object, *, schema: str, field: str) -> list[Mapping[str, object]]:
    """Return a list of mappings with an exact field-specific diagnostic."""
    if not isinstance(value, list):
        raise ReportSchemaError(f"{schema} payload field {field!r} must be a list")
    records: list[Mapping[str, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ReportSchemaError(f"{schema} payload field {field}[{index}] must be a mapping")
        records.append(item)
    return records


def _record_ids(
    records: list[Mapping[str, object]],
    *,
    schema: str,
    field: str,
) -> list[str]:
    """Return unique, non-empty string ids from a record list."""
    identifiers: list[str] = []
    for index, record in enumerate(records):
        identifier = record.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ReportSchemaError(f"{schema} payload field {field}[{index}].id must be a non-empty string")
        identifiers.append(identifier)
    if len(set(identifiers)) != len(identifiers):
        raise ReportSchemaError(f"{schema} payload field {field!r} contains duplicate ids")
    return identifiers


def _check_edges(
    records: list[Mapping[str, object]],
    *,
    schema: str,
    field: str,
    valid_nodes: set[str],
    allowed_dispositions: set[str],
) -> None:
    """Validate directed edge records against a declared node inventory."""
    seen: set[tuple[str, str]] = set()
    for index, edge in enumerate(records):
        source = edge.get("source")
        target = edge.get("target")
        label = edge.get("label")
        disposition = edge.get("disposition")
        if source not in valid_nodes or target not in valid_nodes:
            raise ReportSchemaError(f"{schema} payload field {field}[{index}] references an unknown node")
        if not isinstance(label, str) or not label:
            raise ReportSchemaError(f"{schema} payload field {field}[{index}].label must be non-empty")
        if disposition not in allowed_dispositions:
            raise ReportSchemaError(f"{schema} payload field {field}[{index}] has invalid disposition")
        # A directed dependency is identified by its endpoints.  Treating the
        # human-readable label as part of the identity would allow the same
        # producer or invalidation relation to be duplicated under two labels.
        identity = (str(source), str(target))
        if identity in seen:
            raise ReportSchemaError(f"{schema} payload field {field!r} contains duplicate edges")
        seen.add(identity)


def _check_nonempty_string_fields(
    record: Mapping[str, object],
    *,
    schema: str,
    field: str,
    names: tuple[str, ...],
) -> None:
    """Require the listed record fields to contain non-empty strings."""
    for name in names:
        value = record.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ReportSchemaError(f"{schema} payload field {field}.{name} must be a non-empty string")
