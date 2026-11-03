"""Figure exact values report contracts."""

from __future__ import annotations

from collections.abc import Mapping

from analysis.figure_exact_values import EXACT_VALUE_GENERATORS, EXACT_VALUE_REPORT_BINDINGS
from analysis.schemas.primitives import ReportSchemaError, SchemaDefinition, _check

_FIGURE_EXACT_VALUES_SCHEMA = SchemaDefinition(
    required={
        "schema_version": "str",
        "generated_by": "str",
        "tables": "list",
    }
)


def _check_figure_exact_values(payload: Mapping[str, object]) -> None:
    """Validate identifier-addressable tables used as non-visual fallbacks."""
    _check("figure_exact_values", payload, _FIGURE_EXACT_VALUES_SCHEMA)
    if payload["schema_version"] != "1.0":
        raise ReportSchemaError("figure_exact_values schema_version must be '1.0'")
    if payload["generated_by"] != "analysis.figure_exact_values.build_figure_exact_values":
        raise ReportSchemaError("figure_exact_values generated_by is not the canonical producer")
    tables = payload["tables"]
    assert isinstance(tables, list)
    if not tables:
        raise ReportSchemaError("figure_exact_values requires at least one table")
    identifiers: list[str] = []
    generators: list[str] = []
    for index, table in enumerate(tables):
        if not isinstance(table, Mapping):
            raise ReportSchemaError(f"figure_exact_values tables[{index}] must be a mapping")
        _check(
            f"figure_exact_values tables[{index}]",
            table,
            SchemaDefinition(
                required={
                    "identifier": "str",
                    "generator": "str",
                    "source_report": "str",
                    "columns": "list",
                    "rows": "list",
                    "note": "str",
                }
            ),
        )
        expected_table_fields = {
            "identifier",
            "generator",
            "source_report",
            "columns",
            "rows",
            "note",
        }
        if set(table) != expected_table_fields:
            raise ReportSchemaError(f"figure_exact_values tables[{index}] has unknown or missing fields")
        identifier = str(table["identifier"])
        generator = str(table["generator"])
        expected_report = dict(EXACT_VALUE_REPORT_BINDINGS).get(generator)
        if identifier != f"fig-values:{generator.replace('_', '-')}":
            raise ReportSchemaError(f"figure_exact_values identifier/generator mismatch at tables[{index}]")
        if expected_report is not None and table["source_report"] != f"{expected_report}.json":
            raise ReportSchemaError(f"figure_exact_values source_report mismatch at tables[{index}]")
        if not str(table["note"]).strip():
            raise ReportSchemaError(f"figure_exact_values tables[{index}] note must be non-empty")
        columns = table["columns"]
        rows = table["rows"]
        assert isinstance(columns, list) and isinstance(rows, list)
        if (
            not columns
            or any(not isinstance(column, str) or not column for column in columns)
            or len(columns) != len(set(columns))
        ):
            raise ReportSchemaError(f"figure_exact_values tables[{index}] has invalid columns")
        if not rows:
            raise ReportSchemaError(f"figure_exact_values tables[{index}] has no rows")
        for row_index, row in enumerate(rows):
            if not isinstance(row, Mapping) or set(row) != set(columns):
                raise ReportSchemaError(
                    f"figure_exact_values tables[{index}].rows[{row_index}] does not match columns"
                )
            if any(
                value is not None and not isinstance(value, (str, int, float, bool)) for value in row.values()
            ):
                raise ReportSchemaError(
                    f"figure_exact_values tables[{index}].rows[{row_index}] contains a non-scalar value"
                )
        identifiers.append(identifier)
        generators.append(generator)
    if identifiers != sorted(set(identifiers)) or len(generators) != len(set(generators)):
        raise ReportSchemaError("figure_exact_values tables must be sorted and unique")
    expected_generators = set(EXACT_VALUE_GENERATORS)
    if set(generators) != expected_generators:
        raise ReportSchemaError(
            "figure_exact_values generator inventory mismatch: "
            f"missing={sorted(expected_generators - set(generators))}, "
            f"extra={sorted(set(generators) - expected_generators)}"
        )
