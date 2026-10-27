"""Review values report contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping

from analysis.schemas.primitives import ReportSchemaError


def _review_mapping(value: object, *, field: str) -> Mapping[str, object]:
    """Return a nested review-grid mapping or raise a named schema error."""
    if not isinstance(value, Mapping):
        raise ReportSchemaError(f"robustness_review_grid field {field!r} must be a mapping")
    return value


def _review_list(value: object, *, field: str) -> list[object]:
    """Return a nested review-grid list or raise a named schema error."""
    if not isinstance(value, list):
        raise ReportSchemaError(f"robustness_review_grid field {field!r} must be a list")
    return value


def _review_number(value: object, *, field: str) -> float:
    """Validate a finite non-bool number in a review-grid payload."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportSchemaError(f"robustness_review_grid field {field!r} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ReportSchemaError(f"robustness_review_grid field {field!r} must be a finite number")
    return number


def _review_string_set(value: object, *, field: str) -> set[str]:
    """Validate a unique list of strings and return its set representation."""
    values = _review_list(value, field=field)
    if not values or any(not isinstance(item, str) for item in values):
        raise ReportSchemaError(f"robustness_review_grid field {field!r} must be a non-empty list of strings")
    result = {str(item) for item in values}
    if len(result) != len(values):
        raise ReportSchemaError(f"robustness_review_grid field {field!r} must not contain duplicate strings")
    return result
