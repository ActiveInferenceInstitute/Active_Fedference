"""Presentation report contracts."""

from __future__ import annotations

from collections.abc import Mapping

from analysis.schemas.primitives import ReportSchemaError, _is_number


def _check_presentation_manifest(payload: Mapping[str, object]) -> None:
    """Require the exact bounded Template panel transport schema."""
    if set(payload) != {"schema_version", "panels"} or payload["schema_version"] != "1.0":
        raise ReportSchemaError("presentation_manifest requires version 1.0 and exactly panels")
    panels = payload["panels"]
    if not isinstance(panels, list) or not 1 <= len(panels) <= 64:
        raise ReportSchemaError("presentation_manifest requires 1 to 64 panels")
    sources: set[str] = set()
    for panel in panels:
        if not isinstance(panel, Mapping) or set(panel) != {"src", "alt", "sha256", "minimum_label_px"}:
            raise ReportSchemaError("presentation_manifest panel fields are invalid")
        source = panel["src"]
        alternative = panel["alt"]
        digest = panel["sha256"]
        measurement = panel["minimum_label_px"]
        if (
            not isinstance(source, str)
            or not source.startswith("../figures/")
            or not source.endswith(".png")
            or "/" in source[len("../figures/") :]
            or any(character in source for character in "\\#%{}")
            or source in sources
        ):
            raise ReportSchemaError("presentation_manifest panel source is unsafe or repeated")
        if not isinstance(alternative, str) or not alternative.strip() or len(alternative) > 500:
            raise ReportSchemaError("presentation_manifest panel alternative is invalid")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ReportSchemaError("presentation_manifest panel digest is invalid")
        if not _is_number(measurement) or float(measurement) <= 0:
            raise ReportSchemaError("presentation_manifest label measurement must be finite and positive")
        sources.add(source)
