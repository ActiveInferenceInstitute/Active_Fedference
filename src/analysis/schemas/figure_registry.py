"""Figure registry report contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict

from typing_extensions import NotRequired

from analysis.schemas.primitives import ReportSchemaError, SchemaDefinition, _check, _describe_value
from analysis.visual_contracts import COMPLEX_FIGURE_GENERATORS


class FigureMetadataEntry(TypedDict):
    """Per-figure metadata payload written inside ``figure_registry.json``."""

    label: str
    filename: str
    path: str
    source_manuscript: str
    caption: str
    generated_by: str
    status: str
    source_relation: str
    source_figure: str
    source_equation: str
    source_citation: str
    estimand: str
    unit: str
    uncertainty: str
    replication_unit: str
    alt_text: str
    long_description: NotRequired[str]
    exact_value_fallback: NotRequired[str]
    presentation_manifest: NotRequired[str]


class FigureRegistryPayload(TypedDict):
    """Top-level figure registry payload."""

    schema_version: str
    generated_by: str
    figures: list[FigureMetadataEntry]
    exact_value_artifact: NotRequired[dict[str, object]]


_FIGURE_METADATA_SCHEMA = SchemaDefinition(
    required={
        "label": "str",
        "filename": "str",
        "path": "str",
        "source_manuscript": "str",
        "caption": "str",
        "generated_by": "str",
        "status": "str",
        "source_relation": "str",
        "source_figure": "str",
        "source_equation": "str",
        "source_citation": "str",
        "estimand": "str",
        "unit": "str",
        "uncertainty": "str",
        "replication_unit": "str",
        "alt_text": "str",
    },
    optional={
        "long_description": "str",
        "exact_value_fallback": "str",
        "presentation_manifest": "str",
    },
)


_FIGURE_REGISTRY_SCHEMA = SchemaDefinition(
    required={
        "schema_version": "str",
        "generated_by": "str",
        "figures": "list",
    },
    optional={"exact_value_artifact": "dict"},
)


def _check_figure_registry(payload: Mapping[str, object]) -> None:
    _check("figure_registry", payload, _FIGURE_REGISTRY_SCHEMA)
    if payload["schema_version"] != "1.2":
        raise ReportSchemaError("figure_registry schema_version must be '1.2'")
    if payload["generated_by"] != "analysis.workflow.run_analysis_pipeline":
        raise ReportSchemaError("figure_registry generated_by is not the canonical producer")
    figures = payload["figures"]
    assert isinstance(figures, list)
    if not figures:
        raise ReportSchemaError("figure_registry requires at least one figure")
    labels: list[str] = []
    filenames: list[str] = []
    generators: list[str] = []
    for index, entry in enumerate(figures):
        if not isinstance(entry, Mapping):
            raise ReportSchemaError(
                "figure_registry payload field 'figures' expected list of dict, "
                f"got {_describe_value(entry)} at index {index}"
            )
        label = entry.get("label")
        entry_name = (
            f"figure_registry figure {label!r}"
            if isinstance(label, str)
            else f"figure_registry figures[{index}]"
        )
        _check(entry_name, entry, _FIGURE_METADATA_SCHEMA)
        label = str(entry["label"])
        filename = str(entry["filename"])
        generator = str(entry["generated_by"])
        if not label.startswith("fig:"):
            raise ReportSchemaError(f"{entry_name} label must start with 'fig:'")
        if filename != f"{generator}.png":
            raise ReportSchemaError(
                f"{entry_name} filename must be the canonical PNG for generator {generator!r}"
            )
        if entry["path"] != f"output/figures/{filename}":
            raise ReportSchemaError(f"{entry_name} path does not match its filename")
        if "presentation_manifest" in entry and entry["presentation_manifest"] != (
            f"output/figures/{generator}.slides.json"
        ):
            raise ReportSchemaError(f"{entry_name} presentation manifest does not match its generator")
        source_manuscript = str(entry["source_manuscript"])
        if not source_manuscript.startswith("manuscript/") or not source_manuscript.endswith(".md"):
            raise ReportSchemaError(f"{entry_name} has an unsafe source_manuscript path")
        if len(str(entry["alt_text"])) > 500:
            raise ReportSchemaError(f"{entry_name} alt_text exceeds 500 characters")
        if "long_description" in entry and not str(entry["long_description"]).strip():
            raise ReportSchemaError(f"{entry_name} long_description must be non-empty")
        if generator in COMPLEX_FIGURE_GENERATORS and not str(entry.get("long_description", "")).strip():
            raise ReportSchemaError(f"{entry_name} complex figure requires a structured long_description")
        fallback = entry.get("exact_value_fallback")
        if fallback is not None and not str(fallback).startswith("fig-values:"):
            raise ReportSchemaError(f"{entry_name} exact_value_fallback is malformed")
        labels.append(label)
        filenames.append(filename)
        generators.append(generator)
    if labels != sorted(labels):
        raise ReportSchemaError("figure_registry figures must be sorted by label")
    for field_name, values in (
        ("label", labels),
        ("filename", filenames),
        ("generated_by", generators),
    ):
        if len(values) != len(set(values)):
            raise ReportSchemaError(f"figure_registry contains duplicate {field_name} entries")
    exact_artifact = payload.get("exact_value_artifact")
    declared = sorted(
        str(entry["exact_value_fallback"])
        for entry in figures
        if isinstance(entry, Mapping) and entry.get("exact_value_fallback")
    )
    if declared and exact_artifact is None:
        raise ReportSchemaError("figure_registry exact-value fallbacks require exact_value_artifact")
    if exact_artifact is not None:
        assert isinstance(exact_artifact, Mapping)
        _check(
            "figure_registry exact_value_artifact",
            exact_artifact,
            SchemaDefinition(
                required={
                    "json_path": "str",
                    "markdown_path": "str",
                    "identifiers": "list",
                }
            ),
        )
        identifiers = exact_artifact["identifiers"]
        assert isinstance(identifiers, list)
        if exact_artifact["json_path"] != "output/figures/figure_exact_values.json":
            raise ReportSchemaError("figure_registry exact-value JSON path is not canonical")
        if exact_artifact["markdown_path"] != "output/figures/figure_exact_values.md":
            raise ReportSchemaError("figure_registry exact-value Markdown path is not canonical")
        if any(
            not isinstance(item, str) or not item.startswith("fig-values:") for item in identifiers
        ) or identifiers != sorted(set(identifiers)):
            raise ReportSchemaError(
                "figure_registry exact_value_artifact identifiers must be sorted, unique fig-values:* strings"
            )
        if declared != identifiers:
            raise ReportSchemaError(
                "figure_registry exact-value identifiers do not match per-figure fallback declarations"
            )
