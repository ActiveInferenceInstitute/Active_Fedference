"""Figure requirements for source-preserving HTML publication."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from analysis.report_schemas import ReportSchemaError, validate_report
from analysis.visual_contracts import COMPLEX_FIGURE_GENERATORS


@dataclass(frozen=True)
class _ComplexFigureRequirement:
    """Source-registry contract for one complex HTML figure."""

    filename: str
    generator: str
    label: str
    details_id: str
    long_description: str


def _complex_figure_requirements(
    project_root: Path,
) -> tuple[dict[str, _ComplexFigureRequirement], tuple[str, ...]]:
    """Load complex-figure requirements from the generated source registry."""
    registry_path = project_root / "output" / "figures" / "figure_registry.json"
    if not registry_path.exists():
        return {}, (f"{registry_path}: required figure registry is missing",)
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {}, (f"{registry_path}: figure registry is unreadable: {exc}",)
    if not isinstance(payload, dict) or not isinstance(payload.get("figures"), list):
        return {}, (f"{registry_path}: figure registry has no figures list",)
    try:
        validate_report("figure_registry", payload)
    except ReportSchemaError as exc:
        return {}, (f"{registry_path}: figure registry schema is invalid: {exc}",)

    requirements: dict[str, _ComplexFigureRequirement] = {}
    labels: set[str] = set()
    observed_complex_generators: set[str] = set()
    issues: list[str] = []
    for index, raw in enumerate(payload["figures"]):
        if not isinstance(raw, dict):
            continue
        generator = raw.get("generated_by")
        long_description_value = str(raw.get("long_description", "")).strip()
        is_required_complex = generator in COMPLEX_FIGURE_GENERATORS
        if is_required_complex:
            observed_complex_generators.add(str(generator))
            if not long_description_value:
                issues.append(f"{registry_path}: complex figure entry {index} lacks long_description")
                continue
        elif not long_description_value:
            continue
        required_values = {
            key: raw.get(key) for key in ("filename", "generated_by", "label", "long_description")
        }
        if any(not isinstance(value, str) or not value.strip() for value in required_values.values()):
            issues.append(f"{registry_path}: complex figure entry {index} has malformed identity fields")
            continue
        filename = str(required_values["filename"])
        generator = str(required_values["generated_by"])
        label = str(required_values["label"])
        long_description = str(required_values["long_description"])
        if filename in requirements or label in labels:
            issues.append(f"{registry_path}: duplicate complex figure filename or label at entry {index}")
            continue
        labels.add(label)
        requirements[filename] = _ComplexFigureRequirement(
            filename=filename,
            generator=generator,
            label=label,
            details_id=f"{label.replace(':', '-')}-long-description",
            long_description=long_description,
        )
    for missing_generator in sorted(COMPLEX_FIGURE_GENERATORS - observed_complex_generators):
        issues.append(f"{registry_path}: required complex figure {missing_generator!r} is absent")
    return requirements, tuple(issues)
