"""Accessibility for source-preserving HTML publication."""
from __future__ import annotations

from html import unescape
from pathlib import Path
from urllib.parse import unquote, urlparse

from publication.web.accessibility_parser import _AccessibilityParser
from publication.web.figure_requirements import _ComplexFigureRequirement


def _normalize_accessibility_text(value: str) -> str:
    """Normalize renderer whitespace without weakening textual identity."""
    return " ".join(value.split())


def _figure_filename(raw_src: str) -> str:
    parsed = urlparse(unescape(raw_src))
    return Path(unquote(parsed.path)).name


def _complex_figure_issues(
    path: Path,
    parser: _AccessibilityParser,
    requirements_by_filename: dict[str, _ComplexFigureRequirement],
) -> tuple[tuple[str, ...], set[str]]:
    """Validate complex figure disclosure identity and exact association."""
    issues: list[str] = []
    observed: set[str] = set()
    requirements_by_label = {
        requirement.label: requirement for requirement in requirements_by_filename.values()
    }
    for facts in parser.figure_facts:
        candidates = {
            requirement.label: requirement
            for source in facts.image_sources
            if (requirement := requirements_by_filename.get(_figure_filename(source)))
        }
        if facts.figure_id in requirements_by_label:
            requirement = requirements_by_label[facts.figure_id]
            candidates[requirement.label] = requirement
        if not candidates:
            continue
        if len(candidates) != 1:
            issues.append(
                f"{path}: complex figure element ambiguously contains {', '.join(sorted(candidates))}"
            )
            continue

        requirement = next(iter(candidates.values()))
        observed.add(requirement.label)
        prefix = f"{path}: {requirement.label}"
        if facts.figure_id != requirement.label:
            issues.append(f"{prefix} figure id must equal the registry label")
        matching_sources = [
            source for source in facts.image_sources if _figure_filename(source) == requirement.filename
        ]
        if len(matching_sources) != 1:
            issues.append(f"{prefix} must contain exactly one registered image {requirement.filename}")
        if len(facts.image_aria_details) != 1 or (facts.image_aria_details[0] != requirement.details_id):
            issues.append(f"{prefix} image aria-details must reference {requirement.details_id!r}")
        if facts.details_count != 1:
            issues.append(f"{prefix} must contain exactly one details disclosure")
        if facts.caption_count != 1:
            issues.append(f"{prefix} must contain exactly one figcaption")
        if facts.caption_details_dom_order != ["figcaption", "details"]:
            issues.append(f"{prefix} labeled details disclosure must follow figcaption in DOM child order")
        if facts.details_id != requirement.details_id:
            issues.append(f"{prefix} details id must equal {requirement.details_id!r}")
        if "figure-long-description" not in facts.details_classes:
            issues.append(f"{prefix} details must use the figure-long-description class")
        if facts.details_figure_label != requirement.label:
            issues.append(f"{prefix} details data-figure-label must match the registry label")
        if facts.details_expanded_initially:
            issues.append(f"{prefix} details disclosure must be initially collapsed")
        if facts.summary_count != 1:
            issues.append(f"{prefix} details disclosure must contain exactly one summary")
        if facts.description_paragraph_count < 2:
            issues.append(f"{prefix} structured long description must contain at least two paragraphs")
        summary = _normalize_accessibility_text(" ".join(facts.summary_chunks))
        expected_summary = f"Detailed description of figure {requirement.generator.replace('_', ' ')}"
        if summary.casefold() != expected_summary.casefold():
            issues.append(f"{prefix} details summary must equal {expected_summary!r}")
        observed_description = _normalize_accessibility_text(" ".join(facts.description_chunks))
        expected_description = _normalize_accessibility_text(requirement.long_description)
        if observed_description != expected_description:
            issues.append(f"{prefix} details text does not match the registry long description")
    return tuple(issues), observed


def _accessibility_issues_in(
    path: Path,
    text: str,
    requirements_by_filename: dict[str, _ComplexFigureRequirement] | None = None,
) -> tuple[tuple[str, ...], set[str]]:
    """Return reader-facing HTML accessibility contract violations."""
    parser = _AccessibilityParser()
    parser.feed(text)
    parser.close()

    issues: list[str] = []
    if len(parser.html_languages) != 1 or not parser.html_languages[0]:
        issues.append(f"{path}: expected one non-empty html lang attribute")
    if len(parser.titles) != 1 or not parser.titles[0]:
        issues.append(f"{path}: expected one non-empty document title")
    if parser.main_count != 1 or parser.main_content_id_count != 1:
        issues.append(f"{path}: expected one main element with id='main-content'")
    elif parser.focusable_main_content_count != 1:
        issues.append(f"{path}: main-content skip target must use tabindex='-1'")
    if parser.skip_link_count != 1:
        issues.append(f"{path}: expected one .skip-link targeting '#main-content'")
    elif not parser.first_interactive_is_skip:
        issues.append(f"{path}: skip link must be the first interactive element")
    if parser.missing_image_alt:
        issues.append(f"{path}: {parser.missing_image_alt} image(s) lack non-empty alt text")
    if parser.figures_missing_captions:
        issues.append(f"{path}: {parser.figures_missing_captions} image figure(s) lack figcaption")
    if parser.unlabelled_full_size_links:
        issues.append(f"{path}: {parser.unlabelled_full_size_links} full-size figure link(s) lack aria-label")
    if parser.generic_full_size_links:
        issues.append(
            f"{path}: {parser.generic_full_size_links} full-size figure link(s) "
            "lack a contextual Figure number or title in aria-label"
        )
    if parser.duplicate_ids:
        issues.append(f"{path}: duplicate id(s): {', '.join(sorted(parser.duplicate_ids))}")
    complex_issues, observed = _complex_figure_issues(
        path,
        parser,
        requirements_by_filename or {},
    )
    issues.extend(complex_issues)
    return tuple(issues), observed
