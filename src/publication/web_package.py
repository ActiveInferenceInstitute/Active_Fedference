"""Canonical HTML package validation and preparation entrypoints."""
from __future__ import annotations

from html import unescape
from pathlib import Path
from urllib.parse import unquote, urlparse

from publication.web.accessibility import _accessibility_issues_in
from publication.web.assets import mirror_web_figures as mirror_web_figures
from publication.web.assets import sanitize_machine_paths as sanitize_machine_paths
from publication.web.attributes import _attribute_occurrences
from publication.web.figure_requirements import _complex_figure_requirements
from publication.web.lexical import _line_number_at
from publication.web.links import _asset_path as _asset_path
from publication.web.links import _fragment_exists
from publication.web.links import _hrefs_in as _hrefs_in
from publication.web.links import _local_href_target as _local_href_target
from publication.web.markup_findings import (
    _LEAKED_FIGURE_RE,
    _UNRESOLVED_TOKEN_RE,
    _accessible_name_markup_in,
    _pattern_lines_in_text_tokens,
    _raw_xrefs_in,
)
from publication.web.models import WebPackageValidation as WebPackageValidation
from publication.web.models import _HTMLTokenizationError
from publication.web.normalization import normalize_web_xrefs as normalize_web_xrefs
from publication.web.paths import _root, _safe_html_paths, _web_dir
from publication.web.reference_labels import _xref_ids
from publication.web.resources import _document_base_issues, _resource_target, _resource_urls
from publication.web.script_integrity import _external_script_issue
from publication.web.tokenizer import _tokenize_html
from publication.web.transaction import _commit_html_writes as _commit_html_writes
from publication.web.transaction_files import _link_no_clobber as _link_no_clobber
from publication.web.transaction_rollback import _restore_committed_html as _restore_committed_html


def validate_web_package(project_root: str | Path | None = None) -> WebPackageValidation:
    """Check generated HTML assets, links, markup, and accessibility structure."""
    root = _root(project_root)
    web_dir = _web_dir(root)
    if not web_dir.exists():
        return WebPackageValidation(
            html_files=0,
            assets_checked=0,
            missing_assets=(f"missing web output directory: {web_dir}",),
            raw_xrefs=(),
            broken_xrefs=(),
            malformed_markup=(),
            accessibility_issues=(),
        )

    html_files, html_path_issues = _safe_html_paths(root, web_dir)
    missing: list[str] = []
    raw_xrefs: list[str] = []
    broken_xrefs: list[str] = []
    malformed_markup: list[str] = []
    malformed_markup.extend(html_path_issues)
    accessibility_issues: list[str] = []
    complex_requirements, registry_issues = _complex_figure_requirements(root)
    accessibility_issues.extend(registry_issues)
    observed_complex_labels: set[str] = set()
    assets_checked = 0
    target_text_cache: dict[Path, str] = {}
    target_read_errors: dict[Path, str] = {}
    target_html_ids: dict[Path, set[str]] = {}
    for html_path in html_files:
        try:
            text = html_path.read_bytes().decode("utf-8")
        except (OSError, UnicodeError) as exc:
            accessibility_issues.append(f"{html_path}: accessibility parse failed: {exc}")
            malformed_markup.append(f"{html_path}: unreadable HTML: {exc}")
            continue
        try:
            tokens = _tokenize_html(text)
        except _HTMLTokenizationError as exc:
            malformed_markup.append(f"{html_path}: {exc}")
            continue
        page_issues, page_complex_labels = _accessibility_issues_in(
            html_path,
            text,
            complex_requirements,
        )
        accessibility_issues.extend(page_issues)
        observed_complex_labels.update(page_complex_labels)
        raw_xrefs.extend(_raw_xrefs_in(html_path, text, tokens))
        malformed_markup.extend(_accessible_name_markup_in(html_path, text, tokens))
        malformed_markup.extend(_document_base_issues(html_path, text, tokens))
        malformed_markup.extend(
            _pattern_lines_in_text_tokens(
                html_path,
                text,
                tokens,
                _LEAKED_FIGURE_RE,
                "leaked Markdown figure syntax",
            )
        )
        malformed_markup.extend(
            _pattern_lines_in_text_tokens(
                html_path,
                text,
                tokens,
                _UNRESOLVED_TOKEN_RE,
                "unresolved manuscript token",
                include_attributes=True,
            )
        )
        for srcdoc in _attribute_occurrences(tokens, frozenset({"srcdoc"})):
            line_number = _line_number_at(text, srcdoc.start)
            malformed_markup.append(
                f"{html_path}:{line_number}: embedded iframe srcdoc is not allowed"
            )
        for occurrence, url, resource_offset in _resource_urls(tokens):
            line_number = _line_number_at(text, resource_offset)
            asset, resource_issue = _resource_target(web_dir, html_path, url)
            if resource_issue is not None:
                malformed_markup.append(
                    f"{html_path}:{line_number}: {url} ({resource_issue})"
                )
                continue
            script_issue = _external_script_issue(occurrence, url)
            if script_issue is not None:
                malformed_markup.append(
                    f"{html_path}:{line_number}: {url} ({script_issue})"
                )
                continue
            if asset is not None:
                assets_checked += 1
            if asset is not None and not asset.exists():
                missing.append(f"{html_path}:{line_number}: {url} -> {asset}")
        for href in _attribute_occurrences(tokens, frozenset({"href"})):
            url = href.value
            line_number = _line_number_at(text, href.start)
            parsed = urlparse(unescape(url))
            target_file, target_issue = _local_href_target(
                web_dir.parent,
                html_path,
                url,
            )
            if target_issue is not None:
                broken_xrefs.append(f"{html_path}:{line_number}: {url} ({target_issue})")
                continue
            if target_file is None:
                continue
            fragment = unquote(parsed.fragment)
            if fragment:
                target_key = target_file.resolve()
                if target_key not in target_text_cache and target_key not in target_read_errors:
                    try:
                        target_text_cache[target_key] = target_file.read_bytes().decode("utf-8")
                    except (OSError, UnicodeError) as exc:
                        target_read_errors[target_key] = str(exc)
                    else:
                        if target_file.suffix.casefold() in {".htm", ".html"}:
                            try:
                                target_html_ids[target_key] = _xref_ids(
                                    _tokenize_html(target_text_cache[target_key])
                                )
                            except _HTMLTokenizationError:
                                target_html_ids[target_key] = set()
                if target_key in target_read_errors:
                    broken_xrefs.append(
                        f"{html_path}:{line_number}: {url} "
                        f"(fragment target is unreadable: {target_read_errors[target_key]})"
                    )
                else:
                    if not _fragment_exists(
                        target_file,
                        target_text_cache[target_key],
                        fragment,
                        html_ids=target_html_ids.get(target_key),
                    ):
                        broken_xrefs.append(
                            f"{html_path}:{line_number}: {url} (missing fragment)"
                        )

    required_complex_labels = {requirement.label for requirement in complex_requirements.values()}
    for missing_label in sorted(required_complex_labels - observed_complex_labels):
        accessibility_issues.append(
            f"{web_dir}: registry-declared complex figure {missing_label} has no "
            "associated labeled details long description"
        )

    return WebPackageValidation(
        html_files=len(html_files),
        assets_checked=assets_checked,
        missing_assets=tuple(sorted(set(missing))),
        raw_xrefs=tuple(sorted(set(raw_xrefs))),
        broken_xrefs=tuple(sorted(set(broken_xrefs))),
        malformed_markup=tuple(sorted(set(malformed_markup))),
        accessibility_issues=tuple(sorted(set(accessibility_issues))),
    )
