"""Normalization for source-preserving HTML publication."""
from __future__ import annotations

import stat
from pathlib import Path

from publication.web.citations import _citation_labels
from publication.web.models import _HTMLDocument, _HTMLRollbackError, _HTMLTokenizationError, _StagedHTMLWrite
from publication.web.paths import _root, _safe_html_paths, _web_dir, _xref_target_prefix
from publication.web.reference_labels import _xref_ids, _xref_labels
from publication.web.reference_markup import _normalize_html_tokens
from publication.web.tokenizer import _tokenize_html
from publication.web.transaction import _commit_html_writes
from publication.web.transaction_files import _stage_bytes


def normalize_web_xrefs(project_root: str | Path | None = None) -> int:
    """Atomically replace visible references with self-contained HTML links."""
    root = _root(project_root)
    web_dir = _web_dir(root)
    if not web_dir.exists():
        raise FileNotFoundError(f"missing web output directory: {web_dir}")

    html_paths, path_issues = _safe_html_paths(root, web_dir)
    if path_issues:
        raise ValueError("; ".join(path_issues))

    citation_labels = _citation_labels(root)
    documents: list[_HTMLDocument] = []
    for html_path in html_paths:
        source_stat = html_path.lstat()
        if not stat.S_IMODE(source_stat.st_mode) & 0o222:
            raise PermissionError(f"{html_path}: HTML input is not writable")
        original = html_path.read_bytes()
        try:
            text = original.decode("utf-8")
            tokens = _tokenize_html(text)
        except (UnicodeError, _HTMLTokenizationError) as exc:
            raise ValueError(f"{html_path}: {exc}") from exc
        documents.append(
            _HTMLDocument(
                path=html_path,
                original=original,
                text=text,
                tokens=tokens,
                source_stat=source_stat,
            )
        )

    index_document = next(
        (document for document in documents if document.path == web_dir / "index.html"),
        None,
    )
    index_tokens = () if index_document is None else index_document.tokens
    xref_labels = _xref_labels(
        web_dir,
        index_text=None if index_document is None else index_document.text,
        index_tokens=index_tokens,
    )
    xref_ids = _xref_ids(index_tokens)

    normalized_documents: list[tuple[_HTMLDocument, bytes]] = []
    replacements = 0
    for document in documents:
        updated, document_count = _normalize_html_tokens(
            document.tokens,
            xref_labels=xref_labels,
            citation_labels=citation_labels,
            xref_ids=xref_ids,
            target_prefix=_xref_target_prefix(web_dir, document.path),
        )
        normalized_documents.append((document, updated.encode("utf-8")))
        replacements += document_count

    staged: list[_StagedHTMLWrite] = []
    temporary_paths: set[Path] = set()
    retained_paths: set[Path] = set()
    try:
        for document, updated_bytes in normalized_documents:
            if updated_bytes == document.original:
                continue
            replacement_path = _stage_bytes(
                document.path,
                updated_bytes,
                document.source_stat.st_mode,
                "replacement",
            )
            temporary_paths.add(replacement_path)
            replacement_stat = replacement_path.lstat()
            rollback_path = _stage_bytes(
                document.path,
                document.original,
                document.source_stat.st_mode,
                "rollback",
            )
            temporary_paths.add(rollback_path)
            staged.append(
                _StagedHTMLWrite(
                    document=document,
                    replacement_path=replacement_path,
                    replacement_bytes=updated_bytes,
                    replacement_stat=replacement_stat,
                    rollback_path=rollback_path,
                )
            )
        try:
            _commit_html_writes(staged)
        except _HTMLRollbackError as exc:
            retained_paths.update(exc.retained_paths)
            raise
    finally:
        for temporary_path in temporary_paths - retained_paths:
            temporary_path.unlink(missing_ok=True)
    return replacements
