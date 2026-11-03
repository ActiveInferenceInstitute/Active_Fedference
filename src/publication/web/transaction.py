"""Transaction for source-preserving HTML publication."""
from __future__ import annotations

import os
from pathlib import Path

from publication.web.models import _HTMLRollbackError, _StagedHTMLWrite
from publication.web.transaction_files import (
    _document_is_unchanged,
    _link_no_clobber,
    _path_is_captured_original,
    _path_is_staged_replacement,
)
from publication.web.transaction_rollback import _restore_committed_html


def _commit_html_writes(staged: list[_StagedHTMLWrite]) -> None:
    """Commit with no-clobber links and restore originals after a later failure."""
    committed: list[_StagedHTMLWrite] = []
    try:
        for item in staged:
            if not _document_is_unchanged(item.document):
                raise RuntimeError(
                    f"{item.document.path}: HTML input changed during normalization"
                )

            # Atomically capture the current destination first. A change in the
            # former check-to-replace window is now moved into ``rollback_path``
            # and compared before any replacement can be installed.
            try:
                os.replace(item.document.path, item.rollback_path)
            except BaseException as exc:
                restored = False
                link_exception: BaseException | None = None
                if not os.path.lexists(item.document.path):
                    restored, link_exception = _link_no_clobber(
                        item.rollback_path,
                        item.document.path,
                    )
                raise _HTMLRollbackError(
                    f"{item.document.path}: forward capture was interrupted or failed: {exc}; "
                    f"destination restored without clobber={restored}; "
                    f"restoration signal={link_exception}; recovery files retained at "
                    f"{item.replacement_path} and {item.rollback_path}",
                    frozenset({item.replacement_path, item.rollback_path}),
                ) from exc

            try:
                captured_original = _path_is_captured_original(
                    item,
                    item.rollback_path,
                )
            except BaseException as exc:
                raise _HTMLRollbackError(
                    f"{item.document.path}: forward capture inspection was interrupted: {exc}; "
                    f"recovery files retained at {item.replacement_path} and "
                    f"{item.rollback_path}",
                    frozenset({item.replacement_path, item.rollback_path}),
                ) from exc
            if not captured_original:
                restored, link_exception = _link_no_clobber(
                    item.rollback_path,
                    item.document.path,
                )
                if restored and link_exception is None:
                    raise RuntimeError(
                        f"{item.document.path}: HTML input changed during normalization"
                    )
                raise _HTMLRollbackError(
                    f"{item.document.path}: changed input could not be restored cleanly; "
                    f"restored={restored}; restoration signal={link_exception}; "
                    f"recovery files retained at {item.replacement_path} and "
                    f"{item.rollback_path}",
                    frozenset({item.replacement_path, item.rollback_path}),
                ) from link_exception

            installed, link_exception = _link_no_clobber(
                item.replacement_path,
                item.document.path,
            )
            if not installed:
                restored, restore_exception = _link_no_clobber(
                    item.rollback_path,
                    item.document.path,
                )
                if restored and restore_exception is None and link_exception is not None:
                    raise link_exception
                raise _HTMLRollbackError(
                    f"{item.document.path}: no-clobber forward install failed; "
                    f"signal={link_exception}; original restored={restored}; "
                    f"restoration signal={restore_exception}; recovery files retained at "
                    f"{item.replacement_path} and {item.rollback_path}",
                    frozenset({item.replacement_path, item.rollback_path}),
                ) from link_exception
            committed.append(item)
            item.replacement_path.unlink()
            if link_exception is not None:
                raise link_exception

        for item in committed:
            if not _path_is_staged_replacement(item, item.document.path):
                raise RuntimeError(
                    f"{item.document.path}: HTML output changed during normalization"
                )
    except BaseException as exc:
        rollback_failures: list[str] = []
        retained_paths: set[Path] = set()
        if isinstance(exc, _HTMLRollbackError):
            rollback_failures.append(str(exc))
            retained_paths.update(exc.retained_paths)
        for item in reversed(committed):
            rollback_failure, item_retained_paths = _restore_committed_html(item)
            if rollback_failure is not None:
                rollback_failures.append(rollback_failure)
                retained_paths.update(item_retained_paths)
        if rollback_failures:
            if isinstance(exc, _HTMLRollbackError) and len(rollback_failures) == 1:
                raise
            raise _HTMLRollbackError(
                "HTML normalization failed and rollback was incomplete: "
                + "; ".join(rollback_failures),
                frozenset(retained_paths),
            ) from exc
        raise
