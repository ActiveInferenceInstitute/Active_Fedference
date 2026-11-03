"""Transaction rollback for source-preserving HTML publication."""
from __future__ import annotations

import os
from pathlib import Path

from publication.web.models import _StagedHTMLWrite
from publication.web.transaction_files import _link_no_clobber, _path_is_staged_replacement


def _restore_committed_html(
    item: _StagedHTMLWrite,
) -> tuple[str | None, frozenset[Path]]:
    """Restore one commit without replacing a concurrently changed destination.

    Move the current destination into the consumed replacement staging name,
    then use a no-clobber hard link to reinstall either the original bytes or,
    when the destination changed, the concurrent bytes. Any incomplete or
    concurrency-preserving outcome retains both available recovery files.
    """
    retained_paths = frozenset({item.replacement_path, item.rollback_path})
    try:
        os.replace(item.document.path, item.replacement_path)
    except BaseException as exc:
        restored = False
        link_exception: BaseException | None = None
        if not os.path.lexists(item.document.path):
            restore_source = (
                item.rollback_path
                if _path_is_staged_replacement(item, item.replacement_path)
                else item.replacement_path
            )
            restored, link_exception = _link_no_clobber(
                restore_source,
                item.document.path,
            )
        return (
            f"{item.document.path}: rollback move was interrupted or failed: {exc}; "
            f"destination restored without clobber={restored}; "
            f"restoration signal={link_exception}; current bytes retained at "
            f"{item.replacement_path}; original bytes retained at {item.rollback_path}",
            retained_paths,
        )

    try:
        destination_changed = not _path_is_staged_replacement(
            item,
            item.replacement_path,
        )
    except BaseException as exc:
        return (
            f"{item.document.path}: rollback inspection was interrupted: {exc}; "
            f"current bytes retained at {item.replacement_path}; "
            f"original bytes retained at {item.rollback_path}",
            retained_paths,
        )
    restore_source = item.replacement_path if destination_changed else item.rollback_path
    restored, link_exception = _link_no_clobber(
        restore_source,
        item.document.path,
    )
    if not restored or link_exception is not None:
        return (
            f"{item.document.path}: no-clobber restoration was interrupted or failed: "
            f"{link_exception}; restored={restored}; "
            f"current bytes retained at {item.replacement_path}; "
            f"original bytes retained at {item.rollback_path}",
            retained_paths,
        )

    if destination_changed:
        return (
            f"{item.document.path}: destination changed concurrently after commit; "
            f"concurrent bytes were preserved at the destination and {item.replacement_path}; "
            f"original bytes retained at {item.rollback_path}",
            retained_paths,
        )
    return None, frozenset()
