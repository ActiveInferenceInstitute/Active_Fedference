"""Research run artifact verification."""

from __future__ import annotations

from pathlib import Path

from .hashing import canonical_sha256, sha256_file
from .run_schema import RunReceipt
from .source_verification import _git_revision
from .strict_json import _loads_json_strict


def verify_run_receipt(
    receipt: RunReceipt,
    *,
    root: str | Path,
    require_clean_git: bool = False,
    project_root: str | Path | None = None,
) -> tuple[str, ...]:
    """Return exact provenance/artifact findings; empty means verified.

    Development receipts may bind a dirty tree and still verify their config and
    output bytes. Publication callers should pass ``require_clean_git=True`` and
    ``project_root`` so both the recorded and live source tree, commit, and
    environment lock are checked.
    """
    findings: list[str] = []
    if receipt.status != "completed":
        findings.append(f"run status is {receipt.status!r}, not 'completed'")
    if require_clean_git and receipt.git_tree_state != "clean":
        findings.append(f"git tree state is {receipt.git_tree_state!r}, not 'clean'")
    if project_root is not None:
        source_root = Path(project_root).resolve()
        live_commit, live_state = _git_revision(source_root)
        if live_commit == "unavailable" or live_state == "unavailable":
            findings.append(f"live Git state is unavailable: {source_root}")
        else:
            if live_commit != receipt.git_commit:
                findings.append(
                    "live Git commit does not match receipt: "
                    f"{live_commit} != {receipt.git_commit}"
                )
            if live_state != receipt.git_tree_state:
                findings.append(
                    "live Git tree state does not match receipt: "
                    f"{live_state!r} != {receipt.git_tree_state!r}"
                )
            if require_clean_git and live_state != "clean":
                findings.append(f"live Git tree state is {live_state!r}, not 'clean'")
            if receipt.git_tree_state != "clean":
                findings.append(
                    "receipt binds an unhashed dirty tree; live source equivalence "
                    "cannot be verified"
                )
        lock_path = source_root / "uv.lock"
        if not lock_path.is_file():
            findings.append(f"live environment lock is missing: {lock_path}")
        elif sha256_file(lock_path) != receipt.environment_lock_sha256:
            findings.append("live environment lock digest does not match receipt")
    root_path = Path(root).resolve()
    for artifact in receipt.outputs:
        path = (root_path / artifact.path).resolve()
        try:
            path.relative_to(root_path)
        except ValueError:
            findings.append(f"artifact escapes receipt root: {artifact.path}")
            continue
        if not path.is_file():
            findings.append(f"missing artifact: {artifact.path}")
            continue
        if path.stat().st_size != artifact.bytes:
            findings.append(f"artifact byte-size mismatch: {artifact.path}")
        if sha256_file(path) != artifact.sha256:
            findings.append(f"artifact digest mismatch: {artifact.path}")
    config_records = [record for record in receipt.outputs if record.name == "config"]
    if len(config_records) == 1:
        config_path = (root_path / config_records[0].path).resolve()
        try:
            config_path.relative_to(root_path)
        except ValueError:
            pass
        else:
            if config_path.is_file():
                try:
                    config = _loads_json_strict(
                        config_path.read_text(encoding="utf-8")
                    )
                except (OSError, ValueError):
                    findings.append(f"config artifact is not valid JSON: {config_records[0].path}")
                else:
                    if canonical_sha256(config) != receipt.config_sha256:
                        findings.append(f"configuration hash mismatch: {config_records[0].path}")
    return tuple(findings)

