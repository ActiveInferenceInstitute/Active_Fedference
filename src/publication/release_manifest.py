"""Release manifest and checksum package (MED-1).

Builds a reproducible release bundle at ``output/release/``:

* ``manifest.json`` — one entry per artifact: relative path, byte size, SHA-256.
* ``sha256sums.txt`` — the same digests in ``sha256sum -c`` format.
* ``README.md`` — a provenance one-pager whose counts are derived from the
  walk itself (never hand-typed).

The artifact set is every file under the release roots below, excluding the
bundle's own directory and log files. :func:`verify_release` recomputes every
digest against the shipped manifest, so a tampered or stale artifact fails
loudly (``sha256sum -c`` compatible).

Provenance fingerprint (MIN-1): the manifest additionally records a
``fingerprint`` — a SHA-256 over the sorted ``(path, content-sha256)`` set of
the declared source, manuscript, documentation, configuration, dependency-lock,
and pipeline-script inputs (``fingerprint_inputs``). This binds the bundle to
the tree and producer code that generated it, including prose and references
that can change rendered claims, so a bundle that is internally consistent
(every byte digest matches) but was built from a different source/config state
still fails :func:`verify_release`. The fingerprint is deterministic: no
timestamps or file metadata enter the hash. The manifest also records the
pipeline profile and generator version so a reviewer can identify how the
snapshot was produced. Unreleased builds omit ``generated_at`` by default so
two builds from the same evidence tree are byte-identical. An approved release
may supply an explicit UTC timestamp; the CLI also translates the standard
reproducible-build variable ``SOURCE_DATE_EPOCH`` into that field.

This module is the byte-level manifest primitive. The release-facing CLI first
requires fresh publication-profile analysis, test/coverage, hydration, and
render receipts; checksum verification alone is not scientific evidence or
external publication authorization.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Increment when the manifest metadata contract changes incompatibly.
RELEASE_MANIFEST_SCHEMA_VERSION = 3
RELEASE_GENERATOR_VERSION = "4"

_UTC_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

#: Directories (relative to the project root) whose files enter the bundle.
RELEASE_ROOTS: tuple[str, ...] = (
    "output/pdf",
    "output/figures",
    "output/reports",
    "output/slides",
    "output/web",
    "output/docs",
    "output/data",
    "output/manuscript",
)

#: Single files included alongside the artifact trees.
RELEASE_FILES: tuple[str, ...] = (
    "LICENSE",
    "CITATION.cff",
    ".zenodo.json",
    "codemeta.json",
    "coverage_project.json",
)

#: File suffixes excluded from the bundle (transient/log noise).
_EXCLUDED_SUFFIXES: tuple[str, ...] = (".log", ".aux", ".bbl", ".blg", ".out", ".toc")

#: Declared source/config/producer inputs of the provenance fingerprint, as glob
#: patterns relative to the project root. Recorded verbatim in the manifest as
#: ``fingerprint_inputs`` so the fingerprint boundary is self-documenting.
#: Mutable output directories are deliberately outside this set.
FINGERPRINT_INPUTS: tuple[str, ...] = (
    "LICENSE",
    "README.md",
    "_fedference_build_backend.py",
    "MANIFEST.in",
    "src/**/*.py",
    "tests/**/*.py",
    "scripts/**/*.py",
    "manuscript/**/*.md",
    "manuscript/**/*.bib",
    "manuscript/**/*.tex",
    "manuscript/**/*.yaml",
    "docs/**/*.md",
    ".github/workflows/*.yml",
    "experiment_plan.yaml",
    "pyproject.toml",
    "uv.lock",
    "ISA.md",
    "TODO.md",
    "REDTEAM_REVIEW.md",
)

#: Path components that never count as fingerprint inputs (build/cache noise).
_FINGERPRINT_EXCLUDED_PARTS: tuple[str, ...] = ("__pycache__",)
_FINGERPRINT_EXCLUDED_NAMES: tuple[str, ...] = ("AGENTS.md",)


def _project_root(project_root: Path | None) -> Path:
    return project_root or Path(__file__).resolve().parent.parent.parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_artifacts(root: Path) -> list[Path]:
    files: list[Path] = []
    for rel in RELEASE_ROOTS:
        base = root / rel
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix in _EXCLUDED_SUFFIXES:
                continue
            if "release" in path.relative_to(root).parts:
                continue
            files.append(path)
    for rel in RELEASE_FILES:
        path = root / rel
        if path.exists():
            files.append(path)
    return files


def _iter_fingerprint_inputs(root: Path) -> list[tuple[str, Path]]:
    """Resolve ``FINGERPRINT_INPUTS`` to ``(relative-posix-path, path)`` pairs.

    Sorted by relative path; cache directories (``__pycache__``) and anything
    under ``*.egg-info`` are excluded so only tracked source/config content
    enters the fingerprint.
    """
    seen: dict[str, Path] = {}
    for pattern in FINGERPRINT_INPUTS:
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            parts = path.relative_to(root).parts
            if (
                any(part in _FINGERPRINT_EXCLUDED_PARTS or part.endswith(".egg-info") for part in parts)
                or path.name in _FINGERPRINT_EXCLUDED_NAMES
            ):
                continue
            seen[path.relative_to(root).as_posix()] = path
    return sorted(seen.items())


def _fingerprint_file_digests(root: Path) -> dict[str, str]:
    """Return the individual input digests recorded for diagnostics."""
    return {relative: _sha256(path) for relative, path in _iter_fingerprint_inputs(root)}


def _package_version(root: Path) -> str | None:
    """Read the PEP 621 project version without importing the package."""
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        return None
    content = pyproject.read_text(encoding="utf-8")
    project_match = re.search(r"(?ms)^\[project\]\s*(.*?)(?=^\[|\Z)", content)
    if project_match is None:
        return None
    version_match = re.search(r"(?m)^version\s*=\s*['\"]([^'\"]+)['\"]", project_match.group(1))
    return version_match.group(1) if version_match else None


def compute_fingerprint(project_root: Path | None = None) -> str:
    """SHA-256 over the sorted ``(path, content-sha256)`` set of the inputs.

    Deterministic by construction: only relative POSIX paths and content
    digests enter the hash — no timestamps, sizes, or permissions.
    """
    root = _project_root(project_root)
    digest = hashlib.sha256()
    for rel, file_digest in _fingerprint_file_digests(root).items():
        digest.update(f"{file_digest}  {rel}\n".encode("utf-8"))
    return digest.hexdigest()


def validate_utc_timestamp(timestamp: str | None) -> str | None:
    """Validate an optional canonical UTC timestamp.

    ``None`` is the deterministic default for unreleased reviewer snapshots.
    A populated value must be second-resolution UTC in the manifest's
    canonical form, rather than a locale-dependent or offset-bearing spelling.
    """
    if timestamp is None:
        return None
    if not isinstance(timestamp, str) or not timestamp:
        raise ValueError("timestamp must be a non-empty UTC string or None")
    try:
        parsed = datetime.strptime(timestamp, _UTC_TIMESTAMP_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError("timestamp must use canonical UTC form YYYY-MM-DDTHH:MM:SSZ") from exc
    if parsed.strftime(_UTC_TIMESTAMP_FORMAT) != timestamp:
        raise ValueError("timestamp must use canonical UTC form YYYY-MM-DDTHH:MM:SSZ")
    return timestamp


def timestamp_from_source_date_epoch(value: str) -> str:
    """Convert ``SOURCE_DATE_EPOCH`` seconds to the canonical UTC timestamp."""
    normalized = value.strip()
    if not normalized or not normalized.isascii() or not normalized.isdecimal():
        raise ValueError("SOURCE_DATE_EPOCH must be a nonnegative integer")
    if len(normalized) > 20:
        raise ValueError("SOURCE_DATE_EPOCH is outside the supported range")
    epoch = int(normalized)
    try:
        timestamp = datetime.fromtimestamp(epoch, timezone.utc)
    except (OverflowError, OSError, ValueError) as exc:
        raise ValueError("SOURCE_DATE_EPOCH is outside the supported range") from exc
    return timestamp.strftime(_UTC_TIMESTAMP_FORMAT)


def build_release(
    project_root: Path | None = None,
    *,
    timestamp: str | None = None,
    profile: str = "publication",
) -> dict[str, Any]:
    """Write ``output/release/`` and return the manifest mapping.

    The default ``timestamp=None`` is intentional: an unreleased reviewer
    snapshot has no release time and repeated builds must be byte-identical.
    Release tooling may pass a canonical UTC timestamp after approval. This
    low-level byte-manifest function does not itself establish a fresh evidence
    chain; use ``scripts/build_release.py`` for that guarded operation.
    """
    root = _project_root(project_root)
    if not profile.strip():
        raise ValueError("release pipeline profile must be non-empty")
    release_dir = root / "output" / "release"
    release_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for path in _iter_artifacts(root):
        rel = path.relative_to(root).as_posix()
        entries.append({"path": rel, "bytes": path.stat().st_size, "sha256": _sha256(path)})

    stamp = validate_utc_timestamp(timestamp)
    fingerprint_files = _fingerprint_file_digests(root)
    manifest: dict[str, Any] = {
        "manifest_schema_version": RELEASE_MANIFEST_SCHEMA_VERSION,
        "generated_at": stamp,
        "timestamp_policy": "recorded" if stamp is not None else "omitted",
        "pipeline_profile": profile,
        "generator": "src/publication/release_manifest.py",
        "generator_version": RELEASE_GENERATOR_VERSION,
        "package_version": _package_version(root),
        "n_artifacts": len(entries),
        "total_bytes": sum(e["bytes"] for e in entries),
        "fingerprint": compute_fingerprint(root),
        "fingerprint_inputs": list(FINGERPRINT_INPUTS),
        "fingerprint_files": fingerprint_files,
        "artifacts": entries,
    }
    (release_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (release_dir / "sha256sums.txt").write_text(
        "".join(f"{e['sha256']}  {e['path']}\n" for e in entries), encoding="utf-8"
    )

    by_root: dict[str, int] = {}
    for e in entries:
        rel = str(e["path"])
        top = "/".join(rel.split("/")[:2]) if rel.startswith("output/") else rel
        by_root[top] = by_root.get(top, 0) + 1
    lines = [
        "# Active Fedference — release bundle provenance",
        "",
        (
            f"Generated at: {stamp} by `src/publication/release_manifest.py`"
            if stamp is not None
            else "Generated at: omitted for a byte-reproducible unreleased build."
        ),
        "(invoked via `uv run python scripts/build_release.py`).",
        "",
        f"Pipeline profile: `{profile}`; generator version: `{RELEASE_GENERATOR_VERSION}`.",
        "",
        f"Artifacts: {len(entries)} files, {manifest['total_bytes']} bytes, over:",
        "",
    ]
    lines += [f"- `{k}`: {v} file(s)" for k, v in sorted(by_root.items())]
    lines += [
        "",
        "Verify integrity from the project root:",
        "",
        "```bash",
        "shasum -a 256 -c output/release/sha256sums.txt",
        "# or: uv run python scripts/build_release.py --verify",
        "```",
        "",
        "Scientific reports are regenerated under the seeds in",
        "`manuscript/config.yaml`; rendered containers are rebuilt by the pinned",
        "publication toolchain. The manifest is re-derived on each build and",
        "records the actual bytes; it is never hand-edited.",
        "",
    ]
    (release_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return manifest


def verify_release(project_root: Path | None = None) -> list[str]:
    """Verify the exact artifact set and every digest in ``manifest.json``.

    A missing manifest raises. A missing, resized, altered, duplicated, or
    unexpected artifact is a mismatch; metadata counts are checked too. Empty
    return == bundle verified. Exact-set checking matters because digesting all
    listed files alone cannot detect an unreviewed file added to a publication
    bundle.

    Beyond byte digests, the recorded provenance ``fingerprint`` and individual
    ``fingerprint_files`` are recomputed from the CURRENT declared input tree:
    a bundle that is internally consistent but was built from a different
    source/config/producer state is stale and fails with an explicit diagnostic
    naming changed inputs where available. This function intentionally verifies
    byte/source consistency only; the CLI performs receipt-chain preflight and
    neither operation substitutes for external release approval.
    """
    root = _project_root(project_root)
    manifest_path = root / "output" / "release" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bad: list[str] = []
    entries = manifest.get("artifacts")
    if not isinstance(entries, list):
        return ["manifest: malformed artifacts list"]

    if manifest.get("manifest_schema_version") != RELEASE_MANIFEST_SCHEMA_VERSION:
        bad.append("manifest: unsupported or missing manifest_schema_version")
    generated_at_present = "generated_at" in manifest
    generated_at = manifest.get("generated_at")
    timestamp_policy = manifest.get("timestamp_policy")
    if not generated_at_present:
        bad.append("manifest: generated_at field missing")
    elif generated_at is None:
        if timestamp_policy != "omitted":
            bad.append("manifest: omitted generated_at requires timestamp_policy=omitted")
    elif isinstance(generated_at, str):
        try:
            validate_utc_timestamp(generated_at)
        except ValueError:
            bad.append("manifest: generated_at is not canonical UTC")
        if timestamp_policy != "recorded":
            bad.append("manifest: populated generated_at requires timestamp_policy=recorded")
    else:
        bad.append("manifest: generated_at must be a canonical UTC string or null")
    if not isinstance(manifest.get("pipeline_profile"), str) or not manifest["pipeline_profile"].strip():
        bad.append("manifest: pipeline_profile missing")
    if manifest.get("generator") != "src/publication/release_manifest.py":
        bad.append("manifest: generator identity drift")
    if manifest.get("generator_version") != RELEASE_GENERATOR_VERSION:
        bad.append("manifest: generator_version drift")

    listed: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            bad.append("manifest: malformed artifact entry")
            continue
        raw_path = str(entry.get("path", ""))
        if raw_path in listed:
            bad.append(f"manifest: duplicate artifact {raw_path}")
        listed.add(raw_path)
        relative = Path(raw_path)
        if not raw_path or relative.is_absolute() or ".." in relative.parts:
            bad.append(f"manifest: unsafe artifact path {raw_path}")
            continue
        path = root / relative
        raw_bytes = entry.get("bytes")
        expected_bytes = (
            raw_bytes
            if isinstance(raw_bytes, int) and not isinstance(raw_bytes, bool) and raw_bytes >= 0
            else None
        )
        if expected_bytes is None:
            bad.append(f"manifest: invalid bytes metadata {raw_path}")
        if (
            expected_bytes is None
            or not path.is_file()
            or path.stat().st_size != expected_bytes
            or _sha256(path) != entry.get("sha256")
        ):
            bad.append(raw_path)

    actual = {path.relative_to(root).as_posix() for path in _iter_artifacts(root)}
    for unexpected in sorted(actual - listed):
        bad.append(f"manifest: unexpected artifact {unexpected}")
    if manifest.get("n_artifacts") != len(entries):
        bad.append("manifest: n_artifacts mismatch")
    declared_bytes = sum(
        entry["bytes"]
        for entry in entries
        if isinstance(entry, dict)
        and isinstance(entry.get("bytes"), int)
        and not isinstance(entry.get("bytes"), bool)
        and entry["bytes"] >= 0
    )
    if manifest.get("total_bytes") != declared_bytes or any(
        not (
            isinstance(entry, dict)
            and isinstance(entry.get("bytes"), int)
            and not isinstance(entry.get("bytes"), bool)
            and entry["bytes"] >= 0
        )
        for entry in entries
    ):
        bad.append("manifest: total_bytes mismatch")

    recorded = manifest.get("fingerprint")
    current_files = _fingerprint_file_digests(root)
    current = compute_fingerprint(root)
    if not isinstance(recorded, str) or not recorded:
        bad.append("manifest: provenance fingerprint missing")
    recorded_files = manifest.get("fingerprint_files")
    if not isinstance(recorded_files, dict):
        bad.append("manifest: fingerprint_files missing")
    else:
        recorded_map = {str(key): str(value) for key, value in recorded_files.items()}
        changed_inputs = sorted(
            path
            for path in set(recorded_map) | set(current_files)
            if recorded_map.get(path) != current_files.get(path)
        )
        if recorded != current:
            recorded_prefix = recorded[:12] if isinstance(recorded, str) else "<missing>"
            bad.append(
                "manifest: provenance fingerprint mismatch — bundle records "
                f"{recorded_prefix} but the current declared input tree computes {current[:12]} "
                f"(changed inputs: {', '.join(changed_inputs[:20]) or 'not available'}); "
                "the bundle is stale relative to the tree — rebuild with scripts/build_release.py"
            )
        elif changed_inputs:
            bad.append(
                "manifest: fingerprint_files mismatch — changed inputs: " + ", ".join(changed_inputs[:20])
            )
    if (
        isinstance(recorded, str)
        and recorded
        and recorded != current
        and not isinstance(recorded_files, dict)
    ):
        bad.append(
            "manifest: provenance fingerprint mismatch — bundle records "
            f"{recorded[:12]} but the current declared input tree computes {current[:12]} "
            "(changed inputs unavailable); the bundle is stale relative to the tree — "
            "rebuild with scripts/build_release.py"
        )
    if manifest.get("fingerprint_inputs") != list(FINGERPRINT_INPUTS):
        bad.append("manifest: fingerprint_inputs drift from declared input set")
    return bad
