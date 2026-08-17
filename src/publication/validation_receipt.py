"""Source-bound receipts for a successful full test-and-coverage gate.

The publication pipeline has to hydrate the test-count, coverage, and runtime
environment tokens from a completed gate rather than from a new ad-hoc test
collection performed during rendering.  This module records that gate after a
full successful run and binds it to both the current validation inputs and the
already-fresh analysis receipt it will support.

The receipt deliberately remains outside ``PIPELINE_STAGES``.  The complete
test suite runs after a provisional hydrate/render pass, so making it a formal
hydration dependency would form a cycle.  Final hydration instead validates
this standalone receipt before it writes anything and fingerprints the receipt
as a direct hydration input.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable, Mapping

from publication.pipeline_freshness import (
    pipeline_stage_record,
    validate_pipeline_freshness,
)
from publication.release_manifest import validate_utc_timestamp

VALIDATION_RECEIPT_SCHEMA_VERSION = 2
VALIDATION_RECEIPT_PATH = Path("output/data/test_coverage_receipt.json")
VALIDATION_INPUT_PATTERNS: tuple[str, ...] = (
    "src/**/*.py",
    "src/**/*.md",
    "tests/**/*.py",
    "tests/**/*.md",
    "scripts/**/*.py",
    "scripts/**/*.md",
    "data/**/*.md",
    "docs/**/*.md",
    ".github/workflows/*.yml",
    # Final hydration consumes every manuscript source format below.  Binding
    # them here prevents a post-test caption, formalism, bibliography, or
    # template-config edit from being released under an older test receipt.
    "manuscript/**/*.md",
    "manuscript/**/*.bib",
    "manuscript/**/*.yaml",
    "manuscript/**/*.tex",
    # Source-owned documentation and release/packaging metadata are validated
    # by the full suite too.  Bind them here so the receipt honestly describes
    # the tree whose documentation checks passed, without pulling generated
    # output, review scratch, environments, or caches into the boundary.
    "AGENTS.md",
    "README.md",
    "STANDALONE.md",
    "TODO.md",
    "REDTEAM_REVIEW.md",
    "ISA.md",
    ".gitignore",
    ".zenodo.json",
    "CITATION.cff",
    "codemeta.json",
    "MANIFEST.in",
    "LICENSE",
    "_fedference_build_backend.py",
    "domain_profile.yaml",
    "experiment_plan.yaml",
    "pyproject.toml",
    "uv.lock",
)
# The repository currently has no manuscript ``.tex`` source, but hydration
# would consume one if it were added.  Keep the pattern in the contract so a
# later addition invalidates an older receipt, without making today's release
# workflow fail solely because that optional source class is absent.
# These files are part of the source-bound contract when present.  Keeping
# them optional preserves the fixture trees used by focused receipt tests,
# while a real checkout still hashes the license as a release input.
_OPTIONAL_VALIDATION_INPUT_PATTERNS = frozenset({"manuscript/**/*.tex", "LICENSE"})
MINIMUM_COVERAGE_THRESHOLD = 90.0
_ENVIRONMENT_FIELDS: tuple[str, ...] = (
    "python_version",
    "python_implementation",
    "platform",
    "numpy_version",
    "scipy_version",
    "pytest_version",
)
_MACHINE_PATH_RE = re.compile(
    r"(?P<prefix>/private/tmp|/tmp|/Users|/home|/Volumes)/[^/\s\"'<>]+"
)
_MACHINE_PATH_REPLACEMENTS = {
    "/private/tmp": "<tmp>",
    "/tmp": "<tmp>",
    "/Users": "<home>",
    "/home": "<home>",
    "/Volumes": "<volume>",
}


class ValidationReceiptError(ValueError):
    """Raised when a validation receipt is absent, malformed, or stale."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _map_digest(file_hashes: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for relative, file_hash in sorted(file_hashes.items()):
        digest.update(f"{file_hash}  {relative}\n".encode("utf-8"))
    return digest.hexdigest()


def _canonicalize_command(command: Iterable[str]) -> list[str]:
    """Remove machine-specific prefixes from the durable command evidence.

    A receipt is committed and later passed through the web-package sanitizer.
    If the live command embeds an absolute interpreter or temporary-report path,
    that sanitizer would mutate the receipt after hydration and invalidate the
    source-bound pipeline fingerprint.  Canonicalizing at the write boundary
    keeps the receipt stable across local macOS, Linux CI, and external-volume
    checkouts while preserving the useful path suffix.
    """

    def replace(match: re.Match[str]) -> str:
        return _MACHINE_PATH_REPLACEMENTS[match.group("prefix")]

    return [_MACHINE_PATH_RE.sub(replace, part) for part in command]


def validation_input_hashes(project_root: Path) -> dict[str, str]:
    """Hash every declared test/producer/dependency-lock input, fail closed."""
    root = Path(project_root).resolve()
    files: dict[str, Path] = {}
    for pattern in VALIDATION_INPUT_PATTERNS:
        matches = [path for path in root.glob(pattern) if path.is_file()]
        if not matches:
            if pattern in _OPTIONAL_VALIDATION_INPUT_PATTERNS:
                continue
            raise ValidationReceiptError(f"validation input pattern matched no files: {pattern}")
        for path in matches:
            files[path.relative_to(root).as_posix()] = path
    return {relative: _sha256(path) for relative, path in sorted(files.items())}


def validation_environment() -> dict[str, str]:
    """Return the execution-environment fields surfaced in the manuscript."""

    def distribution_version(distribution: str) -> str:
        try:
            return version(distribution)
        except PackageNotFoundError:
            return "not installed"

    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": f"{platform.system()} {platform.machine()}",
        "numpy_version": distribution_version("numpy"),
        "scipy_version": distribution_version("scipy"),
        "pytest_version": distribution_version("pytest"),
    }


def _receipt_path(root: Path) -> Path:
    return root / VALIDATION_RECEIPT_PATH


def _load_receipt(root: Path) -> dict[str, Any]:
    path = _receipt_path(root)
    if not path.is_file():
        raise ValidationReceiptError(f"validation receipt is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationReceiptError(f"invalid validation receipt: {path}") from exc
    if not isinstance(payload, dict):
        raise ValidationReceiptError(f"invalid validation receipt shape: {path}")
    return payload


def _analysis_stage_digests(root: Path) -> dict[str, str]:
    findings = validate_pipeline_freshness(root, stages=("analysis",))
    if findings:
        raise ValidationReceiptError("analysis receipt is not fresh: " + "; ".join(findings))
    record = pipeline_stage_record(root, "analysis")
    digests: dict[str, str] = {}
    for key in ("input_digest", "output_digest"):
        value = record.get(key)
        if not isinstance(value, str) or not value:
            raise ValidationReceiptError(f"analysis receipt is missing a valid {key}")
        digests[key] = value
    return digests


def _validation_snapshot_from_mapping(
    payload: object,
    *,
    label: str,
) -> dict[str, Any]:
    """Validate and normalize one pre- or post-test validation snapshot."""
    if not isinstance(payload, Mapping):
        raise ValidationReceiptError(f"{label} validation snapshot must be a mapping")
    input_hashes = payload.get("input_hashes")
    if not isinstance(input_hashes, Mapping) or not input_hashes:
        raise ValidationReceiptError(f"{label} validation snapshot has invalid input hashes")
    normalized_hashes: dict[str, str] = {}
    for relative, digest in input_hashes.items():
        if not isinstance(relative, str) or not relative or not isinstance(digest, str) or not digest:
            raise ValidationReceiptError(f"{label} validation snapshot has invalid input hashes")
        normalized_hashes[relative] = digest
    expected_input_digest = _map_digest(normalized_hashes)
    if payload.get("input_digest") != expected_input_digest:
        raise ValidationReceiptError(f"{label} validation snapshot has an invalid input digest")

    analysis_stage = payload.get("analysis_stage")
    if not isinstance(analysis_stage, Mapping):
        raise ValidationReceiptError(f"{label} validation snapshot has invalid analysis-stage digests")
    normalized_analysis: dict[str, str] = {}
    for key in ("input_digest", "output_digest"):
        value = analysis_stage.get(key)
        if not isinstance(value, str) or not value:
            raise ValidationReceiptError(f"{label} validation snapshot has invalid analysis-stage digests")
        normalized_analysis[key] = value
    return {
        "input_hashes": dict(sorted(normalized_hashes.items())),
        "input_digest": expected_input_digest,
        "analysis_stage": normalized_analysis,
    }


def capture_validation_snapshot(project_root: Path) -> dict[str, Any]:
    """Capture a coherent source and analysis boundary immediately around a test run.

    A source-bound receipt cannot infer the tested tree after a long test suite
    has completed.  The caller captures this snapshot immediately before the
    suite; :func:`write_validation_receipt` captures the post-run state and
    refuses to attest it unless the two snapshots agree exactly.
    """
    root = Path(project_root).resolve()
    input_hashes = validation_input_hashes(root)
    analysis_stage = _analysis_stage_digests(root)
    # Detect a mutation while this small snapshot itself was being assembled.
    # The second map makes the returned input/analysis pair a coherent boundary
    # rather than an arbitrary mixture of pre- and post-edit values.
    if validation_input_hashes(root) != input_hashes:
        raise ValidationReceiptError(
            "validation inputs changed while capturing the test-run snapshot; retry the gate"
        )
    return {
        "input_hashes": input_hashes,
        "input_digest": _map_digest(input_hashes),
        "analysis_stage": analysis_stage,
    }


def _require_nonnegative_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationReceiptError(f"validation receipt {key} must be a non-negative integer")
    return value


def _require_number(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationReceiptError(f"validation receipt {key} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValidationReceiptError(f"validation receipt {key} must be finite")
    return numeric


def write_validation_receipt(
    project_root: Path,
    *,
    command: Iterable[str],
    test_summary: Mapping[str, int],
    coverage_percent: float,
    pre_run_snapshot: Mapping[str, Any],
    coverage_threshold: float = MINIMUM_COVERAGE_THRESHOLD,
    environment: Mapping[str, str] | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Atomically write a receipt for a successful, fresh full validation gate.

    The caller must only invoke this after its actual test process exits zero.
    This writer independently rejects failed summaries, a below-policy coverage
    threshold, insufficient achieved coverage, stale analysis evidence, or a
    changed source/analysis boundary while the test suite was running.
    """
    root = Path(project_root).resolve()
    command_parts = _canonicalize_command(command)
    if not command_parts or any(not isinstance(part, str) or not part for part in command_parts):
        raise ValidationReceiptError("validation receipt command must be non-empty strings")
    summary: dict[str, Any] = dict(test_summary)
    collected = _require_nonnegative_int(summary, "collected")
    passed = _require_nonnegative_int(summary, "passed")
    failed = _require_nonnegative_int(summary, "failed")
    skipped = _require_nonnegative_int(summary, "skipped")
    if collected != passed + failed + skipped:
        raise ValidationReceiptError("validation receipt test summary does not partition collected tests")
    if failed:
        raise ValidationReceiptError("cannot write a successful validation receipt with failures")
    try:
        coverage = float(coverage_percent)
        threshold = float(coverage_threshold)
    except (TypeError, ValueError) as exc:
        raise ValidationReceiptError("coverage values must be numeric") from exc
    if not math.isfinite(coverage) or not math.isfinite(threshold):
        raise ValidationReceiptError("coverage values must be finite")
    if threshold < MINIMUM_COVERAGE_THRESHOLD:
        raise ValidationReceiptError(f"coverage threshold must be at least {MINIMUM_COVERAGE_THRESHOLD:.0f}")
    if coverage < threshold:
        raise ValidationReceiptError(
            "cannot write a successful validation receipt below its coverage threshold"
        )
    environment_values = dict(validation_environment() if environment is None else environment)
    missing_environment = [
        key
        for key in _ENVIRONMENT_FIELDS
        if not isinstance(environment_values.get(key), str) or not environment_values[key]
    ]
    if missing_environment:
        raise ValidationReceiptError(
            "validation receipt environment is missing fields: " + ", ".join(missing_environment)
        )
    pre_snapshot = _validation_snapshot_from_mapping(
        pre_run_snapshot,
        label="pre-run",
    )
    post_snapshot = capture_validation_snapshot(root)
    if pre_snapshot != post_snapshot:
        raise ValidationReceiptError(
            "validation inputs or analysis-stage evidence changed while the test suite ran"
        )
    input_hashes = post_snapshot["input_hashes"]
    analysis_digests = post_snapshot["analysis_stage"]
    stamp = validate_utc_timestamp(timestamp)
    receipt: dict[str, Any] = {
        "schema_version": VALIDATION_RECEIPT_SCHEMA_VERSION,
        "receipt_type": "test_coverage",
        "success": True,
        "command": command_parts,
        "test_summary": {
            "collected": collected,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
        },
        "coverage": {
            "percent": coverage,
            "threshold": threshold,
        },
        "environment": environment_values,
        "input_patterns": list(VALIDATION_INPUT_PATTERNS),
        "input_hashes": input_hashes,
        "input_digest": _map_digest(input_hashes),
        "source_lock_sha256": input_hashes["uv.lock"],
        "analysis_stage": analysis_digests,
        "run_snapshots": {
            "pre": pre_snapshot,
            "post": post_snapshot,
        },
        "recorded_at": stamp,
        "timestamp_policy": "recorded" if stamp is not None else "omitted",
    }
    path = _receipt_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return receipt


def validation_receipt_findings(project_root: Path) -> list[str]:
    """Return fail-closed findings for the standalone validation receipt."""
    root = Path(project_root).resolve()
    try:
        receipt = _load_receipt(root)
    except ValidationReceiptError as exc:
        return [str(exc)]
    findings: list[str] = []
    if receipt.get("schema_version") != VALIDATION_RECEIPT_SCHEMA_VERSION:
        findings.append("validation receipt schema version is unsupported")
    if receipt.get("receipt_type") != "test_coverage":
        findings.append("validation receipt type is invalid")
    if receipt.get("success") is not True:
        findings.append("validation receipt does not record success=true")
    command = receipt.get("command")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(part, str) or not part for part in command)
    ):
        findings.append("validation receipt command is invalid")
    if receipt.get("input_patterns") != list(VALIDATION_INPUT_PATTERNS):
        findings.append("validation receipt input pattern contract drift")
    snapshots = receipt.get("run_snapshots")
    if not isinstance(snapshots, Mapping):
        findings.append("validation receipt run snapshots are invalid")
        pre_snapshot = None
        post_snapshot = None
    else:
        try:
            pre_snapshot = _validation_snapshot_from_mapping(
                snapshots.get("pre"),
                label="recorded pre-run",
            )
            post_snapshot = _validation_snapshot_from_mapping(
                snapshots.get("post"),
                label="recorded post-run",
            )
        except ValidationReceiptError as exc:
            findings.append(str(exc))
            pre_snapshot = None
            post_snapshot = None
        else:
            if pre_snapshot != post_snapshot:
                findings.append("validation receipt pre/post snapshots differ")
            if receipt.get("input_hashes") != post_snapshot["input_hashes"]:
                findings.append("validation receipt input hashes differ from its post-run snapshot")
            if receipt.get("input_digest") != post_snapshot["input_digest"]:
                findings.append("validation receipt input digest differs from its post-run snapshot")
            if receipt.get("analysis_stage") != post_snapshot["analysis_stage"]:
                findings.append("validation receipt analysis-stage digest differs from its post-run snapshot")
    timestamp = receipt.get("recorded_at")
    timestamp_policy = receipt.get("timestamp_policy")
    if timestamp is None:
        if timestamp_policy != "omitted":
            findings.append("validation receipt omitted recorded_at requires timestamp_policy=omitted")
    elif isinstance(timestamp, str):
        try:
            validate_utc_timestamp(timestamp)
        except ValueError:
            findings.append("validation receipt recorded_at is not canonical UTC")
        if timestamp_policy != "recorded":
            findings.append("validation receipt populated recorded_at requires timestamp_policy=recorded")
    else:
        findings.append("validation receipt recorded_at must be a canonical UTC string or null")

    summary = receipt.get("test_summary")
    if not isinstance(summary, dict):
        findings.append("validation receipt test summary is invalid")
    else:
        try:
            collected = _require_nonnegative_int(summary, "collected")
            passed = _require_nonnegative_int(summary, "passed")
            failed = _require_nonnegative_int(summary, "failed")
            skipped = _require_nonnegative_int(summary, "skipped")
        except ValidationReceiptError as exc:
            findings.append(str(exc))
        else:
            if collected != passed + failed + skipped:
                findings.append("validation receipt test summary does not partition collected tests")
            if failed:
                findings.append("validation receipt records failed tests")

    coverage = receipt.get("coverage")
    if not isinstance(coverage, dict):
        findings.append("validation receipt coverage is invalid")
    else:
        try:
            percent = _require_number(coverage, "percent")
            threshold = _require_number(coverage, "threshold")
        except ValidationReceiptError as exc:
            findings.append(str(exc))
        else:
            if threshold < MINIMUM_COVERAGE_THRESHOLD:
                findings.append("validation receipt coverage threshold is below the project policy")
            if percent < threshold:
                findings.append("validation receipt coverage is below its recorded threshold")

    environment = receipt.get("environment")
    if not isinstance(environment, dict):
        findings.append("validation receipt environment is invalid")
    else:
        missing_environment = [
            key
            for key in _ENVIRONMENT_FIELDS
            if not isinstance(environment.get(key), str) or not environment[key]
        ]
        if missing_environment:
            findings.append(
                "validation receipt environment is missing fields: " + ", ".join(missing_environment)
            )

    try:
        current_inputs = validation_input_hashes(root)
    except ValidationReceiptError as exc:
        findings.append(str(exc))
    else:
        recorded_inputs = receipt.get("input_hashes")
        if recorded_inputs != current_inputs:
            findings.append("validation receipt input hashes are stale")
        if receipt.get("input_digest") != _map_digest(current_inputs):
            findings.append("validation receipt input digest is stale")
        if receipt.get("source_lock_sha256") != current_inputs.get("uv.lock"):
            findings.append("validation receipt dependency-lock digest is stale")

    try:
        current_analysis = _analysis_stage_digests(root)
    except ValidationReceiptError as exc:
        findings.append(str(exc))
    else:
        if receipt.get("analysis_stage") != current_analysis:
            findings.append("validation receipt analysis-stage digest is stale")
    return findings


def require_fresh_validation_receipt(project_root: Path) -> dict[str, Any]:
    """Return the receipt only when its test, source, and analysis evidence is fresh."""
    root = Path(project_root).resolve()
    findings = validation_receipt_findings(root)
    if findings:
        raise ValidationReceiptError("validation receipt preflight failed: " + "; ".join(findings))
    return _load_receipt(root)


def validation_receipt_tokens(project_root: Path) -> dict[str, str]:
    """Load final manuscript provenance tokens from a fresh successful receipt."""
    receipt = require_fresh_validation_receipt(project_root)
    summary = receipt["test_summary"]
    coverage = receipt["coverage"]
    environment = receipt["environment"]
    return {
        "TEST_COUNT": str(summary["collected"]),
        "COVERAGE_PERCENT": f"{float(coverage['percent']):.2f}",
        "PYTHON_VERSION": str(environment["python_version"]),
        "NUMPY_VERSION": str(environment["numpy_version"]),
        "SCIPY_VERSION": str(environment["scipy_version"]),
        "PLATFORM": str(environment["platform"]),
    }


__all__ = [
    "MINIMUM_COVERAGE_THRESHOLD",
    "VALIDATION_INPUT_PATTERNS",
    "VALIDATION_RECEIPT_PATH",
    "VALIDATION_RECEIPT_SCHEMA_VERSION",
    "ValidationReceiptError",
    "capture_validation_snapshot",
    "require_fresh_validation_receipt",
    "validation_environment",
    "validation_input_hashes",
    "validation_receipt_findings",
    "validation_receipt_tokens",
    "write_validation_receipt",
]
