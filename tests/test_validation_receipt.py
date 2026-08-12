"""Tests for the standalone full-suite validation receipt."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Mapping, cast

import pytest

from manuscript_vars.loaders import _validation_receipt_variables
from publication.pipeline_freshness import (
    ANALYSIS_EXECUTION_PATH,
    ANALYSIS_EXECUTION_SCHEMA_VERSION,
    capture_analysis_input_snapshot,
    record_pipeline_stage,
    record_publication_analysis_stage,
)
from publication.validation_receipt import (
    ValidationReceiptError,
    capture_validation_snapshot,
    require_fresh_validation_receipt,
    validation_environment,
    validation_input_hashes,
    validation_receipt_findings,
    validation_receipt_tokens,
    write_validation_receipt,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_HYDRATE_SCRIPT = _PROJECT_ROOT / "scripts" / "z_generate_manuscript_variables.py"


def _write(root: Path, relative: str, text: str = "content\n") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_validation_tree(root: Path, *, analysis_profile: str = "smoke") -> None:
    for relative, text in (
        ("src/model.py", "VALUE = 1\n"),
        ("src/README.md", "# Source\n"),
        ("tests/test_model.py", "def test_model():\n    assert True\n"),
        ("tests/README.md", "# Tests\n"),
        ("scripts/02_run_analysis.py", "# producer\n"),
        ("scripts/z_generate_manuscript_variables.py", "# hydrator\n"),
        ("scripts/README.md", "# Scripts\n"),
        ("data/README.md", "# Data\n"),
        ("docs/README.md", "# Documentation\n"),
        (".github/workflows/ci.yml", "name: fixture\n"),
        (
            "manuscript/config.yaml",
            f"paper: {{}}\nexperiment:\n  analysis_profile: {analysis_profile}\n",
        ),
        ("manuscript/01_intro.md", "# Intro\n"),
        ("manuscript/references.bib", "@misc{fixture, title={Fixture}}\n"),
        ("manuscript/preamble.tex", "% fixture\n"),
        ("ISA.md", "- [x] ISC-1: fixture\n"),
        ("AGENTS.md", "# Guidance\n"),
        ("README.md", "# Fixture\n"),
        ("STANDALONE.md", "# Standalone\n"),
        ("TODO.md", "# TODO\n"),
        ("REDTEAM_REVIEW.md", "# Review\n"),
        (".gitignore", ".tmp/\n"),
        (".zenodo.json", "{}\n"),
        ("CITATION.cff", "cff-version: 1.2.0\n"),
        ("codemeta.json", "{}\n"),
        ("MANIFEST.in", "include README.md\n"),
        ("_fedference_build_backend.py", "# fixture backend\n"),
        ("domain_profile.yaml", "profile: fixture\n"),
        ("experiment_plan.yaml", "profile: smoke\n"),
        ("pyproject.toml", "[project]\nname = 'fixture'\nversion = '0.0.0'\n"),
        ("uv.lock", "version = 1\n"),
        ("output/reports/belief_sharing.json", "{}\n"),
        ("output/figures/example.png", "not-a-real-png\n"),
        ("output/data/stage_timings.json", "{}\n"),
    ):
        _write(root, relative, text)
    record_pipeline_stage(root, "analysis")


def test_validation_boundary_includes_the_package_license() -> None:
    hashes = validation_input_hashes(_PROJECT_ROOT)

    assert "LICENSE" in hashes


def _write_successful_receipt(root: Path) -> dict[str, object]:
    return write_validation_receipt(
        root,
        command=("python", "-m", "pytest", "tests/", "--cov=src"),
        test_summary={"collected": 3, "passed": 3, "failed": 0, "skipped": 0},
        coverage_percent=93.25,
        pre_run_snapshot=capture_validation_snapshot(root),
        environment=validation_environment(),
        timestamp="2026-08-02T00:00:00Z",
    )


def _mutated_receipt_findings(
    root: Path,
    mutate: Callable[[dict[str, Any]], None],
) -> list[str]:
    """Return validation findings after one realistic on-disk receipt corruption."""
    _make_validation_tree(root)
    _write_successful_receipt(root)
    path = root / "output" / "data" / "test_coverage_receipt.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return validation_receipt_findings(root)


def _set_divergent_snapshot(payload: dict[str, Any]) -> None:
    snapshots = payload["run_snapshots"]
    assert isinstance(snapshots, dict)
    post = snapshots["post"]
    assert isinstance(post, dict)
    analysis_stage = post["analysis_stage"]
    assert isinstance(analysis_stage, dict)
    analysis_stage["output_digest"] = "different"


def _set_invalid_recorded_snapshot(payload: dict[str, Any]) -> None:
    snapshots = payload["run_snapshots"]
    assert isinstance(snapshots, dict)
    pre = snapshots["pre"]
    assert isinstance(pre, dict)
    pre["input_hashes"] = {}


def test_successful_receipt_binds_test_inputs_analysis_and_tokens(tmp_path: Path) -> None:
    _make_validation_tree(tmp_path)
    receipt = _write_successful_receipt(tmp_path)

    assert receipt["success"] is True
    run_snapshots = receipt["run_snapshots"]
    assert isinstance(run_snapshots, dict)
    assert run_snapshots["pre"] == run_snapshots["post"]
    assert validation_receipt_findings(tmp_path) == []
    assert require_fresh_validation_receipt(tmp_path)["input_digest"] == receipt["input_digest"]
    expected_tokens = {
        "TEST_COUNT": "3",
        "COVERAGE_PERCENT": "93.25",
        "PYTHON_VERSION": validation_environment()["python_version"],
        "NUMPY_VERSION": validation_environment()["numpy_version"],
        "SCIPY_VERSION": validation_environment()["scipy_version"],
        "PLATFORM": validation_environment()["platform"],
    }
    assert validation_receipt_tokens(tmp_path) == expected_tokens
    assert _validation_receipt_variables(tmp_path) == expected_tokens

    (tmp_path / "tests" / "test_model.py").write_text(
        "def test_model():\n    assert 2 == 2\n",
        encoding="utf-8",
    )
    assert "validation receipt input hashes are stale" in validation_receipt_findings(tmp_path)


def test_writer_refuses_post_launch_test_tree_drift(tmp_path: Path) -> None:
    """A long suite must not attest a source tree it did not start against."""
    _make_validation_tree(tmp_path)
    pre_run_snapshot = capture_validation_snapshot(tmp_path)
    # Tests are receipt inputs but not analysis inputs, so this specifically
    # exercises the pre/post test-run comparison rather than analysis freshness.
    (tmp_path / "tests" / "test_model.py").write_text(
        "def test_model():\n    assert 2 == 2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationReceiptError, match="changed while the test suite ran"):
        write_validation_receipt(
            tmp_path,
            command=("python", "-m", "pytest", "tests/", "--cov=src"),
            test_summary={"collected": 3, "passed": 3, "failed": 0, "skipped": 0},
            coverage_percent=93.25,
            pre_run_snapshot=pre_run_snapshot,
            environment=validation_environment(),
        )
    assert not (tmp_path / "output" / "data" / "test_coverage_receipt.json").exists()


def test_final_hydration_cli_rechecks_receipt_after_output_writes() -> None:
    """Guard the final post-write receipt check against a TOCTOU regression."""
    source = _HYDRATE_SCRIPT.read_text(encoding="utf-8")
    render_index = source.index("manuscript_dir = render_manuscript_tree(root, variables)")
    post_write_check = source.rindex("require_fresh_validation_receipt(root)")
    stage_record = source.index('record_pipeline_stage(root, "hydration", timestamp=timestamp)')
    assert render_index < post_write_check < stage_record


def test_provisional_hydration_cli_never_records_a_publication_receipt() -> None:
    """The pre-test render may write scratch outputs but cannot advance freshness."""
    source = _HYDRATE_SCRIPT.read_text(encoding="utf-8")
    record_index = source.index('record_pipeline_stage(root, "hydration", timestamp=timestamp)')
    guard_index = source.index(
        'if not args.allow_draft and not args.provisional_validation and (root / "src").is_dir():'
    )
    assert guard_index < record_index


@pytest.mark.parametrize(
    ("relative", "replacement"),
    (
        ("manuscript/01_intro.md", "# Corrected caption\n"),
        ("ISA.md", "- [ ] ISC-1: changed after validation\n"),
    ),
)
def test_receipt_fails_when_a_hydration_source_changes_after_tests(
    tmp_path: Path,
    relative: str,
    replacement: str,
) -> None:
    _make_validation_tree(tmp_path)
    _write_successful_receipt(tmp_path)
    (tmp_path / relative).write_text(replacement, encoding="utf-8")

    findings = validation_receipt_findings(tmp_path)

    assert "validation receipt input hashes are stale" in findings


def test_receipt_fails_when_a_validated_document_changes_after_tests(tmp_path: Path) -> None:
    """Documentation validation cannot be attributed to a later edited guide."""
    _make_validation_tree(tmp_path)
    _write_successful_receipt(tmp_path)
    (tmp_path / "docs" / "README.md").write_text("# Corrected documentation\n", encoding="utf-8")

    assert "validation receipt input hashes are stale" in validation_receipt_findings(tmp_path)


def test_receipt_fails_when_its_bound_analysis_output_changes(tmp_path: Path) -> None:
    _make_validation_tree(tmp_path)
    _write_successful_receipt(tmp_path)
    (tmp_path / "output" / "reports" / "belief_sharing.json").write_text(
        '{"changed": true}\n',
        encoding="utf-8",
    )

    findings = validation_receipt_findings(tmp_path)
    assert any("analysis receipt is not fresh" in finding for finding in findings)
    with pytest.raises(ValidationReceiptError, match="validation receipt preflight failed"):
        require_fresh_validation_receipt(tmp_path)


def test_optional_template_source_can_be_added_only_after_revalidation(
    tmp_path: Path,
) -> None:
    _make_validation_tree(tmp_path)
    (tmp_path / "manuscript" / "preamble.tex").unlink()
    _write_successful_receipt(tmp_path)
    (tmp_path / "manuscript" / "preamble.tex").write_text(
        "% newly added template source\n",
        encoding="utf-8",
    )

    assert "validation receipt input hashes are stale" in validation_receipt_findings(tmp_path)


def test_writer_rejects_failed_or_below_policy_results_without_writing(tmp_path: Path) -> None:
    _make_validation_tree(tmp_path)
    with pytest.raises(ValidationReceiptError, match="failures"):
        write_validation_receipt(
            tmp_path,
            command=("python", "-m", "pytest"),
            test_summary={"collected": 2, "passed": 1, "failed": 1, "skipped": 0},
            coverage_percent=99.0,
            pre_run_snapshot=capture_validation_snapshot(tmp_path),
        )
    with pytest.raises(ValidationReceiptError, match="at least 90"):
        write_validation_receipt(
            tmp_path,
            command=("python", "-m", "pytest"),
            test_summary={"collected": 2, "passed": 2, "failed": 0, "skipped": 0},
            coverage_percent=99.0,
            pre_run_snapshot=capture_validation_snapshot(tmp_path),
            coverage_threshold=89.0,
        )
    with pytest.raises(ValidationReceiptError, match="must be finite"):
        write_validation_receipt(
            tmp_path,
            command=("python", "-m", "pytest"),
            test_summary={"collected": 2, "passed": 2, "failed": 0, "skipped": 0},
            coverage_percent=float("nan"),
            pre_run_snapshot=capture_validation_snapshot(tmp_path),
        )
    assert not (tmp_path / "output" / "data" / "test_coverage_receipt.json").exists()


@pytest.mark.parametrize(
    ("label", "mutate", "expected"),
    (
        (
            "unsupported schema",
            lambda payload: payload.__setitem__("schema_version", 1),
            "validation receipt schema version is unsupported",
        ),
        (
            "wrong receipt type",
            lambda payload: payload.__setitem__("receipt_type", "other"),
            "validation receipt type is invalid",
        ),
        (
            "missing success marker",
            lambda payload: payload.__setitem__("success", False),
            "validation receipt does not record success=true",
        ),
        (
            "malformed command",
            lambda payload: payload.__setitem__("command", ["python", 3]),
            "validation receipt command is invalid",
        ),
        (
            "empty command",
            lambda payload: payload.__setitem__("command", []),
            "validation receipt command is invalid",
        ),
        (
            "drifted input contract",
            lambda payload: payload.__setitem__("input_patterns", []),
            "validation receipt input pattern contract drift",
        ),
        (
            "nonmapping snapshots",
            lambda payload: payload.__setitem__("run_snapshots", []),
            "validation receipt run snapshots are invalid",
        ),
        (
            "invalid pre snapshot",
            _set_invalid_recorded_snapshot,
            "recorded pre-run validation snapshot has invalid input hashes",
        ),
        (
            "invalid post snapshot",
            lambda payload: payload["run_snapshots"]["post"].__setitem__(
                "input_hashes", {"": "digest"}
            ),
            "recorded post-run validation snapshot has invalid input hashes",
        ),
        (
            "mismatched snapshots",
            _set_divergent_snapshot,
            "validation receipt pre/post snapshots differ",
        ),
        (
            "omitted timestamp with recorded policy",
            lambda payload: payload.update(recorded_at=None, timestamp_policy="recorded"),
            "validation receipt omitted recorded_at requires timestamp_policy=omitted",
        ),
        (
            "invalid timestamp",
            lambda payload: payload.__setitem__("recorded_at", "not-a-timestamp"),
            "validation receipt recorded_at is not canonical UTC",
        ),
        (
            "nonstring timestamp",
            lambda payload: payload.__setitem__("recorded_at", 1),
            "validation receipt recorded_at must be a canonical UTC string or null",
        ),
        (
            "invalid summary container",
            lambda payload: payload.__setitem__("test_summary", []),
            "validation receipt test summary is invalid",
        ),
        (
            "invalid coverage container",
            lambda payload: payload.__setitem__("coverage", []),
            "validation receipt coverage is invalid",
        ),
        (
            "invalid environment container",
            lambda payload: payload.__setitem__("environment", []),
            "validation receipt environment is invalid",
        ),
        (
            "invalid timestamp policy",
            lambda payload: payload.__setitem__("timestamp_policy", "other"),
            "validation receipt populated recorded_at requires timestamp_policy=recorded",
        ),
    ),
    ids=lambda case: str(case),
)
def test_receipt_findings_fail_closed_on_structural_corruption(
    tmp_path: Path,
    label: str,
    mutate: Callable[[dict[str, Any]], None],
    expected: str,
) -> None:
    """Every receipt-shape contract must reject a plausible persisted corruption."""
    del label
    assert any(expected in finding for finding in _mutated_receipt_findings(tmp_path, mutate))


@pytest.mark.parametrize(
    ("label", "mutate", "expected"),
    (
        (
            "top-level hashes disagree with post snapshot",
            lambda payload: payload.__setitem__("input_hashes", {}),
            "validation receipt input hashes differ from its post-run snapshot",
        ),
        (
            "top-level digest disagrees with post snapshot",
            lambda payload: payload.__setitem__("input_digest", "different"),
            "validation receipt input digest differs from its post-run snapshot",
        ),
        (
            "top-level analysis differs from post snapshot",
            lambda payload: payload.__setitem__("analysis_stage", {}),
            "validation receipt analysis-stage digest differs from its post-run snapshot",
        ),
        (
            "timestamp policy disagrees with a timestamp",
            lambda payload: payload.__setitem__("timestamp_policy", "omitted"),
            "validation receipt populated recorded_at requires timestamp_policy=recorded",
        ),
        (
            "summary does not partition collection",
            lambda payload: payload.__setitem__(
                "test_summary",
                {"collected": 3, "passed": 2, "failed": 0, "skipped": 0},
            ),
            "validation receipt test summary does not partition collected tests",
        ),
        (
            "summary records a failure",
            lambda payload: payload.__setitem__(
                "test_summary",
                {"collected": 3, "passed": 2, "failed": 1, "skipped": 0},
            ),
            "validation receipt records failed tests",
        ),
        (
            "summary has an invalid count",
            lambda payload: payload.__setitem__(
                "test_summary",
                {"collected": True, "passed": 3, "failed": 0, "skipped": 0},
            ),
            "validation receipt collected must be a non-negative integer",
        ),
        (
            "coverage has an invalid number",
            lambda payload: payload.__setitem__(
                "coverage",
                {"percent": True, "threshold": 90.0},
            ),
            "validation receipt percent must be numeric",
        ),
        (
            "coverage threshold has an invalid number",
            lambda payload: payload.__setitem__(
                "coverage",
                {"percent": 93.0, "threshold": "90"},
            ),
            "validation receipt threshold must be numeric",
        ),
        (
            "coverage threshold is below policy",
            lambda payload: payload.__setitem__(
                "coverage",
                {"percent": 93.0, "threshold": 89.0},
            ),
            "validation receipt coverage threshold is below the project policy",
        ),
        (
            "coverage is below recorded threshold",
            lambda payload: payload.__setitem__(
                "coverage",
                {"percent": 89.0, "threshold": 90.0},
            ),
            "validation receipt coverage is below its recorded threshold",
        ),
        (
            "environment lacks a required field",
            lambda payload: payload.__setitem__("environment", {"python_version": "3.13"}),
            "validation receipt environment is missing fields",
        ),
        (
            "summary has a negative count",
            lambda payload: payload["test_summary"].__setitem__("skipped", -1),
            "validation receipt skipped must be a non-negative integer",
        ),
        (
            "dependency lock digest disagrees with inputs",
            lambda payload: payload.__setitem__("source_lock_sha256", "different"),
            "validation receipt dependency-lock digest is stale",
        ),
        (
            "analysis stage digest is stale",
            lambda payload: payload.__setitem__(
                "analysis_stage",
                {"input_digest": "different", "output_digest": "different"},
            ),
            "validation receipt analysis-stage digest is stale",
        ),
    ),
    ids=lambda case: str(case),
)
def test_receipt_findings_validate_nested_integrity_contracts(
    tmp_path: Path,
    label: str,
    mutate: Callable[[dict[str, Any]], None],
    expected: str,
) -> None:
    """A syntactically valid receipt still needs coherent provenance fields."""
    del label
    assert any(expected in finding for finding in _mutated_receipt_findings(tmp_path, mutate))


@pytest.mark.parametrize(
    ("snapshot_mutation", "expected"),
    (
        (lambda snapshot: snapshot.clear(), "pre-run validation snapshot has invalid input hashes"),
        (
            lambda snapshot: snapshot.__setitem__("input_hashes", {}),
            "pre-run validation snapshot has invalid input hashes",
        ),
        (
            lambda snapshot: snapshot["input_hashes"].__setitem__("", "digest"),
            "pre-run validation snapshot has invalid input hashes",
        ),
        (
            lambda snapshot: snapshot.__setitem__("input_digest", "different"),
            "pre-run validation snapshot has an invalid input digest",
        ),
        (
            lambda snapshot: snapshot.__setitem__("analysis_stage", []),
            "pre-run validation snapshot has invalid analysis-stage digests",
        ),
    ),
)
def test_writer_rejects_malformed_pre_run_snapshots(
    tmp_path: Path,
    snapshot_mutation: Callable[[dict[str, Any]], None],
    expected: str,
) -> None:
    """The writer must not normalize an attacker-controlled pre-run snapshot."""
    _make_validation_tree(tmp_path)
    snapshot = capture_validation_snapshot(tmp_path)
    snapshot_mutation(snapshot)

    with pytest.raises(ValidationReceiptError, match=expected):
        write_validation_receipt(
            tmp_path,
            command=("python", "-m", "pytest"),
            test_summary={"collected": 3, "passed": 3, "failed": 0, "skipped": 0},
            coverage_percent=93.25,
            pre_run_snapshot=snapshot,
        )


def test_writer_rejects_a_nonmapping_pre_run_snapshot(tmp_path: Path) -> None:
    """The pre-run boundary itself must be a mapping before it is normalized."""
    _make_validation_tree(tmp_path)

    with pytest.raises(ValidationReceiptError, match="pre-run validation snapshot must be a mapping"):
        write_validation_receipt(
            tmp_path,
            command=("python", "-m", "pytest"),
            test_summary={"collected": 3, "passed": 3, "failed": 0, "skipped": 0},
            coverage_percent=93.25,
            pre_run_snapshot=cast(Mapping[str, Any], []),
        )


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    (
        ({"command": ()}, "command must be non-empty strings"),
        ({"command": ("python", "")}, "command must be non-empty strings"),
        (
            {"test_summary": {"collected": 3, "passed": 2, "failed": 0, "skipped": 0}},
            "does not partition collected tests",
        ),
        (
            {"test_summary": {"collected": True, "passed": 1, "failed": 0, "skipped": 0}},
            "collected must be a non-negative integer",
        ),
        ({"coverage_percent": "not-a-number"}, "coverage values must be numeric"),
        ({"coverage_threshold": float("nan")}, "coverage values must be finite"),
        ({"coverage_percent": 89.0}, "below its coverage threshold"),
        ({"environment": {"python_version": "3.13"}}, "environment is missing fields"),
    ),
)
def test_writer_rejects_invalid_live_gate_inputs(
    tmp_path: Path,
    kwargs: dict[str, Any],
    expected: str,
) -> None:
    """Only a complete, successful live gate can produce a provenance receipt."""
    _make_validation_tree(tmp_path)
    defaults: dict[str, Any] = {
        "command": ("python", "-m", "pytest"),
        "test_summary": {"collected": 3, "passed": 3, "failed": 0, "skipped": 0},
        "coverage_percent": 93.25,
        "pre_run_snapshot": capture_validation_snapshot(tmp_path),
    }
    defaults.update(kwargs)

    with pytest.raises(ValidationReceiptError, match=expected):
        write_validation_receipt(tmp_path, **defaults)


def test_input_hash_contract_rejects_a_missing_required_source(tmp_path: Path) -> None:
    """A receipt cannot silently omit a top-level validated documentation source."""
    _make_validation_tree(tmp_path)
    (tmp_path / "README.md").unlink()

    with pytest.raises(ValidationReceiptError, match="validation input pattern matched no files: README.md"):
        validation_input_hashes(tmp_path)


def test_receipt_finds_missing_or_malformed_persisted_payloads(tmp_path: Path) -> None:
    """The reader must fail closed before attempting semantic receipt validation."""
    assert validation_receipt_findings(tmp_path) == [
        f"validation receipt is missing: {tmp_path / 'output/data/test_coverage_receipt.json'}"
    ]

    path = tmp_path / "output" / "data" / "test_coverage_receipt.json"
    path.parent.mkdir(parents=True)
    path.write_text("{", encoding="utf-8")
    assert validation_receipt_findings(tmp_path) == [f"invalid validation receipt: {path}"]

    path.write_text("[]", encoding="utf-8")
    assert validation_receipt_findings(tmp_path) == [f"invalid validation receipt shape: {path}"]


def test_receipt_supports_a_deterministic_omitted_timestamp(tmp_path: Path) -> None:
    """An ordinary local gate has no wall-clock claim but remains verifiable."""
    _make_validation_tree(tmp_path)
    receipt = write_validation_receipt(
        tmp_path,
        command=("python", "-m", "pytest"),
        test_summary={"collected": 3, "passed": 3, "failed": 0, "skipped": 0},
        coverage_percent=93.25,
        pre_run_snapshot=capture_validation_snapshot(tmp_path),
    )

    assert receipt["recorded_at"] is None
    assert receipt["timestamp_policy"] == "omitted"
    assert validation_receipt_findings(tmp_path) == []


def test_receipt_findings_reject_nonfinite_coverage_and_current_hash_failure(tmp_path: Path) -> None:
    """Live receipt verification must reject nonfinite values and unreadable input boundaries."""
    findings = _mutated_receipt_findings(
        tmp_path,
        lambda payload: payload.__setitem__(
            "coverage",
            {"percent": float("nan"), "threshold": 90.0},
        ),
    )
    assert "validation receipt percent must be finite" in findings

    _make_validation_tree(tmp_path)
    _write_successful_receipt(tmp_path)
    (tmp_path / "README.md").unlink()
    findings = validation_receipt_findings(tmp_path)
    assert "validation input pattern matched no files: README.md" in findings


def test_final_hydration_refuses_missing_receipt_before_any_output_write(tmp_path: Path) -> None:
    _make_validation_tree(tmp_path, analysis_profile="publication")
    # Reach the validation-receipt boundary deliberately: the publication
    # analysis producer proof is an earlier, independent preflight.
    _write(
        tmp_path,
        ANALYSIS_EXECUTION_PATH.as_posix(),
        json.dumps(
            {
                "schema_version": ANALYSIS_EXECUTION_SCHEMA_VERSION,
                "configured_profile": "publication",
                "effective_profile": "publication",
                "producer": "analysis.workflow.run_analysis_pipeline",
                "analysis_input_snapshot": capture_analysis_input_snapshot(tmp_path),
            }
        ),
    )
    record_publication_analysis_stage(tmp_path)
    variables_path = tmp_path / "output" / "data" / "manuscript_variables.json"
    manuscript_path = tmp_path / "output" / "manuscript" / "01_intro.md"
    _write(tmp_path, "output/data/manuscript_variables.json", "sentinel variables\n")
    _write(tmp_path, "output/manuscript/01_intro.md", "sentinel manuscript\n")
    before_variables = variables_path.read_bytes()
    before_manuscript = manuscript_path.read_bytes()
    environment = dict(os.environ)
    environment["ACTIVE_FEDFERENCE_PROJECT_ROOT"] = str(tmp_path)

    completed = subprocess.run(
        [sys.executable, str(_HYDRATE_SCRIPT)],
        cwd=_PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode != 0
    assert "validation receipt preflight failed" in completed.stderr
    assert variables_path.read_bytes() == before_variables
    assert manuscript_path.read_bytes() == before_manuscript
