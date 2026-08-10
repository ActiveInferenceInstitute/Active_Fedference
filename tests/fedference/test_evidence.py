"""Versioned experiment and run-receipt contracts."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace

import pytest

from fedference.evidence import (
    ArtifactRecord,
    DatasetSpec,
    ExperimentSpec,
    RunReceipt,
    canonical_sha256,
    load_run_receipt,
    make_artifact_record,
    sha256_file,
    validate_evidence_report,
    verify_run_receipt,
    write_run_receipt,
)

SHA = "a" * 64
HERMETIC_GIT = ("git", "-c", "core.fsmonitor=false", "-c", "core.untrackedcache=false")


def test_evidence_report_write_boundary_requires_claim_contract() -> None:
    validate_evidence_report(
        {
            "status": "pilot",
            "primary_estimand": "held-out log score",
            "independent_unit": "seeded world",
            "no_claim": "pilot is not confirmatory",
            "negative_controls": {},
        }
    )
    with pytest.raises(ValueError, match="primary_estimand"):
        validate_evidence_report(
            {
                "status": "pilot",
                "primary_estimand": "",
                "independent_unit": "seeded world",
                "no_claim": "pilot is not confirmatory",
            }
        )
    with pytest.raises(ValueError, match="negative_controls"):
        validate_evidence_report(
            {
                "status": "pilot",
                "primary_estimand": "held-out log score",
                "independent_unit": "seeded world",
                "no_claim": "pilot is not confirmatory",
                "negative_controls": [],
            }
        )
    with pytest.raises(ValueError, match="mapping"):
        validate_evidence_report(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rows"):
        validate_evidence_report(
            {
                "status": "pilot",
                "primary_estimand": "held-out log score",
                "independent_unit": "seeded world",
                "no_claim": "pilot is not confirmatory",
                "rows": {},
            }
        )


def _receipt(tmp_path) -> RunReceipt:
    config = tmp_path / "config.json"
    config.write_text('{"seed": 0}\n', encoding="utf-8")
    report = tmp_path / "report.json"
    report.write_text('{"status": "ok"}\n', encoding="utf-8")
    return RunReceipt(
        run_id="run-1",
        experiment_id="experiment",
        experiment_version="0.1",
        profile="smoke",
        git_commit="d" * 40,
        git_tree_state="clean",
        environment_lock_sha256=SHA,
        config_sha256=canonical_sha256({"seed": 0}),
        dataset_sha256={"dataset": SHA},
        seeds=(0,),
        device="test-cpu",
        backend="numpy",
        fallbacks=(),
        checkpoints=(),
        outputs=(
            make_artifact_record("config", config, root=tmp_path),
            make_artifact_record("report", report, root=tmp_path),
        ),
        status="completed",
        started_at_utc="2026-07-29T00:00:00+00:00",
        completed_at_utc="2026-07-29T00:00:01+00:00",
    )


def test_run_receipt_round_trip_and_artifact_verification(tmp_path) -> None:
    receipt = _receipt(tmp_path)
    path = write_run_receipt(tmp_path / "receipt.json", receipt)
    loaded = load_run_receipt(path)
    assert loaded.as_dict() == receipt.as_dict()
    assert verify_run_receipt(loaded, root=tmp_path) == ()
    with pytest.raises(TypeError):
        loaded.dataset_sha256["dataset"] = "0" * 64
    with pytest.raises(ValueError, match="Out of range float values"):
        canonical_sha256({"not_json": float("nan")})


def test_run_receipt_detects_tamper_and_noncompleted_status(tmp_path) -> None:
    receipt = _receipt(tmp_path)
    (tmp_path / "report.json").write_text('{"status": "changed"}\n', encoding="utf-8")
    findings = verify_run_receipt(receipt, root=tmp_path)
    assert "artifact byte-size mismatch: report.json" in findings
    assert "artifact digest mismatch: report.json" in findings
    partial = replace(receipt, status="partial")
    assert "run status is 'partial', not 'completed'" in verify_run_receipt(partial, root=tmp_path)
    dirty = replace(receipt, git_tree_state="dirty")
    assert not any("git tree state" in finding for finding in verify_run_receipt(dirty, root=tmp_path))
    assert "git tree state is 'dirty', not 'clean'" in verify_run_receipt(
        dirty,
        root=tmp_path,
        require_clean_git=True,
    )


def test_run_receipt_schema_rejects_missing_or_escaping_fields(tmp_path) -> None:
    receipt = _receipt(tmp_path)
    raw = receipt.as_dict()
    del raw["backend"]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="fields do not match schema"):
        load_run_receipt(path)
    with pytest.raises(ValueError, match="relative"):
        replace(receipt.outputs[0], path="../escape.json")


def test_experiment_and_dataset_specs_validate_research_contract() -> None:
    experiment = ExperimentSpec(
        experiment_id="example",
        version="1",
        title="Example",
        state="planned",
        source_ids=("source",),
        primary_estimand="paired log score",
        independent_unit="world",
        falsifier="interval reverses",
        no_claim="not universal",
        profiles=("smoke",),
        smallest_effect_of_interest="pilot-frozen threshold",
        mcse_stopping_target="pilot-frozen target",
        maximum_budget="one bounded smoke run",
        comparison_family="one primary pair",
    )
    dataset = DatasetSpec(
        dataset_id="dataset",
        name="Dataset",
        source_url="https://example.test/data.zip",
        doi="10.0000/example",
        license="CC BY 4.0",
        archive_sha256=SHA,
        archive_member="data.csv",
        file_format="csv",
        n_rows=2,
        n_features=1,
        n_classes=2,
        has_missing_values=False,
        preprocessing=("parse",),
        schema=("feature: float64", "label: integer"),
        split_policy="seeded holdout with receipt-bound split hash",
    )
    assert experiment.independent_unit == "world"
    assert dataset.archive_sha256 == SHA
    with pytest.raises(ValueError, match="profiles"):
        replace(experiment, profiles=())
    with pytest.raises(ValueError, match="SHA-256"):
        replace(dataset, archive_sha256="bad")


def test_run_receipt_decoder_does_not_coerce_wrong_scalar_types(tmp_path) -> None:
    raw = _receipt(tmp_path).as_dict()
    raw["run_id"] = 17
    path = tmp_path / "wrong-type.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="run_id"):
        load_run_receipt(path)


def test_evidence_contract_rejects_duplicate_units_and_malformed_sequences(
    tmp_path,
) -> None:
    receipt = _receipt(tmp_path)
    with pytest.raises(ValueError, match="seeds must be unique"):
        replace(receipt, seeds=(0, 0))
    with pytest.raises(ValueError, match="at least one output"):
        replace(receipt, outputs=())
    with pytest.raises(ValueError, match="exactly one 'config'"):
        replace(receipt, outputs=(receipt.outputs[1],))
    with pytest.raises(ValueError, match="must not precede"):
        replace(
            receipt,
            completed_at_utc="2026-07-28T23:59:59+00:00",
        )
    with pytest.raises(ValueError, match="explicit UTC"):
        replace(receipt, started_at_utc="2026-07-29T00:00:00")

    dataset = DatasetSpec(
        dataset_id="dataset",
        name="Dataset",
        source_url="https://example.test/data.zip",
        doi="10.0000/example",
        license="CC BY 4.0",
        archive_sha256=SHA,
        archive_member="data.csv",
        file_format="csv",
        n_rows=2,
        n_features=1,
        n_classes=2,
        has_missing_values=False,
        preprocessing=("parse",),
        schema=("feature: float64", "label: integer"),
        split_policy="seeded holdout",
    )
    with pytest.raises(ValueError, match="preprocessing"):
        replace(dataset, preprocessing=(1,))


def test_receipt_verifies_config_hash_and_rejects_symlink_escape(tmp_path) -> None:
    receipt = _receipt(tmp_path)
    (tmp_path / "config.json").write_text('{"seed": 1}\n', encoding="utf-8")
    findings = verify_run_receipt(receipt, root=tmp_path)
    assert "artifact digest mismatch: config.json" in findings
    assert "configuration hash mismatch: config.json" in findings

    (tmp_path / "config.json").write_text('{"seed": NaN}\n', encoding="utf-8")
    findings = verify_run_receipt(receipt, root=tmp_path)
    assert "config artifact is not valid JSON: config.json" in findings

    outside = tmp_path.parent / "outside.json"
    outside.write_text('{"status": "ok"}\n', encoding="utf-8")
    (tmp_path / "linked.json").symlink_to(outside)
    escaped = replace(
        receipt,
        outputs=(
            receipt.outputs[0],
            ArtifactRecord(
                name="report",
                path="linked.json",
                sha256=sha256_file(outside),
                bytes=outside.stat().st_size,
            ),
        ),
    )
    assert "artifact escapes receipt root: linked.json" in verify_run_receipt(
        escaped,
        root=tmp_path,
    )


def test_strict_receipt_verification_checks_live_commit_tree_and_lock(
    tmp_path,
) -> None:
    project = tmp_path / "project"
    artifacts = tmp_path / "artifacts"
    project.mkdir()
    artifacts.mkdir()
    (project / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    source = project / "source.txt"
    source.write_text("source\n", encoding="utf-8")
    for command in (
        (*HERMETIC_GIT, "init", "-q"),
        (*HERMETIC_GIT, "config", "user.email", "test@example.com"),
        (*HERMETIC_GIT, "config", "user.name", "Evidence Test"),
        (*HERMETIC_GIT, "add", "source.txt", "uv.lock"),
        (*HERMETIC_GIT, "commit", "-qm", "fixture"),
    ):
        subprocess.run(command, cwd=project, check=True)
    commit = subprocess.run(
        (*HERMETIC_GIT, "rev-parse", "HEAD"),
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    receipt = replace(
        _receipt(artifacts),
        git_commit=commit,
        environment_lock_sha256=sha256_file(project / "uv.lock"),
    )
    assert verify_run_receipt(
        receipt,
        root=artifacts,
        project_root=project,
        require_clean_git=True,
    ) == ()

    source.write_text("changed\n", encoding="utf-8")
    findings = verify_run_receipt(
        receipt,
        root=artifacts,
        project_root=project,
        require_clean_git=True,
    )
    assert "live Git tree state is 'dirty', not 'clean'" in findings
    assert any("does not match receipt" in finding for finding in findings)

    source.write_text("source\n", encoding="utf-8")
    (project / "uv.lock").write_text("version = 2\n", encoding="utf-8")
    findings = verify_run_receipt(
        receipt,
        root=artifacts,
        project_root=project,
        require_clean_git=True,
    )
    assert "live environment lock digest does not match receipt" in findings
