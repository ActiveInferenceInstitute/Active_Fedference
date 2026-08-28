"""Versioned experiment and run-receipt contracts."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace

import pytest

from fedference.application import LabeledAggregationRequest, aggregate_labeled
from fedference.evidence import (
    ApplicationReceipt,
    ArtifactRecord,
    DatasetSpec,
    ExperimentSpec,
    RunReceipt,
    canonical_sha256,
    load_application_receipt,
    load_run_receipt,
    make_artifact_record,
    sha256_file,
    validate_evidence_report,
    verify_application_receipt,
    verify_run_receipt,
    write_application_receipt,
    write_run_receipt,
)
from fedference.provenance import SourceProvenance, runtime_provenance

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


def test_legacy_run_receipt_round_trips_exact_fields_with_unavailable_runtime(tmp_path) -> None:
    raw = _receipt(tmp_path).as_dict()
    raw["schema_version"] = "1.1"
    del raw["runtime_provenance"]
    loaded = RunReceipt.from_dict(raw)
    assert loaded.as_dict() == raw
    assert loaded.runtime_provenance.distribution_version == "unavailable"
    assert "legacy schema 1.1" in loaded.runtime_provenance.warnings[0]
    direct = replace(_receipt(tmp_path), schema_version="1.1")
    assert direct.runtime_provenance.distribution_version == "unavailable"
    assert "runtime_provenance" not in direct.as_dict()


def test_application_receipt_round_trip_semantics_and_solver_policy(tmp_path) -> None:
    request = LabeledAggregationRequest.from_dict(
        {
            "schema_version": "1.0",
            "state_labels": ["no", "yes"],
            "agents": [
                {"agent_id": "a", "posterior": [0.8, 0.2]},
                {"agent_id": "b", "posterior": [0.3, 0.7]},
            ],
            "aggregation_config": {
                "method": "naive",
                "robustness": 1.0,
                "entropy_weight": 1.0,
                "max_iter": 64,
                "tol": 1e-9,
                "multistart": True,
            },
        }
    )
    result = aggregate_labeled(request)
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    request_path.write_text(json.dumps(request.as_dict()), encoding="utf-8")
    result_path.write_text(json.dumps(result.as_dict()), encoding="utf-8")
    receipt = ApplicationReceipt(
        request_sha256=request.request_sha256,
        config_fingerprint=request.config.fingerprint,
        solver_status=result.aggregation.solver_status,
        fallback_events=result.aggregation.fallback_events,
        outputs=(
            make_artifact_record("request", request_path, root=tmp_path),
            make_artifact_record("result", result_path, root=tmp_path),
        ),
        runtime_provenance=runtime_provenance(),
        source_provenance=SourceProvenance(
            source_kind="unavailable",
            warnings=("test fixture",),
        ),
        started_at_utc="2026-08-25T00:00:00+00:00",
        completed_at_utc="2026-08-25T00:00:01+00:00",
    )
    path = write_application_receipt(tmp_path / "receipt.json", receipt)
    loaded = load_application_receipt(path)
    assert loaded.as_dict() == receipt.as_dict()
    assert verify_application_receipt(loaded, root=tmp_path) == ()
    nonnominal_findings = verify_application_receipt(
        replace(loaded, solver_status="not_converged"),
        root=tmp_path,
        require_nominal_solver=True,
    )
    assert "solver status is 'not_converged', not 'nominal'" in nonnominal_findings
    assert "result solver status does not match receipt" in nonnominal_findings

    version_payload = result.as_dict()
    version_payload["software_version"] = "0.0.0"
    result_path.write_text(json.dumps(version_payload), encoding="utf-8")
    version_tampered = replace(
        loaded,
        outputs=(
            loaded.outputs[0],
            make_artifact_record("result", result_path, root=tmp_path),
        ),
    )
    assert (
        "result software version does not match runtime provenance"
        in verify_application_receipt(version_tampered, root=tmp_path)
    )

    semantic_payload = result.as_dict()
    semantic_payload["normalized_local_posteriors"] = [
        semantic_payload["normalized_local_posteriors"][1],
        semantic_payload["normalized_local_posteriors"][0],
    ]
    result_path.write_text(json.dumps(semantic_payload), encoding="utf-8")
    semantically_tampered = replace(
        loaded,
        outputs=(
            loaded.outputs[0],
            make_artifact_record("result", result_path, root=tmp_path),
        ),
    )
    assert (
        "result normalized local posteriors do not match the request"
        in verify_application_receipt(semantically_tampered, root=tmp_path)
    )

    result_path.write_text("{}\n", encoding="utf-8")
    findings = verify_application_receipt(loaded, root=tmp_path)
    assert "artifact digest mismatch: result.json" in findings
    assert any("result artifact is invalid" in finding for finding in findings)


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
    load_application_receipt,
    verify_application_receipt,
    write_application_receipt,
