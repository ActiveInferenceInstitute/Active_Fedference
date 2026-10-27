"""Independent byte vectors and malformed evidence boundary controls."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import TypeAlias

import pytest

from fedference.aggregation import AggregationConfig
from fedference.application import LabeledAggregationRequest, aggregate_labeled
from fedference.evidence import (
    ApplicationReceipt,
    DatasetSpec,
    ExperimentSpec,
    RunReceipt,
    SourceReference,
    canonical_sha256,
    load_application_receipt,
    load_receipt,
    load_run_receipt,
    make_artifact_record,
    sha256_file,
    verify_application_receipt,
    verify_run_receipt,
    write_application_receipt,
    write_run_receipt,
)
from fedference.provenance import (
    RuntimeProvenance,
    SourceProvenance,
    collect_source_provenance,
    runtime_provenance,
)

InvalidFieldValue: TypeAlias = str | int | None | tuple[str, ...] | list[str] | list[int] | dict[str, str]


def _run(root: Path) -> RunReceipt:
    config = root / "config.json"
    config.write_bytes(b"{}")
    return RunReceipt(
        run_id="byte-vector", experiment_id="control", experiment_version="1",
        profile="smoke", git_commit="unavailable", git_tree_state="unavailable",
        environment_lock_sha256="a" * 64,
        config_sha256="44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
        dataset_sha256={}, seeds=(0,), device="cpu", backend="numpy",
        fallbacks=(), checkpoints=(),
        outputs=(make_artifact_record("config", config, root=root),),
        status="completed", started_at_utc="2026-07-29T00:00:00+00:00",
        completed_at_utc="2026-07-29T00:00:01+00:00",
        runtime_provenance=RuntimeProvenance.unavailable(warning="synthetic byte fixture"),
    )


def test_known_bytes_and_historical_receipts(tmp_path: Path) -> None:
    """Schema 1.1 is a synthetic fixture, never invented historical runtime."""
    vector = tmp_path / "abc.bin"
    vector.write_bytes(b"abc")
    assert sha256_file(vector) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert canonical_sha256({}) == "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
    independent = '{"a":[1,true,null],"z":"é"}'.encode()
    assert canonical_sha256({"z": "é", "a": [1, True, None]}) == hashlib.sha256(independent).hexdigest()
    current = _run(tmp_path)
    for version in ("1.2", "1.1"):
        receipt = replace(current, schema_version=version)
        path = write_run_receipt(tmp_path / f"schema-{version}.json", receipt)
        before = path.read_bytes()
        loaded = load_receipt(path)
        assert isinstance(loaded, RunReceipt)
        assert loaded.as_dict() == receipt.as_dict()
        assert verify_run_receipt(loaded, root=tmp_path) == ()
        assert path.read_bytes() == before
        if version == "1.1":
            assert "runtime_provenance" not in loaded.as_dict()
            assert loaded.runtime_provenance.distribution_version == "unavailable"
            assert loaded.runtime_provenance.warnings == (
                "runtime provenance is unavailable in legacy schema 1.1",
            )
        print(version, sha256_file(path), loaded.runtime_provenance.warnings)


@pytest.mark.parametrize("invalid", ['{"seed":0,"seed":1}', '{"seed":NaN}', '{"seed":Infinity}'])
def test_duplicate_json_nonfinite_or_tampered_artifact(tmp_path: Path, invalid: str) -> None:
    receipt = _run(tmp_path)
    receipt_path = write_run_receipt(tmp_path / "receipt.json", receipt)
    original = receipt_path.read_bytes()
    config = tmp_path / "config.json"
    config.write_text(invalid, encoding="utf-8")
    findings = verify_run_receipt(receipt, root=tmp_path)
    assert findings == (
        "artifact byte-size mismatch: config.json",
        "artifact digest mismatch: config.json",
        "config artifact is not valid JSON: config.json",
    )
    invalid_receipt = tmp_path / "invalid-receipt.json"
    invalid_receipt.write_text(invalid, encoding="utf-8")
    with pytest.raises(ValueError, match="invalid receipt file") as error:
        load_receipt(invalid_receipt)
    assert error.value.__cause__ is not None
    print(str(error.value.__cause__), findings)
    assert receipt_path.read_bytes() == original


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_hashing_rejects_nonfinite_values(value: float) -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        canonical_sha256({"value": value})


@pytest.mark.parametrize("invalid", ["[]", "null", "{", '{"schema_version":"99"}'])
def test_loading_rejects_non_receipts(tmp_path: Path, invalid: str) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(invalid, encoding="utf-8")
    with pytest.raises(ValueError):
        load_receipt(path)
    assert path.read_text(encoding="utf-8") == invalid


def test_rehashed_config_still_requires_semantic_binding(tmp_path: Path) -> None:
    receipt = _run(tmp_path)
    config = tmp_path / "config.json"
    config.write_bytes(b'{"changed":true}')
    rebound = replace(receipt, outputs=(make_artifact_record("config", config, root=tmp_path),))
    assert verify_run_receipt(rebound, root=tmp_path) == ("configuration hash mismatch: config.json",)
    print(verify_run_receipt(rebound, root=tmp_path))


def test_receipt_schema_preserves_rejection(tmp_path: Path) -> None:
    raw = _run(tmp_path).as_dict()
    raw["unrecognized"] = True
    path = tmp_path / "extra.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="extra=\\['unrecognized'\\]"):
        load_receipt(path)



@pytest.mark.parametrize("changes", [
    {"dataset_sha256": []}, {"seeds": None}, {"git_commit": "bad"},
    {"git_tree_state": "bad"}, {"git_tree_state": "clean"}, {"schema_version": "99"},
    {"runtime_provenance": None}, {"seeds": [-1]}, {"status": "bad"},
    {"fallbacks": [""]}, {"checkpoints": [3]}, {"outputs": ["bad"]},
    {"started_at_utc": "bad"}, {"config_sha256": "z" * 64},
])
def test_run_constructor_invalid_values(tmp_path: Path, changes: Mapping[str, InvalidFieldValue]) -> None:
    with pytest.raises(ValueError):
        replace(_run(tmp_path), **changes)


@pytest.mark.parametrize("changes", [
    {"outputs": {}}, {"dataset_sha256": []}, {"seeds": 0},
])
def test_run_decoder_invalid_shapes(tmp_path: Path, changes: Mapping[str, InvalidFieldValue]) -> None:
    raw = _run(tmp_path).as_dict()
    raw.update(changes)
    with pytest.raises(ValueError):
        RunReceipt.from_dict(raw)


def test_duplicate_output_identity_and_missing_custody(tmp_path: Path) -> None:
    receipt = _run(tmp_path)
    record = receipt.outputs[0]
    for duplicate in (record, replace(record, name="other")):
        with pytest.raises(ValueError, match="unique"):
            replace(receipt, outputs=(record, duplicate))
    with pytest.raises(ValueError, match="non-negative"):
        replace(record, bytes=-1)
    for path in (tmp_path / "absent", tmp_path.parent):
        with pytest.raises(ValueError):
            make_artifact_record("missing", path, root=tmp_path)
    missing = replace(receipt, outputs=(replace(record, path="missing.json"),))
    assert verify_run_receipt(missing, root=tmp_path) == ("missing artifact: missing.json",)
    findings = verify_run_receipt(receipt, root=tmp_path, project_root=tmp_path)
    assert any("live Git state is unavailable" in item for item in findings)
    assert any("live environment lock is missing" in item for item in findings)


@pytest.mark.parametrize("changes", [
    {"preprocessing": None}, {"schema": None}, {"file_format": "bad"}, {"n_rows": 0},
    {"has_missing_values": 1}, {"schema": [""]}, {"split_seed": -1},
    {"split_sha256": "a" * 64},
])
def test_dataset_rejects_invalid_fields(changes: Mapping[str, InvalidFieldValue]) -> None:
    dataset = DatasetSpec(
        dataset_id="d", name="D", source_url="https://example.test", doi="10.0/test",
        license="test", archive_sha256="a" * 64, archive_member="data.csv",
        file_format="csv", n_rows=2, n_features=1, n_classes=2, has_missing_values=False,
        preprocessing=("parse",), schema=("label",), split_policy="seeded",
    )
    with pytest.raises(ValueError):
        replace(dataset, **changes)


@pytest.mark.parametrize("changes", [
    {"source_ids": None}, {"profiles": None}, {"state": "bad"},
    {"source_ids": ()}, {"confirmatory_ready": 1}, {"runner": ""},
])
def test_experiment_rejects_invalid_fields(changes: Mapping[str, InvalidFieldValue]) -> None:
    experiment = ExperimentSpec(
        experiment_id="e", version="1", title="E", state="planned", source_ids=("s",),
        primary_estimand="score", independent_unit="world", falsifier="reversal",
        no_claim="synthetic", profiles=("smoke",), smallest_effect_of_interest="pilot",
        mcse_stopping_target="pilot", maximum_budget="one", comparison_family="one",
    )
    with pytest.raises(ValueError):
        replace(experiment, **changes)


def _application(root: Path) -> ApplicationReceipt:
    request = LabeledAggregationRequest.from_dict({
        "schema_version": "1.0", "state_labels": ["no", "yes"],
        "agents": [{"agent_id": "a", "posterior": [0.8, 0.2]}],
        "aggregation_config": AggregationConfig(method="naive").as_dict(),
    })
    result = aggregate_labeled(request)
    records = []
    for name, payload in (("request", request.as_dict()), ("result", result.as_dict())):
        path = root / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        records.append(make_artifact_record(name, path, root=root))
    return ApplicationReceipt(
        request_sha256=request.request_sha256, config_fingerprint=request.config.fingerprint,
        solver_status=result.aggregation.solver_status, fallback_events=result.aggregation.fallback_events,
        outputs=tuple(records), runtime_provenance=runtime_provenance(),
        source_provenance=SourceProvenance(source_kind="unavailable"),
        started_at_utc="2026-07-29T00:00:00+00:00", completed_at_utc="2026-07-29T00:00:01+00:00",
    )


@pytest.mark.parametrize("changes", [
    {"schema_version": "99"}, {"receipt_type": "run"}, {"operation": "bad"}, {"status": "bad"},
    {"solver_status": "bad"}, {"fallback_events": [""]}, {"fallback_events": ["fallback"]},
    {"outputs": ["bad"]}, {"outputs": ()}, {"runtime_provenance": None}, {"source_provenance": None},
    {"completed_at_utc": "2026-07-28T00:00:00+00:00"},
])
def test_application_rejects_invalid_fields(tmp_path: Path, changes: Mapping[str, InvalidFieldValue]) -> None:
    with pytest.raises(ValueError):
        replace(_application(tmp_path), **changes)


@pytest.mark.parametrize("changes", [
    {"outputs": {}}, {"fallback_events": None}, {"unrecognized": 1}, {"solver_status": "bad"},
])
def test_application_decoder_rejects_shapes(tmp_path: Path, changes: Mapping[str, InvalidFieldValue]) -> None:
    raw = _application(tmp_path).as_dict()
    raw.update(changes)
    with pytest.raises(ValueError):
        ApplicationReceipt.from_dict(raw)


def test_application_custody_and_solver_scopes(tmp_path: Path) -> None:
    receipt = _application(tmp_path)
    path = write_application_receipt(tmp_path / "receipt.json", receipt)
    assert load_receipt(path) == load_application_receipt(path)
    assert verify_application_receipt(receipt, root=tmp_path, scope="solver") == ()
    failed = replace(receipt, solver_status="not_converged")
    assert verify_application_receipt(failed, root=tmp_path, scope="solver", require_nominal_solver=True)
    assert verify_application_receipt(receipt, root=tmp_path, scope="source", require_clean_git=True)
    for name in ("request", "result"):
        record = receipt.outputs[0 if name == "request" else 1]
        for new_path in (receipt.outputs[1].path, "different.json"):
            records = (replace(receipt.outputs[0], path=new_path), receipt.outputs[1])
            with pytest.raises(ValueError):
                replace(receipt, outputs=records)
        (tmp_path / record.path).unlink()
    assert verify_application_receipt(receipt, root=tmp_path) == (
        "missing artifact: request.json", "missing artifact: result.json",
    )
    for loader in (load_run_receipt, load_application_receipt, load_receipt):
        with pytest.raises(ValueError):
            loader(tmp_path / "missing")
        path.write_text("[]", encoding="utf-8")
        with pytest.raises(ValueError):
            loader(path)



@pytest.mark.parametrize("changes, expected", [
    ({"request_sha256": "b" * 64}, "request semantic hash does not match receipt"),
    ({"config_fingerprint": "b" * 64}, "request configuration fingerprint does not match receipt"),
    ({"solver_status": "converged_with_fallback", "fallback_events": ("control",)},
     "result fallback events do not match receipt"),
])
def test_application_semantic_receipt_mismatch(
    tmp_path: Path, changes: Mapping[str, InvalidFieldValue], expected: str
) -> None:
    receipt = replace(_application(tmp_path), **changes)
    assert expected in verify_application_receipt(receipt, root=tmp_path)


@pytest.mark.parametrize("field, value, expected", [
    ("state_labels", ["yes", "no"], "request and result state-label order do not match"),
    ("agent_ids", ["other"], "request and result agent order do not match"),
])
def test_application_result_identity_mismatch(
    tmp_path: Path, field: str, value: list[str], expected: str
) -> None:
    receipt = _application(tmp_path)
    result = tmp_path / "result.json"
    raw = json.loads(result.read_text(encoding="utf-8"))
    raw[field] = value
    result.write_text(json.dumps(raw), encoding="utf-8")
    assert expected in verify_application_receipt(receipt, root=tmp_path)


def test_application_source_mismatch_uses_real_checkout(tmp_path: Path) -> None:
    receipt = _application(tmp_path)
    project = Path(__file__).resolve().parents[2]
    source = collect_source_provenance(project)
    assert source.source_kind == "git_checkout"
    mismatched = replace(source, git_commit="0" * 40, git_tree_state="dirty", uv_lock_sha256="0" * 64)
    bound = replace(receipt, source_provenance=mismatched)
    findings = verify_application_receipt(bound, root=tmp_path, project_root=project, scope="source")
    assert any("live Git commit does not match receipt" in item for item in findings)
    assert "live uv.lock digest does not match receipt" in findings
    assert any("unhashed dirty tree" in item for item in findings)
    assert "receipt does not bind an explicit Git checkout" in verify_application_receipt(
        receipt, root=tmp_path, project_root=project, scope="source",
    )
    assert any("live Git source is unavailable" in item for item in verify_application_receipt(
        receipt, root=tmp_path, project_root=tmp_path, scope="source",
    ))


@pytest.mark.parametrize("field", ["source_id", "title", "url", "role", "doi", "revision"])
@pytest.mark.parametrize("invalid", ["", 42])
def test_source_reference_rejects_invalid_declared_text(field: str, invalid: str | int) -> None:
    reference = SourceReference(
        "source", "Declared source", "https://example.test/source", "implementation",
        doi="10.0/example", revision="a" * 40,
    )
    with pytest.raises(ValueError, match=field):
        replace(reference, **{field: invalid})
