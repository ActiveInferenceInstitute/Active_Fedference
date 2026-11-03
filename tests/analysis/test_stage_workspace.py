"""Real staged-analysis boundary regressions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.stage_checkpoint import import_report, load_checkpoint
from analysis.stage_io import StageError, confined, digest, encode, read_json
from analysis.stage_workspace import admit_workspace, prepare_workspace
from tests.analysis.stage_fixtures import cli, run
from tests.analysis.stage_fixtures import prepared as prepared


def test_checkpoint_reuse_does_not_start_another_producer(prepared: tuple[Path, Path]) -> None:
    _, workspace = prepared
    first = run(workspace)
    assert first["status"] == "complete"
    completed = workspace / "completed/belief_sharing"
    before = {p.name: p.read_bytes() for p in completed.iterdir() if p.is_file()}
    attempts = list((workspace / "attempts").iterdir())
    second = run(workspace)
    assert second["status"] == "reused"
    assert attempts == list((workspace / "attempts").iterdir())
    assert before == {p.name: p.read_bytes() for p in completed.iterdir() if p.is_file()}
    assert first["identity"] == second["identity"]
    assert cli(workspace, "status").returncode == 0


@pytest.mark.parametrize("damage", ["bytes", "marker", "schema", "symlink"])
def test_bad_completed_checkpoint_is_refused_without_recompute(
    prepared: tuple[Path, Path],
    damage: str,
) -> None:
    _, workspace = prepared
    run(workspace)
    folder = workspace / "completed/belief_sharing"
    report = folder / "report.json"
    marker = folder / "complete.json"
    if damage == "bytes":
        report.write_bytes(report.read_bytes() + b" ")
    elif damage == "marker":
        value = read_json(marker)
        value["identity"] = "0" * 64
        marker.write_bytes(encode(value))
    elif damage == "schema":
        report.write_bytes(encode({"bad": "report"}))
        value = read_json(marker)
        value["report_sha256"] = digest(report.read_bytes())
        marker.write_bytes(encode(value))
    else:
        target = workspace / "retained-report.json"
        report.rename(target)
        report.symlink_to(target)
    before = list((workspace / "attempts").iterdir())
    result = cli(workspace, "run", "--stage", "belief_sharing", "--max-seconds", "10")
    assert result.returncode != 0
    assert list((workspace / "attempts").iterdir()) == before


@pytest.mark.parametrize("operand", ["src/fedference/aggregation.py", "manuscript/config.yaml", "uv.lock"])
def test_changed_inputs_refuse_compute_and_import(prepared: tuple[Path, Path], operand: str) -> None:
    project, workspace = prepared
    run(workspace)
    destination = project / operand
    destination.write_bytes(destination.read_bytes() + b"\n# changed input\n")
    with pytest.raises(StageError, match="source input changed"):
        import_report(workspace, project, "belief_sharing", "belief_sharing")
    assert not (project / "output/reports/belief_sharing.json").exists()
    source = workspace / "source" / operand
    source.write_bytes(source.read_bytes() + b"\n# changed input\n")
    result = cli(workspace, "run", "--stage", "belief_sharing", "--max-seconds", "10")
    assert result.returncode != 0 and "source input changed" in result.stderr


def test_short_report_import_checks_schema_and_does_not_mint_analysis_acceptance(
    prepared: tuple[Path, Path],
) -> None:
    project, workspace = prepared
    run(workspace)
    result = cli(workspace, "import", "--stage", "belief_sharing", "--project-root", str(project))
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["analysis_acceptance"] is False
    assert (project / "output/reports/belief_sharing.json").read_bytes() == (
        workspace / "completed/belief_sharing/report.json"
    ).read_bytes()
    assert not (project / "output/data/pipeline_provenance.json").exists()


def test_workspace_identity_runtime_and_paths_are_strict(prepared: tuple[Path, Path]) -> None:
    project, workspace = prepared
    with pytest.raises(StageError, match="independent"):
        prepare_workspace(project, project / "nested", profile="smoke")
    for value in ("../escape", "/absolute", "./not-canonical"):
        with pytest.raises(StageError, match="unsafe"):
            confined(workspace, value)
    path = workspace / "bad.json"
    for raw in ('{"x":1,"x":2}', '{"x":NaN}', "[]", "{"):
        path.write_text(raw)
        with pytest.raises(StageError):
            read_json(path)
    manifest = read_json(workspace / "workspace.json")
    manifest["runtime"]["python"] = "0.0.0"
    manifest["identity"] = digest(encode({k: v for k, v in manifest.items() if k != "identity"}))
    (workspace / "workspace.json").write_bytes(encode(manifest))
    with pytest.raises(StageError, match="runtime identity"):
        admit_workspace(workspace)


def test_missing_checkpoint_is_not_a_partial_success(prepared: tuple[Path, Path]) -> None:
    _, workspace = prepared
    manifest = admit_workspace(workspace)
    with pytest.raises(StageError, match="unreadable"):
        load_checkpoint(workspace, "belief_sharing", "belief_sharing", manifest["identity"])


def test_import_refuses_peer_output_and_preserves_admitted_prior_bytes(prepared: tuple[Path, Path]) -> None:
    project, workspace = prepared
    run(workspace)
    target = project / "output/reports/belief_sharing.json"
    target.parent.mkdir(parents=True)
    target.write_text('{"peer":"work"}\n')
    with pytest.raises(StageError, match="destination output changed"):
        import_report(workspace, project, "belief_sharing", "belief_sharing")
    assert target.read_text() == '{"peer":"work"}\n'
    # A new explicit preparation admits the actual prior output; no recovery
    # evidence or previous workspace is removed to obtain success.
    successor = workspace.with_name("successor")
    prepare_workspace(project, successor, profile="smoke")
    run(successor)
    import_report(successor, project, "belief_sharing", "belief_sharing")
    old = b'{"peer":"work"}\n'
    assert (successor / "prior-outputs" / f"{digest(old)}.bin").read_bytes() == old
    assert target.read_bytes() != old
    import_report(successor, project, "belief_sharing", "belief_sharing")


def test_terminal_receipt_is_part_of_checkpoint_admission(prepared: tuple[Path, Path]) -> None:
    _, workspace = prepared
    run(workspace)
    folder = workspace / "completed/belief_sharing"
    result = read_json(folder / "result.json")
    result["status"] = "cancelled"
    (folder / "result.json").write_bytes(encode(result))
    failed = cli(workspace, "status")
    assert failed.returncode != 0 and "result digest mismatch" in failed.stderr
    marker = read_json(folder / "complete.json")
    marker["result_sha256"] = digest((folder / "result.json").read_bytes())
    (folder / "complete.json").write_bytes(encode(marker))
    failed = cli(workspace, "status")
    assert failed.returncode != 0 and "invalid completed process receipt" in failed.stderr


@pytest.mark.parametrize("name", ["source", "completed/belief_sharing"])
def test_workspace_roots_cannot_be_replaced_by_aliases(prepared: tuple[Path, Path], name: str) -> None:
    _, workspace = prepared
    run(workspace)
    target = workspace / name
    retained = target.with_name(target.name + "-retained")
    target.rename(retained)
    target.symlink_to(retained, target_is_directory=True)
    with pytest.raises(StageError, match="symlink|canonical"):
        if name == "source":
            admit_workspace(workspace)
        else:
            manifest = admit_workspace(workspace)
            load_checkpoint(workspace, "belief_sharing", "belief_sharing", manifest["identity"])


def test_worker_refuses_execution_from_a_different_source_copy(prepared: tuple[Path, Path]) -> None:
    _, workspace = prepared
    identity = admit_workspace(workspace)["identity"]
    token = "0" * 32
    attempt = workspace / "attempts" / token
    attempt.mkdir(parents=True)
    (attempt / "started.json").write_bytes(encode({"stage": "belief_sharing", "identity": identity}))
    result = cli(workspace, "worker", "--stage", "belief_sharing", "--attempt", token)
    assert result.returncode != 0 and "executed" in result.stderr and "frozen" in result.stderr
    assert not (attempt / "report.json").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", True),
        ("profile", []),
        ("files", []),
        ("output_baseline", []),
        ("output_baseline", {"../../escape": "0" * 64}),
        ("extra", "unexpected"),
    ],
)
def test_malformed_workspace_fields_are_refused(
    prepared: tuple[Path, Path], field: str, value: object
) -> None:
    _, workspace = prepared
    manifest = read_json(workspace / "workspace.json")
    manifest[field] = value
    manifest["identity"] = digest(encode({k: v for k, v in manifest.items() if k != "identity"}))
    (workspace / "workspace.json").write_bytes(encode(manifest))
    with pytest.raises(StageError):
        admit_workspace(workspace)


def test_added_source_resource_is_not_invisible_to_reuse(prepared: tuple[Path, Path]) -> None:
    project, workspace = prepared
    (workspace / "source/src/new-input.json").write_text("{}")
    with pytest.raises(StageError, match="resource inventory"):
        admit_workspace(workspace)
    # An independently changed destination is refused too.
    (workspace / "source/src/new-input.json").rename(workspace / "retained-new-input.json")
    (project / "src/new-input.json").write_text("{}")
    with pytest.raises(StageError, match="resource inventory"):
        admit_workspace(workspace, project)


def test_new_resource_symlink_is_refused(prepared: tuple[Path, Path]) -> None:
    _, workspace = prepared
    (workspace / "source/src/alias-directory").symlink_to(
        workspace / "source/src/fedference", target_is_directory=True
    )
    with pytest.raises(StageError, match="symlink"):
        admit_workspace(workspace)
