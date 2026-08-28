"""Installed CLI registry, evidence run, and verification workflow."""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from fedference import __version__
from fedference.aggregation import AggregationConfig
from fedference.evidence import load_run_receipt
from fedference.federation import run_socket_round
from fedference_cli import _report_fallbacks, main
from fedference_cli._parser import main as parser_main
from fedference_cli._support import _report_fallbacks as support_report_fallbacks

ROOT = Path(__file__).resolve().parents[1]


def _write_active_checkout_identity(
    root: Path,
    *,
    version: str = __version__,
    project_name: str = "active_fedference",
) -> None:
    package = root / "src" / "fedference"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("\n", encoding="utf-8")
    (package / "aggregation.py").write_text("\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        "[project]\n"
        f'name = "{project_name}"\n'
        f'version = "{version}"\n',
        encoding="utf-8",
    )


def _labeled_request(*, method: str = "naive", max_iter: int = 64) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "state_labels": ["clear", "warning", "critical"],
        "agents": [
            {"agent_id": "agent-0", "posterior": [0.9, 0.09, 0.01]},
            {"agent_id": "agent-1", "posterior": [0.01, 0.09, 0.9]},
            {"agent_id": "agent-2", "posterior": [0.4, 0.3, 0.3]},
        ],
        "aggregation_config": {
            "method": method,
            "robustness": 1.5,
            "entropy_weight": 1.0,
            "max_iter": max_iter,
            "tol": 0.0 if max_iter == 1 else 1e-9,
            "multistart": False,
        },
    }


def test_cli_facade_preserves_public_imports_and_delegates_by_responsibility() -> None:
    facade = (ROOT / "src/fedference_cli/__init__.py").read_text(encoding="utf-8")
    assert main is parser_main
    assert _report_fallbacks is support_report_fallbacks
    assert "argparse" not in facade
    assert "run_external_benchmark_pack" not in facade
    assert "def _build_parser" not in facade
    assert len(facade.splitlines()) <= 24


def test_report_fallbacks_summarizes_solver_health_without_pseudoreplication() -> None:
    report = {
        "rows": [
            {
                "dataset_id": "uci-dry-bean",
                "seed": 7,
                "n_test": 100,
                "variational_fallback_predictions": 3,
                "variational_nonconverged_predictions": 4,
            }
        ]
    }
    assert _report_fallbacks(report) == (
        "aggregation fallback: dataset=uci-dry-bean seed=7 "
        "method=variational predictions=3/100",
        "aggregation non-convergence: dataset=uci-dry-bean seed=7 "
        "method=variational predictions=4/100",
    )


def test_cli_list_is_machine_readable(capsys) -> None:
    assert main(["list", "--json"]) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["schema_version"] == "1.0"
    assert {row["experiment_id"] for row in manifest["experiments"]} >= {
        "server-theory",
        "external-tabular",
    }


def test_cli_server_theory_smoke_writes_and_verifies_receipt(tmp_path, capsys) -> None:
    output_dir = tmp_path / "run"
    assert (
        main(
            [
                "run",
                "server-theory",
                "--profile",
                "smoke",
                "--seed",
                "0",
                "--output-dir",
                str(output_dir),
                "--project-root",
                str(ROOT),
            ]
        )
        == 0
    )
    paths = json.loads(capsys.readouterr().out)
    receipt_path = Path(paths["receipt"])
    receipt = load_run_receipt(receipt_path)
    assert receipt.experiment_id == "server-theory"
    assert receipt.status == "completed"
    assert receipt.git_tree_state in {"clean", "dirty"}
    assert len(receipt.git_commit) == 40
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    assert config["experiment_id"] == "server-theory"
    assert {artifact.name for artifact in receipt.outputs} == {"config", "report"}
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    declaration = report["experiment_spec"]
    assert declaration["source_bundle"]
    assert declaration["smallest_effect_of_interest"]
    assert declaration["mcse_stopping_target"]
    assert declaration["maximum_budget"]
    assert declaration["comparison_family"]
    assert declaration["confirmatory_ready"] is False
    assert main(["verify", str(receipt_path)]) == 0
    assert "PASS:" in capsys.readouterr().out


def test_cli_refuses_committed_reviewer_output(tmp_path) -> None:
    project = tmp_path / "project"
    (project / "output").mkdir(parents=True)
    _write_active_checkout_identity(project)
    try:
        main(
            [
                "run",
                "server-theory",
                "--output-dir",
                str(project / "output" / "run"),
                "--project-root",
                str(project),
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("CLI accepted the committed reviewer output tree")


def test_python_module_entrypoint_lists_registry() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "fedference_cli", "list"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "server-theory" in result.stdout


def test_confirmatory_profile_is_blocked_until_pilot_freezes_it(
    tmp_path,
    capsys,
) -> None:
    output = tmp_path / "confirmatory"
    with pytest.raises(SystemExit):
        main(
            [
                "benchmark",
                "--dataset-id",
                "uci-banknote",
                "--profile",
                "confirmatory",
                "--cache-dir",
                str(tmp_path / "cache"),
                "--output-dir",
                str(output),
            ]
        )
    assert "blocked until its pilot freezes" in capsys.readouterr().err
    assert not output.exists()


def test_cli_validates_before_creating_output_directory(tmp_path, capsys) -> None:
    output = tmp_path / "invalid"
    with pytest.raises(SystemExit):
        main(
            [
                "run",
                "server-theory",
                "--seed",
                "1",
                "--seed",
                "1",
                "--output-dir",
                str(output),
                "--project-root",
                str(ROOT),
            ]
        )
    assert "seeds must be unique" in capsys.readouterr().err
    assert not output.exists()

    with pytest.raises(SystemExit):
        main(
            [
                "run",
                "multi-node-emulator",
                "--output-dir",
                str(output),
                "--project-root",
                str(ROOT),
            ]
        )
    assert "invalid choice" in capsys.readouterr().err
    assert not output.exists()


def test_cli_calibration_pilot_writes_a_verifiable_receipt(tmp_path, capsys) -> None:
    output = tmp_path / "calibration"
    assert main(
        [
            "run",
            "robustness-calibration",
            "--profile",
            "smoke",
            "--seed",
            "0",
            "--output-dir",
            str(output),
            "--project-root",
            str(ROOT),
        ]
    ) == 0
    paths = json.loads(capsys.readouterr().out)
    receipt = load_run_receipt(paths["receipt"])
    assert receipt.experiment_id == "robustness-calibration"
    assert main(["verify", paths["receipt"]]) == 0
    assert "overlap_negative_control" in json.loads((output / "report.json").read_text())


def test_nested_project_root_still_protects_reviewer_snapshot(tmp_path, capsys) -> None:
    protected = ROOT / "output" / "cli-must-not-exist"
    with pytest.raises(SystemExit):
        main(
            [
                "run",
                "server-theory",
                "--output-dir",
                str(protected),
                "--project-root",
                str(ROOT / "docs"),
            ]
        )
    assert "committed reviewer output tree" in capsys.readouterr().err
    assert not protected.exists()


def test_cli_strict_verification_rejects_dirty_receipt(tmp_path, capsys) -> None:
    output = tmp_path / "run"
    assert (
        main(
            [
                "run",
                "server-theory",
                "--output-dir",
                str(output),
                "--project-root",
                str(ROOT),
            ]
        )
        == 0
    )
    receipt_path = json.loads(capsys.readouterr().out)["receipt"]
    receipt = load_run_receipt(receipt_path)
    result = main(["verify", receipt_path, "--require-clean-git"])
    output_text = capsys.readouterr().out
    if receipt.git_tree_state == "clean":
        assert result == 0
        assert "PASS:" in output_text
    else:
        assert result == 1
        assert "git tree state" in output_text


def test_cli_replay_verifies_a_real_socket_round(tmp_path, capsys) -> None:
    beliefs = np.asarray([[0.8, 0.2], [0.7, 0.3], [0.1, 0.9]])
    replay_path = tmp_path / "replay.json"
    result = run_socket_round(
        beliefs,
        robustness=1.5,
        round_id="cli-replay",
        replay_path=replay_path,
    )
    assert result["aggregation_config"]["max_iter"] == 32
    beliefs_path = tmp_path / "beliefs.json"
    consensus_path = tmp_path / "consensus.json"
    beliefs_path.write_text(json.dumps(beliefs.tolist()), encoding="utf-8")
    consensus_path.write_text(
        json.dumps(result["consensus"].tolist()),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "replay",
                "--replay",
                str(replay_path),
                "--beliefs",
                str(beliefs_path),
                "--consensus",
                str(consensus_path),
                "--robustness",
                "1.5",
            ]
        )
        == 0
    )
    replay_output = capsys.readouterr().out.strip()
    assert replay_output.startswith("PASS: replay integrity verified")
    assert "solver_status=nominal" in replay_output


def test_cli_replay_reads_recorded_config_and_emits_structured_findings(
    tmp_path,
    capsys,
) -> None:
    beliefs = np.asarray([[0.8, 0.2], [0.7, 0.3], [0.1, 0.9]])
    replay_path = tmp_path / "replay.json"
    result = run_socket_round(
        beliefs,
        robustness=1.5,
        round_id="cli-recorded-config",
        replay_path=replay_path,
    )
    beliefs_path = tmp_path / "beliefs.json"
    consensus_path = tmp_path / "consensus.json"
    beliefs_path.write_text(json.dumps(beliefs.tolist()), encoding="utf-8")
    consensus_path.write_text(json.dumps(result["consensus"].tolist()), encoding="utf-8")
    base_args = [
        "replay",
        "--replay",
        str(replay_path),
        "--beliefs",
        str(beliefs_path),
        "--consensus",
        str(consensus_path),
    ]
    assert main([*base_args, "--json"]) == 0
    verdict = json.loads(capsys.readouterr().out)
    assert verdict["integrity_valid"] is True
    assert verdict["aggregation_config"]["robustness"] == 1.5
    assert verdict["solver_status"] == "nominal"

    assert main([*base_args, "--robustness", "2.0"]) == 1
    failure = capsys.readouterr().out
    assert "configuration.mismatch" in failure
    assert "PASS" not in failure


def test_cli_replay_can_require_nominal_solver_health(tmp_path, capsys) -> None:
    beliefs = np.asarray([[0.9, 0.09, 0.01], [0.01, 0.09, 0.9], [0.4, 0.3, 0.3]])
    config = AggregationConfig(
        method="robust",
        robustness=1.5,
        max_iter=1,
        tol=0.0,
        multistart=False,
    )
    replay_path = tmp_path / "replay.json"
    result = run_socket_round(
        beliefs,
        config=config,
        round_id="cli-nonconverged",
        replay_path=replay_path,
    )
    beliefs_path = tmp_path / "beliefs.json"
    consensus_path = tmp_path / "consensus.json"
    beliefs_path.write_text(json.dumps(beliefs.tolist()), encoding="utf-8")
    consensus_path.write_text(json.dumps(result["consensus"].tolist()), encoding="utf-8")
    args = [
        "replay",
        "--replay",
        str(replay_path),
        "--beliefs",
        str(beliefs_path),
        "--consensus",
        str(consensus_path),
    ]
    assert main(args) == 0
    assert "solver.nonconvergence" in capsys.readouterr().out
    assert main([*args, "--require-nominal-solver"]) == 1
    output = capsys.readouterr().out
    assert "FAIL: solver.nonconvergence" in output


def test_cli_replay_maps_malformed_numeric_json_to_parser_errors(tmp_path, capsys) -> None:
    beliefs = np.asarray([[0.8, 0.2], [0.7, 0.3], [0.1, 0.9]])
    replay_path = tmp_path / "replay.json"
    result = run_socket_round(
        beliefs,
        robustness=1.5,
        round_id="cli-malformed-json",
        replay_path=replay_path,
    )
    beliefs_path = tmp_path / "beliefs.json"
    consensus_path = tmp_path / "consensus.json"
    valid_beliefs = json.dumps(beliefs.tolist())
    valid_consensus = json.dumps(result["consensus"].tolist())

    for malformed_path, valid_other, expected in (
        (beliefs_path, (consensus_path, valid_consensus), "beliefs must contain a numeric JSON array"),
        (
            consensus_path,
            (beliefs_path, valid_beliefs),
            "consensus must contain a numeric JSON array",
        ),
    ):
        malformed_path.write_text("{}\n", encoding="utf-8")
        valid_other[0].write_text(valid_other[1], encoding="utf-8")
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "replay",
                    "--replay",
                    str(replay_path),
                    "--beliefs",
                    str(beliefs_path),
                    "--consensus",
                    str(consensus_path),
                    "--robustness",
                    "1.5",
                ]
            )
        assert exc_info.value.code == 2
        stderr = capsys.readouterr().err
        assert expected in stderr
        assert "Traceback" not in stderr

    for invalid_beliefs, expected in (
        ([[-0.1, 1.1], [0.7, 0.3], [0.1, 0.9]], "must be non-negative"),
        ([[0.0, 0.0], [0.7, 0.3], [0.1, 0.9]], "positive finite sum"),
    ):
        beliefs_path.write_text(json.dumps(invalid_beliefs), encoding="utf-8")
        consensus_path.write_text(valid_consensus, encoding="utf-8")
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "replay",
                    "--replay",
                    str(replay_path),
                    "--beliefs",
                    str(beliefs_path),
                    "--consensus",
                    str(consensus_path),
                ]
            )
        assert exc_info.value.code == 2
        assert expected in capsys.readouterr().err

    beliefs_path.write_text(valid_beliefs, encoding="utf-8")
    consensus_path.write_text(json.dumps([2.0, 1.0]), encoding="utf-8")
    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "replay",
                "--replay",
                str(replay_path),
                "--beliefs",
                str(beliefs_path),
                "--consensus",
                str(consensus_path),
            ]
        )
    assert exc_info.value.code == 2
    assert "unit probability vector" in capsys.readouterr().err

    invalid_typed_inputs = (
        (
            beliefs_path,
            [[True, False], [0.7, 0.3], [0.1, 0.9]],
            consensus_path,
            json.loads(valid_consensus),
            "beliefs",
        ),
        (
            beliefs_path,
            [["0.8", "0.2"], [0.7, 0.3], [0.1, 0.9]],
            consensus_path,
            json.loads(valid_consensus),
            "beliefs",
        ),
        (
            consensus_path,
            [True, False],
            beliefs_path,
            json.loads(valid_beliefs),
            "consensus",
        ),
        (
            consensus_path,
            ["0.5", "0.5"],
            beliefs_path,
            json.loads(valid_beliefs),
            "consensus",
        ),
    )
    for invalid_path, invalid_value, other_path, other_value, name in invalid_typed_inputs:
        invalid_path.write_text(json.dumps(invalid_value), encoding="utf-8")
        other_path.write_text(json.dumps(other_value), encoding="utf-8")
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "replay",
                    "--replay",
                    str(replay_path),
                    "--beliefs",
                    str(beliefs_path),
                    "--consensus",
                    str(consensus_path),
                ]
            )
        assert exc_info.value.code == 2
        stderr = capsys.readouterr().err
        assert f"{name} must contain only numeric non-boolean JSON values" in stderr
        assert "Traceback" not in stderr


def test_cli_replay_accepts_positive_nonunit_belief_masses(tmp_path, capsys) -> None:
    beliefs = np.asarray([[8.0, 2.0], [3.5, 1.5], [1.0, 9.0]])
    replay_path = tmp_path / "replay.json"
    result = run_socket_round(
        beliefs,
        robustness=1.5,
        round_id="cli-nonunit-beliefs",
        replay_path=replay_path,
    )
    beliefs_path = tmp_path / "beliefs.json"
    consensus_path = tmp_path / "consensus.json"
    beliefs_path.write_text(json.dumps(beliefs.tolist()), encoding="utf-8")
    consensus_path.write_text(
        json.dumps(result["consensus"].tolist()),
        encoding="utf-8",
    )
    assert main(
        [
            "replay",
            "--replay",
            str(replay_path),
            "--beliefs",
            str(beliefs_path),
            "--consensus",
            str(consensus_path),
        ]
    ) == 0
    assert "PASS: replay integrity verified" in capsys.readouterr().out


def test_cli_replay_maps_malformed_worker_ids_to_usage_errors(
    tmp_path,
    capsys,
) -> None:
    beliefs = np.asarray([[0.8, 0.2], [0.7, 0.3], [0.1, 0.9]])
    replay_path = tmp_path / "replay.json"
    result = run_socket_round(
        beliefs,
        robustness=1.5,
        round_id="cli-malformed-worker-ids",
    )
    beliefs_path = tmp_path / "beliefs.json"
    consensus_path = tmp_path / "consensus.json"
    beliefs_path.write_text(json.dumps(beliefs.tolist()), encoding="utf-8")
    consensus_path.write_text(
        json.dumps(result["consensus"].tolist()),
        encoding="utf-8",
    )
    args = [
        "replay",
        "--replay",
        str(replay_path),
        "--beliefs",
        str(beliefs_path),
        "--consensus",
        str(consensus_path),
    ]
    for event_name in ("belief_received", "consensus_broadcast"):
        for invalid in (False, 0.0, "0", [], {}):
            replay = deepcopy(result["replay"])
            event = next(
                event
                for event in replay
                if event["event"] == event_name and event["worker_id"] == 0
            )
            event["worker_id"] = invalid
            replay_path.write_text(json.dumps(replay), encoding="utf-8")
            with pytest.raises(SystemExit) as exc_info:
                main(args)
            assert exc_info.value.code == 2
            stderr = capsys.readouterr().err
            assert "worker_id values must be exact non-boolean JSON integers" in stderr
            assert "Traceback" not in stderr


def test_cli_aggregate_writes_canonical_application_artifacts_and_verifies(
    tmp_path,
    capsys,
) -> None:
    input_path = tmp_path / "input.json"
    output_dir = tmp_path / "application"
    input_path.write_text(json.dumps(_labeled_request()), encoding="utf-8")
    assert main(
        [
            "aggregate",
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
        ]
    ) == 0
    paths = json.loads(capsys.readouterr().out)
    assert paths["solver_status"] == "nominal"
    assert set(path.name for path in output_dir.iterdir()) == {
        "request.json",
        "result.json",
        "receipt.json",
    }
    canonical_request = json.loads((output_dir / "request.json").read_text())
    assert canonical_request["agents"][0]["base_weight"] == 1.0
    result = json.loads((output_dir / "result.json").read_text())
    assert result["operation"] == "labeled_categorical_aggregation"
    assert "decision" not in result
    receipt = json.loads((output_dir / "receipt.json").read_text())
    assert receipt["receipt_type"] == "application"
    assert [artifact["name"] for artifact in receipt["outputs"]] == [
        "request",
        "result",
    ]
    assert main(["verify", paths["receipt"]]) == 0
    verification = capsys.readouterr().out
    assert "application artifact integrity verified" in verification
    assert "source equivalence" in verification


def test_cli_aggregate_nonconvergence_is_retained_and_policy_visible(tmp_path, capsys) -> None:
    input_path = tmp_path / "input.json"
    output_dir = tmp_path / "application"
    input_path.write_text(
        json.dumps(_labeled_request(method="robust", max_iter=1)),
        encoding="utf-8",
    )
    assert main(
        [
            "aggregate",
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
        ]
    ) == 1
    paths = json.loads(capsys.readouterr().out)
    assert paths["solver_status"] == "not_converged"
    assert (output_dir / "receipt.json").is_file()
    assert main(["verify", paths["receipt"]]) == 0
    capsys.readouterr()
    assert main(
        ["verify", paths["receipt"], "--require-nominal-solver"]
    ) == 1
    assert "not 'nominal'" in capsys.readouterr().out


def test_cli_aggregate_rejects_malformed_input_and_unsafe_destinations_before_write(
    tmp_path,
    capsys,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version":"1.0","schema_version":"1.0"}\n',
        encoding="utf-8",
    )
    output = tmp_path / "not-created"
    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "aggregate",
                "--input",
                str(duplicate),
                "--output-dir",
                str(output),
            ]
        )
    assert exc_info.value.code == 2
    assert "duplicate JSON key" in capsys.readouterr().err
    assert not output.exists()

    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps(_labeled_request()), encoding="utf-8")
    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "owned.txt").write_text("keep\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        main(
            [
                "aggregate",
                "--input",
                str(valid),
                "--output-dir",
                str(nonempty),
            ]
        )
    assert "must be empty" in capsys.readouterr().err
    assert (nonempty / "owned.txt").read_text() == "keep\n"


def test_cli_application_source_equivalence_is_checkout_path_independent(
    tmp_path,
    capsys,
) -> None:
    source = tmp_path / "source"
    clone = tmp_path / "clone"
    source.mkdir()
    _write_active_checkout_identity(source)
    (source / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (source / "tracked.txt").write_text("source\n", encoding="utf-8")
    for command in (
        ("git", "init", "-q"),
        ("git", "config", "user.email", "test@example.com"),
        ("git", "config", "user.name", "CLI Test"),
        ("git", "add", "."),
        ("git", "commit", "-qm", "fixture"),
    ):
        subprocess.run(command, cwd=source, check=True)
    subprocess.run(("git", "clone", "-q", str(source), str(clone)), check=True)
    request = tmp_path / "input.json"
    request.write_text(json.dumps(_labeled_request()), encoding="utf-8")
    output = tmp_path / "application"
    assert main(
        [
            "aggregate",
            "--input",
            str(request),
            "--output-dir",
            str(output),
            "--project-root",
            str(source),
            "--require-clean-git",
        ]
    ) == 0
    receipt = json.loads(capsys.readouterr().out)["receipt"]
    assert main(
        [
            "verify",
            receipt,
            "--project-root",
            str(clone),
            "--require-clean-git",
        ]
    ) == 0
    assert "source equivalence verified" in capsys.readouterr().out

    (clone / "tracked.txt").write_text("changed\n", encoding="utf-8")
    assert main(
        [
            "verify",
            receipt,
            "--project-root",
            str(clone),
            "--require-clean-git",
        ]
    ) == 1
    source_failure = capsys.readouterr().out
    assert "PASS: application artifact integrity verified" in source_failure
    assert "FAIL: source equivalence:" in source_failure

    (clone / "tracked.txt").write_text("source\n", encoding="utf-8")
    result_path = output / "result.json"
    result_path.write_text(
        result_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    assert main(
        [
            "verify",
            receipt,
            "--project-root",
            str(clone),
            "--require-clean-git",
        ]
    ) == 1
    artifact_failure = capsys.readouterr().out
    assert "FAIL: application artifact integrity:" in artifact_failure
    assert "PASS: source equivalence verified" in artifact_failure


def test_cli_installed_aggregate_allows_output_tree_in_unrelated_repo(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    subprocess.run(("git", "init", "-q"), cwd=unrelated, check=True)
    request = unrelated / "request.json"
    request.write_text(json.dumps(_labeled_request()), encoding="utf-8")
    output = unrelated / "output" / "run"
    monkeypatch.chdir(unrelated)
    assert main(
        [
            "aggregate",
            "--input",
            str(request),
            "--output-dir",
            str(output),
        ]
    ) == 0
    capsys.readouterr()
    assert (output / "receipt.json").is_file()


@pytest.mark.parametrize(
    ("project_name", "version", "expected"),
    (
        ("other_project", __version__, "Active Fedference checkout"),
        ("active_fedference", "0.0.0", "does not match the running distribution"),
    ),
)
def test_cli_aggregate_rejects_wrong_or_version_mismatched_project_root(
    tmp_path,
    capsys,
    project_name: str,
    version: str,
    expected: str,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    _write_active_checkout_identity(
        project,
        version=version,
        project_name=project_name,
    )
    (project / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    for command in (
        ("git", "init", "-q"),
        ("git", "config", "user.email", "test@example.com"),
        ("git", "config", "user.name", "CLI Test"),
        ("git", "add", "."),
        ("git", "commit", "-qm", "fixture"),
    ):
        subprocess.run(command, cwd=project, check=True)
    request = tmp_path / "request.json"
    request.write_text(json.dumps(_labeled_request()), encoding="utf-8")
    output = tmp_path / "application"
    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "aggregate",
                "--input",
                str(request),
                "--output-dir",
                str(output),
                "--project-root",
                str(project),
            ]
        )
    assert exc_info.value.code == 2
    assert expected in capsys.readouterr().err
    assert not output.exists()


def test_cli_rejects_nominal_solver_policy_for_research_receipt(tmp_path, capsys) -> None:
    output = tmp_path / "research"
    assert main(
        [
            "run",
            "server-theory",
            "--output-dir",
            str(output),
            "--project-root",
            str(ROOT),
        ]
    ) == 0
    receipt = json.loads(capsys.readouterr().out)["receipt"]
    with pytest.raises(SystemExit) as exc_info:
        main(["verify", receipt, "--require-nominal-solver"])
    assert exc_info.value.code == 2
    assert "applies only to application receipts" in capsys.readouterr().err
