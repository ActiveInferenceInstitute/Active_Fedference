"""Installed CLI registry, evidence run, and verification workflow."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from fedference.evidence import load_run_receipt
from fedference.federation import run_socket_round
from fedference_cli import _report_fallbacks, main

ROOT = Path(__file__).resolve().parents[1]


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
    assert "no executable CLI runner" in capsys.readouterr().err
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
                "--max-iter",
                "32",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == "PASS"
