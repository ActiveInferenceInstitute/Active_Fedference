"""Real-subprocess regression coverage for the public runnable examples."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
SRC = ROOT / "src"


def _environment() -> dict[str, str]:
    """Expose the source checkout exactly as the locked editable install does."""
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(SRC) if not existing else f"{SRC}{os.pathsep}{existing}"
    environment.setdefault("MPLBACKEND", "Agg")
    return environment


def _run_example(
    name: str,
    *arguments: str,
    cwd: Path,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    """Execute one example with the real interpreter and public package."""
    return subprocess.run(
        [sys.executable, str(EXAMPLES / name), *arguments],
        cwd=cwd,
        capture_output=True,
        text=True,
        env=_environment(),
        check=False,
        timeout=timeout,
    )


def _successful_json(result: subprocess.CompletedProcess[str]) -> dict:
    assert result.returncode == 0, (
        f"example exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    payload = json.loads(result.stdout)
    assert isinstance(payload, dict)
    return payload


def test_minimal_aggregation_is_deterministic_and_uses_rich_result(tmp_path: Path) -> None:
    first = _run_example("01_minimal_aggregation.py", cwd=tmp_path)
    second = _run_example("01_minimal_aggregation.py", cwd=tmp_path)
    payload = _successful_json(first)
    _successful_json(second)
    assert first.stdout == second.stdout
    assert payload["example"] == "minimal_aggregation"
    assert payload["config"]["method"] == "naive"
    assert payload["result"] == {
        "consensus": [0.929889, 0.066421, 0.00369],
        "converged": True,
        "fallback_events": [],
        "iterations": 0,
        "normalized_effective_weights": [0.333333, 0.666667],
        "raw_effective_weights": [1.0, 2.0],
        "sums_to_one": True,
    }
    assert payload["validation_error"] == {
        "message": "local_posteriors[1] must be non-negative",
        "type": "ValueError",
    }


def test_method_comparison_pins_declared_rules_and_scope(tmp_path: Path) -> None:
    first = _run_example("02_compare_aggregation_methods.py", cwd=tmp_path)
    second = _run_example("02_compare_aggregation_methods.py", cwd=tmp_path)
    payload = _successful_json(first)
    _successful_json(second)
    assert first.stdout == second.stdout
    assert payload["example"] == "compare_aggregation_methods"
    expected = {
        "naive": (
            [0.75, 0.107143, 0.142857],
            [0.333333, 0.333333, 0.333333],
            0,
        ),
        "robust": (
            [0.818356, 0.147523, 0.034121],
            [0.518309, 0.463017, 0.018674],
            27,
        ),
        "variational": (
            [0.416799, 0.309848, 0.273353],
            [0.384618, 0.421713, 0.193669],
            17,
        ),
    }
    for method, (consensus, weights, iterations) in expected.items():
        result = payload["methods"][method]
        assert result["consensus"] == consensus
        assert result["normalized_effective_weights"] == weights
        assert result["iterations"] == iterations
        assert result["converged"] is True
        assert result["fallback_events"] == []
        assert len(result["config_fingerprint"]) == 64
    assert payload["scope"]["robust"] == "project server heuristic"
    assert payload["zero_robustness_recovery"] == {
        "robust": {
            "bit_identical_to_naive": True,
            "consensus": [0.75, 0.107143, 0.142857],
            "entropy_weight": 1.0,
            "robustness": 0.0,
        },
        "variational": {
            "bit_identical_to_naive": True,
            "consensus": [0.75, 0.107143, 0.142857],
            "entropy_weight": 1.0,
            "robustness": 0.0,
        },
    }


@pytest.mark.integration
def test_federation_example_has_bit_exact_parity_and_replay(tmp_path: Path) -> None:
    output_dir = tmp_path / "federation"
    result = _run_example(
        "03_federation_boundaries.py",
        "--output-dir",
        str(output_dir),
        cwd=tmp_path,
    )
    payload = _successful_json(result)
    assert payload["example"] == "federation_boundaries"
    assert payload["artifacts"] == ["beliefs.json", "consensus.json", "replay.json"]
    assert payload["parity"] == {
        "direct_bit_identical_to_reference": True,
        "process_bit_identical_to_reference": True,
        "socket_bit_identical_to_reference": True,
        "socket_reference_bit_identical": True,
    }
    assert payload["replay"] == {
        "persisted_replay_valid": True,
        "socket_replay_valid": True,
    }
    assert payload["socket"] == {
        "authenticated": True,
        "n_workers": 3,
        "protocol_version": 1,
        "round_id": "example-round",
    }
    assert payload["direct_sharing"] == {
        "exclude_self": True,
        "global_consensus": [0.818356, 0.147523, 0.034121],
        "mean_accuracy": 0.683462,
        "mean_surprise": 0.391955,
        "shared_posteriors": [
            [0.567262, 0.286426, 0.146312],
            [0.664215, 0.192749, 0.143036],
            [0.818909, 0.148758, 0.032333],
        ],
    }

    replay_command = [
        sys.executable,
        "-m",
        "fedference_cli",
        "replay",
        "--replay",
        str(output_dir / "replay.json"),
        "--beliefs",
        str(output_dir / "beliefs.json"),
        "--consensus",
        str(output_dir / "consensus.json"),
        "--method",
        "robust",
        "--robustness",
        "1.5",
        "--max-iter",
        "64",
        "--tol",
        "1e-9",
    ]
    replay = subprocess.run(
        replay_command,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=_environment(),
        check=False,
        timeout=60,
    )
    assert replay.returncode == 0, replay.stderr
    assert replay.stdout.startswith("PASS: replay integrity verified")
    assert "solver_status=nominal" in replay.stdout

    tampered_path = output_dir / "tampered-consensus.json"
    tampered_path.write_text("[0.1, 0.2, 0.7]\n", encoding="utf-8")
    tampered_command = list(replay_command)
    tampered_command[tampered_command.index(str(output_dir / "consensus.json"))] = str(tampered_path)
    tampered = subprocess.run(
        tampered_command,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=_environment(),
        check=False,
        timeout=60,
    )
    assert tampered.returncode == 1
    assert tampered.stdout.startswith("FAIL:")
    assert "consensus" in tampered.stdout


def test_cli_receipt_example_runs_verifies_and_protects_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "cli"
    result = _run_example(
        "04_cli_receipt_workflow.py",
        "--output-dir",
        str(output_dir),
        "--project-root",
        str(ROOT),
        cwd=tmp_path,
    )
    payload = _successful_json(result)
    assert payload == {
        "committed_output_rejected": True,
        "example": "cli_receipt_workflow",
        "receipt": {
            "experiment_id": "server-theory",
            "git_metadata_bound": True,
            "output_names": ["config", "report"],
            "profile": "smoke",
            "schema_version": "1.2",
            "seeds": [0],
            "status": "completed",
        },
        "report_declares_no_claim": True,
        "rerun_rejected": True,
        "run_files": ["config.json", "receipt.json", "report.json"],
        "verify_passed": True,
    }
    assert (output_dir / "server-theory-smoke" / "receipt.json").is_file()
    assert not (ROOT / "output" / ".example-cli-isolation-probe").exists()


def test_labeled_application_example_matches_python_cli_and_receipt(tmp_path: Path) -> None:
    output_dir = tmp_path / "labeled"
    first = _run_example(
        "05_labeled_application.py",
        "--output-dir",
        str(output_dir),
        "--project-root",
        str(ROOT),
        cwd=tmp_path,
    )
    payload = _successful_json(first)
    assert payload == {
        "artifacts": ["receipt.json", "request.json", "result.json"],
        "canonical_request": {
            "agent_ids": ["agent-0", "agent-1", "agent-2"],
            "default_weight_expanded": True,
            "request_sha256": (
                "baa58addbd0fcdb5e496bd3dfc14fafbcce8c8e657a778a40a1c0ee12d5bff84"
            ),
            "state_labels": ["clear", "warning", "critical"],
        },
        "example": "labeled_application",
        "receipt": {
            "artifact_integrity_verified": True,
            "nominal_solver_required": True,
            "proves_scientific_validity": False,
        },
        "result": {
            "consensus": [0.818512, 0.147846, 0.033642],
            "normalized_first_row": [0.7, 0.2, 0.1],
            "python_cli_bit_identical": True,
            "solver_status": "nominal",
        },
    }
    assert (output_dir / "request.json").is_file()
    assert (output_dir / "result.json").is_file()
    assert (output_dir / "receipt.json").is_file()


@pytest.mark.parametrize(
    "name",
    [
        "03_federation_boundaries.py",
        "04_cli_receipt_workflow.py",
        "05_labeled_application.py",
    ],
)
def test_artifact_examples_refuse_nonempty_output_directories(name: str, tmp_path: Path) -> None:
    output_dir = tmp_path / name
    output_dir.mkdir()
    sentinel = output_dir / "caller-owned.txt"
    sentinel.write_text("preserve me\n", encoding="utf-8")
    result = _run_example(name, "--output-dir", str(output_dir), cwd=tmp_path)
    assert result.returncode == 2
    assert "output directory must be empty" in result.stderr
    assert sentinel.read_text(encoding="utf-8") == "preserve me\n"
