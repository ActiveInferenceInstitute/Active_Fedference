"""Real-boundary CLI and publication-surface failure matrix.

These tests intentionally avoid mocks and test doubles.  They exercise the
installed parser, real Git subprocesses, caller-owned temporary paths, the
available PDF command-line tools, and the committed tagged-PDF probe.
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fedference import __version__
from fedference.aggregation import AggregationConfig
from fedference.application import LabeledAggregationRequest, aggregate_labeled
from fedference.federation import run_socket_round
from fedference.provenance import SourceProvenance
from fedference_cli import main
from fedference_cli._commands import _load_numeric_json_array, _run_command
from fedference_cli._support import (
    _dataset_digests,
    _environment_lock_digest,
    _git_revision,
    _prepare_output_dir,
    _report_fallbacks,
    _require_locked_profile,
    _resolve_project_root,
    _validate_benchmark_controls,
    _validate_output_dir,
    _validate_seeds,
    _write_application_run,
    _write_evidence_run,
    _write_json,
)
from publication.surface_validation import (
    SurfaceValidation,
    _log_findings,
    _pdf_structure_findings,
    _pdf_tagging_findings,
    _pdf_text_findings,
    _pdfinfo_tagging_status,
    _publication_text_findings,
    _qpdf_tagging_status,
    _tagged_pdf_requested,
    validate_rendered_surfaces,
)
from publication.web_package import WebPackageValidation

ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 64


def _request_payload(*, method: str = "naive") -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "state_labels": ["no", "yes"],
        "agents": [
            {"agent_id": "agent-a", "posterior": [0.8, 0.2]},
            {"agent_id": "agent-b", "posterior": [0.3, 0.7]},
        ],
        "aggregation_config": {
            "method": method,
            "robustness": 1.5,
            "entropy_weight": 1.0,
            "max_iter": 64,
            "tol": 1e-9,
            "multistart": False,
        },
    }


def _write_active_identity(root: Path) -> None:
    package = root / "src" / "fedference"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("\n", encoding="utf-8")
    (package / "aggregation.py").write_text("\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        "[project]\n"
        'name = "active_fedference"\n'
        f'version = "{__version__}"\n',
        encoding="utf-8",
    )


def _initialize_git_checkout(root: Path, *, dirty: bool = False) -> None:
    _write_active_identity(root)
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    for command in (
        ("git", "init", "-q"),
        ("git", "config", "user.email", "boundary@example.test"),
        ("git", "config", "user.name", "Boundary Test"),
        ("git", "add", "."),
        ("git", "commit", "-qm", "fixture"),
    ):
        subprocess.run(command, cwd=root, check=True)
    if dirty:
        (root / "dirty.txt").write_text("caller-owned\n", encoding="utf-8")


def _expect_usage_error(
    argv: list[str],
    expected: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(argv)
    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert expected in stderr
    assert "Traceback" not in stderr


def test_numeric_json_loader_rejects_syntax_constants_raggedness_and_rank(
    tmp_path: Path,
) -> None:
    path = tmp_path / "values.json"
    cases = (
        ("NaN\n", 1, "valid finite JSON"),
        ("[\n", 1, "valid finite JSON"),
        ('"not-a-number"\n', 1, "numeric JSON array"),
        ("[[1], [2, 3]]\n", 2, "numeric JSON array"),
        ("[]\n", 1, "non-empty finite one-dimensional"),
        ("[[1, 2]]\n", 1, "non-empty finite one-dimensional"),
        ("[1, 2]\n", 3, "non-empty finite 3-dimensional"),
    )
    for contents, ndim, expected in cases:
        path.write_text(contents, encoding="utf-8")
        with pytest.raises(ValueError, match=expected):
            _load_numeric_json_array(str(path), name="values", ndim=ndim)

    path.write_text("[1, 2.5]\n", encoding="utf-8")
    assert _load_numeric_json_array(str(path), name="values", ndim=1).tolist() == [
        1.0,
        2.5,
    ]


def test_support_helpers_fail_before_writes_and_bind_real_git_state(
    tmp_path: Path,
) -> None:
    invalid_json = tmp_path / "atomic" / "payload.json"
    with pytest.raises(ValueError, match="Out of range float values"):
        _write_json(invalid_json, {"value": float("nan")})
    assert not invalid_json.exists()
    assert not list(invalid_json.parent.glob("*.tmp"))

    nongit = tmp_path / "nongit"
    nongit.mkdir()
    assert _resolve_project_root(nongit) == nongit.resolve()
    assert _git_revision(nongit) == ("unavailable", "unavailable")

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    _initialize_git_checkout(checkout)
    nested = checkout / "docs"
    nested.mkdir()
    assert _resolve_project_root(nested) == checkout.resolve()
    revision, state = _git_revision(checkout)
    assert len(revision) == 40
    assert state == "clean"
    (nested / "note.txt").write_text("untracked\n", encoding="utf-8")
    assert _git_revision(checkout) == (revision, "dirty")

    destination_file = tmp_path / "destination-file"
    destination_file.write_text("preserve\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a directory"):
        _validate_output_dir(str(destination_file), checkout)

    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "owned.txt").write_text("preserve\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be empty"):
        _prepare_output_dir(str(nonempty), checkout)
    assert (nonempty / "owned.txt").read_text(encoding="utf-8") == "preserve\n"

    fresh = _prepare_output_dir(str(tmp_path / "fresh"), checkout)
    assert fresh.is_dir()

    with pytest.raises(ValueError, match="committed reviewer output tree"):
        _validate_output_dir(str(checkout / "output" / "nested"), tmp_path)


def test_support_validation_matrix_and_receipt_extraction(tmp_path: Path) -> None:
    _require_locked_profile("smoke", confirmatory_ready=False)
    _require_locked_profile("confirmatory", confirmatory_ready=True)
    with pytest.raises(ValueError, match="blocked until its pilot freezes"):
        _require_locked_profile("m4_confirmatory", confirmatory_ready=False)

    assert _validate_seeds([0, 2]) == (0, 2)
    for seeds, expected in (
        ([], "non-empty"),
        ([False], "non-negative integers"),
        ([-1], "non-negative integers"),
        ([3, 3], "unique"),
    ):
        with pytest.raises(ValueError, match=expected):
            _validate_seeds(seeds)

    base = {
        "n_clients": 5,
        "n_contaminated": 2,
        "contamination_rate": 0.5,
        "robustness": 1.5,
        "entropy_weight": 1.0,
    }
    _validate_benchmark_controls(argparse.Namespace(**base))
    for changes, expected in (
        ({"n_clients": 0}, "n_clients must be positive"),
        ({"n_contaminated": 6}, "n_contaminated"),
        ({"contamination_rate": float("inf")}, "contamination_rate"),
        ({"robustness": -1.0}, "robustness"),
        ({"entropy_weight": -1.0}, "entropy_weight"),
    ):
        values = {**base, **changes}
        with pytest.raises(ValueError, match=expected):
            _validate_benchmark_controls(argparse.Namespace(**values))

    fallback_digest, warnings = _environment_lock_digest(tmp_path)
    assert len(fallback_digest) == 64
    assert warnings == ("uv.lock was unavailable to the installed CLI",)
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    lock_digest, warnings = _environment_lock_digest(tmp_path)
    assert len(lock_digest) == 64
    assert warnings == ()

    rows = {
        "rows": [
            {"dataset_id": "a", "dataset_archive_sha256": "1" * 64},
            {"dataset_id": "b", "dataset_archive_sha256": "2" * 64},
        ]
    }
    assert _dataset_digests(rows) == {"a": "1" * 64, "b": "2" * 64}
    assert _dataset_digests(
        {"dataset_id": "single", "dataset_archive_sha256": "3" * 64}
    ) == {"single": "3" * 64}
    assert _dataset_digests({"rows": [{"dataset_id": "incomplete"}]}) == {}

    report = {
        "device": {"fallback": "mps unavailable; used cpu"},
        "rows": [
            "ignored",
            {
                "dataset": "fixture",
                "seed": 9,
                "n_test": 12,
                "robust_fallback_predictions": 2,
                "robust_nonconverged_predictions": 1,
                "naive_fallback_predictions": True,
            },
            {
                "dataset": "fixture",
                "seed": 9,
                "n_test": 12,
                "robust_fallback_predictions": 2,
            },
        ],
    }
    fallbacks = _report_fallbacks(report)
    assert fallbacks[0] == "device fallback: mps unavailable; used cpu"
    assert len(fallbacks) == 3
    assert not any("method=naive" in event for event in fallbacks)


def test_application_writer_rejects_wrong_boundary_types(tmp_path: Path) -> None:
    request = LabeledAggregationRequest.from_dict(_request_payload())
    result = aggregate_labeled(request)
    source = SourceProvenance(source_kind="unavailable", warnings=("fixture",))
    arguments: dict[str, Any] = {
        "output_dir": tmp_path,
        "request": request,
        "result": result,
        "source_provenance": source,
        "started_at": "2026-08-25T00:00:00+00:00",
    }
    for field, invalid, expected in (
        ("request", {}, "request must be"),
        ("result", {}, "result must be"),
        ("source_provenance", {}, "source_provenance must be"),
    ):
        with pytest.raises(ValueError, match=expected):
            _write_application_run(**{**arguments, field: invalid})
    assert not list(tmp_path.iterdir())


def test_cli_maps_preexecution_contract_failures_to_usage_exit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "must-not-exist"
    common = ["--output-dir", str(output), "--project-root", str(ROOT)]
    cases = (
        (
            ["run", "server-theory", "--profile", "unknown", *common],
            "not declared",
        ),
        (
            ["run", "server-theory", "--seed", "0", "--seed", "1", *common],
            "exactly one seed",
        ),
        (["run", "external-tabular", *common], "requires --cache-dir"),
        (
            [
                "run",
                "robustness-calibration",
                "--seed",
                "0",
                "--seed",
                "1",
                *common,
            ],
            "exactly one seed",
        ),
        (
            ["run", "fedgvi-bnn", "--profile", "source_5090", *common],
            "declarative and is not executed",
        ),
        (
            ["run", "fedgvi-bnn", "--seed", "0", "--seed", "1", *common],
            "one seed per evidence directory",
        ),
        (
            ["run", "friston-protocol", "--seed", "0", "--seed", "1", *common],
            "accepts one seed",
        ),
    )
    for argv, expected in cases:
        _expect_usage_error(argv, expected, capsys)
        assert not output.exists()

    with pytest.raises(ValueError, match="no executable CLI runner"):
        _run_command(
            argparse.Namespace(
                project_root=str(ROOT),
                experiment_id="multi-node-emulator",
                profile="smoke",
            )
        )


def test_benchmark_cli_rejects_invalid_controls_before_cache_or_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    base = [
        "benchmark",
        "--dataset-id",
        "uci-banknote",
        "--cache-dir",
        str(tmp_path / "unused-cache"),
        "--output-dir",
        str(tmp_path / "must-not-exist"),
        "--project-root",
        str(ROOT),
    ]
    cases = (
        (["--n-clients", "0"], "n_clients must be positive"),
        (["--n-contaminated", "6"], "n_contaminated"),
        (["--contamination-rate", "nan"], "contamination_rate"),
        (["--robustness", "-1"], "robustness"),
        (["--entropy-weight", "-1"], "entropy_weight"),
    )
    for controls, expected in cases:
        _expect_usage_error([*base, *controls], expected, capsys)
        assert not (tmp_path / "must-not-exist").exists()
        assert not (tmp_path / "unused-cache").exists()


def test_cli_executes_all_lightweight_research_adapters_and_tamper_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs = (
        ("hybrid-tracking", ["--seed", "0"], "single"),
        ("hybrid-tracking", ["--seed", "1", "--seed", "2"], "multiple"),
        ("hierarchy-tasks", ["--seed", "3"], "hierarchy"),
        ("friston-protocol", ["--seed", "4"], "friston"),
    )
    receipts: list[Path] = []
    for experiment_id, seed_args, label in runs:
        output = tmp_path / label
        assert main(
            [
                "run",
                experiment_id,
                *seed_args,
                "--output-dir",
                str(output),
                "--project-root",
                str(ROOT),
            ]
        ) == 0
        paths = json.loads(capsys.readouterr().out)
        receipts.append(Path(paths["receipt"]))
        report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
        assert report["experiment_spec"]["experiment_id"] == experiment_id
        if label == "multiple":
            assert report["status"] == "pilot"
            assert report["seeds"] == [1, 2]

    assert main(["verify", str(receipts[-1])]) == 0
    assert "PASS:" in capsys.readouterr().out
    tampered = receipts[-1].parent / "report.json"
    tampered.write_text(tampered.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert main(["verify", str(receipts[-1])]) == 1
    assert "artifact" in capsys.readouterr().out


def test_cli_executes_real_torch_bnn_smoke_and_pilot_profiles(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("torch")
    for profile in ("smoke", "pilot"):
        output = tmp_path / profile
        assert main(
            [
                "run",
                "fedgvi-bnn",
                "--profile",
                profile,
                "--seed",
                "5",
                "--device",
                "cpu",
                "--output-dir",
                str(output),
                "--project-root",
                str(ROOT),
            ]
        ) == 0
        paths = json.loads(capsys.readouterr().out)
        report = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
        assert report["backend"] == "torch-mean-field-diagonal-gaussian"
        assert report["device"]["resolved"] == "cpu"
        assert report["checkpoint_fingerprints"]


def test_cli_source_requirements_fail_before_application_destination_creation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(_request_payload()), encoding="utf-8")

    active_nongit = tmp_path / "active-nongit"
    active_nongit.mkdir()
    _write_active_identity(active_nongit)
    explicit_output = tmp_path / "explicit-output"
    _expect_usage_error(
        [
            "aggregate",
            "--input",
            str(request_path),
            "--output-dir",
            str(explicit_output),
            "--project-root",
            str(active_nongit),
        ],
        "readable Git checkout",
        capsys,
    )
    assert not explicit_output.exists()

    original_cwd = Path.cwd()
    try:
        os.chdir(active_nongit)
        implicit_output = tmp_path / "implicit-output"
        _expect_usage_error(
            [
                "aggregate",
                "--input",
                str(request_path),
                "--output-dir",
                str(implicit_output),
                "--require-clean-git",
            ],
            "requires an explicit readable Git checkout",
            capsys,
        )
    finally:
        os.chdir(original_cwd)
    assert not implicit_output.exists()

    dirty_checkout = tmp_path / "dirty-checkout"
    dirty_checkout.mkdir()
    _initialize_git_checkout(dirty_checkout, dirty=True)
    dirty_output = tmp_path / "dirty-output"
    _expect_usage_error(
        [
            "aggregate",
            "--input",
            str(request_path),
            "--output-dir",
            str(dirty_output),
            "--project-root",
            str(dirty_checkout),
            "--require-clean-git",
        ],
        "requires a clean Git tree",
        capsys,
    )
    assert not dirty_output.exists()


def test_verify_distinguishes_unrequested_source_and_invalid_checkout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    request = LabeledAggregationRequest.from_dict(_request_payload())
    result = aggregate_labeled(request)
    run_dir = tmp_path / "application"
    run_dir.mkdir()
    _, _, receipt = _write_application_run(
        output_dir=run_dir,
        request=request,
        result=result,
        source_provenance=SourceProvenance(
            source_kind="installed_archive",
            installed_archive_sha256=SHA,
        ),
        started_at="2026-08-25T00:00:00+00:00",
    )
    assert main(["verify", str(receipt)]) == 0
    assert "source equivalence not requested" in capsys.readouterr().out

    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    _expect_usage_error(
        ["verify", str(receipt), "--project-root", str(unrelated)],
        "must select an Active Fedference checkout",
        capsys,
    )


def test_cli_clean_checkout_application_receipt_and_tamper_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    _initialize_git_checkout(checkout)
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(_request_payload()), encoding="utf-8")
    output = tmp_path / "application"

    assert main(
        [
            "aggregate",
            "--input",
            str(request_path),
            "--output-dir",
            str(output),
            "--project-root",
            str(checkout),
            "--require-clean-git",
        ]
    ) == 0
    paths = json.loads(capsys.readouterr().out)
    assert paths["solver_status"] == "nominal"
    assert main(
        [
            "verify",
            paths["receipt"],
            "--project-root",
            str(checkout),
            "--require-clean-git",
            "--require-nominal-solver",
        ]
    ) == 0
    verification = capsys.readouterr().out
    assert "application artifact integrity verified" in verification
    assert "source equivalence verified" in verification
    assert "solver_status=nominal" in verification

    result_path = Path(paths["result"])
    result_path.write_text(
        result_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    assert main(["verify", paths["receipt"]]) == 1
    failure = capsys.readouterr().out
    assert "FAIL: application artifact integrity" in failure
    assert "source equivalence not requested" in failure
    assert "source equivalence unavailable" not in failure


def test_cli_real_replay_reports_integrity_and_solver_health_separately(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    beliefs = np.asarray(
        [[0.9, 0.09, 0.01], [0.01, 0.09, 0.9], [0.4, 0.3, 0.3]]
    )
    config = AggregationConfig(
        method="robust",
        robustness=1.5,
        entropy_weight=1.0,
        max_iter=1,
        tol=0.0,
        multistart=False,
    )
    replay_path = tmp_path / "replay.json"
    round_result = run_socket_round(
        beliefs,
        config=config,
        round_id="failure-matrix",
        replay_path=replay_path,
    )
    beliefs_path = tmp_path / "beliefs.json"
    consensus_path = tmp_path / "consensus.json"
    beliefs_path.write_text(json.dumps(beliefs.tolist()), encoding="utf-8")
    consensus_path.write_text(
        json.dumps(round_result["consensus"].tolist()),
        encoding="utf-8",
    )
    base = [
        "replay",
        "--replay",
        str(replay_path),
        "--beliefs",
        str(beliefs_path),
        "--consensus",
        str(consensus_path),
        "--method",
        "robust",
        "--robustness",
        "1.5",
        "--entropy-weight",
        "1.0",
        "--max-iter",
        "1",
        "--tol",
        "0.0",
        "--single-start",
    ]
    assert main([*base, "--json"]) == 0
    verdict = json.loads(capsys.readouterr().out)
    assert verdict["integrity_valid"] is True
    assert verdict["solver_status"] == "not_converged"
    assert any(
        finding["code"] == "solver.nonconvergence"
        for finding in verdict["findings"]
    )

    assert main([*base, "--require-nominal-solver"]) == 1
    output = capsys.readouterr().out
    assert "PASS: replay integrity verified" in output
    assert "FAIL: solver.nonconvergence" in output

    assert main([*base, "--robustness", "2.0"]) == 1
    mismatch = capsys.readouterr().out
    assert "FAIL: configuration.mismatch" in mismatch
    assert "PASS:" not in mismatch


def test_research_receipt_writer_records_missing_lock_and_dataset_digest(
    tmp_path: Path,
) -> None:
    output = tmp_path / "receipt"
    output.mkdir()
    report = {
        "status": "pilot",
        "primary_estimand": "fixture score",
        "independent_unit": "seed",
        "negative_controls": {},
        "no_claim": "fixture only",
        "dataset_id": "fixture-dataset",
        "dataset_archive_sha256": SHA,
        "device": {"fallback": "requested accelerator unavailable"},
    }
    report_path, receipt_path = _write_evidence_run(
        output_dir=output,
        project_root=tmp_path,
        experiment_id="server-theory",
        profile="smoke",
        seeds=(0,),
        config={"seed": 0},
        report=report,
        started_at="2026-08-25T00:00:00+00:00",
        git_revision=("unavailable", "unavailable"),
    )
    assert report_path.is_file()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["dataset_sha256"] == {"fixture-dataset": SHA}
    assert "uv.lock was unavailable" in receipt["fallbacks"][0]
    assert any("device fallback" in finding for finding in receipt["fallbacks"])


def test_module_entrypoint_executes_in_process_and_maps_parser_exit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_argv = sys.argv
    try:
        sys.argv = ["fedference", "list"]
        with pytest.raises(SystemExit) as success:
            runpy.run_module("fedference_cli.__main__", run_name="__main__")
        assert success.value.code == 0
        assert "server-theory" in capsys.readouterr().out
    finally:
        sys.argv = original_argv

    process = subprocess.run(
        [sys.executable, "-m", "fedference_cli"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 2
    assert "required" in process.stderr
    assert "Traceback" not in process.stderr


def test_pdf_helpers_fail_closed_with_missing_and_malformed_real_tools(
    tmp_path: Path,
) -> None:
    malformed_pdf = tmp_path / "malformed.pdf"
    malformed_pdf.write_bytes(b"not-a-pdf")

    original_path = os.environ.get("PATH")
    try:
        os.environ["PATH"] = ""
        assert _pdf_text_findings(malformed_pdf) == [
            "pdftotext is required for reviewer-surface PDF text validation"
        ]
        assert _pdf_structure_findings(malformed_pdf) == [
            "qpdf is required for reviewer-surface PDF structural validation"
        ]
        assert _pdf_tagging_findings(malformed_pdf, required=True) == [
            f"{malformed_pdf}: pdfinfo is required when metadata.tagged_pdf is enabled"
        ]
    finally:
        if original_path is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = original_path

    if shutil.which("pdftotext") is not None:
        text_findings = _pdf_text_findings(malformed_pdf)
        assert any("pdftotext failed with exit code" in finding for finding in text_findings)
    if shutil.which("qpdf") is not None:
        structure_findings = _pdf_structure_findings(malformed_pdf)
        assert any("qpdf structural check failed" in finding for finding in structure_findings)
    if shutil.which("pdfinfo") is not None:
        tagging_findings = _pdf_tagging_findings(malformed_pdf, required=True)
        assert any("pdfinfo failed" in finding for finding in tagging_findings)

    assert _pdf_tagging_findings(malformed_pdf, required=False) == []


def test_tagging_parsers_and_source_configuration_cover_nested_shapes(
    tmp_path: Path,
) -> None:
    payload = json.dumps(
        {
            "objects": [
                {"ignored": None},
                {"catalog": [{"StructTreeRoot": "2 0 R"}, {"Lang": "en"}]},
            ]
        }
    )
    assert _qpdf_tagging_status(payload) == (True, True)
    assert _qpdf_tagging_status(json.dumps([None, "text", 2])) == (False, False)
    assert _qpdf_tagging_status("{") == (False, False)
    assert _pdfinfo_tagging_status("ignored line\nTagged: YES\nLanguage:\n") == (
        True,
        None,
    )

    absent = tmp_path / "absent"
    assert not _tagged_pdf_requested(absent)

    unreadable = tmp_path / "unreadable" / "manuscript" / "config.yaml"
    unreadable.mkdir(parents=True)
    assert not _tagged_pdf_requested(unreadable.parents[1])

    sequence = tmp_path / "sequence" / "manuscript"
    sequence.mkdir(parents=True)
    (sequence / "config.yaml").write_text("- item\n", encoding="utf-8")
    assert not _tagged_pdf_requested(sequence.parent)

    malformed = tmp_path / "malformed" / "manuscript"
    malformed.mkdir(parents=True)
    (malformed / "config.yaml").write_text("metadata: [\n", encoding="utf-8")
    assert not _tagged_pdf_requested(malformed.parent)

    scalar_metadata = tmp_path / "scalar" / "manuscript"
    scalar_metadata.mkdir(parents=True)
    (scalar_metadata / "config.yaml").write_text("metadata: disabled\n", encoding="utf-8")
    assert not _tagged_pdf_requested(scalar_metadata.parent)

    enabled = tmp_path / "enabled" / "manuscript"
    enabled.mkdir(parents=True)
    (enabled / "config.yaml").write_text(
        "metadata:\n  tagged_pdf: true\n",
        encoding="utf-8",
    )
    assert _tagged_pdf_requested(enabled.parent)


@pytest.mark.skipif(
    not all(shutil.which(tool) for tool in ("pdfinfo", "qpdf")),
    reason="PDF structural tools are unavailable",
)
def test_tagged_pdf_probe_checks_current_source_bound_manuscript() -> None:
    manuscript = ROOT / "output" / "pdf" / "active_fedference_combined.pdf"
    if not manuscript.is_file():
        pytest.skip("source-bound combined PDF is unavailable")
    assert _pdf_structure_findings(manuscript) == []
    assert _pdf_text_findings(manuscript) == []
    assert _pdf_tagging_findings(manuscript, required=True) == []

    untagged = ROOT / "output" / "slides" / "00_abstract_slides.pdf"
    if untagged.is_file():
        findings = _pdf_tagging_findings(untagged, required=True)
        assert any("requires Tagged: yes" in finding for finding in findings)
        assert any("StructTreeRoot" in finding for finding in findings)
        assert any("document language" in finding for finding in findings)


@pytest.mark.skipif(
    shutil.which("pdfinfo") is None,
    reason="pdfinfo is unavailable",
)
def test_tagged_pdf_probe_requires_qpdf_independently(tmp_path: Path) -> None:
    untagged = ROOT / "output" / "slides" / "00_abstract_slides.pdf"
    if not untagged.is_file():
        pytest.skip("source-bound untagged slide PDF is unavailable")
    tool_dir = tmp_path / "tools"
    tool_dir.mkdir()
    (tool_dir / "pdfinfo").symlink_to(Path(shutil.which("pdfinfo") or ""))
    original_path = os.environ.get("PATH")
    try:
        os.environ["PATH"] = str(tool_dir)
        findings = _pdf_tagging_findings(untagged, required=True)
    finally:
        if original_path is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = original_path
    assert any("requires Tagged: yes" in finding for finding in findings)
    assert any("qpdf is required" in finding for finding in findings)
    assert any("document language" in finding for finding in findings)


def test_full_surface_validator_accumulates_pdf_log_inventory_and_web_findings(
    tmp_path: Path,
) -> None:
    manuscript = tmp_path / "output" / "pdf"
    slides = tmp_path / "output" / "slides"
    web = tmp_path / "output" / "web"
    manuscript.mkdir(parents=True)
    slides.mkdir(parents=True)
    web.mkdir(parents=True)

    (manuscript / "active_fedference_combined.pdf").write_bytes(b"invalid pdf")
    (manuscript / "_combined_manuscript.tex").write_text(
        r"\href{https://doi.org/(forthcoming)}{forthcoming}",
        encoding="utf-8",
    )
    (manuscript / "_combined_manuscript.log").write_text(
        "Overfull \\hbox (2.5pt too wide)\n"
        "Missing character: There is no glyph\n"
        "LaTeX Warning: Citation `missing' undefined\n",
        encoding="utf-8",
    )
    config = tmp_path / "manuscript"
    config.mkdir()
    (config / "config.yaml").write_text(
        "metadata:\n  tagged_pdf: true\n",
        encoding="utf-8",
    )

    (slides / "alpha_slides.pdf").write_bytes(b"invalid slide pdf")
    (slides / "beta_slides.tex").write_text("source\n", encoding="utf-8")
    (slides / "gamma_slides.log").write_text(
        "Overfull \\vbox (3.0pt too high)\n",
        encoding="utf-8",
    )
    (web / "index.html").write_text(
        "<html><head><title>Fixture</title></head><body><img src='missing.png'></body></html>",
        encoding="utf-8",
    )

    result = validate_rendered_surfaces(tmp_path)
    assert not result.ok
    assert result.manuscript_pdf
    assert (result.manuscript_logs, result.slide_pdfs, result.slide_tex, result.slide_logs) == (
        1,
        1,
        1,
        1,
    )
    joined = "\n".join(result.findings)
    for expected in (
        "suspiciously small PDF",
        "qpdf structural check failed",
        "pdfinfo failed",
        "pdftotext failed",
        "placeholder DOI resolver URL",
        "missing required manuscript log",
        "Overfull \\hbox",
        "Missing character",
        "missing slide TeX source",
        "orphan slide TeX source",
        "missing slide LaTeX log",
        "orphan slide LaTeX log",
        "Overfull \\vbox",
        "main element with id='main-content'",
        "image(s) lack non-empty alt text",
    ):
        assert expected in joined


def test_surface_validator_handles_empty_outputs_with_valid_web(tmp_path: Path) -> None:
    web = tmp_path / "output" / "web"
    web.mkdir(parents=True)
    (web / "index.html").write_text(
        '<html lang="en"><head><title>Fixture</title></head><body>'
        '<a class="skip-link" href="#main-content">Skip</a>'
        '<main id="main-content" tabindex="-1">Complete reader surface</main>'
        "</body></html>",
        encoding="utf-8",
    )
    result = validate_rendered_surfaces(tmp_path)
    assert result.web.ok
    assert not result.ok
    assert any("missing combined manuscript PDF" in finding for finding in result.findings)
    assert any("missing generated slide PDFs" in finding for finding in result.findings)


def test_publication_text_and_log_parsers_report_each_stable_failure(
    tmp_path: Path,
) -> None:
    findings = _publication_text_findings(
        Path("reader.pdf"),
        "Raw [@sec:method], reference \\ref{sec:missing}, and {{TOKEN}}.",
    )
    assert len(findings) == 3
    log = tmp_path / "warnings.log"
    log.write_text(
        "LaTeX Warning: Citation `missing' undefined\n",
        encoding="utf-8",
    )
    assert "undefined" in _log_findings(log)[0]


def test_surface_validation_ok_requires_each_reader_surface() -> None:
    valid_web = WebPackageValidation(
        html_files=1,
        assets_checked=0,
        missing_assets=(),
        raw_xrefs=(),
    )
    valid = SurfaceValidation(
        manuscript_pdf=True,
        manuscript_logs=1,
        slide_pdfs=1,
        slide_tex=1,
        slide_logs=1,
        findings=(),
        web=valid_web,
    )
    assert valid.ok
    assert not SurfaceValidation(
        manuscript_pdf=True,
        manuscript_logs=1,
        slide_pdfs=0,
        slide_tex=1,
        slide_logs=1,
        findings=(),
        web=valid_web,
    ).ok
