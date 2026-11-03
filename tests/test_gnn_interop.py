"""GNN model exchange: native parity and optional real upstream extraction."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from fedference.pomdp import build_sentinel_world
from tests.test_examples import _run_example, _successful_json


def test_gnn_example_preserves_native_matrices_and_axis_order(tmp_path: Path) -> None:
    # Given the canonical world and the standalone example.
    world = build_sentinel_world()
    # When exporting through both public process formats.
    payload = _successful_json(_run_example("06_gnn_sentinel.py", cwd=tmp_path))
    document = _run_example("06_gnn_sentinel.py", "--format", "gnn", cwd=tmp_path)
    # Then every exported value agrees exactly with the native source.
    assert document.returncode == 0, document.stderr
    assert document.stdout == payload["gnn_model"]
    assert payload["b_axis_order"] == ["next_state", "previous_state", "action"]
    assert payload["control_labels"] == list(world["control_labels"])
    parameter_block = document.stdout.split("## InitialParameterization\n", 1)[1].split("##", 1)[0]
    parameters = dict(line.split("=", 1) for line in parameter_block.splitlines() if "=" in line)
    for name in ("A", "B", "C", "D"):
        expected = world[name][0]
        if name in ("C", "D"):
            expected = expected.reshape(-1)
        np.testing.assert_array_equal(np.asarray(json.loads(parameters[name])), expected)


@pytest.mark.integration
def test_gnn_example_with_explicit_upstream_checkout(tmp_path: Path) -> None:
    # Given a caller-selected real upstream checkout, never a test double.
    root = os.environ.get("GNN_ROOT")
    if root is None:
        pytest.skip("set GNN_ROOT to run the optional real upstream GNN extraction gate")
    upstream_src = Path(root).expanduser().resolve() / "src"
    assert (upstream_src / "gnn" / "extract" / "__init__.py").is_file()
    payload = _successful_json(_run_example("06_gnn_sentinel.py", cwd=tmp_path))
    model_path = tmp_path / "sentinel.gnn.md"
    model_path.write_text(payload["gnn_model"], encoding="utf-8")
    command = [
        sys.executable,
        "-I",
        "-S",
        "-c",
        "import sys,json; sys.path.insert(0,sys.argv[1]); "
        "from gnn.extract import extract_to_json; "
        "result=extract_to_json(sys.argv[2],on_error='raise',strict_validation=True); "
        "print(result); sys.exit(json.loads(result).get('status')=='error')",
        str(upstream_src),
        str(model_path),
    ]
    # When the current upstream extracts the actual exchange document.
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=60)
    # Then native values and canonical orientation survive the external boundary.
    extracted = _successful_json(result)
    assert extracted["extraction_schema_version"] == "1.0.0"
    for key in ("num_states", "num_observations", "num_actions"):
        assert extracted["dimension_provenance"][key]["source"] == "ModelParameters"
    world = build_sentinel_world()
    for name, key in (("A", "A_matrix"), ("B", "B_matrix"), ("C", "C_vector"), ("D", "D_vector")):
        expected = world[name][0]
        if name in ("C", "D"):
            expected = expected.reshape(-1)
        np.testing.assert_array_equal(np.asarray(extracted[key]), expected)
    assert extracted["matrix_provenance"]["B"]["declared_order_explicit"] is True
    assert extracted["matrix_provenance"]["B"]["detected_order"] == payload["b_axis_order"]
    extracted_path = tmp_path / "extracted.json"
    extracted_path.write_text(result.stdout, encoding="utf-8")
    verified = _successful_json(
        _run_example(
            "06_gnn_sentinel.py",
            "--verify-extracted",
            str(extracted_path),
            cwd=tmp_path,
        )
    )
    assert verified["extraction_verified"] is True


@pytest.mark.integration
def test_gnn_verification_rejects_corruption_even_if_upstream_only_warns(tmp_path: Path) -> None:
    # Given an invalid A shape in an otherwise real exported model.
    root = os.environ.get("GNN_ROOT")
    if root is None:
        pytest.skip("set GNN_ROOT to run the optional real upstream GNN negative control")
    upstream_src = Path(root).expanduser().resolve() / "src"
    assert (upstream_src / "gnn" / "extract" / "__init__.py").is_file()
    model = _successful_json(_run_example("06_gnn_sentinel.py", cwd=tmp_path))["gnn_model"]
    lines = ["A=[[1.0]]" if line.startswith("A=") else line for line in model.splitlines()]
    model_path = tmp_path / "invalid.gnn.md"
    model_path.write_text("\n".join(lines), encoding="utf-8")
    # When strict extraction sees the corrupted matrix.
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            "import sys,json; sys.path.insert(0,sys.argv[1]); "
            "from gnn.extract import extract_to_json; "
            "result=extract_to_json(sys.argv[2],on_error='raise',strict_validation=True); "
            "print(result); sys.exit(json.loads(result).get('status')=='error')",
            str(upstream_src),
            str(model_path),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    # Then the local parity boundary rejects either a warning-only result or an error envelope.
    assert result.returncode in (0, 1), result.stdout + result.stderr
    extracted_path = tmp_path / "invalid-extraction.json"
    extracted_path.write_text(result.stdout, encoding="utf-8")
    verified = _run_example(
        "06_gnn_sentinel.py",
        "--verify-extracted",
        str(extracted_path),
        cwd=tmp_path,
    )
    assert verified.returncode == 2, verified.stdout + verified.stderr


@pytest.mark.parametrize(
    "mutation",
    ["shape", "transpose", "missing", "schema", "error", "dimensions", "nonfinite", "invalid-json"],
)
def test_gnn_verifier_rejects_invalid_extractions(mutation: str, tmp_path: Path) -> None:
    # Given an extraction whose validity does not depend on an installed GNN checkout.
    world = build_sentinel_world()
    extracted = {
        "extraction_schema_version": "1.0.0",
        "num_states": 9,
        "num_observations": 9,
        "num_actions": 3,
        "A_matrix": world["A"][0].tolist(),
        "B_matrix": world["B"][0].tolist(),
        "C_vector": world["C"][0].reshape(-1).tolist(),
        "D_vector": world["D"][0].reshape(-1).tolist(),
    }
    if mutation == "shape":
        extracted["A_matrix"] = [[1.0]]
    elif mutation == "transpose":
        extracted["B_matrix"] = world["B"][0].transpose(1, 0, 2).tolist()
    elif mutation == "missing":
        del extracted["C_vector"]
    elif mutation == "schema":
        extracted["extraction_schema_version"] = "2.0.0"
    elif mutation == "error":
        extracted["status"] = "error"
    elif mutation == "dimensions":
        extracted["num_states"] = 8
    elif mutation == "nonfinite":
        extracted["A_matrix"] = [[float("nan")]]
    path = tmp_path / "extraction.json"
    path.write_text("{" if mutation == "invalid-json" else json.dumps(extracted), encoding="utf-8")
    # When verifying that supplied extraction.
    result = _run_example("06_gnn_sentinel.py", "--verify-extracted", str(path), cwd=tmp_path)
    # Then a malformed, incomplete, or changed model cannot pass parity.
    assert result.returncode == 2, result.stdout + result.stderr


def test_gnn_verifier_accepts_exact_native_extraction(tmp_path: Path) -> None:
    # Given every native matrix in the documented extraction representation.
    world = build_sentinel_world()
    extracted = {
        "extraction_schema_version": "1.0.0",
        "num_states": 9,
        "num_observations": 9,
        "num_actions": 3,
        "A_matrix": world["A"][0].tolist(),
        "B_matrix": world["B"][0].tolist(),
        "C_vector": world["C"][0].reshape(-1).tolist(),
        "D_vector": world["D"][0].reshape(-1).tolist(),
    }
    path = tmp_path / "extraction.json"
    path.write_text(json.dumps(extracted), encoding="utf-8")
    # When the example verifies it in a real subprocess.
    result = _run_example("06_gnn_sentinel.py", "--verify-extracted", str(path), cwd=tmp_path)
    # Then the explicit parity disposition is true.
    assert _successful_json(result)["extraction_verified"] is True
