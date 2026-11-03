"""Real CLI model validation with canonical resource inputs and corrupt controls."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from fedference.gnn.admission import content_digest, dumps
from fedference.gnn.loader import SENTINEL_ASSET, load_sentinel
from fedference_cli import main


def test_native_model_validation_preserves_canonical_identity(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    model = load_sentinel()
    path = tmp_path / "model.json"
    path.write_text(dumps(model))
    assert main(["model", "validate", "--model", str(path), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["valid"] is True
    assert result["model_digest"] == content_digest(model)
    assert result["model"] == json.loads(dumps(model))
    assert result["native_configuration"] == "not-provided"
    assert path.read_text() == dumps(model)


@pytest.mark.parametrize("mutation", ["nonfinite", "nonnormalized", "axes", "duplicate", "syntax"])
def test_native_model_rejection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], mutation: str,
) -> None:
    raw = json.loads(dumps(load_sentinel()))
    if mutation == "nonfinite":
        raw["categorical"]["D"]["values"][0] = float("nan")
    elif mutation == "nonnormalized":
        raw["categorical"]["D"]["values"][0] = 2.0
    elif mutation == "axes":
        raw["categorical"]["B"]["axes"] = ["previous_state", "next_state", "action"]
    text = json.dumps(raw)
    if mutation == "duplicate":
        text = text.replace('"kind": "categorical"', '"kind": "categorical", "kind": "categorical"')
    elif mutation == "syntax":
        text = "{"
    path = tmp_path / "invalid.json"
    path.write_text(text)
    with pytest.raises(SystemExit) as failure:
        main(["model", "validate", "--model", str(path), "--json"])
    assert failure.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "GNN_" in captured.err
    assert path.read_text() == text


def test_gnn_requires_explicit_root(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as failure:
        main(["model", "validate", "--model", str(SENTINEL_ASSET), "--json"])
    assert failure.value.code == 2
    assert "--gnn-root" in capsys.readouterr().err


def test_default_cli_import_is_torch_free(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import fedference_cli,sys; "
         "assert not any(n == 'torch' or n.startswith('torch.') for n in sys.modules); print('torch-free')"],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "torch-free"


def test_native_text_and_explicit_format(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "model.data"
    path.write_text(dumps(load_sentinel()))
    assert main(["model", "validate", "--model", str(path), "--format", "native"]) == 0
    assert "no execution" in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        main(["model", "validate", "--model", str(path), "--format", "native", "--gnn-root", "."])
    assert error.value.code == 2
    assert "does not consume" in capsys.readouterr().err


def test_ambiguous_family_and_missing_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "ambiguous.gnn.md"
    path.write_text("## FedferenceScientific\n{}\n## FedferenceStructured\n{}\n")
    with pytest.raises(SystemExit) as error:
        main(["model", "validate", "--model", str(path), "--gnn-root", "."])
    assert error.value.code == 2
    assert "GNN_SECTIONS" in capsys.readouterr().err
    with pytest.raises(SystemExit) as error:
        main(["model", "validate", "--model", str(tmp_path / "missing.json")])
    assert error.value.code == 2
    assert "No such file" in capsys.readouterr().err
