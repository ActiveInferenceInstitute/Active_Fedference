"""Required real-upstream model CLI admission; no synthetic parser or skipped runtime."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from fedference.gnn.admission import content_digest, dumps
from fedference.gnn.loader import SENTINEL_ASSET, load_sentinel
from fedference.gnn.scientific_admission import HYBRID_ASSET, SITES_ASSET, load_scientific
from fedference.gnn.structured_admission import HIERARCHY_ASSET, MOVING_ASSET, load_structured


def test_upstream_sentinel_cli(tmp_path: Path) -> None:
    root = os.environ["GNN_ROOT"]
    result = subprocess.run(
        [sys.executable, "-m", "fedference_cli", "model", "validate", "--model", str(SENTINEL_ASSET),
         "--gnn-root", root, "--json"],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["model_digest"] == content_digest(load_sentinel())
    assert payload["model"] == json.loads(dumps(load_sentinel()))


@pytest.mark.parametrize("asset", [MOVING_ASSET, HIERARCHY_ASSET, HYBRID_ASSET, SITES_ASSET])
def test_upstream_extension_cli(asset: Path, tmp_path: Path) -> None:
    root, dependencies = os.environ["GNN_ROOT"], os.environ["GNN_PARSER_DEPENDENCIES"]
    python = Path(os.environ["GNN_PYTHON"]) if "GNN_PYTHON" in os.environ else None
    loader = load_structured if asset in (MOVING_ASSET, HIERARCHY_ASSET) else load_scientific
    expected = loader(asset, upstream_root=Path(root), dependency_root=Path(dependencies), python=python)
    result = subprocess.run(
        [sys.executable, "-m", "fedference_cli", "model", "validate", "--model", str(asset),
         "--gnn-root", root, "--json"], cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["model"] == json.loads(dumps(expected.model))
    assert payload["program_digest"] == expected.digest
    assert payload["native_configuration"] == "validated"


@pytest.mark.parametrize("field,value", [
    ("prior_mean", [0.0]),
    ("prior_covariance", [[0.0, 0.0], [0.0, 1.0]]),
    ("client_means", [[0.0, 0.0]]),
    ("client_covariances", [[[1.0, 0.0], [0.0, -1.0]]] * 2),
])
def test_upstream_cli_rejects_malformed_native_configuration(
    field: str, value: object, tmp_path: Path,
) -> None:
    root = os.environ["GNN_ROOT"]
    prefix, text = SITES_ASSET.read_text().split("## FedferenceScientific\n")
    raw = json.loads(text)
    raw["native"][field] = value
    path = tmp_path / "bad-sites.gnn.md"
    text = prefix + "## FedferenceScientific\n" + json.dumps(raw) + "\n"
    path.write_text(text)
    result = subprocess.run(
        [sys.executable, "-m", "fedference_cli", "model", "validate", "--model", str(path),
         "--gnn-root", root, "--json"], cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "GNN_" in result.stderr
    assert result.stdout == ""
    assert path.read_text() == text




@pytest.mark.parametrize("existing", [False, True])
def test_sentinel_rejects_unused_dependency_override(tmp_path: Path, existing: bool) -> None:
    """An explicit inapplicable option must not silently succeed."""
    dependencies = os.environ["GNN_PARSER_DEPENDENCIES"] if existing else str(tmp_path / "missing")
    result = subprocess.run(
        [sys.executable, "-m", "fedference_cli", "model", "validate", "--model", str(SENTINEL_ASSET),
         "--gnn-root", os.environ["GNN_ROOT"], "--parser-dependencies", dependencies, "--json"],
        cwd=tmp_path, capture_output=True, text=True, timeout=60, check=False,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "--parser-dependencies applies only to structured/scientific models" in result.stderr
    assert result.stdout == ""
