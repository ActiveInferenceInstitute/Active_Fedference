"""Real source/extractor/native contracts, with independent arithmetic controls."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from fedference.gnn.admission import GNNAdmissionError, dumps
from fedference.gnn.loader import (
    PARSER_MANIFEST,
    SENTINEL_ASSET,
    SENTINEL_CACHE,
    load_gnn,
    load_sentinel,
    verify_sentinel_cache,
)
from fedference.gnn.native import SentinelParameters, native_posterior, native_world, parameterize_sentinel


@pytest.fixture
def upstream() -> Path:
    root = os.environ.get("GNN_ROOT")
    if root is None:
        pytest.skip("real upstream gate requires caller-selected GNN_ROOT")
    return Path(root)


def _changed_asset(tmp_path: Path) -> Path:
    lines = SENTINEL_ASSET.read_text().splitlines()
    for index, line in enumerate(lines):
        if line.startswith("A="):
            a = json.loads(line[2:])
            a[0][0] = 0.5
            a[1][0] += 0.4
            lines[index] = "A=" + json.dumps(a)
    path = tmp_path / "changed.gnn.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_asset_changes_drive_native_sentinel(tmp_path: Path, upstream: Path) -> None:
    # Given an independently specified likelihood edit and uniform prior.
    path = _changed_asset(tmp_path)
    # When the real extractor feeds the existing native inference path.
    model = load_gnn(path, upstream_root=upstream)
    posterior = native_posterior(model, 0, np.zeros(9))
    # Then Bayes arithmetic is .5 / (.5 + 8*.0125) = 5/6.
    np.testing.assert_allclose(posterior, [5 / 6] + [1 / 48] * 8, atol=1e-14)
    assert native_posterior(load_sentinel(), 0, np.zeros(9))[0] == pytest.approx(0.9)
    print(json.dumps({"posterior": posterior.tolist(), "asset": model.metadata.provenance.asset}))


def test_stale_cache_and_wrong_b_orientation(tmp_path: Path, upstream: Path) -> None:
    # Given a changed source with an old cache and a falsely declared axis order.
    path = _changed_asset(tmp_path)
    original_cache = SENTINEL_CACHE.read_bytes()
    wrong = tmp_path / "wrong-axis.gnn.md"
    wrong.write_text(
        SENTINEL_ASSET.read_text().replace(
            "# B[next_state,previous_state,action]", "# B[action,previous_state,next_state]"
        )
    )
    # When both invalid inputs cross their respective boundaries.
    with pytest.raises(GNNAdmissionError, match="GNN_STALE_CACHE") as stale:
        load_sentinel(asset=path)
    with pytest.raises(GNNAdmissionError, match="GNN_AXIS") as axis:
        load_gnn(wrong, upstream_root=upstream)
    # Then neither produced accepted output or overwrote the compiled artifact.
    assert SENTINEL_CACHE.read_bytes() == original_cache
    print(json.dumps({"stale": str(stale.value), "axis": str(axis.value)}))


def test_real_parser_regeneration_matches_cache(upstream: Path) -> None:
    # Given the declared upstream source closure.
    # When regenerating without writes.
    parsed = verify_sentinel_cache(upstream_root=upstream)
    # Then the actual parsed values match every compiled field.
    assert dumps(parsed) == dumps(load_sentinel())


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "section", "schema", "shape", "transpose"])
def test_invalid_raw_source_rejected(mutation: str, tmp_path: Path, upstream: Path) -> None:
    # Given a real malformed document.
    document = SENTINEL_ASSET.read_text()
    lines = document.splitlines()
    a_line = next(line for line in lines if line.startswith("A="))
    if mutation == "duplicate":
        document = document.replace(a_line, a_line + "\n" + a_line)
    elif mutation == "missing":
        document = document.replace(a_line + "\n", "")
    elif mutation == "section":
        document += "\n## UnsupportedDynamics\nx=y\n"
    elif mutation == "schema":
        document = document.replace("fedference.gnn/1", "fedference.gnn/999")
    elif mutation == "shape":
        document = document.replace(a_line, "A=[[1.0]]")
    else:
        b_line = next(line for line in lines if line.startswith("B="))
        b = np.asarray(json.loads(b_line[2:])).transpose(2, 1, 0)
        document = document.replace(b_line, "B=" + json.dumps(b.tolist()))
    path = tmp_path / "bad.gnn.md"
    path.write_text(document)
    # When source admission and real extraction are attempted.
    with pytest.raises(GNNAdmissionError) as error:
        load_gnn(path, upstream_root=upstream)
    # Then the refusal is explicit and the source stays byte-identical.
    assert path.read_text() == document
    print(mutation, error.value)


def test_parser_closure_drift_rejected(tmp_path: Path, upstream: Path) -> None:
    # Given a copy of the parser closure with one changed byte.
    manifest = json.loads(PARSER_MANIFEST.read_text())
    for relative in manifest["files"]:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(upstream / relative, destination)
    target = tmp_path / "src/gnn/extract/pomdp_state.py"
    target.write_text(target.read_text() + "\n# altered closure\n")
    # When the caller selects the altered parser.
    with pytest.raises(GNNAdmissionError, match="GNN_PARSER_DRIFT"):
        load_gnn(SENTINEL_ASSET, upstream_root=tmp_path)


def test_cache_payload_tampering_rejected(tmp_path: Path) -> None:
    # Given a valid PMF edit retaining the original asset digest.
    cache = json.loads(SENTINEL_CACHE.read_text())
    cache["categorical"]["A"]["values"][0] -= 0.1
    cache["categorical"]["A"]["values"][9] += 0.1
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(cache))
    # When offline loading checks source and compiled content.
    with pytest.raises(GNNAdmissionError, match="GNN_EXTRACTION_MISMATCH"):
        load_sentinel(cache=path)


@pytest.mark.parametrize("acuity", [0.2, 0.9, 1.0])
def test_explicit_parameterization_retains_native_semantics(acuity: float) -> None:
    # Given the canonical source with explicit overrides.
    # When the native boundary adapts a parameterized model.
    world = native_world(parameterize_sentinel(load_sentinel(), SentinelParameters(acuity, -3)))
    # Then independent characterization formulas hold.
    expected = np.full((9, 9), (1 - acuity) / 8)
    np.fill_diagonal(expected, acuity)
    np.testing.assert_allclose(world["A"][0], expected, atol=1e-15)
    np.testing.assert_array_equal(world["C"][0].ravel(), [0, 0, 0, 0, -3, 0, 0, 0, 0])
    for state in range(9):
        row, col = divmod(state, 3)
        assert world["B"][0][row * 3 + max(0, col - 1), state, 1] == 1
        assert world["B"][0][row * 3 + min(2, col + 1), state, 2] == 1


@pytest.mark.parametrize("acuity", [0, -1, 1.1, float("nan")])
def test_invalid_parameterization_rejected(acuity: float) -> None:
    # Given an invalid explicit sensor probability.
    # When it crosses the native parameter boundary.
    with pytest.raises(GNNAdmissionError, match="GNN_ACUITY"):
        parameterize_sentinel(load_sentinel(), SentinelParameters(acuity))


def test_missing_executable_is_explicit_refusal(upstream: Path, tmp_path: Path) -> None:
    # Given a caller-selected missing executable.
    # When real extraction is requested.
    with pytest.raises(GNNAdmissionError, match="GNN_PROCESS"):
        load_gnn(SENTINEL_ASSET, upstream_root=upstream, python=tmp_path / "absent-python")


def test_unmanifested_parser_shadow_never_executes(tmp_path: Path, upstream: Path) -> None:
    import typing

    closure = tmp_path / "upstream"
    manifest = json.loads(PARSER_MANIFEST.read_text())
    for relative in manifest["files"]:
        destination = closure / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(upstream / relative, destination)
    marker = tmp_path / "unmanifested-code-ran"
    shadow = (
        f"open({str(marker)!r}, \"w\").write(\"unmanifested typing executed\")\n"
        f"exec(compile(open({typing.__file__!r}).read(), {typing.__file__!r}, \"exec\"))\n"
    )
    (closure / "src/typing.py").write_text(shadow)
    model = load_gnn(SENTINEL_ASSET, upstream_root=closure)
    assert dumps(model) == dumps(load_sentinel())
    assert not marker.exists(), "unmanifested code executed outside the pinned closure"


def test_raw_oversized_json_integer_has_stable_refusal(tmp_path: Path, upstream: Path) -> None:
    lines = SENTINEL_ASSET.read_text().splitlines()
    lines = ["D=[" + "1" * 5000 + "]" if line.startswith("D=") else line for line in lines]
    path = tmp_path / "oversized.gnn.md"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(GNNAdmissionError) as error:
        load_gnn(path, upstream_root=upstream)
    assert error.value.code == "GNN_JSON"


def test_newline_byte_change_invalidates_offline_cache(tmp_path: Path) -> None:
    asset = tmp_path / "changed-newlines.gnn.md"
    asset.write_bytes(SENTINEL_ASSET.read_bytes().replace(b"\n", b"\r\n"))
    with pytest.raises(GNNAdmissionError, match="GNN_STALE_CACHE"):
        load_sentinel(asset=asset)


@pytest.mark.parametrize("ending", ["\n", "\r\n", "\r"])
def test_source_digest_binds_original_unicode_and_newline_bytes(
    tmp_path: Path, upstream: Path, ending: str
) -> None:
    import hashlib

    prefix, heading, tail = SENTINEL_ASSET.read_text().partition("## ModelName\n")
    _, separator, rest = tail.partition("\n\n## ")
    document = prefix + heading + "Sentinel α" + separator + rest
    asset = tmp_path / "unicode.gnn.md"
    payload = document.replace("\n", ending).encode("utf-8")
    asset.write_bytes(payload)
    model = load_gnn(asset, upstream_root=upstream)
    assert model.metadata.name == "Sentinel α"
    assert model.metadata.provenance.asset == hashlib.sha256(payload).hexdigest()
    assert model.metadata.provenance.source == hashlib.sha256(payload).hexdigest()
    assert model.categorical.A == load_sentinel().categorical.A
    compiled = tmp_path / "unicode.ir.json"
    compiled.write_text(dumps(model), encoding="utf-8")
    assert load_sentinel(asset=asset, cache=compiled) == model
