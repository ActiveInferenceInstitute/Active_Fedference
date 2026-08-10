"""Release manifest tests (MED-1) — no mocks, real files and real digests."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from publication.release_manifest import (
    FINGERPRINT_INPUTS,
    build_release,
    compute_fingerprint,
    timestamp_from_source_date_epoch,
    verify_release,
)

_PDF_FIXTURE = b"%PDF-1.4 minimal"
_PNG_FIXTURE = b"\x89PNG minimal"


def _make_artifacts(tmp_path: Path) -> None:
    (tmp_path / "output" / "pdf").mkdir(parents=True)
    (tmp_path / "output" / "figures").mkdir(parents=True)
    (tmp_path / "output" / "pdf" / "paper.pdf").write_bytes(_PDF_FIXTURE)
    (tmp_path / "output" / "figures" / "fig.png").write_bytes(_PNG_FIXTURE)
    (tmp_path / "output" / "figures" / "junk.log").write_text("excluded")
    (tmp_path / "CITATION.cff").write_text("cff-version: 1.2.0\n")


def _make_source_tree(tmp_path: Path) -> None:
    """Fingerprint inputs: code, producers, manuscript, claims, and configs."""
    (tmp_path / "src" / "analysis").mkdir(parents=True)
    (tmp_path / "src" / "analysis" / "model.py").write_text("SEED = 7\n")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "producer.py").write_text("PROFILE = 'publication'\n")
    (tmp_path / "manuscript").mkdir()
    (tmp_path / "manuscript" / "config.yaml").write_text("seed: 7\n")
    (tmp_path / "manuscript" / "01_introduction.md").write_text("Claim A.\n")
    (tmp_path / "manuscript" / "references.bib").write_text("@article{a,\n  year = {2026}\n}\n")
    (tmp_path / "experiment_plan.yaml").write_text("plan: baseline\n")
    (tmp_path / "docs" / "research").mkdir(parents=True)
    (tmp_path / "docs" / "research" / "claim-audit.md").write_text("Scoped.\n")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "fixture"\nversion = "0.0.1"\n')
    (tmp_path / "uv.lock").write_text("version = 1\n")
    (tmp_path / "ISA.md").write_text("phase: learn\n")
    (tmp_path / "TODO.md").write_text("- open\n")
    (tmp_path / "REDTEAM_REVIEW.md").write_text("Reviewed.\n")


def test_build_writes_bundle_with_true_digests(tmp_path: Path) -> None:
    _make_artifacts(tmp_path)
    manifest = build_release(tmp_path, timestamp="2026-07-06T00:00:00Z")
    release = tmp_path / "output" / "release"
    assert (release / "manifest.json").exists()
    assert (release / "sha256sums.txt").exists()
    assert (release / "README.md").exists()
    paths = {e["path"] for e in manifest["artifacts"]}
    assert paths == {"output/pdf/paper.pdf", "output/figures/fig.png", "CITATION.cff"}
    # .log excluded; digests are the REAL sha256 of the bytes on disk.
    entry = next(e for e in manifest["artifacts"] if e["path"].endswith("paper.pdf"))
    assert entry["sha256"] == hashlib.sha256(_PDF_FIXTURE).hexdigest()
    assert manifest["n_artifacts"] == 3
    assert manifest["generated_at"] == "2026-07-06T00:00:00Z"
    assert manifest["timestamp_policy"] == "recorded"
    # README counts are derived from the walk, not hand-typed.
    assert "3 files" in (release / "README.md").read_text()


def test_bundle_excludes_its_own_directory_on_rebuild(tmp_path: Path) -> None:
    _make_artifacts(tmp_path)
    build_release(tmp_path, timestamp="2026-07-06T00:00:00Z")
    second = build_release(tmp_path, timestamp="2026-07-06T00:00:00Z")
    assert not any("output/release" in e["path"] for e in second["artifacts"])
    assert second["n_artifacts"] == 3
    assert verify_release(tmp_path) == []


def test_default_build_is_byte_idempotent_and_omits_release_time(
    tmp_path: Path,
) -> None:
    """An unreleased clean-clone rebuild must not drift with wall-clock time."""
    _make_artifacts(tmp_path)
    _make_source_tree(tmp_path)
    first = build_release(tmp_path)
    release = tmp_path / "output" / "release"
    first_bytes = {
        path.name: path.read_bytes()
        for path in sorted(release.iterdir())
        if path.is_file()
    }
    second = build_release(tmp_path)
    second_bytes = {
        path.name: path.read_bytes()
        for path in sorted(release.iterdir())
        if path.is_file()
    }
    assert first == second
    assert first_bytes == second_bytes
    assert first["generated_at"] is None
    assert first["timestamp_policy"] == "omitted"
    assert "byte-reproducible unreleased build" in (
        release / "README.md"
    ).read_text(encoding="utf-8")
    assert verify_release(tmp_path) == []


@pytest.mark.parametrize(
    ("epoch", "expected"),
    [
        ("0", "1970-01-01T00:00:00Z"),
        ("1782864000", "2026-07-01T00:00:00Z"),
    ],
)
def test_source_date_epoch_conversion(epoch: str, expected: str) -> None:
    assert timestamp_from_source_date_epoch(epoch) == expected


@pytest.mark.parametrize(
    "epoch",
    ["", "-1", "1.5", "tomorrow", "９", "9" * 21],
)
def test_source_date_epoch_rejects_invalid_values(epoch: str) -> None:
    with pytest.raises(ValueError, match="SOURCE_DATE_EPOCH"):
        timestamp_from_source_date_epoch(epoch)


@pytest.mark.parametrize(
    "timestamp",
    ["", "2026-07-01", "2026-07-01T00:00:00+00:00", "not-a-date"],
)
def test_build_rejects_noncanonical_release_timestamp(
    tmp_path: Path, timestamp: str
) -> None:
    _make_artifacts(tmp_path)
    with pytest.raises(ValueError, match="timestamp"):
        build_release(tmp_path, timestamp=timestamp)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ({"generated_at": "2026-07-01"}, "generated_at is not canonical UTC"),
        (
            {"generated_at": None, "timestamp_policy": "recorded"},
            "omitted generated_at requires timestamp_policy=omitted",
        ),
        (
            {
                "generated_at": "2026-07-01T00:00:00Z",
                "timestamp_policy": "omitted",
            },
            "populated generated_at requires timestamp_policy=recorded",
        ),
        ({"generated_at": False}, "generated_at must be a canonical UTC string or null"),
    ],
)
def test_verify_rejects_timestamp_contract_drift(
    tmp_path: Path, mutation: dict[str, object], expected: str
) -> None:
    _make_artifacts(tmp_path)
    build_release(tmp_path)
    manifest_path = tmp_path / "output" / "release" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(mutation)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert any(expected in item for item in verify_release(tmp_path))


def test_verify_rejects_missing_generated_at_field(tmp_path: Path) -> None:
    _make_artifacts(tmp_path)
    build_release(tmp_path)
    manifest_path = tmp_path / "output" / "release" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["generated_at"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert "manifest: generated_at field missing" in verify_release(tmp_path)


def test_verify_detects_tamper_and_shasum_agrees(tmp_path: Path) -> None:
    """Proof-of-detection: altering one artifact must fail verify AND shasum -c."""
    _make_artifacts(tmp_path)
    build_release(tmp_path, timestamp="2026-07-06T00:00:00Z")
    assert verify_release(tmp_path) == []
    (tmp_path / "output" / "pdf" / "paper.pdf").write_bytes(b"%PDF-1.4 TAMPERED")
    assert verify_release(tmp_path) == ["output/pdf/paper.pdf"]
    proc = subprocess.run(
        ["shasum", "-a", "256", "-c", "output/release/sha256sums.txt"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "FAILED" in proc.stdout


def test_verify_detects_missing_artifact(tmp_path: Path) -> None:
    _make_artifacts(tmp_path)
    build_release(tmp_path, timestamp="2026-07-06T00:00:00Z")
    (tmp_path / "output" / "figures" / "fig.png").unlink()
    assert verify_release(tmp_path) == ["output/figures/fig.png"]


def test_verify_detects_unlisted_artifact(tmp_path: Path) -> None:
    _make_artifacts(tmp_path)
    build_release(tmp_path, timestamp="2026-07-06T00:00:00Z")
    extra = tmp_path / "output" / "figures" / "unreviewed.png"
    extra.write_bytes(b"not in the manifest")
    assert verify_release(tmp_path) == ["manifest: unexpected artifact output/figures/unreviewed.png"]


def test_verify_detects_manifest_metadata_drift_and_duplicate(tmp_path: Path) -> None:
    _make_artifacts(tmp_path)
    build_release(tmp_path, timestamp="2026-07-06T00:00:00Z")
    manifest_path = tmp_path / "output" / "release" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"][0]["bytes"] += 1
    manifest["artifacts"].append(dict(manifest["artifacts"][0]))
    manifest_path.write_text(json.dumps(manifest))
    bad = verify_release(tmp_path)
    assert any("duplicate artifact" in item for item in bad)
    assert any("n_artifacts mismatch" in item for item in bad)
    assert any(item.startswith("output/") for item in bad)


def test_verify_rejects_non_numeric_bytes_metadata(tmp_path: Path) -> None:
    _make_artifacts(tmp_path)
    build_release(tmp_path, timestamp="2026-07-06T00:00:00Z")
    manifest_path = tmp_path / "output" / "release" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"][0]["bytes"] = "not-an-integer"
    manifest_path.write_text(json.dumps(manifest))
    bad = verify_release(tmp_path)
    assert any("invalid bytes metadata" in item for item in bad)
    assert "manifest: total_bytes mismatch" in bad


def test_manifest_carries_fingerprint_stable_across_rebuild(tmp_path: Path) -> None:
    """Fingerprint is present, deterministic, and unchanged by a no-op rebuild."""
    _make_artifacts(tmp_path)
    _make_source_tree(tmp_path)
    first = build_release(tmp_path, timestamp="2026-07-06T00:00:00Z")
    second = build_release(tmp_path, timestamp="2026-07-07T00:00:00Z")
    assert isinstance(first["fingerprint"], str) and len(first["fingerprint"]) == 64
    assert first["fingerprint"] == second["fingerprint"]  # timestamp does not enter it
    assert first["fingerprint"] == compute_fingerprint(tmp_path)
    assert first["fingerprint_inputs"] == list(FINGERPRINT_INPUTS)
    assert first["pipeline_profile"] == "publication"
    assert first["generator_version"] == "4"
    assert first["package_version"] == "0.0.1"
    assert first["fingerprint_files"]["scripts/producer.py"]
    assert verify_release(tmp_path) == []


def test_verify_rejects_stale_bundle_after_source_change(tmp_path: Path) -> None:
    """Negative control: source mutation with intact byte digests must fail."""
    _make_artifacts(tmp_path)
    _make_source_tree(tmp_path)
    build_release(tmp_path, timestamp="2026-07-06T00:00:00Z")
    assert verify_release(tmp_path) == []
    # Mutate a fingerprint input WITHOUT rebuilding: every artifact digest in
    # the bundle still matches, so only the provenance layer can catch this.
    (tmp_path / "src" / "analysis" / "model.py").write_text("SEED = 8\n")
    bad = verify_release(tmp_path)
    assert len(bad) == 1  # byte-digest layer stays green; staleness alone fails
    assert "provenance fingerprint mismatch" in bad[0]
    assert "stale" in bad[0]


def test_verify_rejects_stale_bundle_after_test_change(tmp_path: Path) -> None:
    """The release fingerprint must bind the test evidence it reports."""
    _make_artifacts(tmp_path)
    _make_source_tree(tmp_path)
    test_path = tmp_path / "tests" / "test_model.py"
    test_path.parent.mkdir()
    test_path.write_text("def test_model():\n    assert True\n")
    build_release(tmp_path, timestamp="2026-07-06T00:00:00Z")
    test_path.write_text("def test_model():\n    assert 2 == 2\n")
    bad = verify_release(tmp_path)
    assert len(bad) == 1
    assert "tests/test_model.py" in bad[0]


def test_verify_rejects_stale_bundle_after_config_change(tmp_path: Path) -> None:
    _make_artifacts(tmp_path)
    _make_source_tree(tmp_path)
    build_release(tmp_path, timestamp="2026-07-06T00:00:00Z")
    (tmp_path / "manuscript" / "config.yaml").write_text("seed: 8\n")
    bad = verify_release(tmp_path)
    assert len(bad) == 1
    assert "provenance fingerprint mismatch" in bad[0]


def test_verify_rejects_stale_bundle_after_manuscript_change(tmp_path: Path) -> None:
    """Rendered claims cannot change without invalidating the release bundle."""
    _make_artifacts(tmp_path)
    _make_source_tree(tmp_path)
    build_release(tmp_path, timestamp="2026-07-06T00:00:00Z")
    assert verify_release(tmp_path) == []
    (tmp_path / "manuscript" / "01_introduction.md").write_text("Claim B.\n")
    bad = verify_release(tmp_path)
    assert len(bad) == 1
    assert "provenance fingerprint mismatch" in bad[0]


def test_verify_rejects_stale_bundle_after_claim_audit_change(tmp_path: Path) -> None:
    """Claim-boundary edits require a fresh manifest too."""
    _make_artifacts(tmp_path)
    _make_source_tree(tmp_path)
    build_release(tmp_path, timestamp="2026-07-06T00:00:00Z")
    assert verify_release(tmp_path) == []
    (tmp_path / "docs" / "research" / "claim-audit.md").write_text("Rescoped.\n")
    bad = verify_release(tmp_path)
    assert len(bad) == 1
    assert "provenance fingerprint mismatch" in bad[0]
    assert "docs/research/claim-audit.md" in bad[0]


def test_verify_rejects_stale_bundle_after_producer_change(tmp_path: Path) -> None:
    """Pipeline code changes invalidate an otherwise byte-consistent bundle."""
    _make_artifacts(tmp_path)
    _make_source_tree(tmp_path)
    build_release(tmp_path, timestamp="2026-07-06T00:00:00Z")
    (tmp_path / "scripts" / "producer.py").write_text("PROFILE = 'smoke'\n")
    bad = verify_release(tmp_path)
    assert len(bad) == 1
    assert "scripts/producer.py" in bad[0]


def test_manifest_json_is_sha256sum_c_compatible(tmp_path: Path) -> None:
    _make_artifacts(tmp_path)
    build_release(tmp_path, timestamp="2026-07-06T00:00:00Z")
    proc = subprocess.run(
        ["shasum", "-a", "256", "-c", "output/release/sha256sums.txt"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    manifest = json.loads((tmp_path / "output" / "release" / "manifest.json").read_text())
    assert manifest["total_bytes"] == sum(e["bytes"] for e in manifest["artifacts"])
