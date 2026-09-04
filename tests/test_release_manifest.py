"""Release manifest tests (MED-1) — no mocks, real files and real digests."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from publication.clean_checkout import IMMUTABLE_RELEASE_PDFS
from publication.release_manifest import (
    DOWNSTREAM_CONTROL_ARTIFACTS,
    DOWNSTREAM_CONTROL_PREFIXES,
    FINGERPRINT_INPUTS,
    RELEASE_ARTIFACT_SCOPE,
    build_release,
    compute_fingerprint,
    timestamp_from_source_date_epoch,
    validate_utc_timestamp,
    verify_release,
)

_PDF_FIXTURE = b"%PDF-1.4 minimal"
_PNG_FIXTURE = b"\x89PNG minimal"


def _write_artifact(root: Path, relative: str, content: bytes) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _symlink_or_skip(link: Path, target: Path, *, target_is_directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks are unavailable on this filesystem: {exc}")


def _make_artifacts(tmp_path: Path) -> None:
    (tmp_path / "output" / "pdf").mkdir(parents=True)
    (tmp_path / "output" / "figures").mkdir(parents=True)
    (tmp_path / "output" / "pdf" / "paper.pdf").write_bytes(_PDF_FIXTURE)
    (tmp_path / "output" / "figures" / "fig.png").write_bytes(_PNG_FIXTURE)
    (tmp_path / "output" / "figures" / "junk.log").write_text("excluded")
    for suffix in ("aux", "bbl", "blg", "lof", "lot", "nav", "out", "snm", "toc", "vrb"):
        (tmp_path / "output" / "pdf" / f"renderer.{suffix}").write_text("excluded")
    (tmp_path / "CITATION.cff").write_text("cff-version: 1.2.0\n")


def _make_source_tree(tmp_path: Path) -> None:
    """Fingerprint inputs: code, producers, manuscript, claims, and configs."""
    (tmp_path / "src" / "analysis").mkdir(parents=True)
    (tmp_path / "src" / "analysis" / "model.py").write_text("SEED = 7\n")
    (tmp_path / "src" / "AGENTS.md").write_text("# Source guidance\n")
    (tmp_path / "src" / "fedference" / "config").mkdir(parents=True)
    (tmp_path / "src" / "fedference" / "config" / "hierarchical_layers.yaml").write_text(
        "layers: 3\n"
    )
    (tmp_path / "src" / "fedference" / "data").mkdir()
    (tmp_path / "src" / "fedference" / "data" / "README.md").write_text("# Package data\n")
    (tmp_path / "src" / "fedference" / "data" / "synthetic_tabular.csv").write_text(
        "feature,label\n0,0\n"
    )
    (tmp_path / "src" / "fedference" / "py.typed").write_text("")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "producer.py").write_text("PROFILE = 'publication'\n")
    (tmp_path / "scripts" / "AGENTS.md").write_text("# Script guidance\n")
    (tmp_path / "scripts" / "CONVENTIONS.md").write_text("# Script conventions\n")
    (tmp_path / "scripts" / "README.md").write_text("# Scripts\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "AGENTS.md").write_text("# Test guidance\n")
    (tmp_path / "tests" / "PATTERNS.md").write_text("# Test patterns\n")
    (tmp_path / "tests" / "README.md").write_text("# Tests\n")
    (tmp_path / "examples").mkdir()
    (tmp_path / "examples" / "README.md").write_text("# Runnable examples\n")
    (tmp_path / "examples" / "minimal.py").write_text("print('ok')\n")
    (tmp_path / "examples" / "data").mkdir()
    (tmp_path / "examples" / "data" / "labeled_aggregation_request.json").write_text(
        '{"schema_version":"1.0","state_labels":["clear"],"agents":[]}\n'
    )
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "README.md").write_text("# Scientific data\n")
    (tmp_path / "data" / "AGENTS.md").write_text("# Data guidance\n")
    (tmp_path / "data" / "claim_ledger.yaml").write_text("claims: []\n")
    (tmp_path / "data" / "synthetic_tabular.csv").write_text("feature,label\n0,0\n")
    (tmp_path / "manuscript").mkdir()
    (tmp_path / "manuscript" / "config.yaml").write_text("seed: 7\n")
    (tmp_path / "manuscript" / "config.yaml.example").write_text("seed: 0\n")
    (tmp_path / "manuscript" / "01_introduction.md").write_text("Claim A.\n")
    (tmp_path / "manuscript" / "AGENTS.md").write_text("# Manuscript guidance\n")
    (tmp_path / "manuscript" / "references.bib").write_text("@article{a,\n  year = {2026}\n}\n")
    (tmp_path / "manuscript" / "cover_image.png").write_text("fixture image bytes\n")
    (tmp_path / "experiment_plan.yaml").write_text("plan: baseline\n")
    (tmp_path / "domain_profile.yaml").write_text("profile: fixture\n")
    (tmp_path / "docs" / "research").mkdir(parents=True)
    (tmp_path / "docs" / "research" / "claim-audit.md").write_text("Scoped.\n")
    (tmp_path / "docs" / "research" / "AGENTS.md").write_text("# Research guidance\n")
    (tmp_path / "docs" / "reference").mkdir()
    (tmp_path / "docs" / "reference" / "historical-release-pdfs.json").write_text(
        '{"releases":[]}\n'
    )
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "fixture"\nversion = "0.0.1"\n')
    (tmp_path / "uv.lock").write_text("version = 1\n")
    (tmp_path / "ISA.md").write_text("phase: learn\n")
    (tmp_path / "AGENTS.md").write_text("# Repository guidance\n")
    (tmp_path / "STANDALONE.md").write_text("# Standalone contract\n")
    (tmp_path / "TODO.md").write_text("- open\n")
    (tmp_path / "REDTEAM_REVIEW.md").write_text("Reviewed.\n")
    for filename in IMMUTABLE_RELEASE_PDFS:
        (tmp_path / filename).write_text(f"historical PDF fixture: {filename}\n")


def test_build_writes_bundle_with_true_digests(tmp_path: Path) -> None:
    _make_artifacts(tmp_path)
    manifest = build_release(tmp_path, timestamp="2026-07-06T00:00:00Z")
    release = tmp_path / "output" / "release"
    assert (release / "manifest.json").exists()
    assert (release / "sha256sums.txt").exists()
    assert (release / "README.md").exists()
    paths = {e["path"] for e in manifest["artifacts"]}
    assert paths == {"output/pdf/paper.pdf", "output/figures/fig.png", "CITATION.cff"}
    # Renderer logs/sidecars are excluded; digests are the REAL sha256 of the bytes on disk.
    entry = next(e for e in manifest["artifacts"] if e["path"].endswith("paper.pdf"))
    assert entry["sha256"] == hashlib.sha256(_PDF_FIXTURE).hexdigest()
    assert manifest["n_artifacts"] == 3
    assert manifest["manifest_schema_version"] == 4
    assert manifest["generator_version"] == "6"
    assert manifest["artifact_scope"] == RELEASE_ARTIFACT_SCOPE
    assert manifest["downstream_control_exclusions"] == {
        "paths": list(DOWNSTREAM_CONTROL_ARTIFACTS),
        "prefixes": list(DOWNSTREAM_CONTROL_PREFIXES),
    }
    assert manifest["generated_at"] == "2026-07-06T00:00:00Z"
    assert manifest["timestamp_policy"] == "recorded"
    # README counts are derived from the walk, not hand-typed.
    release_readme = (release / "README.md").read_text(encoding="utf-8")
    assert "3 files" in release_readme
    assert "uv run --locked python scripts/build_release.py" in release_readme
    assert "uv run python scripts/build_release.py" not in release_readme
    assert RELEASE_ARTIFACT_SCOPE in release_readme
    assert "without a checksum cycle" in release_readme


def test_release_bundle_carries_the_declared_license(tmp_path: Path) -> None:
    _make_artifacts(tmp_path)
    license_text = "MIT License\n"
    (tmp_path / "LICENSE").write_text(license_text, encoding="utf-8")

    manifest = build_release(tmp_path)

    entry = next(item for item in manifest["artifacts"] if item["path"] == "LICENSE")
    assert entry["bytes"] == len(license_text.encode("utf-8"))
    assert entry["sha256"] == hashlib.sha256(license_text.encode("utf-8")).hexdigest()
    assert verify_release(tmp_path) == []


def test_release_bundle_includes_visual_accessibility_support_artifacts(
    tmp_path: Path,
) -> None:
    _make_artifacts(tmp_path)
    expected = {
        "output/figures/figure_exact_values.json": b'{"schema_version":"1.0"}\n',
        "output/figures/figure_exact_values.md": b"# Exact values\n",
        "output/figures/figure_registry.json": b'{"schema_version":"1.2"}\n',
        "output/reports/application_integrity_flow.json": b"{}\n",
        "output/reports/evidence_replication_map.json": b"{}\n",
        "output/reports/sensitivity.json": b"{}\n",
        "output/reports/source_render_provenance.json": b"{}\n",
    }
    for relative, content in expected.items():
        _write_artifact(tmp_path, relative, content)

    manifest = build_release(tmp_path)
    paths = {str(entry["path"]) for entry in manifest["artifacts"]}

    assert set(expected) <= paths
    assert verify_release(tmp_path) == []


def test_bundle_excludes_its_own_directory_on_rebuild(tmp_path: Path) -> None:
    _make_artifacts(tmp_path)
    build_release(tmp_path, timestamp="2026-07-06T00:00:00Z")
    second = build_release(tmp_path, timestamp="2026-07-06T00:00:00Z")
    assert not any("output/release" in e["path"] for e in second["artifacts"])
    assert second["n_artifacts"] == 3
    assert verify_release(tmp_path) == []


def test_bundle_excludes_only_declared_downstream_controls(tmp_path: Path) -> None:
    _make_artifacts(tmp_path)
    scientific = _write_artifact(
        tmp_path,
        "output/reports/scientific_result.json",
        b'{"mean": 0.25}\n',
    )
    for relative in DOWNSTREAM_CONTROL_ARTIFACTS:
        _write_artifact(tmp_path, relative, f"control: {relative}\n".encode())
    snapshot = _write_artifact(
        tmp_path,
        DOWNSTREAM_CONTROL_PREFIXES[0] + "stage-99-regression.json",
        b'{"control": true}\n',
    )

    manifest = build_release(tmp_path)
    paths = {entry["path"] for entry in manifest["artifacts"]}

    assert scientific.relative_to(tmp_path).as_posix() in paths
    assert not set(DOWNSTREAM_CONTROL_ARTIFACTS) & paths
    assert snapshot.relative_to(tmp_path).as_posix() not in paths
    assert verify_release(tmp_path) == []


def test_release_payload_and_downstream_controls_form_a_dag(tmp_path: Path) -> None:
    """Control receipts may bind payload bytes without feeding back into them."""
    _make_artifacts(tmp_path)
    _make_source_tree(tmp_path)
    scientific = _write_artifact(
        tmp_path,
        "output/reports/scientific_result.json",
        b'{"paired_difference": 0.125}\n',
    )

    first = build_release(tmp_path)
    release_dir = tmp_path / "output" / "release"
    first_release_bytes = {
        path.name: path.read_bytes() for path in sorted(release_dir.iterdir()) if path.is_file()
    }
    release_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(release_dir.iterdir())
        if path.is_file()
    }

    artifact_manifest = _write_artifact(
        tmp_path,
        "output/reports/artifact_manifest.json",
        (json.dumps({"release_hashes": release_hashes}, sort_keys=True) + "\n").encode(),
    )
    artifact_hash = hashlib.sha256(artifact_manifest.read_bytes()).hexdigest()
    evidence = _write_artifact(
        tmp_path,
        "output/reports/evidence_registry.json",
        (json.dumps({"payload_manifest": release_hashes["manifest.json"]}) + "\n").encode(),
    )
    statistics = _write_artifact(
        tmp_path,
        "output/reports/output_statistics.json",
        (
            json.dumps(
                {"release_bytes": sum(len(value) for value in first_release_bytes.values())}
            )
            + "\n"
        ).encode(),
    )
    validation = _write_artifact(
        tmp_path,
        "output/reports/validation_report.json",
        (json.dumps({"artifact_manifest_sha256": artifact_hash}) + "\n").encode(),
    )
    rendered = _write_artifact(
        tmp_path,
        "output/reports/rendered_provenance.json",
        (
            json.dumps(
                {
                    "artifact_manifest_sha256": artifact_hash,
                    "validation_report_sha256": hashlib.sha256(
                        validation.read_bytes()
                    ).hexdigest(),
                },
                sort_keys=True,
            )
            + "\n"
        ).encode(),
    )
    snapshot = _write_artifact(
        tmp_path,
        DOWNSTREAM_CONTROL_PREFIXES[0] + "stage-04-validation.json",
        (json.dumps({"artifact_manifest_sha256": artifact_hash}) + "\n").encode(),
    )

    second = build_release(tmp_path)
    second_release_bytes = {
        path.name: path.read_bytes() for path in sorted(release_dir.iterdir()) if path.is_file()
    }
    assert second == first
    assert second_release_bytes == first_release_bytes
    assert verify_release(tmp_path) == []

    for control in (artifact_manifest, evidence, statistics, validation, rendered, snapshot):
        control.write_bytes(control.read_bytes() + b"downstream refresh\n")
    third = build_release(tmp_path)
    third_release_bytes = {
        path.name: path.read_bytes() for path in sorted(release_dir.iterdir()) if path.is_file()
    }
    assert third == first
    assert third_release_bytes == first_release_bytes
    assert verify_release(tmp_path) == []

    scientific.write_bytes(b'{"paired_difference": 0.5}\n')
    assert verify_release(tmp_path) == ["output/reports/scientific_result.json"]


def test_build_rejects_symlink_alias_to_excluded_downstream_control(
    tmp_path: Path,
) -> None:
    _make_artifacts(tmp_path)
    control = _write_artifact(
        tmp_path,
        "output/reports/artifact_manifest.json",
        b'{"control": true}\n',
    )
    alias = tmp_path / "output" / "reports" / "control-alias.json"
    _symlink_or_skip(alias, Path(control.name))

    with pytest.raises(
        ValueError,
        match=(
            "release artifact path must not contain symlinks: "
            "output/reports/control-alias.json"
        ),
    ):
        build_release(tmp_path)


def test_verify_rejects_out_of_root_symlink_ancestor(tmp_path: Path) -> None:
    _make_artifacts(tmp_path)
    build_release(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (outside / "private.pdf").write_bytes(b"outside checkout\n")
    alias = tmp_path / "output" / "figures" / "escaped"
    _symlink_or_skip(alias, outside, target_is_directory=True)

    assert verify_release(tmp_path) == [
        "manifest: release artifact path must not contain symlinks: "
        "output/figures/escaped"
    ]


def test_verify_rejects_case_alias_for_excluded_control_when_applicable(
    tmp_path: Path,
) -> None:
    """Case-insensitive filesystems must not alias controls into payload scope."""
    _make_artifacts(tmp_path)
    control = _write_artifact(
        tmp_path,
        "output/reports/artifact_manifest.json",
        b'{"control": true}\n',
    )
    build_release(tmp_path)
    alias = "output/reports/Artifact_Manifest.json"
    alias_path = tmp_path / alias
    if not alias_path.is_file() or not alias_path.samefile(control):
        pytest.skip("filesystem is case-sensitive")

    manifest_path = tmp_path / "output" / "release" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"].append(
        {
            "path": alias,
            "bytes": control.stat().st_size,
            "sha256": hashlib.sha256(control.read_bytes()).hexdigest(),
        }
    )
    manifest["n_artifacts"] += 1
    manifest["total_bytes"] += control.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert f"manifest: non-canonical artifact path {alias}" in verify_release(tmp_path)


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
    assert first["generator_version"] == "6"
    assert first["package_version"] == "0.0.1"
    assert first["fingerprint_files"]["scripts/producer.py"]
    assert first["fingerprint_files"]["examples/minimal.py"]
    assert first["fingerprint_files"]["examples/README.md"]
    assert first["fingerprint_files"]["manuscript/config.yaml.example"]
    assert not any(entry["path"].startswith("examples/") for entry in first["artifacts"])
    assert not any(
        entry["path"] == "manuscript/config.yaml.example" for entry in first["artifacts"]
    )
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
    test_path.parent.mkdir(exist_ok=True)
    test_path.write_text("def test_model():\n    assert True\n")
    build_release(tmp_path, timestamp="2026-07-06T00:00:00Z")
    test_path.write_text("def test_model():\n    assert 2 == 2\n")
    bad = verify_release(tmp_path)
    assert len(bad) == 1
    assert "tests/test_model.py" in bad[0]


@pytest.mark.parametrize(
    ("relative", "replacement"),
    (
        ("examples/minimal.py", "print('changed')\n"),
        ("examples/README.md", "# Revised examples\n"),
        (
            "examples/data/labeled_aggregation_request.json",
            '{"schema_version":"1.0","state_labels":["changed"],"agents":[]}\n',
        ),
        ("manuscript/config.yaml.example", "seed: 1\n"),
    ),
)
def test_verify_rejects_stale_bundle_after_example_contract_change(
    tmp_path: Path,
    relative: str,
    replacement: str,
) -> None:
    """Runnable examples and the copyable config are provenance inputs."""
    _make_artifacts(tmp_path)
    _make_source_tree(tmp_path)
    build_release(tmp_path, timestamp="2026-07-06T00:00:00Z")

    (tmp_path / relative).write_text(replacement, encoding="utf-8")

    bad = verify_release(tmp_path)
    assert len(bad) == 1
    assert "provenance fingerprint mismatch" in bad[0]
    assert relative in bad[0]


@pytest.mark.parametrize(
    ("relative", "replacement"),
    (
        ("src/fedference/data/README.md", "# Revised package data\n"),
        ("src/fedference/config/hierarchical_layers.yaml", "layers: 4\n"),
        ("src/fedference/data/synthetic_tabular.csv", "feature,label\n1,1\n"),
        ("src/fedference/py.typed", "changed\n"),
        ("data/README.md", "# Revised scientific data\n"),
        ("data/claim_ledger.yaml", "claims: [changed]\n"),
        ("data/synthetic_tabular.csv", "feature,label\n1,1\n"),
        ("docs/reference/historical-release-pdfs.json", '{"releases":["changed"]}\n'),
        ("manuscript/cover_image.png", "changed image bytes\n"),
        ("domain_profile.yaml", "profile: changed\n"),
    ),
)
def test_verify_rejects_stale_bundle_after_packaged_or_scientific_data_change(
    tmp_path: Path,
    relative: str,
    replacement: str,
) -> None:
    _make_artifacts(tmp_path)
    _make_source_tree(tmp_path)
    build_release(tmp_path, timestamp="2026-07-06T00:00:00Z")

    (tmp_path / relative).write_text(replacement, encoding="utf-8")

    bad = verify_release(tmp_path)
    assert len(bad) == 1
    assert "provenance fingerprint mismatch" in bad[0]
    assert relative in bad[0]


@pytest.mark.parametrize(
    "relative",
    (
        "AGENTS.md",
        "STANDALONE.md",
        "scripts/README.md",
        "tests/PATTERNS.md",
        "docs/research/AGENTS.md",
        "src/AGENTS.md",
        "data/AGENTS.md",
        "manuscript/AGENTS.md",
    ),
)
def test_verify_rejects_stale_bundle_after_shipped_document_change(
    tmp_path: Path,
    relative: str,
) -> None:
    _make_artifacts(tmp_path)
    _make_source_tree(tmp_path)
    build_release(tmp_path, timestamp="2026-07-06T00:00:00Z")

    (tmp_path / relative).write_text("# Revised source contract\n", encoding="utf-8")

    bad = verify_release(tmp_path)
    assert len(bad) == 1
    assert "provenance fingerprint mismatch" in bad[0]
    assert relative in bad[0]


def test_verify_rejects_stale_bundle_after_immutable_historical_pdf_change(
    tmp_path: Path,
) -> None:
    _make_artifacts(tmp_path)
    _make_source_tree(tmp_path)
    build_release(tmp_path, timestamp="2026-07-06T00:00:00Z")

    target = tmp_path / IMMUTABLE_RELEASE_PDFS[-1]
    target.write_bytes(b"substituted historical PDF\n")

    bad = verify_release(tmp_path)
    assert len(bad) == 1
    assert "provenance fingerprint mismatch" in bad[0]
    assert IMMUTABLE_RELEASE_PDFS[-1] in bad[0]


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


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        (lambda payload: payload.__setitem__("artifacts", {}), "malformed artifacts list"),
        (lambda payload: payload.__setitem__("manifest_schema_version", 1), "schema_version"),
        (lambda payload: payload.__setitem__("pipeline_profile", ""), "pipeline_profile"),
        (lambda payload: payload.__setitem__("generator", "other"), "generator identity"),
        (lambda payload: payload.__setitem__("generator_version", "old"), "generator_version"),
        (lambda payload: payload.pop("artifact_scope"), "artifact_scope drift"),
        (
            lambda payload: payload.__setitem__("artifact_scope", "whole-output-v0"),
            "artifact_scope drift",
        ),
        (
            lambda payload: payload.pop("downstream_control_exclusions"),
            "downstream_control_exclusions drift",
        ),
        (
            lambda payload: payload["downstream_control_exclusions"]["paths"].append(
                "output/reports/unreviewed-control.json"
            ),
            "downstream_control_exclusions drift",
        ),
        (lambda payload: payload.pop("fingerprint"), "fingerprint missing"),
        (lambda payload: payload.__setitem__("fingerprint_files", []), "fingerprint_files missing"),
        (lambda payload: payload.__setitem__("fingerprint_inputs", []), "fingerprint_inputs drift"),
    ),
)
def test_verify_rejects_top_level_manifest_contract_drift(tmp_path: Path, mutation, expected) -> None:
    _make_artifacts(tmp_path)
    _make_source_tree(tmp_path)
    build_release(tmp_path)
    path = tmp_path / "output" / "release" / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutation(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert any(expected in finding for finding in verify_release(tmp_path))


@pytest.mark.parametrize(
    ("path_value", "expected"),
    (
        ("", "unsafe artifact path"),
        ("../outside", "unsafe artifact path"),
        ("/absolute", "unsafe artifact path"),
    ),
)
def test_verify_rejects_unsafe_artifact_paths(tmp_path: Path, path_value: str, expected: str) -> None:
    _make_artifacts(tmp_path)
    build_release(tmp_path)
    path = tmp_path / "output" / "release" / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["artifacts"][0]["path"] = path_value
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert any(expected in finding for finding in verify_release(tmp_path))


@pytest.mark.parametrize(
    "relative",
    (
        "outside-declared-roots.txt",
        "output/reports/artifact_manifest.json",
        "output/reports/snapshots/stage-99.json",
    ),
)
def test_verify_rejects_manifest_entries_outside_declared_payload_scope(
    tmp_path: Path,
    relative: str,
) -> None:
    _make_artifacts(tmp_path)
    target = _write_artifact(tmp_path, relative, b"out-of-scope fixture\n")
    build_release(tmp_path)
    path = tmp_path / "output" / "release" / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["artifacts"].append(
        {
            "path": relative,
            "bytes": target.stat().st_size,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }
    )
    payload["n_artifacts"] += 1
    payload["total_bytes"] += target.stat().st_size
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert (
        f"manifest: artifact outside declared {RELEASE_ARTIFACT_SCOPE} scope: {relative}"
        in verify_release(tmp_path)
    )


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        (lambda entry: entry.__setitem__("bytes", -1), "invalid bytes metadata"),
        (lambda entry: entry.__setitem__("bytes", True), "invalid bytes metadata"),
        (lambda entry: entry.__setitem__("sha256", "wrong"), "output/pdf/paper.pdf"),
        (lambda entry: entry.__setitem__("path", "output/pdf/missing.pdf"), "output/pdf/missing.pdf"),
    ),
)
def test_verify_rejects_invalid_artifact_entries(tmp_path: Path, mutation, expected) -> None:
    _make_artifacts(tmp_path)
    build_release(tmp_path)
    path = tmp_path / "output" / "release" / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutation(payload["artifacts"][0])
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert any(expected in finding for finding in verify_release(tmp_path))


def test_verify_rejects_a_non_mapping_artifact_entry(tmp_path: Path) -> None:
    _make_artifacts(tmp_path)
    build_release(tmp_path)
    path = tmp_path / "output" / "release" / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["artifacts"][0] = None
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert "manifest: malformed artifact entry" in verify_release(tmp_path)


@pytest.mark.parametrize(
    "timestamp",
    ["2026-02-30T00:00:00Z", "2026-07-01T00:00:00.000Z", 1],
)
def test_timestamp_validation_is_canonical_and_type_safe(timestamp) -> None:
    with pytest.raises(ValueError, match="timestamp"):
        validate_utc_timestamp(timestamp)

    assert validate_utc_timestamp(None) is None


def test_fingerprint_ignores_cache_files(tmp_path: Path) -> None:
    _make_source_tree(tmp_path)
    baseline = compute_fingerprint(tmp_path)
    (tmp_path / "src" / "__pycache__").mkdir()
    (tmp_path / "src" / "__pycache__" / "junk.pyc").write_bytes(b"cache")

    assert compute_fingerprint(tmp_path) == baseline
