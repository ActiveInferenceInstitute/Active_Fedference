"""Tests for content-bound upstream/downstream pipeline receipts."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from analysis.artifacts import ANALYSIS_REPORT_FILENAMES
from publication.pipeline_freshness import (
    ANALYSIS_EXECUTION_PATH,
    ANALYSIS_EXECUTION_SCHEMA_VERSION,
    PIPELINE_STAGES,
    capture_analysis_input_snapshot,
    pipeline_stage_record,
    record_pipeline_stage,
    record_publication_analysis_stage,
    require_fresh_pipeline_stages,
    require_fresh_publication_analysis,
    validate_pipeline_freshness,
    validate_publication_pipeline_freshness,
)

_REPORT_NAMES = ANALYSIS_REPORT_FILENAMES


def _write(root: Path, relative: str, text: str = "content\n") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_analysis_execution(root: Path, *, configured: str, effective: str) -> None:
    _write(
        root,
        ANALYSIS_EXECUTION_PATH.as_posix(),
        json.dumps(
            {
                "schema_version": ANALYSIS_EXECUTION_SCHEMA_VERSION,
                "configured_profile": configured,
                "effective_profile": effective,
                "producer": "analysis.workflow.run_analysis_pipeline",
                **(
                    {"analysis_input_snapshot": capture_analysis_input_snapshot(root)}
                    if configured == effective == "publication"
                    else {}
                ),
            }
        ),
    )


def _make_stage_tree(root: Path) -> None:
    for relative in (
        "src/model.py",
        "scripts/02_run_analysis.py",
        "scripts/z_generate_manuscript_variables.py",
        "src/manuscript_vars/render.py",
        "src/experiment_config.py",
        "manuscript/config.yaml",
        "manuscript/01_intro.md",
        "manuscript/references.bib",
        "manuscript/preamble.tex",
        "experiment_plan.yaml",
        "pyproject.toml",
        "uv.lock",
    ):
        _write(root, relative)
    for name in _REPORT_NAMES:
        _write(root, f"output/reports/{name}")
    _write(root, "output/figures/example.png")
    _write(root, "output/data/stage_timings.json", '{"total": 1.0}\n')
    for relative in (
        "output/data/manuscript_variables.json",
        "output/data/test_coverage_receipt.json",
        "output/manuscript/01_intro.md",
        "output/manuscript/references.bib",
        "output/manuscript/config.yaml",
        "output/manuscript/preamble.tex",
        "output/pdf/paper.pdf",
        "output/pdf/paper.log",
        "output/slides/deck.pdf",
        "output/slides/deck.log",
        "output/slides/deck.tex",
        "output/web/index.html",
        "output/web/figures/example.png",
    ):
        _write(root, relative)


def _make_renderer(root: Path) -> Path:
    renderer = root / "template-renderer"
    renderer.mkdir()
    subprocess.run(("git", "-C", str(renderer), "init", "-q"), check=True)
    subprocess.run(
        ("git", "-C", str(renderer), "config", "user.name", "Renderer Test"),
        check=True,
    )
    subprocess.run(
        ("git", "-C", str(renderer), "config", "user.email", "renderer@example.invalid"),
        check=True,
    )
    (renderer / "renderer.py").write_text("RENDERER = 1\n", encoding="utf-8")
    subprocess.run(("git", "-C", str(renderer), "add", "."), check=True)
    subprocess.run(
        ("git", "-C", str(renderer), "commit", "-q", "-m", "test: renderer"),
        check=True,
    )
    subprocess.run(
        (
            "git",
            "-C",
            str(renderer),
            "remote",
            "add",
            "origin",
            "https://github.com/docxology/template.git",
        ),
        check=True,
    )
    commit = subprocess.run(
        ("git", "-C", str(renderer), "rev-parse", "HEAD^{commit}"),
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    _write(
        root,
        "manuscript/config.yaml",
        "rendering:\n"
        "  template_renderer:\n"
        "    schema_version: template-renderer-lock-v1\n"
        "    repository: docxology/template\n"
        f"    git_commit: {commit}\n"
        "    git_tree_state: clean\n",
    )
    return renderer


def test_stage_specs_have_dependency_order() -> None:
    assert [stage.name for stage in PIPELINE_STAGES] == ["analysis", "hydration", "render"]
    assert PIPELINE_STAGES[1].dependencies == ("analysis",)
    assert "output/data/test_coverage_receipt.json" in PIPELINE_STAGES[1].input_patterns
    assert PIPELINE_STAGES[2].dependencies == ("analysis", "hydration")
    assert PIPELINE_STAGES[2].output_excluded_suffixes == (
        ".aux",
        ".bbl",
        ".blg",
        ".lof",
        ".log",
        ".lot",
        ".nav",
        ".out",
        ".snm",
        ".toc",
        ".vrb",
    )
    for report_name in (
        "application_integrity_flow.json",
        "evidence_replication_map.json",
        "sensitivity.json",
        "source_render_provenance.json",
    ):
        assert f"output/reports/{report_name}" in PIPELINE_STAGES[0].output_patterns
        assert f"output/reports/{report_name}" in PIPELINE_STAGES[1].input_patterns


def test_receipt_chain_records_and_validates(tmp_path: Path) -> None:
    _make_stage_tree(tmp_path)
    renderer = _make_renderer(tmp_path)
    record_pipeline_stage(tmp_path, "analysis", timestamp="2026-07-27T00:00:00Z")
    record_pipeline_stage(tmp_path, "hydration", timestamp="2026-07-27T00:01:00Z")
    record_pipeline_stage(
        tmp_path,
        "render",
        template_root=renderer,
        timestamp="2026-07-27T00:02:00Z",
    )
    assert validate_pipeline_freshness(tmp_path) == []


def test_render_receipt_excludes_machine_local_renderer_logs(tmp_path: Path) -> None:
    _make_stage_tree(tmp_path)
    renderer = _make_renderer(tmp_path)
    record_pipeline_stage(tmp_path, "analysis")
    record_pipeline_stage(tmp_path, "hydration")
    record = record_pipeline_stage(tmp_path, "render", template_root=renderer)

    assert record["output_excluded_suffixes"] == list(
        PIPELINE_STAGES[2].output_excluded_suffixes
    )
    assert not any(path.endswith(".log") for path in record["output_hashes"])
    (tmp_path / "output" / "slides" / "deck.log").write_text(
        "/Users/local/private renderer path\n",
        encoding="utf-8",
    )
    assert validate_pipeline_freshness(tmp_path, stages=("render",)) == []


def test_publication_analysis_receipt_requires_producer_execution_metadata(tmp_path: Path) -> None:
    """A generic hash receipt cannot promote smoke-scale analysis outputs."""
    _make_stage_tree(tmp_path)
    _write_analysis_execution(tmp_path, configured="publication", effective="publication")
    record_pipeline_stage(tmp_path, "analysis")

    with pytest.raises(ValueError, match="not bound"):
        require_fresh_publication_analysis(tmp_path)
    assert validate_publication_pipeline_freshness(tmp_path, ("analysis",)) == [
        "analysis: receipt is not bound to the producer-owned publication execution metadata"
    ]

    record = record_publication_analysis_stage(tmp_path)
    assert record["analysis_execution"]["effective_profile"] == "publication"
    assert require_fresh_publication_analysis(tmp_path)["stage"] == "analysis"
    assert validate_publication_pipeline_freshness(tmp_path, ("analysis",)) == []


def test_publication_analysis_receipt_rejects_smoke_execution_sidecar(tmp_path: Path) -> None:
    _make_stage_tree(tmp_path)
    _write_analysis_execution(tmp_path, configured="smoke", effective="smoke")
    with pytest.raises(ValueError, match="configured and effective publication"):
        record_publication_analysis_stage(tmp_path)


def test_publication_analysis_receipt_rejects_input_drift_during_producer(tmp_path: Path) -> None:
    """A receipt must attest the inputs present when analysis actually began."""
    _make_stage_tree(tmp_path)
    _write_analysis_execution(tmp_path, configured="publication", effective="publication")
    (tmp_path / "manuscript" / "config.yaml").write_text("changed after analysis began\n", encoding="utf-8")

    with pytest.raises(ValueError, match="inputs changed while the producer ran"):
        record_publication_analysis_stage(tmp_path)


def test_default_receipts_are_byte_idempotent_and_omit_wall_clock(
    tmp_path: Path,
) -> None:
    _make_stage_tree(tmp_path)
    first = record_pipeline_stage(tmp_path, "analysis")
    receipt_path = tmp_path / "output" / "data" / "pipeline_provenance.json"
    first_bytes = receipt_path.read_bytes()
    second = record_pipeline_stage(tmp_path, "analysis")
    assert first == second
    assert receipt_path.read_bytes() == first_bytes
    assert first["recorded_at"] is None
    assert first["timestamp_policy"] == "omitted"
    assert validate_pipeline_freshness(tmp_path, stages=("analysis",)) == []


@pytest.mark.parametrize(
    "timestamp",
    ["", "2026-07-27", "2026-07-27T00:00:00+00:00", "not-a-date"],
)
def test_receipt_rejects_noncanonical_timestamp(tmp_path: Path, timestamp: str) -> None:
    _make_stage_tree(tmp_path)
    with pytest.raises(ValueError, match="timestamp"):
        record_pipeline_stage(tmp_path, "analysis", timestamp=timestamp)


def test_receipt_timestamp_metadata_fails_closed(tmp_path: Path) -> None:
    _make_stage_tree(tmp_path)
    record_pipeline_stage(tmp_path, "analysis")
    receipt_path = tmp_path / "output" / "data" / "pipeline_provenance.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["stages"]["analysis"]["recorded_at"] = "2026-07-27"
    receipt["stages"]["analysis"]["timestamp_policy"] = "recorded"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    findings = validate_pipeline_freshness(tmp_path, stages=("analysis",))
    assert "analysis: recorded_at is not canonical UTC" in findings


def test_analysis_regeneration_replaces_legacy_receipt_schema(tmp_path: Path) -> None:
    _make_stage_tree(tmp_path)
    receipt_path = tmp_path / "output" / "data" / "pipeline_provenance.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps({"schema_version": 1, "stages": {"analysis": {}}}),
        encoding="utf-8",
    )
    record_pipeline_stage(tmp_path, "analysis")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == 4
    assert list(receipt["stages"]) == ["analysis"]
    assert validate_pipeline_freshness(tmp_path, stages=("analysis",)) == []


def test_changed_source_fails_the_analysis_receipt(tmp_path: Path) -> None:
    _make_stage_tree(tmp_path)
    record_pipeline_stage(tmp_path, "analysis")
    (tmp_path / "src" / "model.py").write_text("changed\n", encoding="utf-8")
    findings = validate_pipeline_freshness(tmp_path, stages=("analysis",))
    assert any("analysis inputs" in finding and "src/model.py" in finding for finding in findings)


def test_changed_upstream_report_blocks_downstream_freshness(tmp_path: Path) -> None:
    _make_stage_tree(tmp_path)
    renderer = _make_renderer(tmp_path)
    record_pipeline_stage(tmp_path, "analysis")
    record_pipeline_stage(tmp_path, "hydration")
    record_pipeline_stage(tmp_path, "render", template_root=renderer)
    (tmp_path / "output" / "reports" / "belief_sharing.json").write_text("changed\n", encoding="utf-8")
    findings = validate_pipeline_freshness(tmp_path, stages=("render",))
    assert any("analysis outputs" in finding for finding in findings)
    assert any("hydration inputs" in finding for finding in findings)


def test_render_requires_explicit_locked_template_checkout(tmp_path: Path) -> None:
    _make_stage_tree(tmp_path)
    record_pipeline_stage(tmp_path, "analysis")
    record_pipeline_stage(tmp_path, "hydration")

    with pytest.raises(ValueError, match="requires an explicit template_root"):
        record_pipeline_stage(tmp_path, "render")


def test_non_render_stage_rejects_template_checkout(tmp_path: Path) -> None:
    _make_stage_tree(tmp_path)
    renderer = _make_renderer(tmp_path)
    with pytest.raises(ValueError, match="valid only for the render stage"):
        record_pipeline_stage(tmp_path, "analysis", template_root=renderer)


def test_render_receipt_is_structured_path_free_and_lock_bound(tmp_path: Path) -> None:
    _make_stage_tree(tmp_path)
    renderer = _make_renderer(tmp_path)
    record_pipeline_stage(tmp_path, "analysis")
    record_pipeline_stage(tmp_path, "hydration")
    record = record_pipeline_stage(tmp_path, "render", template_root=renderer)

    assert record["renderer"] == {
        "schema_version": "template-renderer-lock-v1",
        "repository": "docxology/template",
        "git_commit": subprocess.run(
            ("git", "-C", str(renderer), "rev-parse", "HEAD^{commit}"),
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.strip(),
        "git_tree_state": "clean",
    }
    receipt_bytes = (tmp_path / "output/data/pipeline_provenance.json").read_text(
        encoding="utf-8"
    )
    assert str(renderer) not in receipt_bytes
    assert "github.com" not in receipt_bytes

    receipt_path = tmp_path / "output/data/pipeline_provenance.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["stages"]["render"]["renderer"]["git_commit"] = "0" * 40
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    assert "render: render receipt renderer does not match the committed renderer lock" in (
        validate_pipeline_freshness(tmp_path, stages=("render",))
    )


def test_legacy_renderer_label_fails_closed(tmp_path: Path) -> None:
    _make_stage_tree(tmp_path)
    renderer = _make_renderer(tmp_path)
    record_pipeline_stage(tmp_path, "analysis")
    record_pipeline_stage(tmp_path, "hydration")
    record_pipeline_stage(tmp_path, "render", template_root=renderer)
    receipt_path = tmp_path / "output/data/pipeline_provenance.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["stages"]["render"]["renderer"] = "legacy free-form renderer"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    assert "render: render receipt renderer must be a mapping" in validate_pipeline_freshness(
        tmp_path, stages=("render",)
    )


def test_recording_downstream_stage_requires_fresh_dependencies(tmp_path: Path) -> None:
    _make_stage_tree(tmp_path)
    record_pipeline_stage(tmp_path, "analysis")
    (tmp_path / "output" / "reports" / "belief_sharing.json").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="upstream stages are fresh"):
        record_pipeline_stage(tmp_path, "hydration")


def test_missing_receipt_is_fail_closed(tmp_path: Path) -> None:
    _make_stage_tree(tmp_path)
    findings = validate_pipeline_freshness(tmp_path, stages=("analysis",))
    assert findings == ["analysis: missing pipeline provenance receipt"]


def test_public_preflight_and_stage_reader_fail_closed(tmp_path: Path) -> None:
    _make_stage_tree(tmp_path)
    with pytest.raises(ValueError, match="pipeline freshness preflight failed"):
        require_fresh_pipeline_stages(tmp_path, ("analysis",))
    with pytest.raises(ValueError, match="analysis: missing pipeline provenance receipt"):
        pipeline_stage_record(tmp_path, "analysis")

    recorded = record_pipeline_stage(tmp_path, "analysis")
    require_fresh_pipeline_stages(tmp_path, ("analysis",))
    assert pipeline_stage_record(tmp_path, "analysis") == recorded


@pytest.mark.parametrize(
    ("payload", "expected"),
    (
        ("{", "invalid pipeline provenance receipt"),
        ("[]", "invalid pipeline provenance shape"),
        (
            json.dumps({"schema_version": 1, "stages": {}}),
            "unsupported pipeline provenance schema",
        ),
    ),
)
def test_pipeline_reader_rejects_malformed_or_legacy_receipts(
    tmp_path: Path,
    payload: str,
    expected: str,
) -> None:
    """Read-only validation must not normalize a corrupt or legacy receipt."""
    _make_stage_tree(tmp_path)
    receipt_path = tmp_path / "output" / "data" / "pipeline_provenance.json"
    receipt_path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match=expected):
        validate_pipeline_freshness(tmp_path, stages=("analysis",))


def test_pipeline_stage_reader_rejects_an_unknown_stage(tmp_path: Path) -> None:
    """Public receipt access is deliberately limited to the declared stage graph."""
    _make_stage_tree(tmp_path)

    with pytest.raises(ValueError, match="unknown pipeline stage 'unknown'"):
        pipeline_stage_record(tmp_path, "unknown")


@pytest.mark.parametrize(
    ("label", "mutate", "expected"),
    (
        (
            "input contract drift",
            lambda record: record.__setitem__("input_patterns", []),
            "analysis: input pattern contract drift",
        ),
        (
            "output contract drift",
            lambda record: record.__setitem__("output_patterns", []),
            "analysis: output pattern contract drift",
        ),
        (
            "stage identity drift",
            lambda record: record.__setitem__("stage", "hydration"),
            "analysis: persisted stage identity drift",
        ),
        (
            "dependency contract drift",
            lambda record: record.__setitem__("dependencies", ["hydration"]),
            "analysis: dependency contract drift",
        ),
        (
            "missing timestamp",
            lambda record: record.pop("recorded_at"),
            "analysis: recorded_at field missing",
        ),
        (
            "omitted timestamp policy mismatch",
            lambda record: record.update(recorded_at=None, timestamp_policy="recorded"),
            "analysis: omitted recorded_at requires timestamp_policy=omitted",
        ),
        (
            "recorded timestamp policy mismatch",
            lambda record: record.update(
                recorded_at="2026-07-27T00:00:00Z",
                timestamp_policy="omitted",
            ),
            "analysis: populated recorded_at requires timestamp_policy=recorded",
        ),
        (
            "invalid timestamp type",
            lambda record: record.__setitem__("recorded_at", 1),
            "analysis: recorded_at must be a canonical UTC string or null",
        ),
        (
            "missing input hash map",
            lambda record: record.__setitem__("input_hashes", []),
            "analysis inputs: receipt is missing its hash map",
        ),
    ),
    ids=lambda case: str(case),
)
def test_pipeline_validation_reports_corrupt_stage_metadata(
    tmp_path: Path,
    label: str,
    mutate: Callable[[dict[str, Any]], None],
    expected: str,
) -> None:
    """Every persisted stage field is checked before a downstream stage trusts it."""
    del label
    _make_stage_tree(tmp_path)
    record_pipeline_stage(tmp_path, "analysis")
    receipt_path = tmp_path / "output" / "data" / "pipeline_provenance.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert isinstance(receipt, dict)
    stages = receipt["stages"]
    assert isinstance(stages, dict)
    record = stages["analysis"]
    assert isinstance(record, dict)
    mutate(record)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    assert expected in validate_pipeline_freshness(tmp_path, stages=("analysis",))


def test_pipeline_validation_rejects_tampered_receipt_generator(tmp_path: Path) -> None:
    _make_stage_tree(tmp_path)
    record_pipeline_stage(tmp_path, "analysis")
    receipt_path = tmp_path / "output" / "data" / "pipeline_provenance.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["generated_by"] = "untrusted.writer"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    assert validate_pipeline_freshness(tmp_path, stages=("analysis",)) == [
        "pipeline receipt: generated_by identity drift"
    ]


def test_pipeline_validation_reports_an_empty_declared_output_boundary(tmp_path: Path) -> None:
    """A receipt cannot be refreshed when an entire declared output class vanishes."""
    _make_stage_tree(tmp_path)
    record_pipeline_stage(tmp_path, "analysis")
    for path in (tmp_path / "output" / "reports").glob("*.json"):
        path.unlink()
    (tmp_path / "output" / "figures" / "example.png").unlink()
    (tmp_path / "output" / "data" / "stage_timings.json").unlink()

    findings = validate_pipeline_freshness(tmp_path, stages=("analysis",))
    assert findings == ["analysis: pipeline analysis outputs boundary resolved no files"]


def test_recording_requires_a_nonempty_declared_boundary_and_preflight_requires_stages(
    tmp_path: Path,
) -> None:
    """Producers and preflights reject vacuous provenance boundaries."""
    _make_stage_tree(tmp_path)
    for path in (tmp_path / "output" / "reports").glob("*.json"):
        path.unlink()
    (tmp_path / "output" / "figures" / "example.png").unlink()
    (tmp_path / "output" / "data" / "stage_timings.json").unlink()

    with pytest.raises(ValueError, match="pipeline analysis outputs boundary resolved no files"):
        record_pipeline_stage(tmp_path, "analysis")
    with pytest.raises(ValueError, match="at least one pipeline stage is required"):
        require_fresh_pipeline_stages(tmp_path, ())
