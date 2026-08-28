"""Fail-closed edge coverage for labeled application and provenance records."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from importlib import metadata
from pathlib import Path

import numpy as np
import pytest

import fedference.provenance as provenance_module
from fedference import (
    AgentPosterior,
    AggregationConfig,
    AggregationResult,
    LabeledAggregationRequest,
    LabeledAggregationResult,
    aggregate_labeled,
)
from fedference.application import load_labeled_aggregation_request
from fedference.provenance import (
    RuntimeProvenance,
    SourceProvenance,
    active_fedference_checkout_version,
    collect_source_provenance,
    runtime_provenance,
)

HERMETIC_GIT = ("git", "-c", "core.fsmonitor=false", "-c", "core.untrackedcache=false")
SHA256 = "A" * 64
GIT_COMMIT = "B" * 40


def _request_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "state_labels": ["clear", "critical"],
        "agents": [
            {"agent_id": "agent-0", "posterior": [3.0, 1.0]},
            {"agent_id": "agent-1", "posterior": [1.0, 3.0]},
        ],
        "aggregation_config": AggregationConfig(method="naive").as_dict(),
    }


def _valid_request() -> LabeledAggregationRequest:
    return LabeledAggregationRequest.from_dict(_request_payload())


def _valid_result() -> LabeledAggregationResult:
    return aggregate_labeled(_valid_request())


@pytest.mark.parametrize(
    ("agent_id", "posterior", "base_weight", "message"),
    [
        ("", [1.0, 1.0], 1.0, "non-empty string"),
        (7, [1.0, 1.0], 1.0, "non-empty string"),
        ("agent", 7, 1.0, "numeric sequence"),
        ("agent", "1, 2", 1.0, "numeric sequence"),
        ("agent", [], 1.0, "non-empty"),
        ("agent", [1.0, "two"], 1.0, "only numeric"),
        ("agent", [1.0, np.inf], 1.0, "only finite"),
        ("agent", [1.0, 1.0], True, "finite non-negative"),
        ("agent", [1.0, 1.0], "1", "finite non-negative"),
        ("agent", [1.0, 1.0], np.inf, "finite non-negative"),
        ("agent", [1.0, 1.0], -0.1, "finite non-negative"),
    ],
)
def test_agent_posterior_rejects_ambiguous_or_nonfinite_values(
    agent_id: object,
    posterior: object,
    base_weight: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        AgentPosterior(agent_id, posterior, base_weight)  # type: ignore[arg-type]


def test_agent_and_json_envelopes_reject_non_objects() -> None:
    with pytest.raises(ValueError, match="agent must be an object"):
        AgentPosterior.from_dict([("agent_id", "agent")])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="request must be a JSON object"):
        LabeledAggregationRequest.from_json("[]")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"schema_version": "2.0"}, "schema_version"),
        ({"state_labels": "clear"}, "state_labels must be a sequence"),
        ({"state_labels": ("only",)}, "at least two"),
        ({"agents": "agent-0"}, "agents must be a sequence"),
        ({"agents": ()}, "must contain AgentPosterior"),
        ({"agents": ("agent-0",)}, "must contain AgentPosterior"),
        ({"config": {}}, "config must be an AggregationConfig"),
    ],
)
def test_request_constructor_rejects_invalid_typed_envelope(
    kwargs: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "state_labels": ("clear", "critical"),
        "agents": (AgentPosterior("agent-0", [1.0, 1.0]),),
        "config": AggregationConfig(method="naive"),
        "schema_version": "1.0",
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        LabeledAggregationRequest(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("state_labels", ("clear", "critical"), "state_labels must be a list"),
        ("agents", (), "agents must be a list"),
        (
            "aggregation_config",
            {
                "method": "unsupported",
                "robustness": 1.0,
                "entropy_weight": 1.0,
                "max_iter": 64,
                "tol": 1e-9,
                "multistart": True,
            },
            "invalid aggregation_config",
        ),
    ],
)
def test_request_mapping_rejects_non_json_sequences_and_invalid_config(
    field: str,
    replacement: object,
    message: str,
) -> None:
    raw = _request_payload()
    raw[field] = replacement
    with pytest.raises(ValueError, match=message):
        LabeledAggregationRequest.from_dict(raw)


def test_request_mapping_requires_exact_fields() -> None:
    with pytest.raises(ValueError, match="fields do not match schema"):
        LabeledAggregationRequest.from_dict([])  # type: ignore[arg-type]


def test_request_loader_preserves_context_for_filesystem_and_schema_failures(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="could not read labeled aggregation request"):
        load_labeled_aggregation_request(missing)

    invalid = tmp_path / "invalid.json"
    invalid.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid labeled aggregation request"):
        load_labeled_aggregation_request(invalid)

    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps(_request_payload()), encoding="utf-8")
    assert load_labeled_aggregation_request(valid).as_dict() == _valid_request().as_dict()


def test_aggregate_labeled_rejects_an_unvalidated_request() -> None:
    with pytest.raises(ValueError, match="request must be a LabeledAggregationRequest"):
        aggregate_labeled(_request_payload())  # type: ignore[arg-type]


def _aggregation_with_dimensions(*, states: int, agents: int) -> AggregationResult:
    return AggregationResult(
        consensus=np.full(states, 1.0 / states),
        raw_effective_weights=np.ones(agents),
        normalized_effective_weights=np.full(agents, 1.0 / agents),
        iterations=0,
        converged=True,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"schema_version": "2.0"}, "schema_version"),
        ({"state_labels": ("clear", "clear")}, "must be unique"),
        ({"agent_ids": ("agent-0", "agent-0")}, "must be unique"),
        ({"normalized_local_posteriors": [[0.5, 0.5]]}, "shape must match"),
        (
            {"normalized_local_posteriors": [[np.nan, np.nan], [0.5, 0.5]]},
            "finite probability rows",
        ),
        (
            {"normalized_local_posteriors": [[-0.1, 1.1], [0.5, 0.5]]},
            "finite probability rows",
        ),
        (
            {"normalized_local_posteriors": [[0.4, 0.4], [0.5, 0.5]]},
            "finite probability rows",
        ),
        ({"config": {}}, "config must be an AggregationConfig"),
        ({"aggregation": {}}, "aggregation must be an AggregationResult"),
        (
            {"aggregation": _aggregation_with_dimensions(states=3, agents=2)},
            "consensus must match state_labels",
        ),
        ({"request_sha256": 7}, "must be a SHA-256 digest"),
        ({"request_sha256": "a" * 63}, "must be a SHA-256 digest"),
        ({"request_sha256": "z" * 64}, "must be a SHA-256 digest"),
        ({"software_version": ""}, "non-empty string"),
    ],
)
def test_result_constructor_rejects_unbound_or_nonfinite_content(
    kwargs: dict[str, object],
    message: str,
) -> None:
    result = _valid_result()
    with pytest.raises(ValueError, match=message):
        replace(result, **kwargs)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda raw: raw.update(operation="other"), "operation must be"),
        (lambda raw: raw.update(state_labels=("clear", "critical")), "state_labels must be a list"),
        (lambda raw: raw.update(agent_ids=("agent-0", "agent-1")), "agent_ids must be a list"),
        (
            lambda raw: raw.update(normalized_local_posteriors=((0.5, 0.5), (0.5, 0.5))),
            "normalized_local_posteriors must be a list",
        ),
        (lambda raw: raw.update(aggregation=[]), "aggregation must be an object"),
    ],
)
def test_result_mapping_rejects_noncanonical_json_shapes(mutate, message: str) -> None:
    raw = _valid_result().as_dict()
    mutate(raw)
    with pytest.raises(ValueError, match=message):
        LabeledAggregationResult.from_dict(raw)


def test_result_mapping_requires_exact_fields_and_normalizes_digest_case() -> None:
    raw = _valid_result().as_dict()
    raw["request_sha256"] = raw["request_sha256"].upper()
    assert LabeledAggregationResult.from_dict(raw).request_sha256 == raw["request_sha256"].lower()
    with pytest.raises(ValueError, match="fields do not match schema"):
        LabeledAggregationResult.from_dict({})


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"distribution_name": ""}, "distribution_name"),
        ({"distribution_version": " "}, "distribution_version"),
        ({"python_version": 313}, "python_version"),
        ({"dependency_versions": []}, "must be a mapping"),
        ({"dependency_versions": {"": "1"}}, "dependency name"),
        ({"dependency_versions": {"numpy": ""}}, "dependency version"),
        ({"installed_archive_sha256": "bad"}, "SHA-256"),
        ({"warnings": "warning"}, "warnings must be a sequence"),
        ({"warnings": ("",)}, "warnings must be a sequence"),
    ],
)
def test_runtime_provenance_rejects_malformed_records(
    kwargs: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "distribution_name": "active_fedference",
        "distribution_version": "1.1.0.dev0",
        "python_version": "3.13.0",
        "dependency_versions": {"scipy": "1", "numpy": "2"},
        "installed_archive_sha256": SHA256,
        "warnings": (),
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        RuntimeProvenance(**values)  # type: ignore[arg-type]


def test_runtime_provenance_schema_and_unavailable_state() -> None:
    unavailable = RuntimeProvenance.unavailable(warning="legacy receipt")
    assert unavailable.distribution_version == "unavailable"
    assert unavailable.warnings == ("legacy receipt",)
    with pytest.raises(ValueError, match="fields do not match schema"):
        RuntimeProvenance.from_dict({})


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"source_kind": "other"}, "source_kind"),
        ({"git_commit": "bad"}, "git_commit"),
        ({"git_tree_state": "unknown"}, "git_tree_state"),
        ({"uv_lock_sha256": "bad"}, "uv_lock_sha256"),
        ({"installed_archive_sha256": "bad"}, "installed_archive_sha256"),
        ({"source_kind": "git_checkout"}, "requires commit and tree state"),
        ({"source_kind": "unavailable", "git_commit": GIT_COMMIT}, "cannot declare Git fields"),
        ({"source_kind": "unavailable", "git_tree_state": "clean"}, "cannot declare Git fields"),
        ({"source_kind": "unavailable", "uv_lock_sha256": SHA256}, "cannot declare Git fields"),
        ({"source_kind": "installed_archive"}, "requires an archive digest"),
        (
            {"source_kind": "unavailable", "installed_archive_sha256": SHA256},
            "cannot declare an archive digest",
        ),
        ({"warnings": ("",)}, "warnings must be a sequence"),
    ],
)
def test_source_provenance_rejects_contradictory_records(
    kwargs: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "source_kind": "unavailable",
        "git_commit": None,
        "git_tree_state": None,
        "uv_lock_sha256": None,
        "installed_archive_sha256": None,
        "warnings": (),
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        SourceProvenance(**values)  # type: ignore[arg-type]


def test_source_provenance_canonicalizes_digests_and_requires_exact_schema() -> None:
    source = SourceProvenance(
        source_kind="git_checkout",
        git_commit=GIT_COMMIT,
        git_tree_state="clean",
        uv_lock_sha256=SHA256,
    )
    assert source.git_commit == GIT_COMMIT.lower()
    assert source.uv_lock_sha256 == SHA256.lower()
    assert SourceProvenance.from_dict(source.as_dict()) == source
    with pytest.raises(ValueError, match="fields do not match schema"):
        SourceProvenance.from_dict({})


def _write_checkout_identity(root: Path, pyproject_text: str) -> None:
    package = root / "src" / "fedference"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("\n", encoding="utf-8")
    (package / "aggregation.py").write_text("\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(pyproject_text, encoding="utf-8")


@pytest.mark.parametrize(
    "pyproject_text",
    [
        "[build-system]\nrequires = []\n",
        '[project]\nname = "active_fedference"\n',
        '[project]\nversion = "1.1.0.dev0"\n',
        '[project]\nname = "other"\nversion = "1.1.0.dev0"\n',
        '[project]\nname = "active_fedference"\nversion = " 1.1.0.dev0"\n',
    ],
)
def test_checkout_identity_fails_closed_on_incomplete_metadata(
    tmp_path: Path,
    pyproject_text: str,
) -> None:
    _write_checkout_identity(tmp_path, pyproject_text)
    assert active_fedference_checkout_version(tmp_path) is None


def test_checkout_identity_requires_sources_and_accepts_canonicalized_name(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "active_fedference"\nversion = "1.1.0.dev0"\n',
        encoding="utf-8",
    )
    assert active_fedference_checkout_version(tmp_path) is None
    _write_checkout_identity(
        tmp_path,
        '[project]\nname = "Active.Fedference" # normalized\nversion = "1.1.0.dev0" # current\n',
    )
    assert active_fedference_checkout_version(tmp_path) == "1.1.0.dev0"


def _git(root: Path, *arguments: str) -> None:
    subprocess.run((*HERMETIC_GIT, *arguments), cwd=root, check=True)


def test_git_provenance_handles_unborn_checkout_and_missing_lock(tmp_path: Path) -> None:
    unborn = tmp_path / "unborn"
    unborn.mkdir()
    _write_checkout_identity(
        unborn,
        '[project]\nname = "active_fedference"\nversion = "1.1.0.dev0"\n',
    )
    _git(unborn, "init", "-q")
    unresolved = collect_source_provenance(unborn)
    assert unresolved.source_kind in {"installed_archive", "unavailable"}
    assert unresolved.git_commit is None

    committed = tmp_path / "committed"
    committed.mkdir()
    _write_checkout_identity(
        committed,
        '[project]\nname = "active_fedference"\nversion = "1.1.0.dev0"\n',
    )
    _git(committed, "init", "-q")
    _git(committed, "config", "user.name", "Provenance Edge Test")
    _git(committed, "config", "user.email", "provenance-edge@example.invalid")
    _git(committed, "add", ".")
    _git(committed, "commit", "-qm", "fixture")
    nested = committed / "src" / "fedference"
    source = collect_source_provenance(nested)
    assert source.source_kind == "git_checkout"
    assert source.git_tree_state == "clean"
    assert source.uv_lock_sha256 is None
    assert source.warnings == ("uv.lock is unavailable in the selected Git checkout",)


def _write_distribution_metadata(
    root: Path,
    *,
    direct_url: str | None = None,
    requirements: tuple[str, ...] = (),
) -> metadata.Distribution:
    dist_info = root / "active_fedference-9.9.dist-info"
    dist_info.mkdir(parents=True)
    metadata_text = "Metadata-Version: 2.4\nName: active_fedference\nVersion: 9.9\n"
    metadata_text += "".join(f"Requires-Dist: {requirement}\n" for requirement in requirements)
    (dist_info / "METADATA").write_text(metadata_text, encoding="utf-8")
    if direct_url is not None:
        (dist_info / "direct_url.json").write_text(direct_url, encoding="utf-8")
    return metadata.Distribution.at(dist_info)


@contextmanager
def _prepend_import_path(path: Path) -> Iterator[None]:
    original = sys.path[:]
    sys.path.insert(0, str(path))
    try:
        yield
    finally:
        sys.path[:] = original


@pytest.mark.parametrize(
    ("direct_url", "expected_digest", "warning_fragment"),
    [
        (None, None, None),
        ("not-json", None, "malformed"),
        ("[]", None, "malformed"),
        (json.dumps({"archive_info": []}), None, None),
        (json.dumps({"archive_info": {"hashes": {}}}), None, None),
        (
            json.dumps({"archive_info": {"hash": f"sha256={SHA256}"}}),
            SHA256.lower(),
            None,
        ),
        (
            json.dumps({"archive_info": {"hashes": {"sha256": "bad"}}}),
            None,
            "invalid",
        ),
    ],
)
def test_pep610_archive_hash_parsing_is_fail_closed(
    tmp_path: Path,
    direct_url: str | None,
    expected_digest: str | None,
    warning_fragment: str | None,
) -> None:
    distribution = _write_distribution_metadata(tmp_path, direct_url=direct_url)
    digest, warnings = provenance_module._archive_sha256(distribution)
    assert digest == expected_digest
    if warning_fragment is None:
        assert warnings == ()
    else:
        assert warning_fragment in warnings[0]


def test_runtime_provenance_uses_real_distribution_metadata_and_archive_hash(tmp_path: Path) -> None:
    _write_distribution_metadata(
        tmp_path,
        direct_url=json.dumps({"archive_info": {"hashes": {"sha256": SHA256}}}),
        requirements=("???", "edge-dependency-not-installed>=1", 'skip-me; extra == "test"'),
    )
    with _prepend_import_path(tmp_path):
        observed = runtime_provenance()
    assert observed.distribution_version == "9.9"
    assert observed.installed_archive_sha256 == SHA256.lower()
    assert observed.dependency_versions["edge-dependency-not-installed"] == "unavailable"
    assert any("could not parse" in warning for warning in observed.warnings)
    assert any("dependency version is unavailable" in warning for warning in observed.warnings)
    assert "skip-me" not in observed.dependency_versions


def test_runtime_and_source_provenance_real_metadata_fallbacks(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    original = sys.path[:]
    sys.path[:] = [str(empty)]
    try:
        missing = runtime_provenance()
    finally:
        sys.path[:] = original
    assert missing.distribution_version == "unavailable"
    assert missing.warnings == ("active_fedference distribution metadata is unavailable",)

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    _write_distribution_metadata(
        archive_root,
        direct_url=json.dumps({"archive_info": {"hashes": {"sha256": SHA256}}}),
    )
    invalid_checkout = tmp_path / "not-a-checkout"
    with _prepend_import_path(archive_root):
        source = collect_source_provenance(invalid_checkout)
    assert source.source_kind == "installed_archive"
    assert source.installed_archive_sha256 == SHA256.lower()
    assert source.warnings == (
        "the explicitly selected project root is not a readable Active Fedference Git checkout",
    )


def test_source_provenance_without_explicit_checkout_does_not_invent_git_identity() -> None:
    source = collect_source_provenance()
    assert source.source_kind in {"installed_archive", "unavailable"}
    assert source.git_commit is None
    assert source.git_tree_state is None
    assert source.uv_lock_sha256 is None


def test_file_digest_reads_complete_multi_chunk_content(tmp_path: Path) -> None:
    content = b"a" * (1024 * 1024 + 17)
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(content)
    assert provenance_module._sha256_file(artifact) == hashlib.sha256(content).hexdigest()
