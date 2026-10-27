"""Canonical boundary outputs, retained wire history, and retired Python aliases."""

from __future__ import annotations

import importlib
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from fedference.aggregation import AggregationConfig
from fedference.evidence import RunReceipt, canonical_sha256, make_artifact_record, verify_run_receipt
from fedference.federation.process import run_multiprocess_round, run_multiprocess_round_result
from fedference.federation.socket_transport import (
    inspect_socket_replay,
    run_socket_round,
    run_socket_round_result,
    validate_socket_replay,
)
from fedference.federation.transport import (
    deserialize_envelope,
    deserialize_result,
    serialize_envelope,
    serialize_result,
)
from figures import generate_belief_heatmap, generate_robust_influence_weights
from manuscript_vars.loaders import _federation_demo_local_posteriors
from manuscript_vars.tokens import _format_d_equivalent


def _historical_receipt(root: Path) -> RunReceipt:
    config = root / "config.json"
    config.write_text('{"seed":0}\n')
    return RunReceipt(
        run_id="historical-control", experiment_id="boundary-control", experiment_version="1",
        profile="smoke", git_commit="unavailable", git_tree_state="unavailable",
        environment_lock_sha256="a" * 64, config_sha256=canonical_sha256({"seed": 0}),
        dataset_sha256={}, seeds=(0,), device="cpu", backend="numpy", fallbacks=(), checkpoints=(),
        outputs=(make_artifact_record("config", config, root=root),), status="completed",
        started_at_utc="2026-07-29T00:00:00+00:00", completed_at_utc="2026-07-29T00:00:01+00:00",
        schema_version="1.1",
    )


def test_canonical_transport_figures_and_historical_corpus(tmp_path: Path) -> None:
    # Given finite categorical inputs with the independent product [0.8, 0.2].
    rows = np.array([[0.8, 0.2], [0.5, 0.5]])
    config = AggregationConfig(method="naive")
    expected = np.array([0.8, 0.2])
    # When each current boundary consumes the canonical spellings.
    process = run_multiprocess_round_result(local_posteriors=list(rows), config=config)
    projected_process = run_multiprocess_round(local_posteriors=list(rows), config=config)
    socket = run_socket_round_result(
        local_posteriors=rows, config=config, replay_path=tmp_path / "replay.json",
    )
    projected_socket = run_socket_round(local_posteriors=rows, config=config)
    wire = serialize_result(expected, normalized_effective_weights=np.array([0.5, 0.5]))
    (tmp_path / "protocol-v1-result.npz").write_bytes(wire)
    envelope = serialize_envelope(
        wire, message_type="result", round_id="retirement-control", worker_id=0,
        aggregation_config_hash=config.fingerprint,
    )
    (tmp_path / "protocol-v1-envelope.bin").write_bytes(envelope)
    header, payload = deserialize_envelope(envelope)
    decoded = deserialize_result(payload)
    heatmap = generate_belief_heatmap(local_posteriors=rows, consensus=expected, project_root=tmp_path)
    influence = generate_robust_influence_weights(
        normalized_effective_weights=[0.5, 0.5], contaminated_indices=[], project_root=tmp_path,
    )
    # Then numerics, protocol-v1 semantics, replay, and real figures survive.
    for consensus in (process.consensus, projected_process, socket.consensus, projected_socket["consensus"]):
        np.testing.assert_allclose(consensus, expected, rtol=0, atol=1e-15)
    assert process.bit_identical and socket.bit_identical
    assert inspect_socket_replay(
        list(socket.replay), local_posteriors=rows, consensus=socket.consensus,
    ).integrity_valid
    assert validate_socket_replay(list(socket.replay), local_posteriors=rows, consensus=socket.consensus)
    assert header.protocol_version == 1
    assert set(decoded) == {"consensus", "agent_weights"}
    np.testing.assert_array_equal(decoded["consensus"], expected)
    np.testing.assert_array_equal(decoded["agent_weights"], [0.5, 0.5])
    for path in (heatmap, influence):
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert path.with_suffix(".pdf").read_bytes().startswith(b"%PDF")
    assert _format_d_equivalent(0.83) == "0.83"
    assert _format_d_equivalent(1e6) == "saturated (r=+1)"
    np.testing.assert_array_equal(_federation_demo_local_posteriors()[0], [0.02] * 7 + [0.84, 0.02])
    (tmp_path / "consensus.json").write_text(json.dumps(process.consensus.tolist()))

    # Given a schema-1.1 artifact binding without invented runtime provenance.
    raw = _historical_receipt(tmp_path).as_dict()
    receipt_path = tmp_path / "historical-receipt.json"
    receipt_path.write_text(json.dumps(raw, sort_keys=True))
    # When the retained historical reader verifies that representation.
    restored = RunReceipt.from_dict(json.loads(receipt_path.read_text()))
    # Then exact fields survive, while the same binding rejects actual tampering.
    assert restored.as_dict() == raw
    assert restored.runtime_provenance.distribution_version == "unavailable"
    assert "runtime_provenance" not in raw
    assert verify_run_receipt(restored, root=tmp_path) == ()
    (tmp_path / "config.json").write_text('{"seed":1}\n')
    assert verify_run_receipt(restored, root=tmp_path)
    (tmp_path / "config.json").write_text('{"seed":0}\n')

    # Given a labeled own-data request for the real CLI.
    request = tmp_path / "request.json"
    request.write_text(json.dumps({
        "schema_version": "1.0", "state_labels": ["a", "b"],
        "agents": [
            {"agent_id": "first", "posterior": rows[0].tolist()},
            {"agent_id": "second", "posterior": rows[1].tolist()},
        ], "aggregation_config": config.as_dict(),
    }))
    # When the process adapter writes and verifies its receipt.
    cli = subprocess.run(
        [sys.executable, "-m", "fedference_cli", "aggregate", "--input", str(request),
         "--output-dir", str(tmp_path / "cli")], capture_output=True, text=True, check=False,
    )
    (tmp_path / "cli.stdout").write_text(cli.stdout)
    (tmp_path / "cli.stderr").write_text(cli.stderr)
    assert cli.returncode == 0, cli.stderr
    verification = subprocess.run(
        [sys.executable, "-m", "fedference_cli", "verify", str(tmp_path / "cli/receipt.json")],
        capture_output=True, text=True, check=False,
    )
    (tmp_path / "verification.stdout").write_text(verification.stdout)
    (tmp_path / "verification.stderr").write_text(verification.stderr)
    # Then the written numeric result is known and its intact artifacts verify.
    assert verification.returncode == 0, verification.stderr
    result = json.loads((tmp_path / "cli/result.json").read_text())
    np.testing.assert_allclose(result["aggregation"]["consensus"], expected, rtol=0, atol=1e-15)
    print(json.dumps({"consensus": process.consensus.tolist(), "wire_version": header.protocol_version,
                      "historical_schema": restored.schema_version, "cli_exit": cli.returncode,
                      "verification_exit": verification.returncode}, sort_keys=True))


@pytest.mark.parametrize(("module", "name", "retired"), [
    ("fedference.federation.process", "run_multiprocess_round", "beliefs"),
    ("fedference.federation.process", "run_multiprocess_round_result", "beliefs"),
    ("fedference.federation.socket_transport", "run_socket_round", "beliefs"),
    ("fedference.federation.socket_transport", "run_socket_round_result", "beliefs"),
    ("fedference.federation.socket_transport", "inspect_socket_replay", "beliefs"),
    ("fedference.federation.socket_transport", "validate_socket_replay", "beliefs"),
    ("fedference.federation.transport", "serialize_result", "agent_weights"),
    ("figures.belief_heatmap", "generate_belief_heatmap", "agent_beliefs"),
    ("figures.robust_influence_weights", "generate_robust_influence_weights", "agent_weights"),
    ("manuscript_variables", "_format_cohens_d", "import"),
    ("manuscript_vars.tokens", "_format_cohens_d", "import"),
    ("manuscript_vars.loaders", "_federation_demo_beliefs", "import"),
    ("wire", "protocol_version", "version"),
    ("receipt", "schema_version", "version"),
])
def test_retired_runtime_alias_and_unknown_schema(
    tmp_path: Path, module: str, name: str, retired: str,
) -> None:
    # Given otherwise valid data that the old runtime alias accepted.
    rows = np.array([[0.8, 0.2], [0.5, 0.5]])
    expected = np.array([0.8, 0.2])
    config = AggregationConfig(method="naive")
    if retired == "import":
        # When importing a removed helper, then Python rejects the exact name.
        with pytest.raises(ImportError) as import_failure:
            exec(f"from {module} import {name}", {})
        print(str(import_failure.value))
        return
    if module == "receipt":
        raw = _historical_receipt(tmp_path).as_dict()
        raw["schema_version"] = "999"
        # When an unknown receipt version crosses the reader, then it fails closed.
        with pytest.raises(ValueError, match="schema_version") as receipt_failure:
            RunReceipt.from_dict(raw)
        print(str(receipt_failure.value))
        return
    if module == "wire":
        wire = serialize_envelope(
            b"payload", message_type="belief", round_id="control", worker_id=0,
            aggregation_config_hash=config.fingerprint,
        )
        size = struct.unpack(">I", wire[:4])[0]
        header = json.loads(wire[4:4 + size])
        header["protocol_version"] = 999
        changed = json.dumps(header).encode()
        malformed = struct.pack(">I", len(changed)) + changed + wire[4 + size:]
        # When an unknown wire version crosses the reader, then it fails closed.
        with pytest.raises(ValueError, match="unsupported protocol_version") as wire_failure:
            deserialize_envelope(malformed)
        print(str(wire_failure.value))
        return
    function = getattr(importlib.import_module(module), name)
    arguments = {}
    if name in {"inspect_socket_replay", "validate_socket_replay"}:
        socket = run_socket_round_result(local_posteriors=rows, config=config)
        arguments = {"replay": list(socket.replay), "consensus": socket.consensus, retired: rows}
    elif name == "serialize_result":
        arguments = {"consensus": expected, retired: np.array([0.5, 0.5])}
    elif name == "generate_belief_heatmap":
        arguments = {retired: rows, "consensus": expected, "project_root": tmp_path}
    elif name == "generate_robust_influence_weights":
        arguments = {retired: [0.5, 0.5], "contaminated_indices": [], "project_root": tmp_path}
    else:
        arguments = {retired: rows, "config": config}
    # When the retired spelling crosses the boundary, then no work is accepted.
    with pytest.raises(TypeError, match=retired) as failure:
        function(**arguments)
    print(f"{module}.{name}: {failure.value}")
    np.testing.assert_array_equal(rows, [[0.8, 0.2], [0.5, 0.5]])
    assert not list(tmp_path.iterdir())
