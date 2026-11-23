"""Real subprocess publication, namespace, substitution and interruption oracles.

Tracing pauses actual production instructions; it never replaces a writer or
algorithm. Pipes supply the happens-before relation. Every child is reaped and
raw streams/statuses stay in its fixture directory for evidence collection.
"""

from __future__ import annotations

import dis
import hashlib
import inspect
import json
import os
import selectors
import signal
import socket
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Any

import numpy as np
import pytest

from fedference.application import aggregate_labeled, load_labeled_aggregation_request
from fedference.bnn_fedgvi import (
    DiagonalGaussian,
    FedGVIServerState,
    load_server_checkpoint,
    save_server_checkpoint,
)
from fedference.evidence import (
    ApplicationReceipt,
    RunReceipt,
    canonical_sha256,
    load_application_receipt,
    load_run_receipt,
    make_artifact_record,
    sha256_file,
    verify_application_receipt,
    verify_run_receipt,
    write_application_receipt,
    write_run_receipt,
)
from fedference.federation import (
    load_socket_replay,
    run_socket_round,
    save_socket_replay,
    validate_socket_replay,
)
from fedference.provenance import RuntimeProvenance, SourceProvenance, runtime_provenance
from fedference_cli import main
from fedference_cli._support import _write_json

ROOT = Path(__file__).resolve().parents[2]
KINDS = ("receipt", "application", "checkpoint", "replay", "json")
BELIEFS = [np.array([0.8, 0.2]), np.array([0.3, 0.7])]


@pytest.fixture
def inputs(tmp_path: Path) -> Path:
    """Real valid artifacts, including a fully closed loopback replay producer."""
    base = tmp_path / "inputs"
    base.mkdir()
    (base / "config.json").write_bytes(b'{"seed":19}\n')
    (base / "report.json").write_bytes(b'{"engineering_only":true}\n')
    request = load_labeled_aggregation_request(ROOT / "examples/data/labeled_aggregation_request.json")
    result = aggregate_labeled(request)
    for name, payload in (("request", request.as_dict()), ("result", result.as_dict())):
        (base / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
    threads = {thread.ident for thread in threading.enumerate()}
    replay = run_socket_round(BELIEFS, robustness=0.0, round_id="custody-engineering")
    save_socket_replay(base / "replay.json", replay["replay"])
    assert {thread.ident for thread in threading.enumerate()} == threads
    with socket.socket() as client:
        client.settimeout(1)
        refusal = client.connect_ex(("127.0.0.1", replay["port"]))
    assert refusal != 0
    print(
        json.dumps(
            {
                "listener_port": replay["port"],
                "closed_connect_errno": refusal,
                "threads_joined": True,
                "fixture_path": str(base),
            }
        )
    )
    return base


def _state(label: str) -> FedGVIServerState:
    return FedGVIServerState.initialize(
        DiagonalGaussian(
            mean=np.array([0.0 if label == "A" else 0.25, -0.125]),
            variance=np.array([1.0, 2.0]),
        ),
        n_clients=512 if label == "LARGE" else 2,
    )


def _receipt(base: Path, label: str) -> RunReceipt:
    return RunReceipt(
        run_id=label,
        experiment_id="custody-engineering",
        experiment_version="1",
        profile="smoke",
        git_commit="unavailable",
        git_tree_state="unavailable",
        environment_lock_sha256="a" * 64,
        config_sha256=canonical_sha256({"seed": 19}),
        dataset_sha256={},
        seeds=(19,),
        device="cpu",
        backend="numpy",
        fallbacks=(),
        checkpoints=(),
        outputs=tuple(
            make_artifact_record(name, base / f"{name}.json", root=base) for name in ("config", "report")
        ),
        status="completed",
        started_at_utc="2026-10-27T00:00:00+00:00",
        completed_at_utc="2026-10-27T00:00:01+00:00",
        runtime_provenance=RuntimeProvenance.unavailable(warning="engineering fixture"),
    )


def _application(base: Path, label: str) -> ApplicationReceipt:
    request = load_labeled_aggregation_request(base / "request.json")
    result = aggregate_labeled(request)
    return ApplicationReceipt(
        request_sha256=request.request_sha256,
        config_fingerprint=request.config.fingerprint,
        solver_status=result.aggregation.solver_status,
        fallback_events=result.aggregation.fallback_events,
        outputs=tuple(
            make_artifact_record(name, base / f"{name}.json", root=base) for name in ("request", "result")
        ),
        runtime_provenance=runtime_provenance(),
        source_provenance=SourceProvenance(source_kind="unavailable", warnings=("engineering",)),
        started_at_utc="2026-10-27T00:00:00+00:00",
        completed_at_utc=f"2026-10-27T00:00:0{1 if label == 'A' else 2}+00:00",
    )


def _produce(kind: str, target: Path, base: Path, label: str) -> None:
    match kind:
        case "receipt":
            write_run_receipt(target, _receipt(base, label))
        case "application":
            write_application_receipt(target, _application(base, label))
        case "checkpoint":
            save_server_checkpoint(target, _state(label))
        case "replay":
            save_socket_replay(target, load_socket_replay(base / "replay.json"))
        case "json":
            _write_json(target, {"label": label, "value": 19})
        case "cli":
            raise SystemExit(
                main(["aggregate", "--input", str(base / "request.json"), "--output-dir", str(target)])
            )
        case _:
            raise AssertionError(kind)


def _pause_trace(kind: str, phase: str) -> None:
    """Pause unchanged production at a source line or actual post-open opcode."""
    functions: dict[str, Callable[..., Any]] = {
        "receipt": write_run_receipt,
        "application": write_application_receipt,
        "checkpoint": save_server_checkpoint,
        "replay": save_socket_replay,
        "json": _write_json,
    }
    if phase == "serialize":
        from json import encoder

        count = 0

        def serialization_trace(frame: FrameType, event: str, argument: Any) -> Any:
            nonlocal count
            if (
                event == "line"
                and frame.f_code.co_filename == encoder.__file__
                and frame.f_code.co_name == "_iterencode_dict"
            ):
                count += 1
                if count == 5000:
                    sys.settrace(None)
                    print(
                        "PAUSED "
                        + json.dumps(
                            {
                                "pid": os.getpid(),
                                "path": frame.f_code.co_filename,
                                "line": frame.f_lineno,
                                "serialization_events": count,
                            }
                        ),
                        flush=True,
                    )
                    assert sys.stdin.readline().strip() == "GO"
                    return None
            return serialization_trace

        sys.settrace(serialization_trace)
        return
    from fedference_cli._support import _write_application_run

    function: Callable[..., Any]
    if kind == "cli":
        function = _write_application_run
        needle = {"publish": "request_path = ", "record": "outputs=(", "commit": "receipt_path = "}.get(
            phase, "request_path = "
        )
        if phase == "ack":
            from fedference_cli._commands import _aggregate_command

            function, needle = _aggregate_command, "print("

    else:
        function = functions[kind]
        # Baseline and hardened origins are both executable regression targets.
        from fedference import evidence_contracts

        atomic_path = Path(evidence_contracts.__file__).parent / "_atomic.py"
        if atomic_path.exists():
            from fedference.evidence_contracts._atomic import DirectoryAuthority, publish_json

            if phase == "before_open":
                function, needle = publish_json, "with open_directory("
            else:
                function, needle = DirectoryAuthority.publish, "os.fsync(stream.fileno())"
        else:
            needle = {
                "before_open": "with tempfile.NamedTemporaryFile(",
                "publish": "os.replace(",
                "created": "temporary = Path(handle.name)",
            }[phase]
    lines, start = inspect.getsourcelines(function)
    lineno = start + next(index for index, line in enumerate(lines) if needle in line)
    offset: int | None = None
    if phase == "created" and function.__name__ == "publish":
        stores = [
            item.offset
            for item in dis.get_instructions(function)
            if item.opname == "STORE_FAST" and item.argval == "stream"
        ]
        offset = stores[0]  # Open returned; context exit is armed before stream binding.
    code = function.__code__

    def trace(frame: FrameType, event: str, argument: Any) -> Any:
        if frame.f_code is code:
            frame.f_trace_opcodes = offset is not None
            hit = (
                (event == "opcode" and frame.f_lasti == offset)
                if offset is not None
                else (event == "line" and frame.f_lineno == lineno)
            )
            if hit:
                sys.settrace(None)
                print(
                    "PAUSED "
                    + json.dumps(
                        {
                            "pid": os.getpid(),
                            "path": code.co_filename,
                            "line": frame.f_lineno,
                            "offset": frame.f_lasti,
                            "phase": phase,
                        }
                    ),
                    flush=True,
                )
                assert sys.stdin.readline().strip() == "GO"
                return None
        return trace

    sys._getframe().f_trace_opcodes = offset is not None
    sys.settrace(trace)


@dataclass
class _Producer:
    process: subprocess.Popen[bytes]
    logs: Path
    prefix: bytes = b""
    finished: bool = False

    def line(self) -> bytes:
        assert self.process.stdout is not None
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            assert selector.select(30), f"producer barrier timeout: {self.process.pid}"
        line = self.process.stdout.readline()
        self.prefix += line
        assert line, f"producer exited before barrier: {self.process.pid}"
        return line

    def go(self) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(b"GO\n")
        self.process.stdin.flush()

    def finish(self) -> tuple[int, bytes]:
        stdout, stderr = self.process.communicate(timeout=30)
        self.finished = True
        self.logs.with_suffix(".stdout").write_bytes(self.prefix + stdout)
        self.logs.with_suffix(".stderr").write_bytes(stderr)
        self.logs.with_suffix(".status.json").write_text(
            json.dumps(
                {
                    "pid": self.process.pid,
                    "exit": self.process.returncode,
                    "reaped": True,
                    "stdout": str(self.logs.with_suffix(".stdout")),
                    "stderr": str(self.logs.with_suffix(".stderr")),
                }
            ),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "pid": self.process.pid,
                    "exit": self.process.returncode,
                    "logs": str(self.logs),
                    "reaped": True,
                }
            )
        )
        assert self.process.returncode is not None
        return self.process.returncode, stderr


@contextmanager
def _start(kind: str, target: Path, inputs: Path, phase: str, label: str = "A") -> Iterator[_Producer]:
    logs = inputs.parent / f"producer-{kind}-{phase}-{label}-{target.parent.name}"
    interpreter = [sys.executable, "-B"]
    coverage_config = os.environ.get("COVERAGE_PROCESS_START")
    if coverage_config is not None:
        interpreter.extend(["-m", "coverage", "run", "--rcfile", coverage_config, "--parallel-mode"])
    command = [
        *interpreter,
        str(Path(__file__).resolve()),
        "--producer",
        kind,
        str(target),
        str(inputs),
        phase,
        label,
    ]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT))),
        "PYTEST_ADDOPTS": "",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    logs.with_suffix(".command.json").write_text(
        json.dumps({"argv": command, "cwd": str(ROOT), "pid": process.pid}), encoding="utf-8"
    )
    producer = _Producer(process, logs)
    try:
        assert producer.line().startswith(b"ORIGIN ")
        assert producer.line() == b"READY\n"
        producer.go()
        if phase != "none":
            assert producer.line().startswith(b"PAUSED ")
        yield producer
    finally:
        if not producer.finished:
            if process.poll() is None:
                process.kill()
                print(json.dumps({"emergency_reap": process.pid}))
            producer.finish()
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()


def _verify(kind: str, target: Path, inputs: Path) -> None:
    match kind:
        case "receipt":
            assert verify_run_receipt(load_run_receipt(target), root=inputs) == ()
        case "application":
            assert verify_application_receipt(load_application_receipt(target), root=inputs) == ()
        case "checkpoint":
            assert load_server_checkpoint(target).fingerprint == _state("A").fingerprint
        case "replay":
            replay = load_socket_replay(target)
            from fedference.aggregation import log_linear_pool

            assert validate_socket_replay(replay, BELIEFS, log_linear_pool(BELIEFS), robustness=0.0)
        case "json":
            assert json.loads(target.read_bytes()) == {"label": "A", "value": 19}
        case _:
            raise AssertionError(kind)


@pytest.mark.parametrize("kind", KINDS)
def test_one_winner_and_verified_atomic_artifact(tmp_path: Path, inputs: Path, kind: str) -> None:
    # Given: two actual producers have completed staging against one destination.
    target = tmp_path / "race" / "artifact.json"
    with (
        _start(kind, target, inputs, "publish") as first,
        _start(kind, target, inputs, "publish", "B") as second,
    ):
        # When: each reaches the actual atomic publication syscall in a fixed order.
        first.go()
        assert first.finish()[0] == 0
        before = (target.read_bytes(), target.stat().st_ino)
        second.go()
        status, stderr = second.finish()
    # Then: exactly one writer succeeds and the loser cannot change a verified artifact.
    after = (target.read_bytes(), target.stat().st_ino)
    print(
        json.dumps(
            {
                "case": "one-winner",
                "kind": kind,
                "loser_exit": status,
                "winner_sha256": hashlib.sha256(before[0]).hexdigest(),
                "winner_inode": before[1],
                "after_sha256": hashlib.sha256(after[0]).hexdigest(),
                "after_inode": after[1],
            }
        )
    )
    assert status != 0 and b"FileExistsError" in stderr
    assert after == before
    _verify(kind, target, inputs)
    assert sorted(path.name for path in target.parent.iterdir()) == [target.name]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("phase", ("before_open", "publish"))
def test_competing_writer_symlink_and_abort(tmp_path: Path, inputs: Path, kind: str, phase: str) -> None:
    # Given: the real writer is paused before parent acquisition or after staging.
    parent = tmp_path / "selected"
    parent.mkdir()
    redirect = tmp_path / "redirect"
    redirect.mkdir()
    target = parent / "artifact.json"
    with _start(kind, target, inputs, phase) as producer:
        # When: another actor renames its parent and substitutes a raw symlink.
        original = tmp_path / "original"
        parent.rename(original)
        parent.symlink_to(redirect, target_is_directory=True)
        producer.go()
        status, stderr = producer.finish()
    # Then: no redirected accepted artifact and exact fd-backed staging cleanup.
    assert status != 0, stderr
    assert list(redirect.iterdir()) == []
    assert list(original.iterdir()) == []


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("symlink", (False, True))
def test_existing_destinations_are_never_replaced(
    tmp_path: Path, inputs: Path, kind: str, symlink: bool
) -> None:
    protected = tmp_path / "protected"
    protected.write_bytes(b"caller-owned bytes")
    target = tmp_path / "artifact.json"
    if symlink:
        target.symlink_to(protected)
    else:
        target.write_bytes(b"caller-owned bytes")
    identity = target.lstat().st_ino
    with _start(kind, target, inputs, "none") as producer:
        status, stderr = producer.finish()
    assert status != 0 and b"FileExistsError" in stderr
    assert target.lstat().st_ino == identity
    assert target.read_bytes() == protected.read_bytes() == b"caller-owned bytes"


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("phase", ("created", "publish"))
@pytest.mark.parametrize("sig", (signal.SIGINT, signal.SIGKILL))
def test_interruption_cleans_or_reaps_exact_owned_stage(
    tmp_path: Path, inputs: Path, kind: str, phase: str, sig: signal.Signals
) -> None:
    target = tmp_path / "abort" / "artifact.json"
    with _start(kind, target, inputs, phase) as producer:
        stages = list(target.parent.iterdir())
        assert len(stages) == 1 and stages[0].name.endswith(".tmp")
        before = stages[0].read_bytes()
        orphan_inode = stages[0].stat().st_ino
        os.kill(producer.process.pid, sig)
        status, stderr = producer.finish()
    assert status != 0, stderr
    assert not target.exists()
    if sig == signal.SIGINT:
        assert list(target.parent.iterdir()) == []
    else:
        # The reaped producer cannot clean after SIGKILL. Preserve the exact
        # orphan, then remove only the name and identity observed at its barrier.
        orphan = stages[0]
        assert orphan.lstat().st_ino == orphan_inode
        assert orphan.read_bytes() == before
        retained = inputs.parent / f"killed-{kind}-{phase}.bytes"
        retained.write_bytes(before)
        if not before and kind == "checkpoint":
            with pytest.raises(ValueError):
                load_server_checkpoint(retained)
        print(
            json.dumps(
                {
                    "killed_pid": producer.process.pid,
                    "orphan": str(orphan),
                    "bytes": len(before),
                    "sha256": hashlib.sha256(before).hexdigest(),
                    "retained": str(retained),
                    "reaped_before_unlink": True,
                }
            )
        )
        orphan.unlink()
        assert list(target.parent.iterdir()) == []


def test_empty_caller_directory_has_one_cli_owner(tmp_path: Path, inputs: Path) -> None:
    target = tmp_path / "application"
    target.mkdir()
    with _start("cli", target, inputs, "publish") as first:
        with _start("cli", target, inputs, "none", "B") as second:
            loser, stderr = second.finish()
        first.go()
        assert first.finish()[0] == 0
    assert loser == 2, stderr
    receipt = load_application_receipt(target / "receipt.json")
    assert verify_application_receipt(receipt, root=target) == ()
    assert sorted(path.name for path in target.iterdir()) == ["receipt.json", "request.json", "result.json"]


def test_cli_retains_authority_after_parent_substitution(tmp_path: Path, inputs: Path) -> None:
    target = tmp_path / "application"
    target.mkdir()
    redirect = tmp_path / "redirect"
    redirect.mkdir()
    with _start("cli", target, inputs, "publish") as producer:
        original = tmp_path / "original"
        target.rename(original)
        target.symlink_to(redirect, target_is_directory=True)
        producer.go()
        status, stderr = producer.finish()
    assert status == 2, stderr
    assert list(redirect.iterdir()) == []
    assert list(original.iterdir()) == []


def test_checksum_expectation_refuses_schema_valid_checkpoint_corruption(tmp_path: Path) -> None:
    target = save_server_checkpoint(tmp_path / "checkpoint.json", _state("A"))
    trusted = sha256_file(target)
    assert (
        load_server_checkpoint(target, expected_sha256=trusted.upper()).fingerprint == _state("A").fingerprint
    )
    raw = json.loads(target.read_bytes())
    raw["prior"]["mean"][0] = 0.75
    target.write_text(json.dumps(raw), encoding="utf-8")
    # No expectation is explicitly schema-only, and still computes its new fingerprint.
    assert load_server_checkpoint(target).fingerprint != _state("A").fingerprint
    print(
        json.dumps(
            {
                "trusted_sha256": trusted,
                "changed_sha256": sha256_file(target),
                "schema_only_fingerprint": load_server_checkpoint(target).fingerprint,
            }
        )
    )
    with pytest.raises(ValueError, match="trusted expected_sha256"):
        load_server_checkpoint(target, expected_sha256=trusted)


@pytest.mark.parametrize(
    "invalid", ('{"prior":NaN}', '{"round_index":0,"round_index":1}', "{", '"text"', "\udcff")
)
def test_checkpoint_strict_readers_refuse_invalid_bytes(tmp_path: Path, invalid: str) -> None:
    target = tmp_path / "invalid.json"
    target.write_bytes(invalid.encode("utf-8", errors="surrogatepass"))
    with pytest.raises(ValueError):
        load_server_checkpoint(target)


@pytest.mark.parametrize("invalid", ("x" * 64, "a" * 63, "", 19))
def test_checkpoint_requires_explicit_well_formed_trusted_digest(tmp_path: Path, invalid: Any) -> None:
    with pytest.raises(ValueError, match="expected_sha256"):
        load_server_checkpoint(tmp_path / "absent", expected_sha256=invalid)


@pytest.mark.parametrize("writer", (_write_json, save_socket_replay))
def test_nonfinite_serialization_never_allocates_stage(tmp_path: Path, writer: Any) -> None:
    with pytest.raises(ValueError):
        writer(tmp_path / "missing-parent" / "invalid.json", [{"bad": float("nan")}])
    assert not (tmp_path / "missing-parent").exists()


if __name__ == "__main__":
    assert sys.argv[1] == "--producer"
    _, _, selected_kind, raw_target, raw_inputs, selected_phase, selected_label = sys.argv
    modules = {}
    for module_name, module in sorted(sys.modules.items()):
        if module_name.startswith(("fedference", "analysis", "project_paths", "experiment_config")):
            filename = getattr(module, "__file__", None)
            if filename:
                module_path = Path(filename).resolve()
                assert module_path.is_relative_to(ROOT)
                modules[module_name] = {
                    "path": str(module_path),
                    "sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
                }
    print("ORIGIN " + json.dumps(modules), flush=True)
    print("READY", flush=True)
    assert sys.stdin.readline().strip() == "GO"
    if selected_phase != "none":
        _pause_trace(selected_kind, selected_phase)
    _produce(selected_kind, Path(raw_target), Path(raw_inputs), selected_label)


@pytest.mark.parametrize("kind", KINDS)
def test_successful_serialization_preserves_canonical_bytes(tmp_path: Path, inputs: Path, kind: str) -> None:
    target = tmp_path / "artifact.json"
    _produce(kind, target, inputs, "A")
    payloads = {
        "receipt": _receipt(inputs, "A").as_dict(),
        "application": _application(inputs, "A").as_dict(),
        "checkpoint": _state("A").as_dict(),
        "replay": load_socket_replay(inputs / "replay.json"),
        "json": {"label": "A", "value": 19},
    }
    expected = (json.dumps(payloads[kind], indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    assert target.read_bytes() == expected
    print(json.dumps({"kind": kind, "bytes": len(expected), "sha256": sha256_file(target)}))


def test_namespace_capability_closes_and_empty_directory_can_be_reclaimed(tmp_path: Path) -> None:
    from fedference_cli._support import _prepare_output_dir

    selected = tmp_path / "empty"
    selected.mkdir()
    with _prepare_output_dir(str(selected), ROOT) as first:
        assert first.path == selected
    with pytest.raises(ValueError, match="closed"):
        first.write_json("forbidden.json", {})
    with _prepare_output_dir(str(selected), ROOT) as second:
        second.write_json("owned.json", {})
    assert sorted(p.name for p in selected.iterdir()) == ["owned.json"]


def test_descriptor_authority_refuses_removed_ancestry_and_special_children(tmp_path: Path) -> None:
    from fedference.evidence_contracts._atomic import open_directory

    selected = tmp_path / "selected"
    with open_directory(selected) as authority:
        for name in ("", "..", "nested/file"):
            with pytest.raises(ValueError, match="one child"):
                authority.publish(name, b"bytes")
        (selected / "directory").mkdir()
        with pytest.raises(ValueError, match="regular file"):
            authority.read_bytes("directory")
        (selected / "directory").rmdir()
        selected.rmdir()
        with pytest.raises(ValueError, match="ancestry disappeared"):
            authority.publish("absent", b"bytes")
    with pytest.raises(ValueError, match="parent traversal"):
        with open_directory(tmp_path / "missing" / ".." / "other"):
            pytest.fail("parent traversal admitted")


def test_descriptor_reads_refuse_symlinks_and_commit_refuses_changed_bytes(
    tmp_path: Path, inputs: Path
) -> None:
    from fedference_cli._support import _prepare_output_dir

    selected = tmp_path / "namespace"
    with _prepare_output_dir(str(selected), ROOT) as namespace:
        namespace.write_json("config.json", {"seed": 19})
        namespace.write_json("report.json", {"engineering_only": True})
        receipt = _receipt(selected, "A")
        (selected / "report.json").write_bytes(b"changed")
        with pytest.raises(ValueError, match="changed before receipt commit"):
            namespace.commit(receipt)
        (selected / "link").symlink_to(inputs / "report.json")
        with pytest.raises(OSError):
            namespace.record("link", "link")
    assert not (selected / "receipt.json").exists()


@pytest.mark.parametrize("phase", ("record", "commit"))
def test_cli_digest_binding_retains_directory_authority(tmp_path: Path, inputs: Path, phase: str) -> None:
    target = tmp_path / "application"
    target.mkdir()
    redirect = tmp_path / "redirect"
    redirect.mkdir()
    with _start("cli", target, inputs, phase) as producer:
        before = {path.name: path.read_bytes() for path in target.iterdir()}
        assert sorted(before) == ["request.json", "result.json"]
        for name, data in before.items():
            (redirect / name).write_bytes(data)
        original = tmp_path / "original"
        target.rename(original)
        target.symlink_to(redirect, target_is_directory=True)
        producer.go()
        status, stderr = producer.finish()
    assert status == 2, stderr
    assert {path.name: path.read_bytes() for path in redirect.iterdir()} == before
    assert {path.name: path.read_bytes() for path in original.iterdir()} == before


@pytest.mark.parametrize("sig", (signal.SIGINT, signal.SIGKILL))
def test_precommit_abort_releases_only_own_empty_namespace(
    tmp_path: Path, inputs: Path, sig: signal.Signals
) -> None:
    target = tmp_path / "application"
    target.mkdir()
    with _start("cli", target, inputs, "publish") as producer:
        os.kill(producer.process.pid, sig)
        assert producer.finish()[0] != 0
    assert list(target.iterdir()) == []
    with _start("cli", target, inputs, "none", "B") as successor:
        assert successor.finish()[0] == 0
    assert verify_application_receipt(load_application_receipt(target / "receipt.json"), root=target) == ()


def test_postcommit_interrupt_can_lose_acknowledgement_but_keeps_valid_receipt(
    tmp_path: Path, inputs: Path
) -> None:
    target = tmp_path / "application"
    with _start("cli", target, inputs, "ack") as producer:
        before = {path.name: path.read_bytes() for path in target.iterdir()}
        assert sorted(before) == ["receipt.json", "request.json", "result.json"]
        os.kill(producer.process.pid, signal.SIGINT)
        assert producer.finish()[0] != 0
    assert {path.name: path.read_bytes() for path in target.iterdir()} == before
    assert verify_application_receipt(load_application_receipt(target / "receipt.json"), root=target) == ()
    with _start("cli", target, inputs, "none", "B") as successor:
        assert successor.finish()[0] == 2
    assert {path.name: path.read_bytes() for path in target.iterdir()} == before


@pytest.mark.parametrize("sig", (signal.SIGINT, signal.SIGKILL))
def test_interrupted_preserialization_has_no_partial_namespace(
    tmp_path: Path, inputs: Path, sig: signal.Signals
) -> None:
    target = tmp_path / "absent" / "checkpoint.json"
    with _start("checkpoint", target, inputs, "serialize", "LARGE") as producer:
        stages = list(target.parent.iterdir()) if target.parent.exists() else []
        for path in stages:
            retained = inputs.parent / (path.name + ".preserved")
            retained.write_bytes(path.read_bytes())
            print(
                json.dumps(
                    {
                        "midserialization_path": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                        "preserved": str(retained),
                    }
                )
            )
        os.kill(producer.process.pid, sig)
        assert producer.finish()[0] != 0
    assert not target.parent.exists()


def test_checkpoint_duplicate_valid_field_is_rejected(tmp_path: Path) -> None:
    original = json.dumps(_state("A").as_dict())
    target = tmp_path / "duplicate.json"
    target.write_text(original[:-1] + ', "round_index": 0}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_server_checkpoint(target)


def test_evidence_facade_preserves_direct_import_identity() -> None:
    from fedference import evidence
    from fedference.evidence_contracts import custody

    assert evidence.write_run_receipt is custody.write_run_receipt
    assert evidence.write_application_receipt is custody.write_application_receipt
    assert evidence.make_artifact_record is custody.make_artifact_record


@pytest.mark.parametrize("phase", ("directory-open", "directory-register", "read-open", "stage-open"))
def test_sigint_before_descriptor_ownership_preserves_resources(tmp_path: Path, phase: str) -> None:
    """A surviving caller retains no descriptor or signal-mask change after SIGINT."""
    import fedference.evidence_contracts._atomic as atomic

    source = Path(atomic.__file__).resolve()
    assert source == ROOT / "src/fedference/evidence_contracts/_atomic.py"
    command = [
        sys.executable,
        "-B",
        "-c",
        """\
from contextlib import ExitStack
from pathlib import Path
from types import CodeType
import dis
import gc
import hashlib
import json
import os
import signal
import sys
import fedference.evidence_contracts._atomic as atomic

phase, location, expected_source, expected_hash = sys.argv[1:]
root = Path(location)
source = Path(atomic.__file__).resolve()
assert source == Path(expected_source).resolve()
assert hashlib.sha256(source.read_bytes()).hexdigest() == expected_hash
signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT})
signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGUSR1})
initial_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())

def descriptors():
    directory = "/proc/self/fd" if Path("/proc/self/fd").is_dir() else "/dev/fd"
    result = {}
    for name in os.listdir(directory):
        try:
            fd = int(name)
            value = os.fstat(fd)
        except (ValueError, OSError):
            continue
        result[fd] = (value.st_dev, value.st_ino, value.st_mode)
    return result

def exercise(code, offset, operation, directory_registration=False):
    before = descriptors()
    fired = caught = False
    retained_errors = []
    def trace(frame, event, argument):
        nonlocal fired
        if frame.f_code is code:
            frame.f_trace_opcodes = True
            matches = True
            if directory_registration:
                manager = frame.f_locals.get("cm")
                generator = getattr(manager, "gen", None)
                matches = getattr(generator, "gi_code", None) is atomic._directory_fd.__wrapped__.__code__
            if event == "opcode" and frame.f_lasti == offset and matches and not fired:
                fired = True
                os.kill(os.getpid(), signal.SIGINT)
        return trace
    previous_trace = sys.gettrace()
    previous_opcodes = sys._getframe().f_trace_opcodes
    try:
        sys._getframe().f_trace_opcodes = True
        sys.settrace(trace)
        try:
            operation()
        except KeyboardInterrupt as error:
            retained_errors.append(error)
            caught = True
    finally:
        sys.settrace(previous_trace)
        sys._getframe().f_trace_opcodes = previous_opcodes
    after = descriptors()
    extra = {fd: value for fd, value in after.items() if fd not in before}
    retained_error_count = len(retained_errors)
    retained_errors.clear()
    gc.collect()
    still_open = descriptors()
    for fd, value in extra.items():
        if fd in still_open and still_open[fd] == value:
            os.close(fd)
    assert descriptors() == before
    return {
        "phase": phase, "pid": os.getpid(), "fired": fired, "caught": caught,
        "offset": offset, "extra_descriptors": extra,
        "exception_retained_during_measurement": retained_error_count == 1,
        "descriptor_set_unchanged": before == after,
        "exact_extra_descriptors_closed": list(extra),
        "signal_mask_unchanged": signal.pthread_sigmask(signal.SIG_BLOCK, set()) == initial_mask,
        "source": str(source), "source_sha256": expected_hash,
    }

if phase in {"directory-open", "directory-register"}:
    def operation():
        with atomic.open_directory(root / "new"):
            pass
    if phase == "directory-open":
        code = atomic._directory_fd.__wrapped__.__code__
        offset = [item.offset for item in dis.get_instructions(code)
                  if item.opname == "STORE_FAST" and item.argval == "descriptor"][1]
    else:
        code = ExitStack.enter_context.__code__
        offset = next(item.offset for item in dis.get_instructions(code)
                      if item.opname == "STORE_FAST" and item.argval == "result")
    result = exercise(code, offset, operation, phase == "directory-register")
else:
    selected = root / "held"
    selected.mkdir()
    (selected / "existing").write_bytes(b"complete fixture")
    with atomic.open_directory(selected) as authority:
        if phase == "read-open":
            code = atomic.DirectoryAuthority.read_bytes.__code__
            offset = [item.offset for item in dis.get_instructions(code)
                      if item.opname == "STORE_FAST" and item.argval == "descriptor"][1]
            result = exercise(code, offset, lambda: authority.read_bytes("existing"))
        else:
            assert phase == "stage-open"
            code = next(item for item in atomic.DirectoryAuthority.publish.__code__.co_consts
                        if isinstance(item, CodeType) and item.co_name == "create_stage")
            offset = next(item.offset for item in dis.get_instructions(code)
                          if item.opname == "RETURN_VALUE")
            result = exercise(code, offset, lambda: authority.publish("final", b"complete fixture"))
        assert not (selected / "final").exists()
        assert not list(selected.glob(".*.tmp"))
assert hashlib.sha256(source.read_bytes()).hexdigest() == expected_hash
print(json.dumps(result, sort_keys=True), flush=True)
""",
        phase,
        str(tmp_path),
        str(source),
        hashlib.sha256(source.read_bytes()).hexdigest(),
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT))),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    for suffix, value in (("stdout", result.stdout), ("stderr", result.stderr)):
        (tmp_path / ("descriptor-" + phase + "." + suffix)).write_text(value, encoding="utf-8")
    (tmp_path / ("descriptor-" + phase + ".command.json")).write_text(
        json.dumps({"argv": command, "exit": result.returncode}, indent=2) + "\n",
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    print(json.dumps(observed, sort_keys=True))
    assert observed["fired"] and observed["caught"], observed
    assert observed["signal_mask_unchanged"], observed
    assert observed["descriptor_set_unchanged"], observed
