"""End-to-end federation transport tests.

Uses threading-safe ``queue.Queue`` and a single-machine OS-process helper to
exercise the worker -> server -> worker round trip and assert bit-identity to
the in-process ``robust_aggregate`` call. The bit-identity rests on the lossless
float64 serialization in :mod:`fedference.federation.transport`.
"""

import queue
from typing import Any

import numpy as np
import pytest

from fedference.aggregation import robust_aggregate
from fedference.federation import FederationServer, FederationWorker, run_multiprocess_round
from fedference.federation.transport import (
    deserialize_belief,
    deserialize_result,
    serialize_belief,
    serialize_result,
)

pytestmark = pytest.mark.integration


def _make_beliefs(n_agents: int = 5, seed: int = 0) -> tuple[list[np.ndarray], int]:
    n_s = 9
    true_s = 3
    wrong_s = 7
    beliefs = [np.full(n_s, 0.07) for _ in range(n_agents)]
    for b in beliefs:
        b[true_s] = 0.44
    b_adv = np.full(n_s, 0.02)
    b_adv[wrong_s] = 0.84
    beliefs[0] = b_adv
    beliefs[1] = b_adv.copy()
    return beliefs, true_s


def test_federation_bit_identical_to_inprocess():
    beliefs, true_s = _make_beliefs()
    ref = robust_aggregate(beliefs, robustness=0.0).consensus
    rq = queue.Queue()
    rqs = {i: queue.Queue() for i in range(5)}
    workers = [FederationWorker(i, rq, rqs[i]) for i in range(5)]
    server = FederationServer(n_workers=5, robustness=0.0)
    for w, b in zip(workers, beliefs):
        w.send_belief(b)
    fed = server.run_round(rq, rqs)
    assert np.array_equal(ref, fed), f"not bit-identical: max diff={np.max(np.abs(ref - fed))}"


def test_federation_robust_bit_identical():
    beliefs, _ = _make_beliefs()
    ref = robust_aggregate(beliefs, robustness=1.5).consensus
    rq = queue.Queue()
    rqs = {i: queue.Queue() for i in range(5)}
    server = FederationServer(n_workers=5, robustness=1.5)
    workers = [FederationWorker(i, rq, rqs[i]) for i in range(5)]
    for w, b in zip(workers, beliefs):
        w.send_belief(b)
    fed = server.run_round(rq, rqs)
    assert np.allclose(ref, fed, rtol=0, atol=0)


def test_federation_robust_bit_identical_when_workers_arrive_out_of_order():
    beliefs, _ = _make_beliefs()
    ref = robust_aggregate(beliefs, robustness=1.5).consensus
    rq = queue.Queue()
    rqs = {i: queue.Queue() for i in range(5)}
    server = FederationServer(n_workers=5, robustness=1.5)
    workers = [FederationWorker(i, rq, rqs[i]) for i in range(5)]
    for worker, belief in reversed(list(zip(workers, beliefs))):
        worker.send_belief(belief)
    fed = server.run_round(rq, rqs)
    assert np.array_equal(ref, fed)


def test_serialization_round_trip():
    rng = np.random.default_rng(7)
    for _ in range(10):
        b = rng.dirichlet(np.ones(9))
        rt = deserialize_belief(serialize_belief(b))
        assert np.array_equal(b, rt)


def test_result_serialization_round_trip():
    rng = np.random.default_rng(8)
    c = rng.dirichlet(np.ones(9))
    w = rng.dirichlet(np.ones(5))
    rt = deserialize_result(serialize_result(c, w))
    assert np.array_equal(c, rt["consensus"])
    assert np.array_equal(w, rt["agent_weights"])


def test_queue_transport_contract_is_exposed():
    from fedference import federation

    assert hasattr(federation, "FederationServer")
    assert hasattr(federation, "FederationWorker")
    assert hasattr(federation, "run_multiprocess_round")


def test_multiprocess_round_bit_identical_to_inprocess():
    beliefs, _ = _make_beliefs()
    ref = robust_aggregate(beliefs, robustness=0.0).consensus
    fed = run_multiprocess_round(beliefs, robustness=0.0, timeout=5.0)
    assert np.array_equal(ref, fed)


def test_multiprocess_round_robust_bit_identical_to_inprocess():
    beliefs, _ = _make_beliefs()
    ref = robust_aggregate(beliefs, robustness=1.5).consensus
    fed = run_multiprocess_round(beliefs, robustness=1.5, timeout=5.0)
    assert np.array_equal(ref, fed)


def test_multiprocess_round_rejects_empty_beliefs():
    with pytest.raises(ValueError, match="beliefs must be non-empty"):
        run_multiprocess_round([], timeout=5.0)


# --- Worker/server unit paths (in-process, deterministic) ---------------------
# The multiprocess tests above exercise FederationWorker.receive_consensus only
# inside spawned children, so those lines are invisible to the parent-process
# coverage tracer. These in-process tests bind the same round-trip and the
# timeout / duplicate-id error paths directly.


def test_worker_receive_consensus_matches_server_broadcast():
    """A worker's received consensus is bit-identical to the server's result."""
    beliefs, _ = _make_beliefs()
    rq = queue.Queue()
    rqs = {i: queue.Queue() for i in range(5)}
    workers = [FederationWorker(i, rq, rqs[i]) for i in range(5)]
    server = FederationServer(n_workers=5, robustness=1.5)
    for w, b in zip(workers, beliefs):
        w.send_belief(b)
    server_consensus = server.run_round(rq, rqs)
    for w in workers:
        received = w.receive_consensus(timeout=5.0)
        assert np.array_equal(received, server_consensus)
        assert np.array_equal(w.consensus, server_consensus)


def test_worker_receive_consensus_times_out_and_raises(caplog):
    """No server response within the timeout raises queue.Empty and warns."""
    worker = FederationWorker(0, queue.Queue(), queue.Queue(), timeout=0.05)
    with caplog.at_level("WARNING"):
        with pytest.raises(queue.Empty):
            worker.receive_consensus()
    assert worker.consensus is None
    assert any("timed out waiting for consensus" in r.message for r in caplog.records)


def test_server_run_round_times_out_when_a_worker_never_sends(caplog):
    """A missing worker belief makes run_round raise queue.Empty with a warning."""
    rq = queue.Queue()
    rqs = {0: queue.Queue()}
    server = FederationServer(n_workers=1, robustness=0.0, timeout=0.05)
    with caplog.at_level("WARNING"):
        with pytest.raises(queue.Empty):
            server.run_round(rq, rqs)
    assert any("timed out waiting for worker" in r.message for r in caplog.records)


def test_server_run_round_rejects_duplicate_worker_id():
    """Two beliefs tagged with the same worker id are rejected, not silently fused."""
    beliefs, _ = _make_beliefs(n_agents=2)
    rq = queue.Queue()
    rqs = {0: queue.Queue(), 1: queue.Queue()}
    first = FederationWorker(0, rq, rqs[0])
    duplicate = FederationWorker(0, rq, rqs[0])
    first.send_belief(beliefs[0])
    duplicate.send_belief(beliefs[1])
    server = FederationServer(n_workers=2, robustness=0.0, timeout=1.0)
    with pytest.raises(ValueError, match="duplicate worker id"):
        server.run_round(rq, rqs)


def test_server_rejects_unknown_worker_and_response_queue_sets() -> None:
    belief = np.asarray([0.5, 0.5])
    request: queue.Queue = queue.Queue()
    request.put((2, serialize_belief(belief)))
    server = FederationServer(n_workers=1, robustness=0.0)
    with pytest.raises(ValueError, match="worker id"):
        server.run_round(request, {0: queue.Queue()})

    with pytest.raises(ValueError, match="exactly one queue"):
        server.run_round(queue.Queue(), {1: queue.Queue()})


def test_federation_timeouts_reject_boolean_values() -> None:
    with pytest.raises(ValueError, match="timeout"):
        FederationServer(n_workers=1, timeout=True)


def test_federation_adapters_reject_non_configuration_objects() -> None:
    invalid: Any = {"method": "naive"}
    with pytest.raises(ValueError, match="config must be an AggregationConfig"):
        FederationServer(n_workers=1, config=invalid)
    with pytest.raises(ValueError, match="config must be an AggregationConfig"):
        run_multiprocess_round([np.asarray([0.5, 0.5])], config=invalid)
    with pytest.raises(ValueError, match="timeout"):
        run_multiprocess_round([np.asarray([0.5, 0.5])], timeout=True)


def test_federation_transport_boundary_is_explicit_not_cross_host():
    """Enforce the transport's documented boundary in code, not only in prose.

    The federation subpackage includes queue, OS-process, and loopback-TCP
    adapters whose consensus is bit-identical to the in-process
    ``robust_aggregate`` call; true multi-machine deployment is explicitly
    deferred to a separate transport adapter. This test keeps the landed
    network slice visible while preventing a cross-host overclaim.
    """
    from fedference import federation

    doc = (federation.__doc__ or "").lower()
    assert "multi-machine" in doc
    assert "separate transport adapter" in doc
    assert "loopback-tcp" in doc
    assert hasattr(federation, "run_socket_round")
    for forbidden in ("TcpTransport", "SocketServer", "GrpcServer", "HttpTransport"):
        assert not hasattr(federation, forbidden)
