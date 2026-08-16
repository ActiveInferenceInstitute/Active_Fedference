from __future__ import annotations

import queue
import warnings
from collections.abc import Sequence
from multiprocessing import get_context
from time import monotonic
from typing import Any

import numpy as np

from fedference.aggregation import AggregationConfig
from fedference.federation.server import FederationServer
from fedference.federation.worker import FederationWorker

DEFAULT_STARTUP_TIMEOUT = 15.0


def _worker_round(
    worker_id: int,
    belief: np.ndarray,
    request_queue: Any,
    response_queue: Any,
    result_queue: Any,
    ready_queue: Any,
    timeout: float,
) -> None:
    worker = FederationWorker(worker_id, request_queue, response_queue, timeout=timeout)
    ready_queue.put(worker_id)
    worker.send_belief(belief)
    consensus = worker.receive_consensus(timeout=timeout)
    result_queue.put((worker_id, consensus))


def _stop_processes(processes: Sequence[Any], timeout: float) -> None:
    deadline = monotonic() + timeout * max(1, len(processes))
    for process in processes:
        process.join(timeout=max(0.0, deadline - monotonic()))
        if process.is_alive():
            process.terminate()
            process.join(timeout=max(0.0, deadline - monotonic()))


def run_multiprocess_round(
    local_posteriors: Sequence[np.ndarray] | None = None,
    *,
    robustness: float | None = None,
    config: AggregationConfig | None = None,
    timeout: float = 5.0,
    startup_timeout: float | None = None,
    **legacy: object,
) -> np.ndarray:
    """Run one real spawned-process federation round and return its consensus.

    The server and workers exchange serialized beliefs through multiprocessing
    queues. Worker responses are checked against the server result before the
    child processes are joined, so a transport or lifecycle mismatch fails the
    round rather than returning a partial result. ``timeout`` bounds each
    federation queue operation. ``startup_timeout`` separately bounds the
    one-time wait for all spawned workers to import and announce readiness;
    when omitted it is at least ``DEFAULT_STARTUP_TIMEOUT`` so macOS spawn
    latency cannot consume the transport timeout under normal system load.
    """
    if "beliefs" in legacy:
        if local_posteriors is not None:
            raise TypeError(
                "local_posteriors and deprecated beliefs cannot both be supplied"
            )
        local_posteriors = legacy.pop("beliefs")  # type: ignore[assignment]
        warnings.warn(
            "beliefs is deprecated; use local_posteriors",
            DeprecationWarning,
            stacklevel=2,
        )
    if legacy:
        raise TypeError(f"unexpected keyword argument(s): {', '.join(sorted(legacy))}")
    if local_posteriors is None:
        raise TypeError("local_posteriors is required")
    belief_arrays = tuple(
        np.asarray(local_posterior, dtype=np.float64)
        for local_posterior in local_posteriors
    )
    if not belief_arrays:
        raise ValueError("beliefs must be non-empty (local_posteriors is the canonical name)")
    if config is not None and not isinstance(config, AggregationConfig):
        raise ValueError("config must be an AggregationConfig or None")
    if config is not None and robustness is not None:
        raise ValueError("config and compatibility robustness are mutually exclusive")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float, np.integer, np.floating))
        or not np.isfinite(timeout)
        or timeout <= 0.0
    ):
        raise ValueError("timeout must be finite and positive")
    if startup_timeout is None:
        startup_timeout = max(float(timeout), DEFAULT_STARTUP_TIMEOUT)
    elif (
        isinstance(startup_timeout, bool)
        or not isinstance(startup_timeout, (int, float, np.integer, np.floating))
        or not np.isfinite(startup_timeout)
        or startup_timeout <= 0.0
    ):
        raise ValueError("startup_timeout must be finite and positive")
    startup_timeout = float(startup_timeout)

    context = get_context("spawn")
    request_queue = context.Queue()
    response_queues = {worker_id: context.Queue() for worker_id in range(len(belief_arrays))}
    result_queue = context.Queue()
    ready_queue = context.Queue()
    processes = [
        context.Process(
            target=_worker_round,
            args=(
                worker_id,
                belief,
                request_queue,
                response_queues[worker_id],
                result_queue,
                ready_queue,
                float(timeout),
            ),
        )
        for worker_id, belief in enumerate(belief_arrays)
    ]

    worker_consensus: list[np.ndarray] = []
    started_processes: list[Any] = []
    try:
        for process in processes:
            process.start()
            started_processes.append(process)
        ready_worker_ids: set[int] = set()
        startup_deadline = monotonic() + startup_timeout
        while len(ready_worker_ids) < len(processes):
            remaining = startup_deadline - monotonic()
            if remaining <= 0.0:
                raise TimeoutError(
                    "multiprocess federation worker startup timed out"
                )
            worker_id = ready_queue.get(timeout=remaining)
            if (
                isinstance(worker_id, bool)
                or not isinstance(worker_id, (int, np.integer))
                or int(worker_id) not in range(len(processes))
            ):
                raise RuntimeError("multiprocess worker announced an invalid id")
            worker_id = int(worker_id)
            if worker_id in ready_worker_ids:
                raise RuntimeError(
                    f"multiprocess worker announced duplicate id: {worker_id}"
                )
            ready_worker_ids.add(worker_id)
        server = FederationServer(
            n_workers=len(belief_arrays),
            robustness=robustness,
            timeout=timeout,
            config=config,
        )
        consensus = server.run_round(request_queue, response_queues, timeout=timeout)
        for _ in processes:
            _, child_consensus = result_queue.get(timeout=timeout)
            worker_consensus.append(child_consensus)
    except queue.Empty as exc:
        raise TimeoutError("multiprocess federation round timed out") from exc
    finally:
        _stop_processes(started_processes, timeout)
        for transport_queue in (
            request_queue,
            result_queue,
            ready_queue,
            *response_queues.values(),
        ):
            transport_queue.close()
            transport_queue.join_thread()

    failed = [process.exitcode for process in started_processes if process.exitcode != 0]
    if failed:
        raise RuntimeError(f"worker process exit codes: {failed}")
    if any(not np.array_equal(consensus, child) for child in worker_consensus):
        raise RuntimeError("worker consensus did not match server consensus")
    return consensus
