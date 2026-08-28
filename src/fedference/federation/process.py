from __future__ import annotations

import queue
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing import get_context
from time import monotonic
from typing import Any

import numpy as np

from fedference._validation import as_pmf_matrix
from fedference.aggregation import AggregationConfig, AggregationResult
from fedference.federation.server import FederationServer
from fedference.federation.worker import FederationWorker

DEFAULT_STARTUP_TIMEOUT = 15.0


@dataclass(frozen=True)
class MultiprocessRoundResult:
    """Server diagnostics and ordered worker results from one process round."""

    server_aggregation: AggregationResult
    worker_consensuses: tuple[np.ndarray, ...]
    config: AggregationConfig
    n_workers: int
    bit_identical: bool

    def __post_init__(self) -> None:
        if not isinstance(self.config, AggregationConfig):
            raise ValueError("config must be an AggregationConfig")
        if self.n_workers != len(self.worker_consensuses) or self.n_workers <= 0:
            raise ValueError("worker_consensuses must contain one result per worker")
        if not isinstance(self.bit_identical, bool):
            raise ValueError("bit_identical must be a boolean")
        frozen_workers: list[np.ndarray] = []
        for consensus in self.worker_consensuses:
            copied = np.asarray(consensus, dtype=np.float64).copy()
            if copied.ndim != 1 or copied.size == 0 or not np.all(np.isfinite(copied)):
                raise ValueError("worker consensuses must be non-empty finite vectors")
            copied.setflags(write=False)
            frozen_workers.append(copied)
        object.__setattr__(self, "worker_consensuses", tuple(frozen_workers))

    @property
    def consensus(self) -> np.ndarray:
        """Return the server consensus broadcast to every worker."""
        return self.server_aggregation.consensus


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


def run_multiprocess_round_result(
    local_posteriors: Sequence[np.ndarray] | None = None,
    *,
    robustness: float | None = None,
    config: AggregationConfig | None = None,
    timeout: float = 5.0,
    startup_timeout: float | None = None,
    **legacy: object,
) -> MultiprocessRoundResult:
    """Run one spawned-process round and return complete server diagnostics.

    The server and workers exchange serialized beliefs through multiprocessing
    queues. Worker responses are checked against the server result before the
    child processes are joined, so a transport or lifecycle mismatch fails the
    round rather than returning a partial result. ``timeout`` bounds each
    federation queue operation. ``startup_timeout`` separately bounds the
    one-time wait for all spawned workers to import and announce readiness;
    when omitted it is at least ``DEFAULT_STARTUP_TIMEOUT`` so macOS spawn
    latency cannot consume the transport timeout under normal system load.

    The helper deliberately uses the ``spawn`` start method on every platform.
    Call it from an importable Python file and protect the invoking entry point
    with ``if __name__ == "__main__":``. An unguarded module body, ``python -c``
    snippet, or equivalent non-importable interactive entry point can recursively
    start children or fail during worker bootstrap.
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
    raw_local_posteriors = tuple(local_posteriors)
    if not raw_local_posteriors:
        raise ValueError("beliefs must be non-empty (local_posteriors is the canonical name)")
    as_pmf_matrix(raw_local_posteriors, name="local_posteriors")
    # Preserve caller mass on protocol v1. The server normalizes each received
    # categorical row exactly once through aggregate_result.
    belief_arrays = tuple(
        np.asarray(local_posterior, dtype=np.float64)
        for local_posterior in raw_local_posteriors
    )
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

    worker_consensus_by_id: dict[int, np.ndarray] = {}
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
        server_result = server.run_round_result(
            request_queue,
            response_queues,
            timeout=timeout,
        )
        for _ in processes:
            worker_id, child_consensus = result_queue.get(timeout=timeout)
            if (
                isinstance(worker_id, bool)
                or not isinstance(worker_id, (int, np.integer))
                or int(worker_id) not in range(len(processes))
            ):
                raise RuntimeError("multiprocess worker returned an invalid id")
            worker_id = int(worker_id)
            if worker_id in worker_consensus_by_id:
                raise RuntimeError(
                    f"multiprocess worker returned duplicate id: {worker_id}"
                )
            worker_consensus_by_id[worker_id] = child_consensus
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
    ordered_workers = tuple(
        worker_consensus_by_id[worker_id]
        for worker_id in range(len(processes))
    )
    bit_identical = all(
        np.array_equal(server_result.consensus, child)
        for child in ordered_workers
    )
    if not bit_identical:
        raise RuntimeError("worker consensus did not match server consensus")
    return MultiprocessRoundResult(
        server_aggregation=server_result,
        worker_consensuses=ordered_workers,
        config=server.config,
        n_workers=len(processes),
        bit_identical=bit_identical,
    )


def run_multiprocess_round(
    local_posteriors: Sequence[np.ndarray] | None = None,
    *,
    robustness: float | None = None,
    config: AggregationConfig | None = None,
    timeout: float = 5.0,
    startup_timeout: float | None = None,
    **legacy: object,
) -> np.ndarray:
    """Run one process round and return the legacy consensus array."""
    return run_multiprocess_round_result(
        local_posteriors,
        robustness=robustness,
        config=config,
        timeout=timeout,
        startup_timeout=startup_timeout,
        **legacy,
    ).consensus


__all__ = [
    "MultiprocessRoundResult",
    "run_multiprocess_round",
    "run_multiprocess_round_result",
]
