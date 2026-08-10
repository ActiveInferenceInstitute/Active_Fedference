"""Federation server: collects beliefs, aggregates, broadcasts consensus."""

from __future__ import annotations

import logging
import queue
from typing import Any

import numpy as np

from fedference.aggregation import (
    AggregationConfig,
    AggregatorProtocol,
    aggregate_result,
)

_log = logging.getLogger(__name__)

#: Default timeout in seconds for queue operations. Override per-instance via
#: the ``timeout`` constructor parameter or pass it directly to ``run_round``.
DEFAULT_TIMEOUT: float = 10.0


class FederationServer:
    """Collects beliefs from n_workers, aggregates, broadcasts consensus."""

    def __init__(
        self,
        n_workers: int,
        robustness: float | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        *,
        config: AggregationConfig | None = None,
        aggregator: AggregatorProtocol = aggregate_result,
    ) -> None:
        if config is not None and not isinstance(config, AggregationConfig):
            raise ValueError("config must be an AggregationConfig or None")
        if config is not None and robustness is not None:
            raise ValueError("config and compatibility robustness are mutually exclusive")
        if isinstance(n_workers, bool) or not isinstance(n_workers, int) or n_workers <= 0:
            raise ValueError("n_workers must be a positive integer")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float, np.integer, np.floating))
            or not np.isfinite(timeout)
            or timeout <= 0.0
        ):
            raise ValueError("timeout must be finite and positive")
        self.n_workers = n_workers
        self.config = config or AggregationConfig(
            method="robust",
            robustness=0.0 if robustness is None else robustness,
            max_iter=32,
        )
        self.robustness = self.config.robustness
        self.timeout = timeout
        self.aggregator = aggregator

    def run_round(
        self,
        request_queue: Any,
        response_queues: dict,
        timeout: float | None = None,
    ) -> np.ndarray:
        """Run one federation round: collect, fuse, broadcast; return consensus.

        Blocks until ``n_workers`` serialized beliefs have arrived on
        ``request_queue``, fuses them through the configured
        :class:`~fedference.aggregation.AggregatorProtocol`, serializes the result, and pushes it to
        each contributing worker's response queue. The returned consensus is
        bit-identical to the same configured in-process aggregation call.

        Args:
            request_queue: Queue from which worker (id, belief_data) tuples are
                received. Must support ``get(timeout=...)`` raising
                ``queue.Empty`` on timeout.
            response_queues: Mapping from worker id to per-worker response queue.
            timeout: Per-get timeout in seconds. Defaults to ``self.timeout``.

        Raises:
            queue.Empty: If a worker does not respond within the timeout,
                propagated with a logged warning identifying which slot timed out.
        """
        from fedference.federation.transport import deserialize_belief, serialize_result

        t = self.timeout if timeout is None else timeout
        if (
            isinstance(t, bool)
            or not isinstance(t, (int, float, np.integer, np.floating))
            or not np.isfinite(t)
            or t <= 0.0
        ):
            raise ValueError("timeout must be finite and positive")
        expected_worker_ids = set(range(self.n_workers))
        try:
            response_worker_ids = set(response_queues)
        except (TypeError, ValueError) as exc:
            raise ValueError("response_queues must be keyed by worker id") from exc
        if response_worker_ids != expected_worker_ids:
            raise ValueError("response_queues must contain exactly one queue per worker")
        local_posteriors_by_worker_id = {}
        worker_ids = []
        for slot in range(self.n_workers):
            try:
                worker_id, data = request_queue.get(timeout=t)
            except queue.Empty:
                _log.warning(
                    "FederationServer.run_round: timed out waiting for worker "
                    "slot %d of %d after %.1fs; round result dropped.",
                    slot,
                    self.n_workers,
                    t,
                )
                raise
            if (
                isinstance(worker_id, bool)
                or not isinstance(worker_id, (int, np.integer))
                or int(worker_id) not in expected_worker_ids
            ):
                raise ValueError(f"worker id must lie in {sorted(expected_worker_ids)}")
            worker_id = int(worker_id)
            if worker_id in local_posteriors_by_worker_id:
                raise ValueError(f"duplicate worker id in federation round: {worker_id}")
            local_posteriors_by_worker_id[worker_id] = deserialize_belief(data)
            worker_ids.append(worker_id)
        local_posteriors = [
            local_posteriors_by_worker_id[worker_id]
            for worker_id in sorted(worker_ids)
        ]
        result = self.aggregator(local_posteriors, config=self.config)
        serialised = serialize_result(
            result.consensus, result.normalized_effective_weights
        )
        for wid in worker_ids:
            response_queues[wid].put(serialised)
        return result.consensus
